from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from .domain import MODIFIERS, Observation, RuleBook
from .induction import InducedRuleModel, MdlRuleSet, fit_induced_rule_model, fit_mdl_rule_set


AVAILABLE_CONDITIONS = (
    "raw_text",
    "selected_text",
    "qa_expansion",
    "induced_rule_expansion",
    "validation_gated_induction",
    "direct_validation_gated_induction",
    "validation_ranked_induction",
    "mdl_rule_expansion",
    "counterfactual_expansion",
    "prioritized_replay",
    "selected_counterfactual_replay",
)

CONDITION_SCOPE = {
    "raw_text": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "selected_text": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "qa_expansion": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "induced_rule_expansion": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "validation_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": True,
        "validation_used_for_transform_selection": True,
    },
    "direct_validation_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": True,
        "validation_used_for_transform_selection": True,
    },
    "validation_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": True,
    },
    "mdl_rule_expansion": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": True,
    },
    "counterfactual_expansion": {
        "oracle_generated_labels": True,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "prioritized_replay": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "selected_counterfactual_replay": {
        "oracle_generated_labels": True,
        "train_only_selection": True,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
}


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: bool
    pair_key: tuple[str, str]
    source_observation_id: str
    source_kind: str

    @property
    def token_count(self) -> int:
        return len(re.findall(r"[A-Za-z0-9_=.]+", self.text))


@dataclass(frozen=True)
class PipelineExamples:
    condition: str
    examples: tuple[TrainingExample, ...]
    external_event_count: int
    selection_cost_tokens: int
    modeling_cost_tokens: int
    transform_cost_tokens: int
    rule_search_cost_tokens: int
    mdl_description_length_tokens: int
    mdl_selected_rule_count: int
    mdl_validation_score: float
    candidate_ranking_cost_tokens: int
    ranked_candidate_count: int
    ranked_kept_candidate_count: int
    ranked_validation_precision: float
    ranked_synthetic_budget_ratio: float

    @property
    def internal_example_count(self) -> int:
        return len(self.examples)

    @property
    def internal_token_count(self) -> int:
        return sum(example.token_count for example in self.examples)


@dataclass(frozen=True)
class _RankedInducedCandidate:
    example: TrainingExample
    prediction: InducedPrediction
    score: float
    validation_precision: float


def raw_observation_example(observation: Observation, source_kind: str = "raw") -> TrainingExample:
    text = (
        "Observation "
        f"material={observation.material} family={observation.family} "
        f"modifier={observation.modifier} stimulus={observation.stimulus}. "
        "Predict outcome=brittle."
    )
    return TrainingExample(
        text=text,
        label=observation.label,
        pair_key=observation.pair_key,
        source_observation_id=observation.observation_id,
        source_kind=source_kind,
    )


def qa_example(observation: Observation, source_kind: str = "qa") -> TrainingExample:
    text = (
        "Question "
        f"material={observation.material} family={observation.family} "
        f"has modifier={observation.modifier} and stimulus={observation.stimulus}. "
        "Will the material become brittle?"
    )
    return TrainingExample(
        text=text,
        label=observation.label,
        pair_key=observation.pair_key,
        source_observation_id=observation.observation_id,
        source_kind=source_kind,
    )


def _counterfactual_examples(observation: Observation, rules: RuleBook) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for modifier in MODIFIERS:
        if modifier == observation.modifier:
            continue
        label = rules.label(observation.family, observation.stimulus, modifier)
        synthetic = Observation(
            observation_id=f"{observation.observation_id}-cf-{modifier}",
            material=observation.material,
            family=observation.family,
            stimulus=observation.stimulus,
            modifier=modifier,
            label=label,
        )
        examples.append(qa_example(synthetic, source_kind="counterfactual"))
    return examples


def _is_high_value(observation: Observation, salience_model: InducedRuleModel) -> bool:
    plain_prediction = salience_model.predict(observation.family, observation.stimulus, "plain")
    is_observed_exception = observation.label != plain_prediction.label and plain_prediction.support >= 2
    is_non_plain_modifier = observation.modifier != "plain"
    is_low_confidence_region = plain_prediction.confidence < 0.67
    return is_observed_exception or is_non_plain_modifier or is_low_confidence_region


def _add_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
) -> int:
    transform_cost_tokens = 0
    for item in observations:
        examples.append(raw_observation_example(item))
        question = qa_example(item)
        examples.append(question)
        transform_cost_tokens += question.token_count
        for modifier in MODIFIERS:
            if modifier == item.modifier:
                continue
            prediction = salience_model.predict(item.family, item.stimulus, modifier)
            if prediction.support < induction_min_support or prediction.confidence < induction_min_confidence:
                continue
            synthetic = Observation(
                observation_id=f"{item.observation_id}-induced-{modifier}",
                material=item.material,
                family=item.family,
                stimulus=item.stimulus,
                modifier=modifier,
                label=prediction.label,
            )
            generated = qa_example(synthetic, source_kind="induced_counterfactual")
            examples.append(generated)
            transform_cost_tokens += generated.token_count
    return transform_cost_tokens


def _validation_source_reliability(
    salience_model: InducedRuleModel,
    validation_observations: tuple[Observation, ...],
    induction_min_support: int,
    induction_min_confidence: float,
) -> tuple[dict[str, float], int]:
    covered_by_source: Counter[str] = Counter()
    correct_by_source: Counter[str] = Counter()
    scoring_cost_tokens = 0
    for item in validation_observations:
        scoring_cost_tokens += raw_observation_example(item).token_count
        prediction = salience_model.predict(item.family, item.stimulus, item.modifier)
        if prediction.support < induction_min_support or prediction.confidence < induction_min_confidence:
            continue
        covered_by_source[prediction.source] += 1
        if prediction.label == item.label:
            correct_by_source[prediction.source] += 1

    reliability: dict[str, float] = {}
    for source, covered in covered_by_source.items():
        reliability[source] = (correct_by_source[source] + 1) / (covered + 2)
    return reliability, scoring_cost_tokens


def _add_validation_ranked_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    validation_observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
    synthetic_budget_ratio: float,
) -> tuple[int, int, int, int, float]:
    transform_cost_tokens = 0
    candidate_ranking_cost_tokens = 0
    candidates: list[_RankedInducedCandidate] = []
    source_reliability, validation_scoring_cost = _validation_source_reliability(
        salience_model=salience_model,
        validation_observations=validation_observations,
        induction_min_support=induction_min_support,
        induction_min_confidence=induction_min_confidence,
    )
    candidate_ranking_cost_tokens += validation_scoring_cost

    for item in observations:
        examples.append(raw_observation_example(item))
        question = qa_example(item)
        examples.append(question)
        transform_cost_tokens += question.token_count
        item_salience_bonus = 0.05 if _is_high_value(item, salience_model) else 0.0
        for modifier in MODIFIERS:
            if modifier == item.modifier:
                continue
            prediction = salience_model.predict(item.family, item.stimulus, modifier)
            if prediction.support < induction_min_support or prediction.confidence < induction_min_confidence:
                continue
            synthetic = Observation(
                observation_id=f"{item.observation_id}-ranked-induced-{modifier}",
                material=item.material,
                family=item.family,
                stimulus=item.stimulus,
                modifier=modifier,
                label=prediction.label,
            )
            generated = qa_example(synthetic, source_kind="validation_ranked_induced")
            candidate_ranking_cost_tokens += generated.token_count
            validation_precision = source_reliability.get(prediction.source, 0.5)
            support_bonus = min(prediction.support, 8) / 16
            score = validation_precision + 0.35 * prediction.confidence + support_bonus + item_salience_bonus
            candidates.append(
                _RankedInducedCandidate(
                    example=generated,
                    prediction=prediction,
                    score=score,
                    validation_precision=validation_precision,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.validation_precision,
            -candidate.prediction.confidence,
            -candidate.prediction.support,
            candidate.example.source_observation_id,
            candidate.example.text,
        )
    )
    synthetic_budget = max(1, int(len(observations) * synthetic_budget_ratio))
    kept = candidates[:synthetic_budget]
    examples.extend(candidate.example for candidate in kept)
    transform_cost_tokens += sum(candidate.example.token_count for candidate in kept)
    ranked_validation_precision = (
        sum(candidate.validation_precision for candidate in kept) / len(kept)
        if kept
        else 0.0
    )
    return (
        transform_cost_tokens,
        candidate_ranking_cost_tokens,
        len(candidates),
        len(kept),
        ranked_validation_precision,
    )


def _add_mdl_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    rule_set: MdlRuleSet,
) -> int:
    transform_cost_tokens = 0
    for item in observations:
        examples.append(raw_observation_example(item))
        question = qa_example(item)
        examples.append(question)
        transform_cost_tokens += question.token_count
        for modifier in MODIFIERS:
            if modifier == item.modifier:
                continue
            synthetic = Observation(
                observation_id=f"{item.observation_id}-mdl-{modifier}",
                material=item.material,
                family=item.family,
                stimulus=item.stimulus,
                modifier=modifier,
                label=False,
            )
            prediction = rule_set.predict(synthetic)
            if prediction is None:
                continue
            generated = qa_example(
                Observation(
                    observation_id=synthetic.observation_id,
                    material=synthetic.material,
                    family=synthetic.family,
                    stimulus=synthetic.stimulus,
                    modifier=synthetic.modifier,
                    label=prediction.label,
                ),
                source_kind="mdl_counterfactual",
            )
            examples.append(generated)
            transform_cost_tokens += generated.token_count
    return transform_cost_tokens


def build_pipeline_examples(
    condition: str,
    observations: Iterable[Observation],
    rules: RuleBook,
    induction_min_support: int = 2,
    induction_min_confidence: float = 0.55,
    validation_observations: Iterable[Observation] | None = None,
    ranked_synthetic_budget_ratio: float = 1.0,
) -> PipelineExamples:
    if condition not in AVAILABLE_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    observations = tuple(observations)
    salience_model = fit_induced_rule_model(observations)
    selection_cost_tokens = 0
    modeling_cost_tokens = 0
    transform_cost_tokens = 0
    rule_search_cost_tokens = 0
    mdl_description_length_tokens = 0
    mdl_selected_rule_count = 0
    mdl_validation_score = 0.0
    candidate_ranking_cost_tokens = 0
    ranked_candidate_count = 0
    ranked_kept_candidate_count = 0
    ranked_validation_precision = 0.0
    exported_ranked_synthetic_budget_ratio = 0.0
    examples: list[TrainingExample] = []

    if condition == "raw_text":
        examples = [raw_observation_example(item) for item in observations]

    elif condition == "selected_text":
        selection_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        selected = [item for item in observations if _is_high_value(item, salience_model)]
        examples = [raw_observation_example(item, source_kind="selected") for item in selected]

    elif condition == "qa_expansion":
        for item in observations:
            examples.append(raw_observation_example(item))
            examples.append(qa_example(item))
            transform_cost_tokens += examples[-1].token_count

    elif condition in {"induced_rule_expansion", "validation_gated_induction", "direct_validation_gated_induction"}:
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        transform_cost_tokens = _add_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
        )

    elif condition == "validation_ranked_induction":
        if validation_observations is None:
            raise ValueError("validation_ranked_induction requires validation_observations")
        validation_observations = tuple(validation_observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = ranked_synthetic_budget_ratio
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
        ) = _add_validation_ranked_induced_examples(
            examples=examples,
            observations=observations,
            validation_observations=validation_observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
            synthetic_budget_ratio=ranked_synthetic_budget_ratio,
        )

    elif condition == "mdl_rule_expansion":
        if validation_observations is None:
            raise ValueError("mdl_rule_expansion requires validation_observations")
        validation_observations = tuple(validation_observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        rule_set = fit_mdl_rule_set(
            train_observations=observations,
            validation_observations=validation_observations,
        )
        rule_search_cost_tokens = rule_set.search_cost_tokens
        mdl_description_length_tokens = rule_set.description_length_tokens
        mdl_selected_rule_count = len(rule_set.rules)
        mdl_validation_score = rule_set.validation_score
        transform_cost_tokens = _add_mdl_examples(
            examples=examples,
            observations=observations,
            rule_set=rule_set,
        )

    elif condition == "counterfactual_expansion":
        for item in observations:
            examples.append(raw_observation_example(item))
            examples.append(qa_example(item))
            counterfactuals = _counterfactual_examples(item, rules)
            examples.extend(counterfactuals)
            transform_cost_tokens += sum(example.token_count for example in counterfactuals)

    elif condition == "prioritized_replay":
        for item in observations:
            base = raw_observation_example(item)
            examples.append(base)
            if _is_high_value(item, salience_model):
                examples.extend([qa_example(item, source_kind="replay-hard") for _ in range(3)])
            else:
                examples.append(qa_example(item, source_kind="replay"))
        transform_cost_tokens = sum(example.token_count for example in examples if example.source_kind.startswith("replay"))

    elif condition == "selected_counterfactual_replay":
        selection_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        selected = [item for item in observations if _is_high_value(item, salience_model)]
        for item in selected:
            examples.append(raw_observation_example(item, source_kind="selected"))
            counterfactuals = _counterfactual_examples(item, rules)
            examples.extend(counterfactuals)
            if _is_high_value(item, salience_model):
                examples.append(qa_example(item, source_kind="replay-hard"))
            transform_cost_tokens += sum(example.token_count for example in counterfactuals)

    return PipelineExamples(
        condition=condition,
        examples=tuple(examples),
        external_event_count=len(observations),
        selection_cost_tokens=selection_cost_tokens,
        modeling_cost_tokens=modeling_cost_tokens,
        transform_cost_tokens=transform_cost_tokens,
        rule_search_cost_tokens=rule_search_cost_tokens,
        mdl_description_length_tokens=mdl_description_length_tokens,
        mdl_selected_rule_count=mdl_selected_rule_count,
        mdl_validation_score=mdl_validation_score,
        candidate_ranking_cost_tokens=candidate_ranking_cost_tokens,
        ranked_candidate_count=ranked_candidate_count,
        ranked_kept_candidate_count=ranked_kept_candidate_count,
        ranked_validation_precision=ranked_validation_precision,
        ranked_synthetic_budget_ratio=exported_ranked_synthetic_budget_ratio,
    )


def build_evaluation_examples(observations: Iterable[Observation]) -> tuple[TrainingExample, ...]:
    return tuple(qa_example(item, source_kind="heldout-eval") for item in observations)
