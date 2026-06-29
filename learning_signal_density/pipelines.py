from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from .domain import MODIFIERS, Observation, RuleBook
from .induction import InducedPrediction, InducedRuleModel, MdlRuleSet, fit_induced_rule_model, fit_mdl_rule_set


AVAILABLE_CONDITIONS = (
    "raw_text",
    "selected_text",
    "qa_expansion",
    "induced_rule_expansion",
    "validation_gated_induction",
    "direct_validation_gated_induction",
    "validation_ranked_induction",
    "train_calibrated_ranked_induction",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "sample_aware_diverse_self_ranked_induction",
    "tempered_sample_aware_self_ranked_induction",
    "train_size_gated_sample_aware_induction",
    "compact_train_size_gated_induction",
    "compact_diverse_train_size_gated_induction",
    "support_ramped_compact_induction",
    "late_confidence_ramped_compact_induction",
    "density_capped_compact_induction",
    "agreement_gated_self_ranked_induction",
    "diverse_self_ranked_induction",
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
    "train_calibrated_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "sample_aware_self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "sample_aware_diverse_self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "tempered_sample_aware_self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "train_size_gated_sample_aware_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
    },
    "compact_train_size_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
        "compact_original_encoding_at_large_samples": True,
    },
    "compact_diverse_train_size_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
        "compact_original_encoding_at_large_samples": True,
        "diversity_penalty_after_compaction": True,
    },
    "support_ramped_compact_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
        "compact_original_encoding_at_large_samples": True,
        "abundant_data_support_ramp": True,
        "abundant_data_support_ramp_min_events": 360,
        "abundant_data_min_support": 4,
    },
    "late_confidence_ramped_compact_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
        "compact_original_encoding_at_large_samples": True,
        "abundant_data_support_ramp": True,
        "abundant_data_support_ramp_min_events": 360,
        "abundant_data_min_support": 4,
        "late_confidence_ramp_min_events": 432,
        "late_confidence_min_confidence": 0.60,
    },
    "density_capped_compact_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
        "validation_used_for_policy_selection": False,
        "compact_original_encoding_at_large_samples": True,
        "abundant_data_raw_fallback": True,
    },
    "agreement_gated_self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
    },
    "diverse_self_ranked_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
        "validation_used_for_transform_selection": False,
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

TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS = 144
COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS = 224
SUPPORT_RAMPED_COMPACT_MIN_EVENTS = 360
SUPPORT_RAMPED_COMPACT_MIN_SUPPORT = 4
LATE_CONFIDENCE_RAMPED_COMPACT_MIN_EVENTS = 432
LATE_CONFIDENCE_RAMPED_COMPACT_MIN_CONFIDENCE = 0.60
DENSITY_CAPPED_COMPACT_RAW_MIN_EVENTS = 360


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
    ranked_induction_min_support: int
    ranked_induction_min_confidence: float
    train_calibration_event_count: int
    validation_calibration_event_count: int
    ranked_diversity_penalty: float
    ranked_unique_modifier_count: int
    ranked_max_modifier_count: int

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
    family: str
    stimulus: str
    modifier: str


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


def _source_reliability(
    salience_model: InducedRuleModel,
    calibration_observations: tuple[Observation, ...],
    induction_min_support: int,
    induction_min_confidence: float,
) -> tuple[dict[str, float], int]:
    covered_by_source: Counter[str] = Counter()
    correct_by_source: Counter[str] = Counter()
    scoring_cost_tokens = 0
    for item in calibration_observations:
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


def _add_ranked_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
    synthetic_budget_ratio: float,
    source_reliability: dict[str, float],
    base_ranking_cost_tokens: int,
    source_kind: str,
    diversity_penalty: float,
    include_original_qa: bool = True,
) -> tuple[int, int, int, int, float, int, int, float]:
    transform_cost_tokens = 0
    candidate_ranking_cost_tokens = base_ranking_cost_tokens
    candidates: list[_RankedInducedCandidate] = []

    for item in observations:
        examples.append(raw_observation_example(item))
        if include_original_qa:
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
            generated = qa_example(synthetic, source_kind=source_kind)
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
                    family=synthetic.family,
                    stimulus=synthetic.stimulus,
                    modifier=synthetic.modifier,
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
    if diversity_penalty <= 0:
        kept = candidates[:synthetic_budget]
    else:
        kept = []
        modifier_counts: Counter[str] = Counter()
        stimulus_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        remaining = list(candidates)
        while remaining and len(kept) < synthetic_budget:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    remaining[index].score
                    - diversity_penalty
                    * (
                        modifier_counts[remaining[index].modifier]
                        + 0.4 * stimulus_counts[remaining[index].stimulus]
                        + 0.25 * family_counts[remaining[index].family]
                    ),
                    remaining[index].validation_precision,
                    remaining[index].prediction.confidence,
                    remaining[index].prediction.support,
                    remaining[index].example.source_observation_id,
                    remaining[index].example.text,
                ),
            )
            candidate = remaining.pop(best_index)
            kept.append(candidate)
            modifier_counts[candidate.modifier] += 1
            stimulus_counts[candidate.stimulus] += 1
            family_counts[candidate.family] += 1
    examples.extend(candidate.example for candidate in kept)
    transform_cost_tokens += sum(candidate.example.token_count for candidate in kept)
    ranked_validation_precision = (
        sum(candidate.validation_precision for candidate in kept) / len(kept)
        if kept
        else 0.0
    )
    kept_modifier_counts = Counter(candidate.modifier for candidate in kept)
    return (
        transform_cost_tokens,
        candidate_ranking_cost_tokens,
        len(candidates),
        len(kept),
        ranked_validation_precision,
        len(kept_modifier_counts),
        max(kept_modifier_counts.values(), default=0),
        diversity_penalty,
    )


def _prediction_from_counts(source: str, counts: Counter[bool]) -> InducedPrediction | None:
    support = sum(counts.values())
    if not support:
        return None
    true_count = counts[True]
    false_count = counts[False]
    label = true_count >= false_count
    label_count = true_count if label else false_count
    return InducedPrediction(
        label=label,
        confidence=label_count / support,
        support=support,
        source=source,
    )


def _source_projection_predictions(
    salience_model: InducedRuleModel,
    family: str,
    stimulus: str,
    modifier: str,
) -> tuple[InducedPrediction, ...]:
    predictions = [
        _prediction_from_counts(
            "exact",
            salience_model.exact_counts.get((family, stimulus, modifier), Counter()),
        ),
        _prediction_from_counts(
            "family_stimulus",
            salience_model.pair_counts.get((family, stimulus), Counter()),
        ),
        _prediction_from_counts(
            "modifier_stimulus",
            salience_model.modifier_stimulus_counts.get((modifier, stimulus), Counter()),
        ),
        _prediction_from_counts(
            "family_modifier",
            salience_model.family_modifier_counts.get((family, modifier), Counter()),
        ),
    ]
    return tuple(prediction for prediction in predictions if prediction is not None)


def _agreement_prediction(
    salience_model: InducedRuleModel,
    family: str,
    stimulus: str,
    modifier: str,
    induction_min_support: int,
    induction_min_confidence: float,
) -> InducedPrediction | None:
    eligible = tuple(
        prediction
        for prediction in _source_projection_predictions(salience_model, family, stimulus, modifier)
        if prediction.support >= induction_min_support
        and prediction.confidence >= induction_min_confidence
        and prediction.source != "exact"
    )
    if len(eligible) < 2:
        return None
    labels = {prediction.label for prediction in eligible}
    if len(labels) != 1:
        return None
    support = sum(prediction.support for prediction in eligible)
    confidence = sum(prediction.confidence * prediction.support for prediction in eligible) / support
    return InducedPrediction(
        label=eligible[0].label,
        confidence=confidence,
        support=support,
        source="agreement:" + "+".join(prediction.source for prediction in eligible),
    )


def _add_agreement_gated_ranked_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
    synthetic_budget_ratio: float,
) -> tuple[int, int, int, int, float, int, int, float]:
    transform_cost_tokens = 0
    candidate_ranking_cost_tokens = 0
    candidates: list[_RankedInducedCandidate] = []

    for item in observations:
        examples.append(raw_observation_example(item))
        question = qa_example(item)
        examples.append(question)
        transform_cost_tokens += question.token_count
        item_salience_bonus = 0.05 if _is_high_value(item, salience_model) else 0.0
        for modifier in MODIFIERS:
            if modifier == item.modifier:
                continue
            prediction = _agreement_prediction(
                salience_model=salience_model,
                family=item.family,
                stimulus=item.stimulus,
                modifier=modifier,
                induction_min_support=induction_min_support,
                induction_min_confidence=induction_min_confidence,
            )
            if prediction is None:
                continue
            synthetic = Observation(
                observation_id=f"{item.observation_id}-agreement-induced-{modifier}",
                material=item.material,
                family=item.family,
                stimulus=item.stimulus,
                modifier=modifier,
                label=prediction.label,
            )
            generated = qa_example(synthetic, source_kind="agreement_gated_self_ranked_induced")
            candidate_ranking_cost_tokens += generated.token_count
            support_bonus = min(prediction.support, 16) / 32
            score = prediction.confidence + support_bonus + item_salience_bonus
            candidates.append(
                _RankedInducedCandidate(
                    example=generated,
                    prediction=prediction,
                    score=score,
                    validation_precision=prediction.confidence,
                    family=synthetic.family,
                    stimulus=synthetic.stimulus,
                    modifier=synthetic.modifier,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
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
    kept_modifier_counts = Counter(candidate.modifier for candidate in kept)
    return (
        transform_cost_tokens,
        candidate_ranking_cost_tokens,
        len(candidates),
        len(kept),
        ranked_validation_precision,
        len(kept_modifier_counts),
        max(kept_modifier_counts.values(), default=0),
        0.0,
    )


def _add_validation_ranked_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    validation_observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
    synthetic_budget_ratio: float,
) -> tuple[int, int, int, int, float, int, int, float]:
    source_reliability, validation_scoring_cost = _source_reliability(
        salience_model=salience_model,
        calibration_observations=validation_observations,
        induction_min_support=induction_min_support,
        induction_min_confidence=induction_min_confidence,
    )
    return _add_ranked_induced_examples(
        examples=examples,
        observations=observations,
        salience_model=salience_model,
        induction_min_support=induction_min_support,
        induction_min_confidence=induction_min_confidence,
        synthetic_budget_ratio=synthetic_budget_ratio,
        source_reliability=source_reliability,
        base_ranking_cost_tokens=validation_scoring_cost,
        source_kind="validation_ranked_induced",
        diversity_penalty=0.0,
    )


def _train_calibration_split(observations: tuple[Observation, ...], calibration_fraction: float = 0.2) -> tuple[tuple[Observation, ...], tuple[Observation, ...]]:
    if len(observations) < 2:
        return observations, ()
    calibration_count = min(len(observations) - 1, max(1, round(len(observations) * calibration_fraction)))
    calibration_ids = {
        item.observation_id
        for index, item in enumerate(sorted(observations, key=lambda item: item.observation_id))
        if index < calibration_count
    }
    calibration = tuple(item for item in observations if item.observation_id in calibration_ids)
    induction = tuple(item for item in observations if item.observation_id not in calibration_ids)
    return induction, calibration


def _add_train_calibrated_ranked_induced_examples(
    examples: list[TrainingExample],
    observations: tuple[Observation, ...],
    salience_model: InducedRuleModel,
    induction_min_support: int,
    induction_min_confidence: float,
    synthetic_budget_ratio: float,
) -> tuple[int, int, int, int, float, int, int, float, int]:
    induction_observations, calibration_observations = _train_calibration_split(observations)
    calibration_model = fit_induced_rule_model(induction_observations)
    calibration_modeling_cost = sum(raw_observation_example(item).token_count for item in induction_observations)
    source_reliability, calibration_scoring_cost = _source_reliability(
        salience_model=calibration_model,
        calibration_observations=calibration_observations,
        induction_min_support=induction_min_support,
        induction_min_confidence=induction_min_confidence,
    )
    (
        transform_cost_tokens,
        candidate_ranking_cost_tokens,
        ranked_candidate_count,
        ranked_kept_candidate_count,
        ranked_validation_precision,
        ranked_unique_modifier_count,
        ranked_max_modifier_count,
        ranked_diversity_penalty,
    ) = _add_ranked_induced_examples(
        examples=examples,
        observations=observations,
        salience_model=salience_model,
        induction_min_support=induction_min_support,
        induction_min_confidence=induction_min_confidence,
        synthetic_budget_ratio=synthetic_budget_ratio,
        source_reliability=source_reliability,
        base_ranking_cost_tokens=calibration_modeling_cost + calibration_scoring_cost,
        source_kind="train_calibrated_ranked_induced",
        diversity_penalty=0.0,
    )
    return (
        transform_cost_tokens,
        candidate_ranking_cost_tokens,
        ranked_candidate_count,
        ranked_kept_candidate_count,
        ranked_validation_precision,
        ranked_unique_modifier_count,
        ranked_max_modifier_count,
        ranked_diversity_penalty,
        len(calibration_observations),
    )


def _sample_aware_ranked_policy(observations: tuple[Observation, ...]) -> tuple[int, float, float]:
    external_events = len(observations)
    if external_events < 72:
        budget_ratio = 0.25
    elif external_events < 144:
        budget_ratio = 0.75
    else:
        budget_ratio = 1.0
    min_support = 3 if external_events >= 224 else 2
    return min_support, 0.55, budget_ratio


def _support_ramped_compact_ranked_policy(observations: tuple[Observation, ...]) -> tuple[int, float, float]:
    min_support, min_confidence, budget_ratio = _sample_aware_ranked_policy(observations)
    if len(observations) >= SUPPORT_RAMPED_COMPACT_MIN_EVENTS:
        min_support = max(min_support, SUPPORT_RAMPED_COMPACT_MIN_SUPPORT)
    return min_support, min_confidence, budget_ratio


def _late_confidence_ramped_compact_ranked_policy(
    observations: tuple[Observation, ...],
) -> tuple[int, float, float]:
    min_support, min_confidence, budget_ratio = _support_ramped_compact_ranked_policy(observations)
    if len(observations) >= LATE_CONFIDENCE_RAMPED_COMPACT_MIN_EVENTS:
        min_confidence = max(min_confidence, LATE_CONFIDENCE_RAMPED_COMPACT_MIN_CONFIDENCE)
    return min_support, min_confidence, budget_ratio


def _tempered_sample_aware_ranked_policy(observations: tuple[Observation, ...]) -> tuple[int, float, float]:
    external_events = len(observations)
    if external_events < 72:
        budget_ratio = 0.25
    elif external_events < 144:
        budget_ratio = 0.5
    else:
        budget_ratio = 1.0
    min_support = 3 if external_events >= 224 else 2
    return min_support, 0.55, budget_ratio


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
    ranked_induction_min_support = 0
    ranked_induction_min_confidence = 0.0
    train_calibration_event_count = 0
    validation_calibration_event_count = 0
    ranked_diversity_penalty = 0.0
    ranked_unique_modifier_count = 0
    ranked_max_modifier_count = 0
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
        validation_calibration_event_count = len(validation_observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = ranked_synthetic_budget_ratio
        ranked_induction_min_support = induction_min_support
        ranked_induction_min_confidence = induction_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_validation_ranked_induced_examples(
            examples=examples,
            observations=observations,
            validation_observations=validation_observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
            synthetic_budget_ratio=ranked_synthetic_budget_ratio,
        )

    elif condition == "train_calibrated_ranked_induction":
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = ranked_synthetic_budget_ratio
        ranked_induction_min_support = induction_min_support
        ranked_induction_min_confidence = induction_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
            train_calibration_event_count,
        ) = _add_train_calibrated_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
            synthetic_budget_ratio=ranked_synthetic_budget_ratio,
        )

    elif condition == "self_ranked_induction":
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = ranked_synthetic_budget_ratio
        ranked_induction_min_support = induction_min_support
        ranked_induction_min_confidence = induction_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
            synthetic_budget_ratio=ranked_synthetic_budget_ratio,
            source_reliability={},
            base_ranking_cost_tokens=0,
            source_kind="self_ranked_induced",
            diversity_penalty=0.0,
        )

    elif condition == "sample_aware_self_ranked_induction":
        (
            effective_min_support,
            effective_min_confidence,
            effective_budget_ratio,
        ) = _sample_aware_ranked_policy(observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = effective_budget_ratio
        ranked_induction_min_support = effective_min_support
        ranked_induction_min_confidence = effective_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=effective_min_support,
            induction_min_confidence=effective_min_confidence,
            synthetic_budget_ratio=effective_budget_ratio,
            source_reliability={},
            base_ranking_cost_tokens=0,
            source_kind="sample_aware_self_ranked_induced",
            diversity_penalty=0.0,
        )

    elif condition == "sample_aware_diverse_self_ranked_induction":
        (
            effective_min_support,
            effective_min_confidence,
            effective_budget_ratio,
        ) = _sample_aware_ranked_policy(observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = effective_budget_ratio
        ranked_induction_min_support = effective_min_support
        ranked_induction_min_confidence = effective_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=effective_min_support,
            induction_min_confidence=effective_min_confidence,
            synthetic_budget_ratio=effective_budget_ratio,
            source_reliability={},
            base_ranking_cost_tokens=0,
            source_kind="sample_aware_diverse_self_ranked_induced",
            diversity_penalty=0.18,
        )

    elif condition == "tempered_sample_aware_self_ranked_induction":
        (
            effective_min_support,
            effective_min_confidence,
            effective_budget_ratio,
        ) = _tempered_sample_aware_ranked_policy(observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = effective_budget_ratio
        ranked_induction_min_support = effective_min_support
        ranked_induction_min_confidence = effective_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=effective_min_support,
            induction_min_confidence=effective_min_confidence,
            synthetic_budget_ratio=effective_budget_ratio,
            source_reliability={},
            base_ranking_cost_tokens=0,
            source_kind="tempered_sample_aware_self_ranked_induced",
            diversity_penalty=0.0,
        )

    elif condition == "train_size_gated_sample_aware_induction":
        if len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS:
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _sample_aware_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind="sample_aware_self_ranked_induced",
                diversity_penalty=0.0,
            )

    elif condition == "compact_train_size_gated_induction":
        if len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS:
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _sample_aware_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            include_original_qa = len(observations) < COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS
            source_kind = (
                "sample_aware_self_ranked_induced"
                if include_original_qa
                else "compact_sample_aware_self_ranked_induced"
            )
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind=source_kind,
                diversity_penalty=0.0,
                include_original_qa=include_original_qa,
            )

    elif condition == "compact_diverse_train_size_gated_induction":
        if len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS:
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _sample_aware_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            include_original_qa = len(observations) < COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS
            source_kind = (
                "sample_aware_self_ranked_induced"
                if include_original_qa
                else "compact_diverse_sample_aware_self_ranked_induced"
            )
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind=source_kind,
                diversity_penalty=0.0 if include_original_qa else 0.18,
                include_original_qa=include_original_qa,
            )

    elif condition == "support_ramped_compact_induction":
        if len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS:
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _support_ramped_compact_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            include_original_qa = len(observations) < COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS
            if include_original_qa:
                source_kind = "sample_aware_self_ranked_induced"
            elif len(observations) >= SUPPORT_RAMPED_COMPACT_MIN_EVENTS:
                source_kind = "support_ramped_compact_sample_aware_self_ranked_induced"
            else:
                source_kind = "compact_sample_aware_self_ranked_induced"
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind=source_kind,
                diversity_penalty=0.0,
                include_original_qa=include_original_qa,
            )

    elif condition == "late_confidence_ramped_compact_induction":
        if len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS:
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _late_confidence_ramped_compact_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            include_original_qa = len(observations) < COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS
            if include_original_qa:
                source_kind = "sample_aware_self_ranked_induced"
            elif len(observations) >= LATE_CONFIDENCE_RAMPED_COMPACT_MIN_EVENTS:
                source_kind = "late_confidence_ramped_compact_sample_aware_self_ranked_induced"
            elif len(observations) >= SUPPORT_RAMPED_COMPACT_MIN_EVENTS:
                source_kind = "support_ramped_compact_sample_aware_self_ranked_induced"
            else:
                source_kind = "compact_sample_aware_self_ranked_induced"
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind=source_kind,
                diversity_penalty=0.0,
                include_original_qa=include_original_qa,
            )

    elif condition == "density_capped_compact_induction":
        if (
            len(observations) < TRAIN_SIZE_GATED_SAMPLE_AWARE_MIN_EVENTS
            or len(observations) >= DENSITY_CAPPED_COMPACT_RAW_MIN_EVENTS
        ):
            examples = [raw_observation_example(item) for item in observations]
        else:
            (
                effective_min_support,
                effective_min_confidence,
                effective_budget_ratio,
            ) = _sample_aware_ranked_policy(observations)
            modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
            exported_ranked_synthetic_budget_ratio = effective_budget_ratio
            ranked_induction_min_support = effective_min_support
            ranked_induction_min_confidence = effective_min_confidence
            include_original_qa = len(observations) < COMPACT_TRAIN_SIZE_GATED_MIN_EVENTS
            source_kind = (
                "sample_aware_self_ranked_induced"
                if include_original_qa
                else "compact_sample_aware_self_ranked_induced"
            )
            (
                transform_cost_tokens,
                candidate_ranking_cost_tokens,
                ranked_candidate_count,
                ranked_kept_candidate_count,
                ranked_validation_precision,
                ranked_unique_modifier_count,
                ranked_max_modifier_count,
                ranked_diversity_penalty,
            ) = _add_ranked_induced_examples(
                examples=examples,
                observations=observations,
                salience_model=salience_model,
                induction_min_support=effective_min_support,
                induction_min_confidence=effective_min_confidence,
                synthetic_budget_ratio=effective_budget_ratio,
                source_reliability={},
                base_ranking_cost_tokens=0,
                source_kind=source_kind,
                diversity_penalty=0.0,
                include_original_qa=include_original_qa,
            )

    elif condition == "agreement_gated_self_ranked_induction":
        (
            effective_min_support,
            effective_min_confidence,
            effective_budget_ratio,
        ) = _sample_aware_ranked_policy(observations)
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = effective_budget_ratio
        ranked_induction_min_support = effective_min_support
        ranked_induction_min_confidence = effective_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_agreement_gated_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=effective_min_support,
            induction_min_confidence=effective_min_confidence,
            synthetic_budget_ratio=effective_budget_ratio,
        )

    elif condition == "diverse_self_ranked_induction":
        modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        exported_ranked_synthetic_budget_ratio = ranked_synthetic_budget_ratio
        ranked_induction_min_support = induction_min_support
        ranked_induction_min_confidence = induction_min_confidence
        (
            transform_cost_tokens,
            candidate_ranking_cost_tokens,
            ranked_candidate_count,
            ranked_kept_candidate_count,
            ranked_validation_precision,
            ranked_unique_modifier_count,
            ranked_max_modifier_count,
            ranked_diversity_penalty,
        ) = _add_ranked_induced_examples(
            examples=examples,
            observations=observations,
            salience_model=salience_model,
            induction_min_support=induction_min_support,
            induction_min_confidence=induction_min_confidence,
            synthetic_budget_ratio=ranked_synthetic_budget_ratio,
            source_reliability={},
            base_ranking_cost_tokens=0,
            source_kind="diverse_self_ranked_induced",
            diversity_penalty=0.18,
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
        ranked_induction_min_support=ranked_induction_min_support,
        ranked_induction_min_confidence=ranked_induction_min_confidence,
        train_calibration_event_count=train_calibration_event_count,
        validation_calibration_event_count=validation_calibration_event_count,
        ranked_diversity_penalty=ranked_diversity_penalty,
        ranked_unique_modifier_count=ranked_unique_modifier_count,
        ranked_max_modifier_count=ranked_max_modifier_count,
    )


def build_evaluation_examples(observations: Iterable[Observation]) -> tuple[TrainingExample, ...]:
    return tuple(qa_example(item, source_kind="heldout-eval") for item in observations)
