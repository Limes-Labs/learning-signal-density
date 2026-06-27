from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .domain import MODIFIERS, Observation, RuleBook
from .induction import InducedRuleModel, fit_induced_rule_model


AVAILABLE_CONDITIONS = (
    "raw_text",
    "selected_text",
    "qa_expansion",
    "induced_rule_expansion",
    "validation_gated_induction",
    "direct_validation_gated_induction",
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
    },
    "selected_text": {
        "oracle_generated_labels": False,
        "train_only_selection": True,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
    },
    "qa_expansion": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
    },
    "induced_rule_expansion": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": False,
    },
    "validation_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": True,
    },
    "direct_validation_gated_induction": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": True,
        "validation_used_for_threshold": True,
    },
    "counterfactual_expansion": {
        "oracle_generated_labels": True,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
    },
    "prioritized_replay": {
        "oracle_generated_labels": False,
        "train_only_selection": False,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
    },
    "selected_counterfactual_replay": {
        "oracle_generated_labels": True,
        "train_only_selection": True,
        "train_only_induction": False,
        "validation_used_for_threshold": False,
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

    @property
    def internal_example_count(self) -> int:
        return len(self.examples)

    @property
    def internal_token_count(self) -> int:
        return sum(example.token_count for example in self.examples)


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


def build_pipeline_examples(
    condition: str,
    observations: Iterable[Observation],
    rules: RuleBook,
    induction_min_support: int = 2,
    induction_min_confidence: float = 0.55,
) -> PipelineExamples:
    if condition not in AVAILABLE_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    observations = tuple(observations)
    salience_model = fit_induced_rule_model(observations)
    selection_cost_tokens = 0
    modeling_cost_tokens = 0
    transform_cost_tokens = 0
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
    )


def build_evaluation_examples(observations: Iterable[Observation]) -> tuple[TrainingExample, ...]:
    return tuple(qa_example(item, source_kind="heldout-eval") for item in observations)
