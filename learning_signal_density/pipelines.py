from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .domain import MODIFIERS, Observation, RuleBook


AVAILABLE_CONDITIONS = (
    "raw_text",
    "selected_text",
    "qa_expansion",
    "counterfactual_expansion",
    "prioritized_replay",
    "selected_counterfactual_replay",
)


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


def _is_high_value(observation: Observation, rules: RuleBook) -> bool:
    key = (observation.family, observation.stimulus)
    base = rules.base_effects[key]
    is_exception = observation.label != base
    is_rare_negative = not observation.label and base
    return is_exception or is_rare_negative or observation.modifier != "plain"


def build_pipeline_examples(
    condition: str,
    observations: Iterable[Observation],
    rules: RuleBook,
) -> PipelineExamples:
    if condition not in AVAILABLE_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    observations = tuple(observations)
    selection_cost_tokens = 0
    transform_cost_tokens = 0
    examples: list[TrainingExample] = []

    if condition == "raw_text":
        examples = [raw_observation_example(item) for item in observations]

    elif condition == "selected_text":
        selection_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        selected = [item for item in observations if _is_high_value(item, rules)]
        examples = [raw_observation_example(item, source_kind="selected") for item in selected]

    elif condition == "qa_expansion":
        for item in observations:
            examples.append(raw_observation_example(item))
            examples.append(qa_example(item))
            transform_cost_tokens += examples[-1].token_count

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
            if _is_high_value(item, rules):
                examples.extend([qa_example(item, source_kind="replay-hard") for _ in range(3)])
            else:
                examples.append(qa_example(item, source_kind="replay"))
        transform_cost_tokens = sum(example.token_count for example in examples if example.source_kind.startswith("replay"))

    elif condition == "selected_counterfactual_replay":
        selection_cost_tokens = sum(raw_observation_example(item).token_count for item in observations)
        selected = [item for item in observations if _is_high_value(item, rules)]
        for item in selected:
            examples.append(raw_observation_example(item, source_kind="selected"))
            counterfactuals = _counterfactual_examples(item, rules)
            examples.extend(counterfactuals)
            if _is_high_value(item, rules):
                examples.append(qa_example(item, source_kind="replay-hard"))
            transform_cost_tokens += sum(example.token_count for example in counterfactuals)

    return PipelineExamples(
        condition=condition,
        examples=tuple(examples),
        external_event_count=len(observations),
        selection_cost_tokens=selection_cost_tokens,
        transform_cost_tokens=transform_cost_tokens,
    )


def build_evaluation_examples(observations: Iterable[Observation]) -> tuple[TrainingExample, ...]:
    return tuple(qa_example(item, source_kind="heldout-eval") for item in observations)
