from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .domain import Observation


@dataclass(frozen=True)
class InducedPrediction:
    label: bool
    confidence: float
    support: int
    source: str


@dataclass(frozen=True)
class InducedRuleModel:
    exact_counts: dict[tuple[str, str, str], Counter[bool]]
    pair_counts: dict[tuple[str, str], Counter[bool]]
    modifier_stimulus_counts: dict[tuple[str, str], Counter[bool]]
    family_modifier_counts: dict[tuple[str, str], Counter[bool]]
    global_counts: Counter[bool]

    def predict(self, family: str, stimulus: str, modifier: str) -> InducedPrediction:
        candidates = (
            ("exact", self.exact_counts.get((family, stimulus, modifier), Counter())),
            ("family_stimulus", self.pair_counts.get((family, stimulus), Counter())),
            ("modifier_stimulus", self.modifier_stimulus_counts.get((modifier, stimulus), Counter())),
            ("family_modifier", self.family_modifier_counts.get((family, modifier), Counter())),
            ("global", self.global_counts),
        )
        for source, counts in candidates:
            support = sum(counts.values())
            if support:
                label, count = _majority_label(counts)
                return InducedPrediction(
                    label=label,
                    confidence=count / support,
                    support=support,
                    source=source,
                )
        return InducedPrediction(label=False, confidence=0.0, support=0, source="empty")


def _majority_label(counts: Counter[bool]) -> tuple[bool, int]:
    true_count = counts[True]
    false_count = counts[False]
    if true_count >= false_count:
        return True, true_count
    return False, false_count


def fit_induced_rule_model(observations: tuple[Observation, ...]) -> InducedRuleModel:
    exact_counts: dict[tuple[str, str, str], Counter[bool]] = {}
    pair_counts: dict[tuple[str, str], Counter[bool]] = {}
    modifier_stimulus_counts: dict[tuple[str, str], Counter[bool]] = {}
    family_modifier_counts: dict[tuple[str, str], Counter[bool]] = {}
    global_counts: Counter[bool] = Counter()

    for item in observations:
        exact_counts.setdefault((item.family, item.stimulus, item.modifier), Counter())[item.label] += 1
        pair_counts.setdefault((item.family, item.stimulus), Counter())[item.label] += 1
        modifier_stimulus_counts.setdefault((item.modifier, item.stimulus), Counter())[item.label] += 1
        family_modifier_counts.setdefault((item.family, item.modifier), Counter())[item.label] += 1
        global_counts[item.label] += 1

    return InducedRuleModel(
        exact_counts=exact_counts,
        pair_counts=pair_counts,
        modifier_stimulus_counts=modifier_stimulus_counts,
        family_modifier_counts=family_modifier_counts,
        global_counts=global_counts,
    )
