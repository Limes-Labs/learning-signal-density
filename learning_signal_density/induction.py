from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class MdlRule:
    fields: tuple[str, ...]
    values: tuple[str, ...]
    label: bool
    support: int
    confidence: float
    description_length_tokens: int
    validation_covered: int
    validation_correct: int
    score: float

    def matches(self, observation: Observation) -> bool:
        return all(_field_value(observation, field) == value for field, value in zip(self.fields, self.values))


@dataclass(frozen=True)
class MdlRuleSet:
    rules: tuple[MdlRule, ...]
    search_cost_tokens: int
    description_length_tokens: int
    validation_score: float

    def predict(self, observation: Observation) -> InducedPrediction | None:
        for rule in self.rules:
            if rule.matches(observation):
                return InducedPrediction(
                    label=rule.label,
                    confidence=rule.confidence,
                    support=rule.support,
                    source="mdl:" + ",".join(rule.fields),
                )
        return None


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


def fit_mdl_rule_set(
    train_observations: tuple[Observation, ...],
    validation_observations: tuple[Observation, ...],
    min_support: int = 3,
    min_confidence: float = 0.67,
    description_penalty: float = 0.08,
    max_rules: int = 12,
) -> MdlRuleSet:
    candidate_fields = (
        ("family", "stimulus", "modifier"),
        ("family", "stimulus"),
        ("modifier", "stimulus"),
        ("family", "modifier"),
        ("stimulus",),
        ("modifier",),
        ("family",),
    )
    candidates: list[MdlRule] = []
    search_cost_tokens = len(train_observations) * sum(len(fields) for fields in candidate_fields)
    for fields in candidate_fields:
        counts_by_values: dict[tuple[str, ...], Counter[bool]] = {}
        for item in train_observations:
            values = tuple(_field_value(item, field) for field in fields)
            counts_by_values.setdefault(values, Counter())[item.label] += 1
        for values, counts in counts_by_values.items():
            support = sum(counts.values())
            if support < min_support:
                continue
            label, label_count = _majority_label(counts)
            confidence = label_count / support
            if confidence < min_confidence:
                continue
            description_length = 2 + 3 * len(fields)
            covered = 0
            correct = 0
            for item in validation_observations:
                search_cost_tokens += len(fields)
                if all(_field_value(item, field) == value for field, value in zip(fields, values)):
                    covered += 1
                    if item.label == label:
                        correct += 1
            if covered == 0:
                continue
            incorrect = covered - correct
            score = correct - incorrect - description_penalty * description_length
            if score <= 0:
                continue
            candidates.append(
                MdlRule(
                    fields=fields,
                    values=values,
                    label=label,
                    support=support,
                    confidence=confidence,
                    description_length_tokens=description_length,
                    validation_covered=covered,
                    validation_correct=correct,
                    score=score,
                )
            )

    candidates.sort(
        key=lambda rule: (
            -rule.score,
            rule.description_length_tokens,
            -len(rule.fields),
            rule.values,
        )
    )
    selected: list[MdlRule] = []
    covered_validation_ids: set[str] = set()
    remaining = list(candidates)
    while len(selected) < max_rules and remaining:
        best_index: int | None = None
        best_rule: MdlRule | None = None
        best_newly_covered: set[str] = set()
        best_correct = 0
        best_score = 0.0
        for index, rule in enumerate(remaining):
            newly_covered: set[str] = set()
            correct = 0
            for item in validation_observations:
                if item.observation_id in covered_validation_ids:
                    continue
                search_cost_tokens += len(rule.fields)
                if rule.matches(item):
                    newly_covered.add(item.observation_id)
                    if item.label == rule.label:
                        correct += 1
            if not newly_covered:
                continue
            incorrect = len(newly_covered) - correct
            score = correct - incorrect - description_penalty * rule.description_length_tokens
            if score <= 0:
                continue
            is_better = (
                best_rule is None
                or score > best_score
                or (
                    score == best_score
                    and (
                        rule.description_length_tokens,
                        -len(rule.fields),
                        rule.values,
                    )
                    < (
                        best_rule.description_length_tokens,
                        -len(best_rule.fields),
                        best_rule.values,
                    )
                )
            )
            if is_better:
                best_index = index
                best_rule = rule
                best_newly_covered = newly_covered
                best_correct = correct
                best_score = score
        if best_rule is None or best_index is None:
            break
        selected.append(
            replace(
                best_rule,
                validation_covered=len(best_newly_covered),
                validation_correct=best_correct,
                score=best_score,
            )
        )
        covered_validation_ids.update(best_newly_covered)
        remaining.pop(best_index)

    description_length_tokens = sum(rule.description_length_tokens for rule in selected)
    validation_score = sum(rule.score for rule in selected)
    return MdlRuleSet(
        rules=tuple(selected),
        search_cost_tokens=search_cost_tokens,
        description_length_tokens=description_length_tokens,
        validation_score=validation_score,
    )


def _field_value(observation: Observation, field: str) -> str:
    if field == "family":
        return observation.family
    if field == "stimulus":
        return observation.stimulus
    if field == "modifier":
        return observation.modifier
    raise ValueError(f"unknown rule field: {field}")
