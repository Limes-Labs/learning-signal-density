from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import random
from statistics import mean
from urllib.request import Request, urlopen
import zipfile

from .learner import PerceptronClassifier
from .pipelines import TrainingExample


UCI_SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
UCI_SMS_SPAM_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
DEFAULT_CACHE_PATH = Path("data/external/sms_spam_collection.zip")
DEFAULT_REAL_TEXT_SEEDS = (211, 223, 227, 229, 233)
DEFAULT_TRAIN_BUDGETS = (32, 64, 128, 256, 512)
DEFAULT_REAL_TEXT_CONDITIONS = (
    "random_sample",
    "class_balanced_sample",
    "label_index_balanced_sample",
    "validation_sample_selector",
    "validation_label_index_selector",
)

SMS_SPAM_DATASET = {
    "name": "UCI SMS Spam Collection",
    "uci_id": 228,
    "url": UCI_SMS_SPAM_URL,
    "doi": "10.24432/C5CC84",
    "license": "CC BY 4.0",
    "sha256": UCI_SMS_SPAM_SHA256,
    "task": "binary text classification",
    "positive_label": "spam",
    "negative_label": "ham",
}

REAL_TEXT_CONDITION_SCOPE = {
    "random_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "label_aware_sampling": False,
    },
    "class_balanced_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "label_aware_sampling": True,
        "selection_cost_model": "full_train_pool_text_scan",
    },
    "label_index_balanced_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "label_aware_sampling": True,
        "selection_cost_model": "one_unit_per_train_pool_label",
    },
    "validation_sample_selector": {
        "real_dataset": True,
        "train_only_selection": True,
        "validation_used_for_policy_selection": True,
        "heldout_used_for_selection": False,
        "label_aware_sampling": True,
        "selection_cost_model": "full_train_pool_text_scan",
    },
    "validation_label_index_selector": {
        "real_dataset": True,
        "train_only_selection": True,
        "validation_used_for_policy_selection": True,
        "heldout_used_for_selection": False,
        "label_aware_sampling": True,
        "selection_cost_model": "one_unit_per_train_pool_label",
    },
}


def _round(value: float) -> float:
    return round(value, 6)


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    label: bool
    text: str

    @property
    def token_count(self) -> int:
        return _to_training_example(self).token_count


@dataclass(frozen=True)
class RealTextSplit:
    train_pool: tuple[TextRecord, ...]
    validation: tuple[TextRecord, ...]
    heldout: tuple[TextRecord, ...]


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    correct: int
    total: int
    spam_precision: float
    spam_recall: float
    spam_f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


@dataclass(frozen=True)
class _CandidateSample:
    policy: str
    records: tuple[TextRecord, ...]
    selection_cost_tokens: int

    @property
    def token_count(self) -> int:
        return sum(record.token_count for record in self.records)


def parse_sms_spam_collection_zip(payload: bytes) -> tuple[TextRecord, ...]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        collection_name = next(
            (name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"),
            None,
        )
        if collection_name is None:
            raise ValueError("SMS Spam Collection zip does not contain SMSSpamCollection")
        raw = archive.read(collection_name).decode("utf-8", errors="replace")

    records: list[TextRecord] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            label_name, message = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"malformed SMS Spam Collection line: {line!r}") from exc
        if label_name not in {"ham", "spam"}:
            raise ValueError(f"unknown SMS Spam Collection label: {label_name!r}")
        records.append(
            TextRecord(
                record_id=f"sms-{len(records) + 1:06d}",
                label=label_name == "spam",
                text=message,
            )
        )
    if not records:
        raise ValueError("SMS Spam Collection archive is empty")
    return tuple(records)


def load_sms_spam_records(cache_path: Path = DEFAULT_CACHE_PATH, timeout: int = 60) -> tuple[TextRecord, ...]:
    payload = _load_or_download_sms_spam_zip(cache_path=cache_path, timeout=timeout)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != UCI_SMS_SPAM_SHA256:
        raise ValueError(f"unexpected SMS Spam Collection sha256 {digest}")
    return parse_sms_spam_collection_zip(payload)


def _load_or_download_sms_spam_zip(cache_path: Path, timeout: int) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(UCI_SMS_SPAM_URL, headers={"User-Agent": "learning-signal-density-research/0.1"})
    payload = urlopen(request, timeout=timeout).read()
    cache_path.write_bytes(payload)
    return payload


def stratified_split(
    records: tuple[TextRecord, ...],
    seed: int,
    validation_size: int,
    heldout_size: int,
) -> RealTextSplit:
    if validation_size <= 0 or heldout_size <= 0:
        raise ValueError("validation_size and heldout_size must be positive")
    if validation_size + heldout_size >= len(records):
        raise ValueError("validation and heldout splits must leave a train pool")

    rng = random.Random(seed)
    positives = [record for record in records if record.label]
    negatives = [record for record in records if not record.label]
    if not positives or not negatives:
        raise ValueError("real text experiment requires both ham and spam records")
    rng.shuffle(positives)
    rng.shuffle(negatives)

    validation, positives, negatives = _take_stratified(
        positives=positives,
        negatives=negatives,
        target_size=validation_size,
        rng=rng,
    )
    heldout, positives, negatives = _take_stratified(
        positives=positives,
        negatives=negatives,
        target_size=heldout_size,
        rng=rng,
    )
    train_pool = positives + negatives
    rng.shuffle(train_pool)
    return RealTextSplit(
        train_pool=tuple(train_pool),
        validation=tuple(validation),
        heldout=tuple(heldout),
    )


def _take_stratified(
    positives: list[TextRecord],
    negatives: list[TextRecord],
    target_size: int,
    rng: random.Random,
) -> tuple[list[TextRecord], list[TextRecord], list[TextRecord]]:
    positive_count, negative_count = _class_counts(
        target_size=target_size,
        available_positive=len(positives),
        available_negative=len(negatives),
    )
    selected = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(selected)
    return selected, positives[positive_count:], negatives[negative_count:]


def _class_counts(target_size: int, available_positive: int, available_negative: int) -> tuple[int, int]:
    available_total = available_positive + available_negative
    if target_size > available_total:
        raise ValueError("target_size exceeds available records")
    positive_count = round(target_size * available_positive / available_total)
    if available_positive and target_size >= 2:
        positive_count = max(1, positive_count)
    if available_negative and target_size >= 2:
        positive_count = min(target_size - 1, positive_count)
    positive_count = min(available_positive, positive_count)
    negative_count = min(available_negative, target_size - positive_count)
    remaining = target_size - positive_count - negative_count
    if remaining and available_positive - positive_count > 0:
        extra = min(remaining, available_positive - positive_count)
        positive_count += extra
        remaining -= extra
    if remaining and available_negative - negative_count > 0:
        negative_count += min(remaining, available_negative - negative_count)
    return positive_count, negative_count


def run_real_text_condition(
    records: tuple[TextRecord, ...],
    seed: int,
    condition: str,
    train_budget: int,
    validation_size: int,
    heldout_size: int,
    epochs: int,
    proxy_epochs: int = 2,
) -> dict:
    if condition not in REAL_TEXT_CONDITION_SCOPE:
        raise ValueError(f"unknown real-text condition: {condition}")
    if train_budget <= 0:
        raise ValueError("train_budget must be positive")
    if epochs <= 0 or proxy_epochs <= 0:
        raise ValueError("epochs and proxy_epochs must be positive")

    split = stratified_split(
        records=records,
        seed=seed,
        validation_size=validation_size,
        heldout_size=heldout_size,
    )
    train_pool_tokens = sum(record.token_count for record in split.train_pool)
    validation_tokens = sum(record.token_count for record in split.validation)

    selected_policy = condition
    validation_tuning_cost_tokens = 0
    validation_selector_scores: dict[str, float] = {}
    if condition == "random_sample":
        candidate = _random_sample(split.train_pool, budget=train_budget, seed=seed + 10_001)
    elif condition == "class_balanced_sample":
        candidate = _class_balanced_sample(split.train_pool, budget=train_budget, seed=seed + 20_001)
    elif condition == "label_index_balanced_sample":
        candidate = _label_index_balanced_sample(split.train_pool, budget=train_budget, seed=seed + 25_001)
    elif condition == "validation_sample_selector":
        candidate, validation_tuning_cost_tokens, validation_selector_scores = _select_candidate_by_validation(
            train_pool=split.train_pool,
            validation=split.validation,
            train_budget=train_budget,
            seed=seed,
            proxy_epochs=proxy_epochs,
            balanced_policy="class_balanced_sample",
        )
        selected_policy = candidate.policy
    else:
        candidate, validation_tuning_cost_tokens, validation_selector_scores = _select_candidate_by_validation(
            train_pool=split.train_pool,
            validation=split.validation,
            train_budget=train_budget,
            seed=seed,
            proxy_epochs=proxy_epochs,
            balanced_policy="label_index_balanced_sample",
        )
        selected_policy = candidate.policy

    train_examples = _to_training_examples(candidate.records)
    validation_examples = _to_training_examples(split.validation)
    heldout_examples = _to_training_examples(split.heldout)

    learner = PerceptronClassifier()
    updates = learner.fit(train_examples, epochs=epochs, seed=seed)
    heldout = _evaluate_model(learner, heldout_examples)
    validation = _evaluate_model(learner, validation_examples)
    baseline = _majority_metrics(train_examples, heldout_examples)

    training_cost_tokens = sum(example.token_count for example in train_examples) * epochs
    charged_compute_units = training_cost_tokens + candidate.selection_cost_tokens + validation_tuning_cost_tokens
    f1_improvement = heldout.spam_f1 - baseline.spam_f1
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_f1_improvement = max(0.0, f1_improvement)

    return {
        "dataset_name": SMS_SPAM_DATASET["name"],
        "dataset_sha256": SMS_SPAM_DATASET["sha256"],
        "seed": seed,
        "condition": condition,
        "selected_sampling_policy": selected_policy,
        "train_budget": train_budget,
        "external_events": len(candidate.records),
        "internal_examples": len(train_examples),
        "internal_tokens": sum(example.token_count for example in train_examples),
        "train_spam_count": sum(1 for record in candidate.records if record.label),
        "train_ham_count": sum(1 for record in candidate.records if not record.label),
        "train_spam_fraction": _round(sum(1 for record in candidate.records if record.label) / len(candidate.records)),
        "train_pool_size": len(split.train_pool),
        "train_pool_tokens": train_pool_tokens,
        "validation_size": len(split.validation),
        "validation_tokens": validation_tokens,
        "heldout_size": len(split.heldout),
        "selection_cost_tokens": candidate.selection_cost_tokens,
        "validation_tuning_cost_tokens": validation_tuning_cost_tokens,
        "charged_compute_units": charged_compute_units,
        "perceptron_updates": updates,
        "heldout_used_for_selection": False,
        "validation_selector_scores": validation_selector_scores,
        "heldout_accuracy": _round(heldout.accuracy),
        "heldout_spam_precision": _round(heldout.spam_precision),
        "heldout_spam_recall": _round(heldout.spam_recall),
        "heldout_spam_f1": _round(heldout.spam_f1),
        "validation_accuracy": _round(validation.accuracy),
        "validation_spam_f1": _round(validation.spam_f1),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "majority_baseline_spam_f1": _round(baseline.spam_f1),
        "accuracy_improvement_over_majority": _round(accuracy_improvement),
        "spam_f1_improvement_over_majority": _round(f1_improvement),
        "signed_external_sample_efficiency": _round(f1_improvement / max(1, len(candidate.records))),
        "clipped_external_sample_efficiency": _round(positive_f1_improvement / max(1, len(candidate.records))),
        "signed_compute_efficiency_per_10k_units": _round(10_000.0 * f1_improvement / max(1, charged_compute_units)),
        "clipped_compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_f1_improvement / max(1, charged_compute_units)
        ),
        "signed_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * f1_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
        "clipped_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_f1_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
        "external_sample_efficiency": _round(positive_f1_improvement / max(1, len(candidate.records))),
        "compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_f1_improvement / max(1, charged_compute_units)
        ),
        "learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_f1_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
    }


def _select_candidate_by_validation(
    train_pool: tuple[TextRecord, ...],
    validation: tuple[TextRecord, ...],
    train_budget: int,
    seed: int,
    proxy_epochs: int,
    balanced_policy: str,
) -> tuple[_CandidateSample, int, dict[str, float]]:
    if balanced_policy == "class_balanced_sample":
        balanced_candidate = _class_balanced_sample(train_pool, budget=train_budget, seed=seed + 40_001)
    elif balanced_policy == "label_index_balanced_sample":
        balanced_candidate = _label_index_balanced_sample(train_pool, budget=train_budget, seed=seed + 40_001)
    else:
        raise ValueError(f"unknown validation balanced policy: {balanced_policy}")
    candidates = (
        _random_sample(train_pool, budget=train_budget, seed=seed + 30_001),
        balanced_candidate,
    )
    validation_examples = _to_training_examples(validation)
    validation_tokens = sum(example.token_count for example in validation_examples)
    scored: list[tuple[float, int, _CandidateSample]] = []
    scores: dict[str, float] = {}
    proxy_and_validation_cost = 0

    for candidate in candidates:
        train_examples = _to_training_examples(candidate.records)
        learner = PerceptronClassifier()
        learner.fit(train_examples, epochs=proxy_epochs, seed=seed + 50_001)
        metrics = _evaluate_model(learner, validation_examples)
        baseline = _majority_metrics(train_examples, validation_examples)
        score = metrics.spam_f1 - baseline.spam_f1
        scores[candidate.policy] = _round(score)
        proxy_and_validation_cost += candidate.token_count * proxy_epochs + validation_tokens
        scored.append((score, -candidate.selection_cost_tokens, candidate))

    scored.sort(key=lambda item: (item[0], item[1], item[2].policy), reverse=True)
    selected = scored[0][2]
    unreused_selection_cost = sum(
        candidate.selection_cost_tokens for candidate in candidates if candidate.policy != selected.policy
    )
    validation_tuning_cost = proxy_and_validation_cost + unreused_selection_cost
    return selected, validation_tuning_cost, scores


def _random_sample(records: tuple[TextRecord, ...], budget: int, seed: int) -> _CandidateSample:
    rng = random.Random(seed)
    count = min(budget, len(records))
    return _CandidateSample(
        policy="random_sample",
        records=tuple(rng.sample(list(records), count)),
        selection_cost_tokens=0,
    )


def _class_balanced_sample(records: tuple[TextRecord, ...], budget: int, seed: int) -> _CandidateSample:
    return _balanced_sample(
        records=records,
        budget=budget,
        seed=seed,
        policy="class_balanced_sample",
        selection_cost_tokens=sum(record.token_count for record in records),
    )


def _label_index_balanced_sample(records: tuple[TextRecord, ...], budget: int, seed: int) -> _CandidateSample:
    return _balanced_sample(
        records=records,
        budget=budget,
        seed=seed,
        policy="label_index_balanced_sample",
        selection_cost_tokens=len(records),
    )


def _balanced_sample(
    records: tuple[TextRecord, ...],
    budget: int,
    seed: int,
    policy: str,
    selection_cost_tokens: int,
) -> _CandidateSample:
    rng = random.Random(seed)
    positives = [record for record in records if record.label]
    negatives = [record for record in records if not record.label]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_count, negative_count = _balanced_class_counts(
        target_size=min(budget, len(records)),
        available_positive=len(positives),
        available_negative=len(negatives),
    )
    selected = positives[:positive_count] + negatives[:negative_count]
    rng.shuffle(selected)
    return _CandidateSample(
        policy=policy,
        records=tuple(selected),
        selection_cost_tokens=selection_cost_tokens,
    )


def _balanced_class_counts(target_size: int, available_positive: int, available_negative: int) -> tuple[int, int]:
    if target_size > available_positive + available_negative:
        raise ValueError("target_size exceeds available records")
    positive_count = min(available_positive, target_size // 2)
    negative_count = min(available_negative, target_size - positive_count)
    remaining = target_size - positive_count - negative_count
    if remaining and available_positive - positive_count > 0:
        extra = min(remaining, available_positive - positive_count)
        positive_count += extra
        remaining -= extra
    if remaining and available_negative - negative_count > 0:
        negative_count += min(remaining, available_negative - negative_count)
    return positive_count, negative_count


def _to_training_example(record: TextRecord) -> TrainingExample:
    return TrainingExample(
        text=record.text,
        label=record.label,
        pair_key=("sms", record.record_id),
        source_observation_id=record.record_id,
        source_kind="sms_raw",
    )


def _to_training_examples(records: tuple[TextRecord, ...]) -> tuple[TrainingExample, ...]:
    return tuple(_to_training_example(record) for record in records)


def _evaluate_model(classifier: PerceptronClassifier, examples: tuple[TrainingExample, ...]) -> BinaryMetrics:
    predictions = tuple(classifier.predict(example.text) for example in examples)
    return _binary_metrics(predictions, tuple(example.label for example in examples))


def _majority_metrics(train_examples: tuple[TrainingExample, ...], eval_examples: tuple[TrainingExample, ...]) -> BinaryMetrics:
    positive = sum(1 for example in train_examples if example.label)
    majority_label = positive > (len(train_examples) - positive)
    predictions = tuple(majority_label for _ in eval_examples)
    return _binary_metrics(predictions, tuple(example.label for example in eval_examples))


def _binary_metrics(predictions: tuple[bool, ...], labels: tuple[bool, ...]) -> BinaryMetrics:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    total = len(labels)
    true_positive = sum(1 for pred, label in zip(predictions, labels) if pred and label)
    false_positive = sum(1 for pred, label in zip(predictions, labels) if pred and not label)
    false_negative = sum(1 for pred, label in zip(predictions, labels) if not pred and label)
    true_negative = sum(1 for pred, label in zip(predictions, labels) if not pred and not label)
    correct = true_positive + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct / total if total else 0.0
    return BinaryMetrics(
        accuracy=accuracy,
        correct=correct,
        total=total,
        spam_precision=precision,
        spam_recall=recall,
        spam_f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def run_real_text_seedset(
    records: tuple[TextRecord, ...],
    seeds: tuple[int, ...] | list[int] = DEFAULT_REAL_TEXT_SEEDS,
    train_budgets: tuple[int, ...] | list[int] = DEFAULT_TRAIN_BUDGETS,
    conditions: tuple[str, ...] | list[str] = DEFAULT_REAL_TEXT_CONDITIONS,
    validation_size: int = 800,
    heldout_size: int = 1200,
    epochs: int = 5,
    proxy_epochs: int = 2,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict:
    unknown = sorted(set(conditions) - set(REAL_TEXT_CONDITION_SCOPE))
    if unknown:
        raise ValueError(f"unknown real-text conditions: {unknown}")

    per_seed: list[dict] = []
    budget_results: dict[str, dict] = {}
    for train_budget in train_budgets:
        grouped: dict[str, list[dict]] = {condition: [] for condition in conditions}
        for seed in seeds:
            for condition in conditions:
                row = run_real_text_condition(
                    records=records,
                    seed=seed,
                    condition=condition,
                    train_budget=train_budget,
                    validation_size=validation_size,
                    heldout_size=heldout_size,
                    epochs=epochs,
                    proxy_epochs=proxy_epochs,
                )
                grouped[condition].append(row)
                per_seed.append(row)
        condition_stats = {condition: _aggregate(rows) for condition, rows in grouped.items()}
        budget_results[str(train_budget)] = {
            "conditions": condition_stats,
            "pareto_frontier_conditions": _pareto_frontier(condition_stats),
        }

    result = {
        "title": "Real Text Selection-Cost Pilot",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": SMS_SPAM_DATASET | {"record_count": len(records)},
        "seeds": list(seeds),
        "train_budgets": list(train_budgets),
        "validation_size": validation_size,
        "heldout_size": heldout_size,
        "epochs": epochs,
        "proxy_epochs": proxy_epochs,
        "claim_scope": {
            "synthetic_domain": False,
            "real_dataset": True,
            "neural_model": False,
            "heldout_used_for_selection": False,
            "validation_used_by_selector": "validation_sample_selector" in conditions,
            "paper_ready_claim": False,
        },
        "condition_scope": {condition: REAL_TEXT_CONDITION_SCOPE[condition] for condition in conditions},
        "budgets": budget_results,
        "per_seed": per_seed,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_markdown(result))
    return result


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = (
        "external_events",
        "internal_examples",
        "internal_tokens",
        "train_spam_count",
        "train_ham_count",
        "train_spam_fraction",
        "train_pool_size",
        "train_pool_tokens",
        "validation_size",
        "validation_tokens",
        "heldout_size",
        "selection_cost_tokens",
        "validation_tuning_cost_tokens",
        "charged_compute_units",
        "perceptron_updates",
        "heldout_accuracy",
        "heldout_spam_precision",
        "heldout_spam_recall",
        "heldout_spam_f1",
        "validation_accuracy",
        "validation_spam_f1",
        "majority_baseline_accuracy",
        "majority_baseline_spam_f1",
        "accuracy_improvement_over_majority",
        "spam_f1_improvement_over_majority",
        "signed_external_sample_efficiency",
        "clipped_external_sample_efficiency",
        "signed_compute_efficiency_per_10k_units",
        "clipped_compute_efficiency_per_10k_units",
        "signed_learning_signal_density_per_1m_event_compute",
        "clipped_learning_signal_density_per_1m_event_compute",
        "external_sample_efficiency",
        "compute_efficiency_per_10k_units",
        "learning_signal_density_per_1m_event_compute",
    )
    selected_policy_counts = Counter(row["selected_sampling_policy"] for row in rows)
    return {
        **{f"{key}_mean": _round(mean(row[key] for row in rows)) for key in numeric_keys},
        "selected_sampling_policy_counts": dict(sorted(selected_policy_counts.items())),
    }


def _pareto_frontier(conditions: dict[str, dict]) -> list[str]:
    frontier: list[str] = []
    for name, stats in conditions.items():
        dominated = False
        for other_name, other in conditions.items():
            if other_name == name:
                continue
            at_least_as_good = other["heldout_spam_f1_mean"] >= stats["heldout_spam_f1_mean"]
            no_more_external = other["external_events_mean"] <= stats["external_events_mean"]
            no_more_compute = other["charged_compute_units_mean"] <= stats["charged_compute_units_mean"]
            strictly_better = (
                other["heldout_spam_f1_mean"] > stats["heldout_spam_f1_mean"]
                or other["external_events_mean"] < stats["external_events_mean"]
                or other["charged_compute_units_mean"] < stats["charged_compute_units_mean"]
            )
            if at_least_as_good and no_more_external and no_more_compute and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def render_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        (
            f"Dataset: {result['dataset']['name']} "
            f"({result['dataset']['doi']}, {result['dataset']['license']}, "
            f"{result['dataset']['record_count']} records)."
        ),
        "",
        "This artifact uses real SMS text, not the synthetic causal-text world.",
        "The heldout split is never used for sampling-policy selection. Validation is used only by the declared selector.",
        "Primary quality metric: spam-class F1 improvement over the train-sample majority baseline.",
        "",
    ]
    for budget, payload in result["budgets"].items():
        lines.extend(
            [
                f"## Train Budget {budget}",
                "",
                "| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition, stats in payload["conditions"].items():
            counts = ", ".join(f"{policy}: {count}" for policy, count in stats["selected_sampling_policy_counts"].items())
            lines.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        counts,
                        f"{stats['heldout_spam_f1_mean']:.3f}",
                        f"{stats['majority_baseline_spam_f1_mean']:.3f}",
                        f"{stats['spam_f1_improvement_over_majority_mean']:.3f}",
                        f"{stats['charged_compute_units_mean']:.1f}",
                        f"{stats['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Pareto frontier: " + ", ".join(f"`{name}`" for name in payload["pareto_frontier_conditions"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Scope Flags",
            "",
            "```json",
            json.dumps(result["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-text SMS Spam selection-cost pilot.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-json", type=Path, default=Path("results/sms_spam_real_text_selection_cost.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/sms_spam_real_text_selection_cost.md"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_REAL_TEXT_SEEDS))
    parser.add_argument("--train-budgets", nargs="+", type=int, default=list(DEFAULT_TRAIN_BUDGETS))
    parser.add_argument("--validation-size", type=int, default=800)
    parser.add_argument("--heldout-size", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--proxy-epochs", type=int, default=2)
    args = parser.parse_args()

    records = load_sms_spam_records(cache_path=args.cache_path)
    result = run_real_text_seedset(
        records=records,
        seeds=tuple(args.seeds),
        train_budgets=tuple(args.train_budgets),
        validation_size=args.validation_size,
        heldout_size=args.heldout_size,
        epochs=args.epochs,
        proxy_epochs=args.proxy_epochs,
        output_json=args.output_json,
        output_markdown=args.output_md,
    )
    print(f"loaded {result['dataset']['record_count']} SMS records")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
