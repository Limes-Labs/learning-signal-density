from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
from statistics import mean
import tarfile
from urllib.request import Request, urlopen
import zipfile

from .learner import featurize


TWENTY_NEWSGROUPS_URL = "https://archive.ics.uci.edu/static/public/113/twenty+newsgroups.zip"
TWENTY_NEWSGROUPS_SHA256 = "cfbb360d6c1e55c06d33a4c5da0789a93b78db74833a70be8ff2e133cc4e6a6e"
DEFAULT_CACHE_PATH = Path("data/external/twenty_newsgroups.zip")
DEFAULT_TAR_MEMBER = "mini_newsgroups.tar.gz"
DEFAULT_CORPUS_ROOT = "mini_newsgroups"
DEFAULT_NEWSGROUPS_SEEDS = (311, 313, 317)
DEFAULT_NEWSGROUPS_BUDGETS = (40, 80, 160)
DEFAULT_NEWSGROUPS_CONDITIONS = (
    "random_sample",
    "class_balanced_sample",
    "length_curriculum_sample",
    "prototype_retrieval_sample",
    "validation_selector",
)

TWENTY_NEWSGROUPS_DATASET = {
    "name": "Twenty Newsgroups",
    "uci_id": 113,
    "url": TWENTY_NEWSGROUPS_URL,
    "license": "CC BY 4.0",
    "sha256": TWENTY_NEWSGROUPS_SHA256,
    "tar_member": DEFAULT_TAR_MEMBER,
    "task": "20-way topic classification",
}

NEWSGROUPS_CONDITION_SCOPE = {
    "random_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "label_aware_sampling": False,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
    },
    "class_balanced_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "label_aware_sampling": True,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "selection_cost_model": "one_unit_per_train_pool_label",
    },
    "length_curriculum_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "label_aware_sampling": False,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "selection_cost_model": "full_train_pool_token_length_scan",
    },
    "prototype_retrieval_sample": {
        "real_dataset": True,
        "train_only_selection": True,
        "label_aware_sampling": True,
        "validation_used_for_policy_selection": False,
        "heldout_used_for_selection": False,
        "selection_cost_model": "label_conditioned_train_pool_prototype_scan",
    },
    "validation_selector": {
        "real_dataset": True,
        "train_only_selection": True,
        "label_aware_sampling": True,
        "validation_used_for_policy_selection": True,
        "heldout_used_for_selection": False,
        "selection_cost_model": "proxy_validation_policy_selection",
    },
}


def _round(value: float) -> float:
    return round(value, 6)


@lru_cache(maxsize=50_000)
def _cached_featurize(text: str) -> Counter[str]:
    return featurize(text)


@dataclass(frozen=True)
class NewsRecord:
    record_id: str
    label: str
    text: str

    @property
    def token_count(self) -> int:
        return sum(_cached_featurize(self.text).values())


@dataclass(frozen=True)
class NewsgroupsSplit:
    train_pool: tuple[NewsRecord, ...]
    validation: tuple[NewsRecord, ...]
    heldout: tuple[NewsRecord, ...]


@dataclass(frozen=True)
class NewsCandidate:
    policy: str
    records: tuple[NewsRecord, ...]
    selection_cost_tokens: int

    @property
    def token_count(self) -> int:
        return sum(record.token_count for record in self.records)


@dataclass(frozen=True)
class MultiClassMetrics:
    accuracy: float
    correct: int
    total: int


class MulticlassPerceptronClassifier:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = tuple(sorted(labels))
        self.weights: dict[str, defaultdict[str, float]] = {
            label: defaultdict(float) for label in self.labels
        }
        self.bias: dict[str, float] = {label: 0.0 for label in self.labels}

    def score(self, label: str, text: str) -> float:
        features = _cached_featurize(text)
        return self.bias[label] + sum(self.weights[label][name] * value for name, value in features.items())

    def predict(self, text: str) -> str:
        return max(self.labels, key=lambda label: (self.score(label, text), label))

    def fit(self, records: tuple[NewsRecord, ...], epochs: int, seed: int) -> int:
        updates = 0
        rng = random.Random(seed)
        indexed = list(records)
        for _ in range(epochs):
            rng.shuffle(indexed)
            for record in indexed:
                prediction = self.predict(record.text)
                if prediction == record.label:
                    continue
                features = _cached_featurize(record.text)
                for name, value in features.items():
                    self.weights[record.label][name] += value
                    self.weights[prediction][name] -= value
                self.bias[record.label] += 1.0
                self.bias[prediction] -= 1.0
                updates += 1
        return updates

    def evaluate(self, records: tuple[NewsRecord, ...]) -> MultiClassMetrics:
        if not records:
            return MultiClassMetrics(accuracy=0.0, correct=0, total=0)
        correct = sum(1 for record in records if self.predict(record.text) == record.label)
        return MultiClassMetrics(accuracy=correct / len(records), correct=correct, total=len(records))


def parse_twenty_newsgroups_tar_gz(payload: bytes, corpus_root: str = DEFAULT_CORPUS_ROOT) -> tuple[NewsRecord, ...]:
    records: list[NewsRecord] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = Path(member.name)
            parts = path.parts
            if len(parts) < 3 or parts[0] != corpus_root:
                continue
            label = parts[1]
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw_text = extracted.read().decode("latin-1", errors="replace")
            text = strip_newsgroups_metadata(raw_text)
            if not text:
                continue
            records.append(NewsRecord(record_id=f"{label}/{path.name}", label=label, text=text))
    if not records:
        raise ValueError(f"no Twenty Newsgroups records found under {corpus_root!r}")
    return tuple(sorted(records, key=lambda record: record.record_id))


def strip_newsgroups_metadata(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body_start = 0
    for index, line in enumerate(lines):
        if not line.strip():
            body_start = index + 1
            break
    body_lines = []
    for line in lines[body_start:]:
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            continue
        if stripped.startswith(">"):
            continue
        if lower.startswith("in article ") or lower.endswith("writes:"):
            continue
        body_lines.append(stripped)
    return " ".join(" ".join(body_lines).split())


def load_twenty_newsgroups_records(
    cache_path: Path = DEFAULT_CACHE_PATH,
    tar_member: str = DEFAULT_TAR_MEMBER,
    corpus_root: str = DEFAULT_CORPUS_ROOT,
    timeout: int = 60,
) -> tuple[NewsRecord, ...]:
    zip_payload = _load_or_download_twenty_newsgroups_zip(cache_path=cache_path, timeout=timeout)
    digest = hashlib.sha256(zip_payload).hexdigest()
    if digest != TWENTY_NEWSGROUPS_SHA256:
        raise ValueError(f"unexpected Twenty Newsgroups sha256 {digest}")
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        try:
            tar_payload = archive.read(tar_member)
        except KeyError as exc:
            raise ValueError(f"Twenty Newsgroups zip does not contain {tar_member}") from exc
    return parse_twenty_newsgroups_tar_gz(tar_payload, corpus_root=corpus_root)


def _load_or_download_twenty_newsgroups_zip(cache_path: Path, timeout: int) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(TWENTY_NEWSGROUPS_URL, headers={"User-Agent": "learning-signal-density-research/0.1"})
    payload = urlopen(request, timeout=timeout).read()
    cache_path.write_bytes(payload)
    return payload


def stratified_newsgroups_split(
    records: tuple[NewsRecord, ...],
    seed: int,
    validation_per_class: int,
    heldout_per_class: int,
) -> NewsgroupsSplit:
    if validation_per_class <= 0 or heldout_per_class <= 0:
        raise ValueError("validation_per_class and heldout_per_class must be positive")
    rng = random.Random(seed)
    by_label: dict[str, list[NewsRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    validation: list[NewsRecord] = []
    heldout: list[NewsRecord] = []
    train_pool: list[NewsRecord] = []
    for label, label_records in sorted(by_label.items()):
        rng.shuffle(label_records)
        needed = validation_per_class + heldout_per_class + 1
        if len(label_records) < needed:
            raise ValueError(f"not enough records for label {label}: need at least {needed}")
        validation.extend(label_records[:validation_per_class])
        heldout.extend(label_records[validation_per_class : validation_per_class + heldout_per_class])
        train_pool.extend(label_records[validation_per_class + heldout_per_class :])
    rng.shuffle(train_pool)
    rng.shuffle(validation)
    rng.shuffle(heldout)
    return NewsgroupsSplit(tuple(train_pool), tuple(validation), tuple(heldout))


def run_newsgroups_condition(
    records: tuple[NewsRecord, ...],
    seed: int,
    condition: str,
    train_budget: int,
    validation_per_class: int,
    heldout_per_class: int,
    epochs: int,
    proxy_epochs: int = 2,
) -> dict:
    if condition not in NEWSGROUPS_CONDITION_SCOPE:
        raise ValueError(f"unknown Twenty Newsgroups condition: {condition}")
    if train_budget <= 0:
        raise ValueError("train_budget must be positive")
    if epochs <= 0 or proxy_epochs <= 0:
        raise ValueError("epochs and proxy_epochs must be positive")

    split = stratified_newsgroups_split(records, seed, validation_per_class, heldout_per_class)
    validation_tuning_cost_tokens = 0
    validation_selector_scores: dict[str, float] = {}
    selected_policy = condition
    if condition == "random_sample":
        candidate = _random_sample(split.train_pool, train_budget, seed + 10_001)
    elif condition == "class_balanced_sample":
        candidate = _class_balanced_sample(split.train_pool, train_budget, seed + 20_001)
    elif condition == "length_curriculum_sample":
        candidate = _length_curriculum_sample(split.train_pool, train_budget)
    elif condition == "prototype_retrieval_sample":
        candidate = _prototype_retrieval_sample(split.train_pool, train_budget)
    else:
        candidate, validation_tuning_cost_tokens, validation_selector_scores = _select_newsgroups_candidate_by_validation(
            train_pool=split.train_pool,
            validation=split.validation,
            train_budget=train_budget,
            seed=seed,
            proxy_epochs=proxy_epochs,
        )
        selected_policy = candidate.policy

    labels = tuple(sorted({record.label for record in records}))
    learner = MulticlassPerceptronClassifier(labels)
    updates = learner.fit(candidate.records, epochs=epochs, seed=seed)
    heldout = learner.evaluate(split.heldout)
    validation = learner.evaluate(split.validation)
    baseline = _majority_metrics(candidate.records, split.heldout)
    validation_baseline = _majority_metrics(candidate.records, split.validation)

    internal_tokens = candidate.token_count
    train_pool_tokens = sum(record.token_count for record in split.train_pool)
    validation_tokens = sum(record.token_count for record in split.validation)
    training_cost_tokens = internal_tokens * epochs
    charged_compute_units = training_cost_tokens + candidate.selection_cost_tokens + validation_tuning_cost_tokens
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_accuracy_improvement = max(0.0, accuracy_improvement)

    return {
        "dataset_name": TWENTY_NEWSGROUPS_DATASET["name"],
        "dataset_sha256": TWENTY_NEWSGROUPS_DATASET["sha256"],
        "seed": seed,
        "condition": condition,
        "selected_sampling_policy": selected_policy,
        "train_budget": train_budget,
        "external_events": len(candidate.records),
        "internal_examples": len(candidate.records),
        "internal_tokens": internal_tokens,
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
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "validation_majority_baseline_accuracy": _round(validation_baseline.accuracy),
        "accuracy_improvement_over_majority": _round(accuracy_improvement),
        "signed_external_sample_efficiency": _round(accuracy_improvement / max(1, len(candidate.records))),
        "clipped_external_sample_efficiency": _round(positive_accuracy_improvement / max(1, len(candidate.records))),
        "signed_compute_efficiency_per_10k_units": _round(10_000.0 * accuracy_improvement / max(1, charged_compute_units)),
        "clipped_compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_accuracy_improvement / max(1, charged_compute_units)
        ),
        "signed_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * accuracy_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
        "clipped_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_accuracy_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
        "external_sample_efficiency": _round(positive_accuracy_improvement / max(1, len(candidate.records))),
        "compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_accuracy_improvement / max(1, charged_compute_units)
        ),
        "learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_accuracy_improvement / max(1, len(candidate.records) * charged_compute_units)
        ),
    }


def _select_newsgroups_candidate_by_validation(
    train_pool: tuple[NewsRecord, ...],
    validation: tuple[NewsRecord, ...],
    train_budget: int,
    seed: int,
    proxy_epochs: int,
) -> tuple[NewsCandidate, int, dict[str, float]]:
    candidates = (
        _random_sample(train_pool, train_budget, seed + 30_001),
        _class_balanced_sample(train_pool, train_budget, seed + 40_001),
        _length_curriculum_sample(train_pool, train_budget),
        _prototype_retrieval_sample(train_pool, train_budget),
    )
    labels = tuple(sorted({record.label for record in train_pool}))
    validation_tokens = sum(record.token_count for record in validation)
    proxy_and_validation_cost = 0
    scored: list[tuple[float, int, str, NewsCandidate]] = []
    scores: dict[str, float] = {}
    for candidate in candidates:
        learner = MulticlassPerceptronClassifier(labels)
        learner.fit(candidate.records, epochs=proxy_epochs, seed=seed + 50_001)
        metrics = learner.evaluate(validation)
        baseline = _majority_metrics(candidate.records, validation)
        score = metrics.accuracy - baseline.accuracy
        scores[candidate.policy] = _round(score)
        proxy_and_validation_cost += candidate.token_count * proxy_epochs + validation_tokens
        scored.append((score, -candidate.selection_cost_tokens, candidate.policy, candidate))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = scored[0][3]
    unreused_selection_cost = sum(
        candidate.selection_cost_tokens for candidate in candidates if candidate.policy != selected.policy
    )
    validation_tuning_cost = proxy_and_validation_cost + unreused_selection_cost
    return selected, validation_tuning_cost, scores


def _random_sample(records: tuple[NewsRecord, ...], budget: int, seed: int) -> NewsCandidate:
    rng = random.Random(seed)
    count = min(budget, len(records))
    return NewsCandidate("random_sample", tuple(rng.sample(list(records), count)), 0)


def _class_balanced_sample(records: tuple[NewsRecord, ...], budget: int, seed: int) -> NewsCandidate:
    selected = _balanced_by_label(records, budget, seed)
    return NewsCandidate("class_balanced_sample", selected, len(records))


def _length_curriculum_sample(records: tuple[NewsRecord, ...], budget: int) -> NewsCandidate:
    selected = tuple(sorted(records, key=lambda record: (record.token_count, record.record_id))[:budget])
    return NewsCandidate("length_curriculum_sample", selected, sum(record.token_count for record in records))


def _prototype_retrieval_sample(records: tuple[NewsRecord, ...], budget: int) -> NewsCandidate:
    by_label: dict[str, list[NewsRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    prototypes: dict[str, Counter[str]] = {}
    for label, label_records in by_label.items():
        prototype: Counter[str] = Counter()
        for record in label_records:
            prototype.update(_cached_featurize(record.text))
        prototypes[label] = prototype
    target_counts = _balanced_counts(sorted(by_label), budget)
    selected: list[NewsRecord] = []
    for label in sorted(by_label):
        prototype = prototypes[label]
        ranked = sorted(
            by_label[label],
            key=lambda record: (_cosine(_cached_featurize(record.text), prototype), -record.token_count, record.record_id),
            reverse=True,
        )
        selected.extend(ranked[: target_counts[label]])
    train_pool_tokens = sum(record.token_count for record in records)
    return NewsCandidate("prototype_retrieval_sample", tuple(selected), train_pool_tokens * 2)


def _balanced_by_label(records: tuple[NewsRecord, ...], budget: int, seed: int) -> tuple[NewsRecord, ...]:
    rng = random.Random(seed)
    by_label: dict[str, list[NewsRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    for label_records in by_label.values():
        rng.shuffle(label_records)
    target_counts = _balanced_counts(sorted(by_label), budget)
    selected: list[NewsRecord] = []
    for label in sorted(by_label):
        selected.extend(by_label[label][: target_counts[label]])
    rng.shuffle(selected)
    return tuple(selected)


def _balanced_counts(labels: list[str], budget: int) -> dict[str, int]:
    if not labels:
        return {}
    base = budget // len(labels)
    remainder = budget % len(labels)
    return {label: base + (1 if index < remainder else 0) for index, label in enumerate(labels)}


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(name, 0) for name, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _majority_metrics(train: tuple[NewsRecord, ...], eval_records: tuple[NewsRecord, ...]) -> MultiClassMetrics:
    if not eval_records:
        return MultiClassMetrics(accuracy=0.0, correct=0, total=0)
    counts = Counter(record.label for record in train)
    majority_label = max(sorted(counts), key=lambda label: (counts[label], label)) if counts else ""
    correct = sum(1 for record in eval_records if record.label == majority_label)
    return MultiClassMetrics(accuracy=correct / len(eval_records), correct=correct, total=len(eval_records))


def run_newsgroups_seedset(
    records: tuple[NewsRecord, ...],
    seeds: tuple[int, ...] | list[int] = DEFAULT_NEWSGROUPS_SEEDS,
    train_budgets: tuple[int, ...] | list[int] = DEFAULT_NEWSGROUPS_BUDGETS,
    conditions: tuple[str, ...] | list[str] = DEFAULT_NEWSGROUPS_CONDITIONS,
    validation_per_class: int = 20,
    heldout_per_class: int = 20,
    epochs: int = 4,
    proxy_epochs: int = 2,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict:
    unknown = sorted(set(conditions) - set(NEWSGROUPS_CONDITION_SCOPE))
    if unknown:
        raise ValueError(f"unknown Twenty Newsgroups conditions: {unknown}")
    per_seed: list[dict] = []
    budget_results: dict[str, dict] = {}
    for train_budget in train_budgets:
        grouped: dict[str, list[dict]] = {condition: [] for condition in conditions}
        for seed in seeds:
            for condition in conditions:
                row = run_newsgroups_condition(
                    records=records,
                    seed=seed,
                    condition=condition,
                    train_budget=train_budget,
                    validation_per_class=validation_per_class,
                    heldout_per_class=heldout_per_class,
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
    label_counts = Counter(record.label for record in records)
    artifact = {
        "title": "Twenty Newsgroups Active Selection-Cost Pilot",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {"record_count": len(records), "label_count": len(label_counts), "label_counts": dict(sorted(label_counts.items()))},
        "seeds": list(seeds),
        "train_budgets": list(train_budgets),
        "validation_per_class": validation_per_class,
        "heldout_per_class": heldout_per_class,
        "epochs": epochs,
        "proxy_epochs": proxy_epochs,
        "claim_scope": {
            "synthetic_domain": False,
            "real_dataset": True,
            "neural_model": False,
            "heldout_used_for_selection": False,
            "metadata_stripped": True,
            "paper_ready_claim": False,
        },
        "condition_scope": {condition: NEWSGROUPS_CONDITION_SCOPE[condition] for condition in conditions},
        "budgets": budget_results,
        "per_seed": per_seed,
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_newsgroups_markdown(artifact))
    return artifact


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        "external_events",
        "internal_examples",
        "internal_tokens",
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
        "validation_accuracy",
        "majority_baseline_accuracy",
        "validation_majority_baseline_accuracy",
        "accuracy_improvement_over_majority",
        "signed_external_sample_efficiency",
        "clipped_external_sample_efficiency",
        "signed_compute_efficiency_per_10k_units",
        "clipped_compute_efficiency_per_10k_units",
        "signed_learning_signal_density_per_1m_event_compute",
        "clipped_learning_signal_density_per_1m_event_compute",
        "external_sample_efficiency",
        "compute_efficiency_per_10k_units",
        "learning_signal_density_per_1m_event_compute",
    ]
    summary = {f"{key}_mean": _round(mean(float(row[key]) for row in rows)) for key in numeric_keys}
    summary["selected_sampling_policy_counts"] = dict(Counter(row["selected_sampling_policy"] for row in rows))
    return summary


def _pareto_frontier(condition_stats: dict[str, dict]) -> list[str]:
    frontier: list[str] = []
    for condition, stats in condition_stats.items():
        dominated = False
        for other_condition, other in condition_stats.items():
            if other_condition == condition:
                continue
            no_less_quality = other["accuracy_improvement_over_majority_mean"] >= stats["accuracy_improvement_over_majority_mean"]
            no_more_compute = other["charged_compute_units_mean"] <= stats["charged_compute_units_mean"]
            strictly_better = (
                other["accuracy_improvement_over_majority_mean"] > stats["accuracy_improvement_over_majority_mean"]
                or other["charged_compute_units_mean"] < stats["charged_compute_units_mean"]
            )
            if no_less_quality and no_more_compute and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(condition)
    return frontier


def render_newsgroups_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        "This is a real NLP classification pilot over UCI Twenty Newsgroups mini.",
        "Headers, quote lines, and reply boilerplate are stripped before splitting.",
        "",
        "| Budget | Condition | Acc. | Gain | LSD | Cost | Selected policy counts |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for budget in artifact["train_budgets"]:
        budget_rows = artifact["budgets"][str(budget)]["conditions"]
        for condition, stats in budget_rows.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(budget),
                        condition,
                        f"{stats['heldout_accuracy_mean']:.3f}",
                        f"{stats['accuracy_improvement_over_majority_mean']:.3f}",
                        f"{stats['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                        f"{stats['charged_compute_units_mean']:.1f}",
                        json.dumps(stats["selected_sampling_policy_counts"], sort_keys=True),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Scope Flags",
            "",
            "```json",
            json.dumps(artifact["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-json", type=Path, default=Path("results/twenty_newsgroups_active_selection.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/twenty_newsgroups_active_selection.md"))
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--heldout-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--proxy-epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_NEWSGROUPS_SEEDS))
    parser.add_argument("--train-budgets", type=int, nargs="+", default=list(DEFAULT_NEWSGROUPS_BUDGETS))
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    artifact = run_newsgroups_seedset(
        records=records,
        seeds=tuple(args.seeds),
        train_budgets=tuple(args.train_budgets),
        validation_per_class=args.validation_per_class,
        heldout_per_class=args.heldout_per_class,
        epochs=args.epochs,
        proxy_epochs=args.proxy_epochs,
        output_json=args.output_json,
        output_markdown=args.output_md,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"records={artifact['dataset']['record_count']} labels={artifact['dataset']['label_count']}")


if __name__ == "__main__":
    main()
