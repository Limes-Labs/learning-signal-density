#!/usr/bin/env python3
"""Audit length-penalized prototype retrieval on Twenty Newsgroups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from learning_signal_density.break_even import break_even_comparison
from learning_signal_density.newsgroups_experiment import (
    DEFAULT_NEWSGROUPS_BUDGETS,
    DEFAULT_NEWSGROUPS_SEEDS,
    TWENTY_NEWSGROUPS_DATASET,
    MulticlassPerceptronClassifier,
    NewsCandidate,
    NewsRecord,
    _balanced_counts,
    _cached_featurize,
    _cosine,
    _majority_metrics,
    _round,
    load_twenty_newsgroups_records,
    stratified_newsgroups_split,
)


SOURCE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
QUALITY_KEY = "accuracy_improvement_over_majority_mean"
QUALITY_UPPER_BOUND = 0.95
REUSABLE_COMPUTE_KEYS = ("selection_cost_tokens_mean", "validation_tuning_cost_tokens_mean")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def length_penalized_prototype_sample(
    records: tuple[NewsRecord, ...],
    budget: int,
    alpha: float,
) -> NewsCandidate:
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

        def score(record: NewsRecord) -> tuple[float, float, int, str]:
            cosine = _cosine(_cached_featurize(record.text), prototype)
            length_penalized = cosine / (max(1, record.token_count) ** alpha)
            return (length_penalized, cosine, -record.token_count, record.record_id)

        ranked = sorted(by_label[label], key=score, reverse=True)
        selected.extend(ranked[: target_counts[label]])
    train_pool_tokens = sum(record.token_count for record in records)
    return NewsCandidate(
        policy=f"length_penalized_prototype_alpha_{_alpha_key(alpha)}",
        records=tuple(selected),
        selection_cost_tokens=train_pool_tokens * 2,
    )


def run_alpha_condition(
    records: tuple[NewsRecord, ...],
    seed: int,
    train_budget: int,
    alpha: float,
    validation_per_class: int,
    heldout_per_class: int,
    epochs: int,
) -> dict:
    split = stratified_newsgroups_split(records, seed, validation_per_class, heldout_per_class)
    candidate = length_penalized_prototype_sample(split.train_pool, train_budget, alpha)
    labels = tuple(sorted({record.label for record in records}))
    learner = MulticlassPerceptronClassifier(labels)
    updates = learner.fit(candidate.records, epochs=epochs, seed=seed)
    heldout = learner.evaluate(split.heldout)
    validation = learner.evaluate(split.validation)
    baseline = _majority_metrics(candidate.records, split.heldout)

    internal_tokens = candidate.token_count
    training_cost_tokens = internal_tokens * epochs
    charged_compute_units = training_cost_tokens + candidate.selection_cost_tokens
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_accuracy_improvement = max(0.0, accuracy_improvement)

    return {
        "seed": seed,
        "condition": candidate.policy,
        "alpha": alpha,
        "train_budget": train_budget,
        "external_events": len(candidate.records),
        "internal_examples": len(candidate.records),
        "internal_tokens": internal_tokens,
        "selection_cost_tokens": candidate.selection_cost_tokens,
        "validation_tuning_cost_tokens": 0,
        "charged_compute_units": charged_compute_units,
        "perceptron_updates": updates,
        "heldout_used_for_selection": False,
        "heldout_accuracy": _round(heldout.accuracy),
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
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
    }


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        "external_events",
        "internal_examples",
        "internal_tokens",
        "selection_cost_tokens",
        "validation_tuning_cost_tokens",
        "charged_compute_units",
        "perceptron_updates",
        "heldout_accuracy",
        "validation_accuracy",
        "majority_baseline_accuracy",
        "accuracy_improvement_over_majority",
        "signed_external_sample_efficiency",
        "clipped_external_sample_efficiency",
        "signed_compute_efficiency_per_10k_units",
        "clipped_compute_efficiency_per_10k_units",
        "signed_learning_signal_density_per_1m_event_compute",
        "clipped_learning_signal_density_per_1m_event_compute",
    ]
    return {f"{key}_mean": _round(mean(float(row[key]) for row in rows)) for key in numeric_keys}


def build_retrieval_cost_audit(
    records: tuple[NewsRecord, ...],
    source_artifact: dict,
    seeds: tuple[int, ...] = DEFAULT_NEWSGROUPS_SEEDS,
    train_budgets: tuple[int, ...] = DEFAULT_NEWSGROUPS_BUDGETS,
    alphas: tuple[float, ...] = ALPHAS,
    validation_per_class: int = 20,
    heldout_per_class: int = 20,
    epochs: int = 3,
) -> dict:
    per_seed: list[dict] = []
    budget_results: dict[str, dict] = {}
    for train_budget in train_budgets:
        alpha_rows: dict[str, list[dict]] = {_alpha_key(alpha): [] for alpha in alphas}
        for seed in seeds:
            for alpha in alphas:
                row = run_alpha_condition(
                    records=records,
                    seed=seed,
                    train_budget=train_budget,
                    alpha=alpha,
                    validation_per_class=validation_per_class,
                    heldout_per_class=heldout_per_class,
                    epochs=epochs,
                )
                alpha_rows[_alpha_key(alpha)].append(row)
                per_seed.append(row)

        reference = {"condition": "random_sample"} | source_artifact["budgets"][str(train_budget)]["conditions"]["random_sample"]
        alpha_results = {}
        for alpha_key, rows in alpha_rows.items():
            aggregate = _aggregate(rows)
            aggregate["condition"] = f"length_penalized_prototype_alpha_{alpha_key}"
            break_even = break_even_comparison(
                reference=reference,
                candidate=aggregate,
                quality_key=QUALITY_KEY,
                quality_upper_bound=QUALITY_UPPER_BOUND,
                reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
            )
            alpha_results[alpha_key] = aggregate | {"break_even_vs_random": break_even}

        best_accuracy_alpha = max(
            alpha_results,
            key=lambda key: (
                alpha_results[key]["heldout_accuracy_mean"],
                alpha_results[key]["signed_learning_signal_density_per_1m_event_compute_mean"],
            ),
        )
        best_density_alpha = max(
            alpha_results,
            key=lambda key: (
                alpha_results[key]["signed_learning_signal_density_per_1m_event_compute_mean"],
                alpha_results[key]["heldout_accuracy_mean"],
            ),
        )
        budget_results[str(train_budget)] = {
            "random_reference": reference,
            "alpha_results": alpha_results,
            "best_accuracy_alpha": best_accuracy_alpha,
            "best_density_alpha": best_density_alpha,
        }

    label_counts = Counter(record.label for record in records)
    return {
        "title": "Twenty Newsgroups Length-Penalized Retrieval-Cost Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(SOURCE_ARTIFACT)],
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {"record_count": len(records), "label_count": len(label_counts), "label_counts": dict(sorted(label_counts.items()))},
        "seeds": list(seeds),
        "train_budgets": list(train_budgets),
        "alphas": list(alphas),
        "validation_per_class": validation_per_class,
        "heldout_per_class": heldout_per_class,
        "epochs": epochs,
        "quality_metric": QUALITY_KEY,
        "quality_upper_bound": QUALITY_UPPER_BOUND,
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "heldout_used_for_selection": False,
            "metadata_stripped": True,
            "post_hoc_optimization_attempt": True,
            "paper_ready_claim": False,
        },
        "condition_scope": {
            "length_penalized_prototype_retrieval": {
                "train_only_selection": True,
                "label_aware_sampling": True,
                "validation_used_for_policy_selection": False,
                "heldout_used_for_selection": False,
                "selection_cost_model": "label_conditioned_full_train_pool_prototype_scan",
                "alpha_interpretation": "score = cosine(record, class_prototype) / token_count(record)^alpha",
            }
        },
        "budgets": budget_results,
        "per_seed": per_seed,
    }


def render_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        "This is a real NLP retrieval-cost optimization audit over UCI Twenty Newsgroups mini.",
        "It tests whether penalizing selected-document length can make prototype retrieval cheaper enough to beat random density.",
        "",
        "| Budget | Best acc alpha | Acc | LSD | Best density alpha | Acc | LSD | Random LSD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_acc = budget_row["alpha_results"][budget_row["best_accuracy_alpha"]]
        best_density = budget_row["alpha_results"][budget_row["best_density_alpha"]]
        random_ref = budget_row["random_reference"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    budget_row["best_accuracy_alpha"],
                    f"{best_acc['heldout_accuracy_mean']:.3f}",
                    f"{best_acc['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    budget_row["best_density_alpha"],
                    f"{best_density['heldout_accuracy_mean']:.3f}",
                    f"{best_density['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{random_ref['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Alpha 0.0 is the original prototype-retrieval score.",
            "- Larger alpha penalizes long selected documents and reduces nonreusable training-token cost.",
            "- In this artifact, length penalties improve some prototype-retrieval rows but do not beat random sampling on density.",
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
    parser.add_argument("--cache-path", type=Path, default=Path("data/external/twenty_newsgroups.zip"))
    parser.add_argument("--source-json", type=Path, default=SOURCE_ARTIFACT)
    parser.add_argument("--output-json", type=Path, default=Path("results/twenty_newsgroups_retrieval_cost_audit.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/twenty_newsgroups_retrieval_cost_audit.md"))
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    source_artifact = load_json(args.source_json)
    artifact = build_retrieval_cost_audit(records=records, source_artifact=source_artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
