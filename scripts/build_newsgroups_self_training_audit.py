#!/usr/bin/env python3
"""Audit confidence-filtered self-training on Twenty Newsgroups."""

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
from learning_signal_density.newsgroups_experiment import (
    DEFAULT_NEWSGROUPS_BUDGETS,
    DEFAULT_NEWSGROUPS_SEEDS,
    TWENTY_NEWSGROUPS_DATASET,
    MulticlassPerceptronClassifier,
    NewsRecord,
    _balanced_counts,
    _class_balanced_sample,
    _majority_metrics,
    _round,
    load_twenty_newsgroups_records,
    stratified_newsgroups_split,
)


SOURCE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
PSEUDO_MULTIPLIERS = (1, 2)
FILTER_MODES = ("global_margin", "balanced_margin")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _teacher_prediction_margin(teacher: MulticlassPerceptronClassifier, labels: tuple[str, ...], text: str) -> tuple[str, float]:
    scores = sorted(((teacher.score(label, text), label) for label in labels), reverse=True)
    return scores[0][1], scores[0][0] - scores[1][0]


def _select_pseudo_rows(
    rows: list[tuple[NewsRecord, str, float]],
    pseudo_count: int,
    filter_mode: str,
) -> list[tuple[NewsRecord, str, float]]:
    if filter_mode == "global_margin":
        return sorted(rows, key=lambda row: (row[2], -row[0].token_count, row[0].record_id), reverse=True)[
            :pseudo_count
        ]
    if filter_mode != "balanced_margin":
        raise ValueError(f"unknown pseudo-label filter mode: {filter_mode}")
    by_prediction: dict[str, list[tuple[NewsRecord, str, float]]] = defaultdict(list)
    for row in rows:
        by_prediction[row[1]].append(row)
    target_counts = _balanced_counts(sorted(by_prediction), pseudo_count)
    selected: list[tuple[NewsRecord, str, float]] = []
    for label in sorted(by_prediction):
        selected.extend(
            sorted(
                by_prediction[label],
                key=lambda row: (row[2], -row[0].token_count, row[0].record_id),
                reverse=True,
            )[: target_counts[label]]
        )
    return selected


def run_self_training_condition(
    records: tuple[NewsRecord, ...],
    seed: int,
    train_budget: int,
    pseudo_multiplier: int,
    filter_mode: str,
    validation_per_class: int,
    heldout_per_class: int,
    teacher_epochs: int,
    student_epochs: int,
) -> dict:
    split = stratified_newsgroups_split(records, seed, validation_per_class, heldout_per_class)
    labels = tuple(sorted({record.label for record in records}))
    labeled_candidate = _class_balanced_sample(split.train_pool, train_budget, seed + 20_001)
    labeled = labeled_candidate.records

    teacher = MulticlassPerceptronClassifier(labels)
    teacher_updates = teacher.fit(labeled, epochs=teacher_epochs, seed=seed + 30_001)
    labeled_ids = {record.record_id for record in labeled}
    unlabeled_pool = tuple(record for record in split.train_pool if record.record_id not in labeled_ids)

    scored_rows: list[tuple[NewsRecord, str, float]] = []
    for record in unlabeled_pool:
        prediction, margin = _teacher_prediction_margin(teacher, labels, record.text)
        scored_rows.append((record, prediction, margin))
    pseudo_count = min(len(scored_rows), train_budget * pseudo_multiplier)
    selected_rows = _select_pseudo_rows(scored_rows, pseudo_count, filter_mode)
    pseudo_records = tuple(
        NewsRecord(record_id=f"pseudo/{record.record_id}", label=prediction, text=record.text)
        for record, prediction, _ in selected_rows
    )

    student = MulticlassPerceptronClassifier(labels)
    student_train = labeled + pseudo_records
    student_updates = student.fit(student_train, epochs=student_epochs, seed=seed + 40_001)
    heldout = student.evaluate(split.heldout)
    validation = student.evaluate(split.validation)
    baseline = _majority_metrics(labeled, split.heldout)

    labeled_tokens = sum(record.token_count for record in labeled)
    pseudo_tokens = sum(record.token_count for record in pseudo_records)
    unlabeled_pool_tokens = sum(record.token_count for record in unlabeled_pool)
    teacher_training_cost_tokens = labeled_tokens * teacher_epochs
    pseudo_scoring_cost_tokens = unlabeled_pool_tokens
    student_training_cost_tokens = (labeled_tokens + pseudo_tokens) * student_epochs
    charged_compute_units = (
        labeled_candidate.selection_cost_tokens
        + teacher_training_cost_tokens
        + pseudo_scoring_cost_tokens
        + student_training_cost_tokens
    )
    pseudo_label_agreement = (
        sum(1 for record, prediction, _ in selected_rows if record.label == prediction) / max(1, len(selected_rows))
    )
    mean_margin = mean(row[2] for row in selected_rows) if selected_rows else 0.0
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_accuracy_improvement = max(0.0, accuracy_improvement)

    condition = f"class_balanced_self_training_{filter_mode}_{pseudo_multiplier}x"
    return {
        "seed": seed,
        "condition": condition,
        "base_policy": "class_balanced_sample",
        "filter_mode": filter_mode,
        "pseudo_multiplier": pseudo_multiplier,
        "train_budget": train_budget,
        "external_events": len(labeled),
        "internal_examples": len(student_train),
        "pseudo_examples": len(pseudo_records),
        "internal_tokens": labeled_tokens + pseudo_tokens,
        "labeled_tokens": labeled_tokens,
        "pseudo_tokens": pseudo_tokens,
        "unlabeled_pool_size": len(unlabeled_pool),
        "unlabeled_pool_tokens": unlabeled_pool_tokens,
        "selection_cost_tokens": labeled_candidate.selection_cost_tokens,
        "teacher_training_cost_tokens": teacher_training_cost_tokens,
        "pseudo_scoring_cost_tokens": pseudo_scoring_cost_tokens,
        "student_training_cost_tokens": student_training_cost_tokens,
        "charged_compute_units": charged_compute_units,
        "teacher_updates": teacher_updates,
        "student_updates": student_updates,
        "pseudo_label_agreement": _round(pseudo_label_agreement),
        "pseudo_label_mean_margin": _round(mean_margin),
        "heldout_used_for_selection": False,
        "heldout_accuracy": _round(heldout.accuracy),
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(accuracy_improvement),
        "signed_external_sample_efficiency": _round(accuracy_improvement / max(1, len(labeled))),
        "clipped_external_sample_efficiency": _round(positive_accuracy_improvement / max(1, len(labeled))),
        "signed_compute_efficiency_per_10k_units": _round(10_000.0 * accuracy_improvement / max(1, charged_compute_units)),
        "clipped_compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_accuracy_improvement / max(1, charged_compute_units)
        ),
        "signed_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * accuracy_improvement / max(1, len(labeled) * charged_compute_units)
        ),
        "clipped_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_accuracy_improvement / max(1, len(labeled) * charged_compute_units)
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        "external_events",
        "internal_examples",
        "pseudo_examples",
        "internal_tokens",
        "labeled_tokens",
        "pseudo_tokens",
        "unlabeled_pool_size",
        "unlabeled_pool_tokens",
        "selection_cost_tokens",
        "teacher_training_cost_tokens",
        "pseudo_scoring_cost_tokens",
        "student_training_cost_tokens",
        "charged_compute_units",
        "teacher_updates",
        "student_updates",
        "pseudo_label_agreement",
        "pseudo_label_mean_margin",
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


def build_self_training_audit(
    records: tuple[NewsRecord, ...],
    source_artifact: dict,
    seeds: tuple[int, ...] = DEFAULT_NEWSGROUPS_SEEDS,
    train_budgets: tuple[int, ...] = DEFAULT_NEWSGROUPS_BUDGETS,
    pseudo_multipliers: tuple[int, ...] = PSEUDO_MULTIPLIERS,
    filter_modes: tuple[str, ...] = FILTER_MODES,
    validation_per_class: int = 20,
    heldout_per_class: int = 20,
    teacher_epochs: int = 2,
    student_epochs: int = 3,
) -> dict:
    per_seed: list[dict] = []
    budget_results: dict[str, dict] = {}
    for train_budget in train_budgets:
        grouped: dict[str, list[dict]] = {}
        for seed in seeds:
            for filter_mode in filter_modes:
                for pseudo_multiplier in pseudo_multipliers:
                    row = run_self_training_condition(
                        records=records,
                        seed=seed,
                        train_budget=train_budget,
                        pseudo_multiplier=pseudo_multiplier,
                        filter_mode=filter_mode,
                        validation_per_class=validation_per_class,
                        heldout_per_class=heldout_per_class,
                        teacher_epochs=teacher_epochs,
                        student_epochs=student_epochs,
                    )
                    grouped.setdefault(row["condition"], []).append(row)
                    per_seed.append(row)
        condition_stats = {condition: _aggregate(rows) for condition, rows in grouped.items()}
        reference = source_artifact["budgets"][str(train_budget)]["conditions"]
        best_self_training_condition = max(
            condition_stats,
            key=lambda condition: (
                condition_stats[condition]["signed_learning_signal_density_per_1m_event_compute_mean"],
                condition_stats[condition]["heldout_accuracy_mean"],
            ),
        )
        budget_results[str(train_budget)] = {
            "random_reference": reference["random_sample"],
            "class_balanced_reference": reference["class_balanced_sample"],
            "condition_results": condition_stats,
            "best_self_training_condition": best_self_training_condition,
        }

    label_counts = Counter(record.label for record in records)
    return {
        "title": "Twenty Newsgroups Self-Training Pseudo-Label Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(SOURCE_ARTIFACT)],
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {"record_count": len(records), "label_count": len(label_counts), "label_counts": dict(sorted(label_counts.items()))},
        "seeds": list(seeds),
        "train_budgets": list(train_budgets),
        "pseudo_multipliers": list(pseudo_multipliers),
        "filter_modes": list(filter_modes),
        "validation_per_class": validation_per_class,
        "heldout_per_class": heldout_per_class,
        "teacher_epochs": teacher_epochs,
        "student_epochs": student_epochs,
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "heldout_used_for_selection": False,
            "metadata_stripped": True,
            "pseudo_labels_use_teacher_predictions": True,
            "oracle_train_labels_used_for_pseudo_label_selection": False,
            "post_hoc_optimization_attempt": True,
            "paper_ready_claim": False,
        },
        "condition_scope": {
            "class_balanced_self_training": {
                "base_policy": "class_balanced_sample",
                "teacher_model": "multiclass_perceptron",
                "student_model": "multiclass_perceptron",
                "pseudo_label_filter": "teacher_margin",
                "heldout_used_for_selection": False,
                "oracle_train_labels_used_for_pseudo_label_selection": False,
                "cost_model": (
                    "class-balanced label-index cost plus teacher training, unlabeled-pool scoring, "
                    "and student training over labeled plus pseudo-labeled examples"
                ),
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
        "This is a real NLP pseudo-label/self-training audit over UCI Twenty Newsgroups mini.",
        "Pseudo-labels are teacher predictions only; oracle train labels are recorded for diagnostics but not used for selection or student labels.",
        "",
        "| Budget | Best condition | Acc | Pseudo agree | LSD | Class-balanced LSD | Random LSD |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_condition = budget_row["best_self_training_condition"]
        best = budget_row["condition_results"][best_condition]
        class_ref = budget_row["class_balanced_reference"]
        random_ref = budget_row["random_reference"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    best_condition,
                    f"{best['heldout_accuracy_mean']:.3f}",
                    f"{best['pseudo_label_agreement_mean']:.3f}",
                    f"{best['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{class_ref['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
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
            "- The teacher pseudo-labels are too noisy in this scarce-label setting.",
            "- Confidence filtering by teacher margin does not rescue learning-signal density once scoring and student training costs are charged.",
            "- This is a negative control for distillation and synthetic-data filtering claims in the current repo.",
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
    parser.add_argument("--output-json", type=Path, default=Path("results/twenty_newsgroups_self_training_audit.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/twenty_newsgroups_self_training_audit.md"))
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    source_artifact = load_json(args.source_json)
    artifact = build_self_training_audit(records=records, source_artifact=source_artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
