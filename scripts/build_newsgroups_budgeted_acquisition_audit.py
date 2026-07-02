#!/usr/bin/env python3
"""Audit budgeted-window active acquisition on Twenty Newsgroups."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import random
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
    NewsRecord,
    _class_balanced_sample,
    _majority_metrics,
    _round,
    load_twenty_newsgroups_records,
    stratified_newsgroups_split,
)
from scripts.build_newsgroups_active_acquisition_audit import (
    _seed_budget,
    _select_acquisition_rows,
    _teacher_prediction_margin,
)


SOURCE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
ACQUISITION_MODES = (
    "margin_uncertainty",
    "balanced_margin_uncertainty",
    "short_margin_uncertainty",
)
SCAN_MULTIPLIERS = (1, 2, 4)
QUALITY_KEY = "accuracy_improvement_over_majority_mean"
QUALITY_UPPER_BOUND = 0.95
REUSABLE_COMPUTE_KEYS = (
    "seed_selection_cost_tokens_mean",
    "teacher_training_cost_tokens_mean",
    "acquisition_scoring_cost_tokens_mean",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _scan_window(
    records: tuple[NewsRecord, ...],
    scan_size: int,
    seed: int,
) -> tuple[NewsRecord, ...]:
    if scan_size >= len(records):
        return records
    rng = random.Random(seed)
    return tuple(rng.sample(list(records), scan_size))


def _select_budgeted_rows(
    scored_rows: list[tuple[NewsRecord, str, float]],
    acquire_count: int,
    mode: str,
) -> list[tuple[NewsRecord, str, float]]:
    selected = list(_select_acquisition_rows(scored_rows, acquire_count, mode))
    if len(selected) >= acquire_count:
        return selected[:acquire_count]
    selected_ids = {row[0].record_id for row in selected}
    remaining = [row for row in scored_rows if row[0].record_id not in selected_ids]
    if mode == "short_margin_uncertainty":
        ranked = sorted(
            remaining,
            key=lambda row: (
                row[2] * max(1, row[0].token_count),
                row[0].token_count,
                row[0].record_id,
            ),
        )
    elif mode == "confidence_curriculum":
        ranked = sorted(remaining, key=lambda row: (row[2], -row[0].token_count, row[0].record_id), reverse=True)
    else:
        ranked = sorted(remaining, key=lambda row: (row[2], row[0].token_count, row[0].record_id))
    selected.extend(ranked[: acquire_count - len(selected)])
    return selected


def run_budgeted_acquisition_condition(
    records: tuple[NewsRecord, ...],
    seed: int,
    train_budget: int,
    acquisition_mode: str,
    scan_multiplier: int,
    validation_per_class: int,
    heldout_per_class: int,
    teacher_epochs: int,
    student_epochs: int,
) -> dict:
    if scan_multiplier <= 0:
        raise ValueError("scan_multiplier must be positive")
    split = stratified_newsgroups_split(records, seed, validation_per_class, heldout_per_class)
    labels = tuple(sorted({record.label for record in records}))
    initial_budget = _seed_budget(train_budget, len(labels))
    seed_candidate = _class_balanced_sample(split.train_pool, initial_budget, seed + 20_001)
    seed_records = seed_candidate.records
    seed_ids = {record.record_id for record in seed_records}
    unlabeled_pool = tuple(record for record in split.train_pool if record.record_id not in seed_ids)

    teacher = MulticlassPerceptronClassifier(labels)
    teacher_updates = teacher.fit(seed_records, epochs=teacher_epochs, seed=seed + 30_001)
    acquire_count = min(len(unlabeled_pool), train_budget - len(seed_records))
    scan_size = min(len(unlabeled_pool), max(acquire_count, acquire_count * scan_multiplier))
    scan_records = _scan_window(unlabeled_pool, scan_size, seed + 50_001 + scan_multiplier)

    scored_rows = []
    for record in scan_records:
        prediction, margin = _teacher_prediction_margin(teacher, labels, record.text)
        scored_rows.append((record, prediction, margin))
    acquired_rows = _select_budgeted_rows(scored_rows, acquire_count, acquisition_mode)
    acquired_records = tuple(row[0] for row in acquired_rows)
    final_records = seed_records + acquired_records

    student = MulticlassPerceptronClassifier(labels)
    student_updates = student.fit(final_records, epochs=student_epochs, seed=seed + 40_001)
    heldout = student.evaluate(split.heldout)
    validation = student.evaluate(split.validation)
    baseline = _majority_metrics(final_records, split.heldout)

    seed_tokens = sum(record.token_count for record in seed_records)
    acquired_tokens = sum(record.token_count for record in acquired_records)
    scan_tokens = sum(record.token_count for record in scan_records)
    teacher_training_cost_tokens = seed_tokens * teacher_epochs
    acquisition_scoring_cost_tokens = scan_tokens
    student_training_cost_tokens = (seed_tokens + acquired_tokens) * student_epochs
    charged_compute_units = (
        seed_candidate.selection_cost_tokens
        + teacher_training_cost_tokens
        + acquisition_scoring_cost_tokens
        + student_training_cost_tokens
    )
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_accuracy_improvement = max(0.0, accuracy_improvement)
    teacher_prediction_counts = Counter(prediction for _, prediction, _ in acquired_rows)
    acquired_true_label_counts = Counter(record.label for record in acquired_records)
    acquired_teacher_agreement = (
        sum(1 for record, prediction, _ in acquired_rows if record.label == prediction) / max(1, len(acquired_rows))
    )
    mean_acquisition_margin = mean(row[2] for row in acquired_rows) if acquired_rows else 0.0
    condition = f"budgeted_active_{acquisition_mode}_{scan_multiplier}x"

    return {
        "seed": seed,
        "condition": condition,
        "base_seed_policy": "class_balanced_sample",
        "acquisition_mode": acquisition_mode,
        "scan_multiplier": scan_multiplier,
        "train_budget": train_budget,
        "seed_examples": len(seed_records),
        "acquired_examples": len(acquired_records),
        "external_events": len(final_records),
        "internal_examples": len(final_records),
        "internal_tokens": seed_tokens + acquired_tokens,
        "seed_tokens": seed_tokens,
        "acquired_tokens": acquired_tokens,
        "unlabeled_pool_size": len(unlabeled_pool),
        "scan_window_size": len(scan_records),
        "scan_window_tokens": scan_tokens,
        "seed_selection_cost_tokens": seed_candidate.selection_cost_tokens,
        "teacher_training_cost_tokens": teacher_training_cost_tokens,
        "acquisition_scoring_cost_tokens": acquisition_scoring_cost_tokens,
        "student_training_cost_tokens": student_training_cost_tokens,
        "charged_compute_units": charged_compute_units,
        "teacher_updates": teacher_updates,
        "student_updates": student_updates,
        "acquired_teacher_agreement": _round(acquired_teacher_agreement),
        "acquired_teacher_prediction_counts": dict(sorted(teacher_prediction_counts.items())),
        "acquired_true_label_counts": dict(sorted(acquired_true_label_counts.items())),
        "mean_acquisition_margin": _round(mean_acquisition_margin),
        "heldout_used_for_selection": False,
        "oracle_train_labels_used_for_acquisition": False,
        "true_labels_acquired_after_selection": True,
        "heldout_accuracy": _round(heldout.accuracy),
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(accuracy_improvement),
        "signed_external_sample_efficiency": _round(accuracy_improvement / max(1, len(final_records))),
        "clipped_external_sample_efficiency": _round(positive_accuracy_improvement / max(1, len(final_records))),
        "signed_compute_efficiency_per_10k_units": _round(10_000.0 * accuracy_improvement / max(1, charged_compute_units)),
        "clipped_compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_accuracy_improvement / max(1, charged_compute_units)
        ),
        "signed_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * accuracy_improvement / max(1, len(final_records) * charged_compute_units)
        ),
        "clipped_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_accuracy_improvement / max(1, len(final_records) * charged_compute_units)
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        "seed_examples",
        "acquired_examples",
        "external_events",
        "internal_examples",
        "internal_tokens",
        "seed_tokens",
        "acquired_tokens",
        "unlabeled_pool_size",
        "scan_window_size",
        "scan_window_tokens",
        "seed_selection_cost_tokens",
        "teacher_training_cost_tokens",
        "acquisition_scoring_cost_tokens",
        "student_training_cost_tokens",
        "charged_compute_units",
        "teacher_updates",
        "student_updates",
        "acquired_teacher_agreement",
        "mean_acquisition_margin",
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


def build_budgeted_acquisition_audit(
    records: tuple[NewsRecord, ...],
    source_artifact: dict,
    seeds: tuple[int, ...] = DEFAULT_NEWSGROUPS_SEEDS,
    train_budgets: tuple[int, ...] = DEFAULT_NEWSGROUPS_BUDGETS,
    acquisition_modes: tuple[str, ...] = ACQUISITION_MODES,
    scan_multipliers: tuple[int, ...] = SCAN_MULTIPLIERS,
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
            for mode in acquisition_modes:
                for scan_multiplier in scan_multipliers:
                    row = run_budgeted_acquisition_condition(
                        records=records,
                        seed=seed,
                        train_budget=train_budget,
                        acquisition_mode=mode,
                        scan_multiplier=scan_multiplier,
                        validation_per_class=validation_per_class,
                        heldout_per_class=heldout_per_class,
                        teacher_epochs=teacher_epochs,
                        student_epochs=student_epochs,
                    )
                    grouped.setdefault(row["condition"], []).append(row)
                    per_seed.append(row)

        source_conditions = source_artifact["budgets"][str(train_budget)]["conditions"]
        reference_random = {"condition": "random_sample"} | source_conditions["random_sample"]
        reference_class = {"condition": "class_balanced_sample"} | source_conditions["class_balanced_sample"]
        condition_results = {}
        for condition, rows in grouped.items():
            aggregate = _aggregate(rows) | {"condition": condition}
            break_even_random = break_even_comparison(
                reference=reference_random,
                candidate=aggregate,
                quality_key=QUALITY_KEY,
                quality_upper_bound=QUALITY_UPPER_BOUND,
                reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
            )
            break_even_class = break_even_comparison(
                reference=reference_class,
                candidate=aggregate,
                quality_key=QUALITY_KEY,
                quality_upper_bound=QUALITY_UPPER_BOUND,
                reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
            )
            condition_results[condition] = aggregate | {
                "break_even_vs_random": break_even_random,
                "break_even_vs_class_balanced": break_even_class,
            }

        best_accuracy_condition = max(
            condition_results,
            key=lambda condition: (
                condition_results[condition]["heldout_accuracy_mean"],
                condition_results[condition]["signed_learning_signal_density_per_1m_event_compute_mean"],
            ),
        )
        best_density_condition = max(
            condition_results,
            key=lambda condition: (
                condition_results[condition]["signed_learning_signal_density_per_1m_event_compute_mean"],
                condition_results[condition]["heldout_accuracy_mean"],
            ),
        )
        budget_results[str(train_budget)] = {
            "random_reference": reference_random,
            "class_balanced_reference": reference_class,
            "condition_results": condition_results,
            "best_accuracy_condition": best_accuracy_condition,
            "best_density_condition": best_density_condition,
        }

    label_counts = Counter(record.label for record in records)
    return {
        "title": "Twenty Newsgroups Budgeted Active Label-Acquisition Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(SOURCE_ARTIFACT)],
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {
            "record_count": len(records),
            "label_count": len(label_counts),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "seeds": list(seeds),
        "train_budgets": list(train_budgets),
        "acquisition_modes": list(acquisition_modes),
        "scan_multipliers": list(scan_multipliers),
        "validation_per_class": validation_per_class,
        "heldout_per_class": heldout_per_class,
        "teacher_epochs": teacher_epochs,
        "student_epochs": student_epochs,
        "quality_metric": QUALITY_KEY,
        "quality_upper_bound": QUALITY_UPPER_BOUND,
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "metadata_stripped": True,
            "heldout_used_for_selection": False,
            "validation_used_for_selection": False,
            "oracle_train_labels_used_for_acquisition": False,
            "true_labels_acquired_after_selection": True,
            "scan_window_sampled_without_text_scoring": True,
            "post_hoc_optimization_attempt": True,
            "paper_ready_claim": False,
        },
        "condition_scope": {
            "budgeted_active_acquisition": {
                "seed_policy": "class_balanced_sample",
                "seed_selection_cost_model": "one_unit_per_train_pool_label",
                "teacher_model": "multiclass_perceptron",
                "student_model": "multiclass_perceptron",
                "acquisition_signal": "teacher margin over sampled train-only unlabeled window",
                "heldout_used_for_selection": False,
                "validation_used_for_selection": False,
                "oracle_train_labels_used_for_acquisition": False,
                "true_labels_acquired_after_selection": True,
                "cost_model": (
                    "class-balanced seed label-index cost plus teacher training, scored-window tokens, "
                    "and final student training over seed plus acquired true-label examples"
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
        "This is a real NLP budgeted active label-acquisition audit over UCI Twenty Newsgroups mini.",
        "The teacher scores only a sampled window of the train-only pool before true labels are acquired.",
        "",
        "| Budget | Best density condition | Acc | Scan size | LSD | Random LSD | Class LSD | Density win? |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_condition = budget_row["best_density_condition"]
        best = budget_row["condition_results"][best_condition]
        random_ref = budget_row["random_reference"]
        class_ref = budget_row["class_balanced_reference"]
        density_win = best["break_even_vs_random"]["candidate_density_wins"] or best["break_even_vs_class_balanced"][
            "candidate_density_wins"
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    best_condition.replace("budgeted_active_", ""),
                    f"{best['heldout_accuracy_mean']:.3f}",
                    f"{best['scan_window_size_mean']:.1f}",
                    f"{best['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{random_ref['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{class_ref['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    str(density_win),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit tests whether reducing active-acquisition scan cost can move the density frontier.",
            "- The scan window is sampled before teacher scoring; true labels are used only after acquisition.",
            "- A density win requires the acquired examples to pay for seed selection, teacher training, window scoring, and final training.",
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/twenty_newsgroups_budgeted_acquisition_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/twenty_newsgroups_budgeted_acquisition_audit.md"),
    )
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    source_artifact = load_json(args.source_json)
    artifact = build_budgeted_acquisition_audit(records=records, source_artifact=source_artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
