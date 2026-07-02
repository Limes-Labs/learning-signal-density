#!/usr/bin/env python3
"""Audit sampled length-window selectors on Twenty Newsgroups."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import math
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
    _majority_metrics,
    _round,
    load_twenty_newsgroups_records,
    run_newsgroups_condition,
    stratified_newsgroups_split,
)


SOURCE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
CONFIRMATION_SEEDS = (331, 337, 347, 349, 353)
QUALITY_KEY = "accuracy_improvement_over_majority_mean"
QUALITY_UPPER_BOUND = 0.95
REUSABLE_COMPUTE_KEYS = ("selection_cost_tokens_mean",)
LENGTH_WINDOW_CONDITIONS = (
    {
        "condition": "length_window_weighted_1.5_2x",
        "mode": "length_weighted",
        "scan_multiplier": 2,
        "exponent": 1.5,
    },
    {
        "condition": "length_window_top_0.25_2x",
        "mode": "top_fraction_random",
        "scan_multiplier": 2,
        "top_fraction": 0.25,
    },
    {
        "condition": "length_window_band_0.20_0.70_1x",
        "mode": "band_random",
        "scan_multiplier": 1,
        "band_low": 0.20,
        "band_high": 0.70,
    },
    {
        "condition": "length_window_shortest_2x",
        "mode": "shortest",
        "scan_multiplier": 2,
    },
    {
        "condition": "length_window_short_diverse_2x",
        "mode": "shortest_diverse_signature",
        "scan_multiplier": 2,
    },
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _scan_window(records: tuple[NewsRecord, ...], scan_size: int, seed: int) -> tuple[NewsRecord, ...]:
    if scan_size >= len(records):
        return records
    rng = random.Random(seed)
    return tuple(rng.sample(list(records), scan_size))


def _select_length_window_records(
    scan_records: tuple[NewsRecord, ...],
    train_budget: int,
    config: dict,
    seed: int,
) -> tuple[NewsRecord, ...]:
    if train_budget <= 0:
        raise ValueError("train_budget must be positive")
    if len(scan_records) < train_budget:
        raise ValueError("scan window must be at least as large as the train budget")
    ranked = sorted(scan_records, key=lambda record: (record.token_count, record.record_id))
    rng = random.Random(seed)
    mode = config["mode"]
    if mode == "shortest":
        return tuple(ranked[:train_budget])
    if mode == "top_fraction_random":
        top_fraction = float(config["top_fraction"])
        candidate_count = max(train_budget, math.ceil(len(ranked) * top_fraction))
        return tuple(rng.sample(ranked[:candidate_count], train_budget))
    if mode == "band_random":
        low = float(config["band_low"])
        high = float(config["band_high"])
        candidates = ranked[int(len(ranked) * low) : max(train_budget, int(len(ranked) * high))]
        if len(candidates) < train_budget:
            candidates = ranked
        return tuple(rng.sample(candidates, train_budget))
    if mode == "length_weighted":
        exponent = float(config["exponent"])
        scored = []
        for record in ranked:
            weight = 1.0 / (max(1, record.token_count) ** exponent)
            scored.append((rng.random() ** (1.0 / weight), record.record_id, record))
        return tuple(record for _, __, record in sorted(scored, reverse=True)[:train_budget])
    if mode == "shortest_diverse_signature":
        selected: list[NewsRecord] = []
        seen_signatures: set[str] = set()
        for record in ranked:
            signature = (record.text.split() or [""])[0].lower()
            if signature in seen_signatures:
                continue
            selected.append(record)
            seen_signatures.add(signature)
            if len(selected) == train_budget:
                return tuple(selected)
        selected_ids = {record.record_id for record in selected}
        selected.extend(record for record in ranked if record.record_id not in selected_ids)
        return tuple(selected[:train_budget])
    raise ValueError(f"unknown length-window mode: {mode}")


def run_length_window_condition(
    records: tuple[NewsRecord, ...],
    seed: int,
    train_budget: int,
    config: dict,
    validation_per_class: int,
    heldout_per_class: int,
    epochs: int,
) -> dict:
    split = stratified_newsgroups_split(records, seed, validation_per_class, heldout_per_class)
    scan_size = min(len(split.train_pool), train_budget * int(config["scan_multiplier"]))
    scan_records = _scan_window(
        split.train_pool,
        scan_size,
        seed + 60_001 + int(config["scan_multiplier"]),
    )
    selected_records = _select_length_window_records(
        scan_records,
        train_budget,
        config,
        seed + 70_001 + int(config["scan_multiplier"]),
    )
    labels = tuple(sorted({record.label for record in records}))
    learner = MulticlassPerceptronClassifier(labels)
    updates = learner.fit(selected_records, epochs=epochs, seed=seed)
    heldout = learner.evaluate(split.heldout)
    validation = learner.evaluate(split.validation)
    baseline = _majority_metrics(selected_records, split.heldout)

    internal_tokens = sum(record.token_count for record in selected_records)
    scan_tokens = sum(record.token_count for record in scan_records)
    training_cost_tokens = internal_tokens * epochs
    charged_compute_units = training_cost_tokens + scan_tokens
    accuracy_improvement = heldout.accuracy - baseline.accuracy
    positive_accuracy_improvement = max(0.0, accuracy_improvement)
    selected_label_counts = Counter(record.label for record in selected_records)

    return {
        "seed": seed,
        "condition": config["condition"],
        "selection_mode": config["mode"],
        "scan_multiplier": config["scan_multiplier"],
        "train_budget": train_budget,
        "external_events": len(selected_records),
        "internal_examples": len(selected_records),
        "internal_tokens": internal_tokens,
        "scan_window_size": len(scan_records),
        "scan_window_tokens": scan_tokens,
        "selection_cost_tokens": scan_tokens,
        "training_cost_tokens": training_cost_tokens,
        "charged_compute_units": charged_compute_units,
        "perceptron_updates": updates,
        "selected_true_label_counts": dict(sorted(selected_label_counts.items())),
        "heldout_used_for_selection": False,
        "validation_used_for_selection": False,
        "teacher_used_for_selection": False,
        "oracle_train_labels_used_for_selection": False,
        "heldout_accuracy": _round(heldout.accuracy),
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(accuracy_improvement),
        "signed_external_sample_efficiency": _round(accuracy_improvement / max(1, len(selected_records))),
        "clipped_external_sample_efficiency": _round(positive_accuracy_improvement / max(1, len(selected_records))),
        "signed_compute_efficiency_per_10k_units": _round(
            10_000.0 * accuracy_improvement / max(1, charged_compute_units)
        ),
        "clipped_compute_efficiency_per_10k_units": _round(
            10_000.0 * positive_accuracy_improvement / max(1, charged_compute_units)
        ),
        "signed_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * accuracy_improvement / max(1, len(selected_records) * charged_compute_units)
        ),
        "clipped_learning_signal_density_per_1m_event_compute": _round(
            1_000_000.0 * positive_accuracy_improvement / max(1, len(selected_records) * charged_compute_units)
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        "external_events",
        "internal_examples",
        "internal_tokens",
        "scan_window_size",
        "scan_window_tokens",
        "selection_cost_tokens",
        "training_cost_tokens",
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
    return {
        f"{key}_mean": _round(mean(float(row[key]) for row in rows))
        for key in numeric_keys
        if key in rows[0]
    }


def _random_reference_rows(
    records: tuple[NewsRecord, ...],
    seeds: tuple[int, ...],
    train_budget: int,
    validation_per_class: int,
    heldout_per_class: int,
    epochs: int,
    proxy_epochs: int,
) -> list[dict]:
    return [
        run_newsgroups_condition(
            records=records,
            seed=seed,
            condition="random_sample",
            train_budget=train_budget,
            validation_per_class=validation_per_class,
            heldout_per_class=heldout_per_class,
            epochs=epochs,
            proxy_epochs=proxy_epochs,
        )
        for seed in seeds
    ]


def build_length_window_confirmation_audit(
    records: tuple[NewsRecord, ...],
    source_artifact: dict,
    development_seeds: tuple[int, ...] = DEFAULT_NEWSGROUPS_SEEDS,
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    train_budgets: tuple[int, ...] = DEFAULT_NEWSGROUPS_BUDGETS,
    conditions: tuple[dict, ...] = LENGTH_WINDOW_CONDITIONS,
    validation_per_class: int = 20,
    heldout_per_class: int = 20,
    epochs: int = 3,
    proxy_epochs: int = 1,
) -> dict:
    phases = {
        "development": tuple(development_seeds),
        "confirmation": tuple(confirmation_seeds),
    }
    phase_results: dict[str, dict] = {}
    per_seed: list[dict] = []
    for phase, seeds in phases.items():
        budget_results: dict[str, dict] = {}
        for train_budget in train_budgets:
            random_rows = _random_reference_rows(
                records,
                seeds,
                train_budget,
                validation_per_class,
                heldout_per_class,
                epochs,
                proxy_epochs,
            )
            random_reference = _aggregate(random_rows) | {"condition": "random_sample"}
            grouped: dict[str, list[dict]] = {}
            for seed in seeds:
                for config in conditions:
                    row = run_length_window_condition(
                        records=records,
                        seed=seed,
                        train_budget=train_budget,
                        config=config,
                        validation_per_class=validation_per_class,
                        heldout_per_class=heldout_per_class,
                        epochs=epochs,
                    )
                    row["phase"] = phase
                    grouped.setdefault(config["condition"], []).append(row)
                    per_seed.append(row)
            condition_results = {}
            for config in conditions:
                condition = config["condition"]
                aggregate = _aggregate(grouped[condition]) | {
                    "condition": condition,
                    "selection_mode": config["mode"],
                    "scan_multiplier": config["scan_multiplier"],
                }
                comparison = break_even_comparison(
                    reference=random_reference,
                    candidate=aggregate,
                    quality_key=QUALITY_KEY,
                    quality_upper_bound=QUALITY_UPPER_BOUND,
                    reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
                )
                condition_results[condition] = aggregate | {"break_even_vs_random": comparison}
            best_density_condition = max(
                condition_results,
                key=lambda condition: (
                    condition_results[condition]["signed_learning_signal_density_per_1m_event_compute_mean"],
                    condition_results[condition]["heldout_accuracy_mean"],
                ),
            )
            density_wins = [
                condition
                for condition, row in condition_results.items()
                if row["break_even_vs_random"]["candidate_density_wins"]
            ]
            budget_results[str(train_budget)] = {
                "random_reference": random_reference,
                "condition_results": condition_results,
                "best_density_condition": best_density_condition,
                "density_win_conditions": density_wins,
            }
        phase_results[phase] = {"seeds": list(seeds), "budgets": budget_results}

    label_counts = Counter(record.label for record in records)
    selected_development_conditions_by_budget = {
        budget: phase_results["development"]["budgets"][str(budget)]["best_density_condition"]
        for budget in train_budgets
    }
    development_density_win_count = sum(
        len(phase_results["development"]["budgets"][str(budget)]["density_win_conditions"])
        for budget in train_budgets
    )
    confirmation_same_condition_density_win_count = sum(
        1
        for budget, condition in selected_development_conditions_by_budget.items()
        if phase_results["confirmation"]["budgets"][str(budget)]["condition_results"][condition]["break_even_vs_random"][
            "candidate_density_wins"
        ]
    )
    return {
        "title": "Twenty Newsgroups Length-Window Confirmation Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(SOURCE_ARTIFACT)],
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {
            "record_count": len(records),
            "label_count": len(label_counts),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "development_seeds": list(development_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "train_budgets": list(train_budgets),
        "conditions": list(conditions),
        "validation_per_class": validation_per_class,
        "heldout_per_class": heldout_per_class,
        "epochs": epochs,
        "quality_metric": QUALITY_KEY,
        "quality_upper_bound": QUALITY_UPPER_BOUND,
        "selected_development_conditions_by_budget": selected_development_conditions_by_budget,
        "development_density_win_count": development_density_win_count,
        "confirmation_same_condition_density_win_count": confirmation_same_condition_density_win_count,
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "metadata_stripped": True,
            "heldout_used_for_selection": False,
            "validation_used_for_selection": False,
            "teacher_used_for_selection": False,
            "oracle_train_labels_used_for_selection": False,
            "scan_window_sampled_before_length_selection": True,
            "post_hoc_development_selection": True,
            "confirmation_seeds_disjoint": True,
            "paper_ready_claim": False,
        },
        "condition_scope": {
            "length_window": {
                "selection_signal": "document token count inside a sampled train-only window",
                "selection_cost_model": "scan-window token count plus final training tokens",
                "label_free": True,
                "heldout_used_for_selection": False,
                "validation_used_for_selection": False,
                "teacher_used_for_selection": False,
            }
        },
        "phases": phase_results,
        "per_seed": per_seed,
    }


def render_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        "This is a real NLP stress audit over UCI Twenty Newsgroups mini.",
        "It tests whether sampled length-window selectors produce a stable random-density win.",
        "",
        "| Budget | Dev best | Dev LSD | Dev random | Dev win? | Confirm same LSD | Confirm random | Same win? | Confirm best |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for budget in artifact["train_budgets"]:
        dev_budget = artifact["phases"]["development"]["budgets"][str(budget)]
        confirm_budget = artifact["phases"]["confirmation"]["budgets"][str(budget)]
        dev_condition = dev_budget["best_density_condition"]
        dev_row = dev_budget["condition_results"][dev_condition]
        confirm_same = confirm_budget["condition_results"][dev_condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    dev_condition,
                    f"{dev_row['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{dev_budget['random_reference']['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    str(dev_row["break_even_vs_random"]["candidate_density_wins"]),
                    f"{confirm_same['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{confirm_budget['random_reference']['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    str(confirm_same["break_even_vs_random"]["candidate_density_wins"]),
                    confirm_budget["best_density_condition"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The development grid finds no sampled length-window row that beats random density.",
            "- Some confirmation-phase best rows beat same-phase random at other budgets, but those are not development-selected wins.",
            "- The result is a mixed stress test for cheap length-window selection, not a deployable policy claim.",
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
        default=Path("results/twenty_newsgroups_length_window_confirmation_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/twenty_newsgroups_length_window_confirmation_audit.md"),
    )
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    source_artifact = load_json(args.source_json)
    artifact = build_length_window_confirmation_audit(records=records, source_artifact=source_artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
