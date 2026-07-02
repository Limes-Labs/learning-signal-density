#!/usr/bin/env python3
"""Run a fresh-seed replication of the Newsgroups density frontier."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from itertools import product
import json
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from learning_signal_density.newsgroups_experiment import (
    TWENTY_NEWSGROUPS_DATASET,
    _round,
    load_twenty_newsgroups_records,
    run_newsgroups_condition,
)
from scripts.build_newsgroups_budgeted_acquisition_audit import run_budgeted_acquisition_condition


FRESH_SEEDS = (439, 443, 449, 457, 461)
QUALITY_KEY = "accuracy_improvement_over_majority"
LSD_KEY = "signed_learning_signal_density_per_1m_event_compute"
ACTIVE_RUNS = (
    (40, "random_sample"),
    (40, "prototype_retrieval_sample"),
    (80, "random_sample"),
    (80, "class_balanced_sample"),
    (160, "random_sample"),
    (160, "class_balanced_sample"),
    (160, "prototype_retrieval_sample"),
)
BUDGETED_RUNS = (
    (160, "margin_uncertainty", 1),
    (160, "margin_uncertainty", 2),
)
COMPARISONS = (
    {
        "comparison": "class_balanced_80_vs_random",
        "budget": 80,
        "candidate_family": "active_selection",
        "candidate_condition": "class_balanced_sample",
        "reference_family": "active_selection",
        "reference_condition": "random_sample",
    },
    {
        "comparison": "prototype_40_vs_random",
        "budget": 40,
        "candidate_family": "active_selection",
        "candidate_condition": "prototype_retrieval_sample",
        "reference_family": "active_selection",
        "reference_condition": "random_sample",
    },
    {
        "comparison": "prototype_160_vs_random",
        "budget": 160,
        "candidate_family": "active_selection",
        "candidate_condition": "prototype_retrieval_sample",
        "reference_family": "active_selection",
        "reference_condition": "random_sample",
    },
    {
        "comparison": "budgeted_margin_1x_160_vs_class_balanced",
        "budget": 160,
        "candidate_family": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_1x",
        "reference_family": "active_selection",
        "reference_condition": "class_balanced_sample",
    },
    {
        "comparison": "budgeted_margin_2x_160_vs_class_balanced",
        "budget": 160,
        "candidate_family": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_2x",
        "reference_family": "active_selection",
        "reference_condition": "class_balanced_sample",
    },
    {
        "comparison": "budgeted_margin_2x_160_vs_random",
        "budget": 160,
        "candidate_family": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_2x",
        "reference_family": "active_selection",
        "reference_condition": "random_sample",
    },
)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _density_ratio(candidate: float, reference: float, *, context: str) -> float | None:
    if reference <= 0:
        return None
    return candidate / reference


def _exact_seed_bootstrap(paired_rows: list[dict]) -> dict:
    n = len(paired_rows)
    density_deltas = []
    density_ratios = []
    quality_deltas = []
    for indices in product(range(n), repeat=n):
        candidate_density = mean(paired_rows[index]["candidate_lsd"] for index in indices)
        reference_density = mean(paired_rows[index]["reference_lsd"] for index in indices)
        candidate_quality = mean(paired_rows[index]["candidate_quality"] for index in indices)
        reference_quality = mean(paired_rows[index]["reference_quality"] for index in indices)
        density_deltas.append(candidate_density - reference_density)
        density_ratio = _density_ratio(candidate_density, reference_density, context="bootstrap sample")
        if density_ratio is not None:
            density_ratios.append(density_ratio)
        quality_deltas.append(candidate_quality - reference_quality)
    return {
        "bootstrap_sample_count": n**n,
        "density_delta_ci95": [
            _round(_quantile(density_deltas, 0.025)),
            _round(_quantile(density_deltas, 0.975)),
        ],
        "density_ratio_ci95": (
            [
                _round(_quantile(density_ratios, 0.025)),
                _round(_quantile(density_ratios, 0.975)),
            ]
            if density_ratios
            else None
        ),
        "density_ratio_defined_sample_count": len(density_ratios),
        "quality_delta_ci95": [
            _round(_quantile(quality_deltas, 0.025)),
            _round(_quantile(quality_deltas, 0.975)),
        ],
    }


def _sign_test_one_sided_p_value(win_count: int, seed_count: int, *, direction: str) -> float:
    if direction == "win":
        numerator = sum(_binomial(seed_count, k) for k in range(win_count, seed_count + 1))
    elif direction == "loss":
        loss_count = seed_count - win_count
        numerator = sum(_binomial(seed_count, k) for k in range(loss_count, seed_count + 1))
    else:
        raise ValueError(f"unknown sign-test direction: {direction}")
    return _round(numerator / (2**seed_count))


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for i in range(1, k + 1):
        numerator *= n - (k - i)
        denominator *= i
    return numerator // denominator


def _format_optional_float(value: float | None, digits: int) -> str:
    if value is None:
        return "undefined"
    return f"{value:.{digits}f}"


def _run_active_rows(records: tuple, seeds: tuple[int, ...]) -> dict[tuple[int, int, str], dict]:
    rows = {}
    for seed in seeds:
        for train_budget, condition in ACTIVE_RUNS:
            row = run_newsgroups_condition(
                records=records,
                seed=seed,
                condition=condition,
                train_budget=train_budget,
                validation_per_class=20,
                heldout_per_class=20,
                epochs=3,
                proxy_epochs=1,
            )
            rows[(seed, train_budget, condition)] = row
    return rows


def _run_budgeted_rows(records: tuple, seeds: tuple[int, ...]) -> dict[tuple[int, int, str], dict]:
    rows = {}
    for seed in seeds:
        for train_budget, acquisition_mode, scan_multiplier in BUDGETED_RUNS:
            row = run_budgeted_acquisition_condition(
                records=records,
                seed=seed,
                train_budget=train_budget,
                acquisition_mode=acquisition_mode,
                scan_multiplier=scan_multiplier,
                validation_per_class=20,
                heldout_per_class=20,
                teacher_epochs=2,
                student_epochs=3,
            )
            rows[(seed, train_budget, row["condition"])] = row
    return rows


def _paired_comparison(config: dict, rows_by_family: dict[str, dict], seeds: tuple[int, ...]) -> dict:
    paired_rows = []
    budget = int(config["budget"])
    for seed in seeds:
        candidate = rows_by_family[config["candidate_family"]][(seed, budget, config["candidate_condition"])]
        reference = rows_by_family[config["reference_family"]][(seed, budget, config["reference_condition"])]
        candidate_lsd = float(candidate[LSD_KEY])
        reference_lsd = float(reference[LSD_KEY])
        candidate_quality = float(candidate[QUALITY_KEY])
        reference_quality = float(reference[QUALITY_KEY])
        density_ratio = _density_ratio(candidate_lsd, reference_lsd, context=f"{config['comparison']} seed {seed}")
        paired_rows.append(
            {
                "seed": seed,
                "candidate_lsd": _round(candidate_lsd),
                "reference_lsd": _round(reference_lsd),
                "density_delta": _round(candidate_lsd - reference_lsd),
                "density_ratio": _round(density_ratio) if density_ratio is not None else None,
                "candidate_quality": _round(candidate_quality),
                "reference_quality": _round(reference_quality),
                "quality_delta": _round(candidate_quality - reference_quality),
                "candidate_density_wins": candidate_lsd > reference_lsd,
            }
        )
    bootstrap = _exact_seed_bootstrap(paired_rows)
    mean_candidate_lsd = mean(row["candidate_lsd"] for row in paired_rows)
    mean_reference_lsd = mean(row["reference_lsd"] for row in paired_rows)
    mean_density_delta = mean_candidate_lsd - mean_reference_lsd
    mean_density_ratio = _density_ratio(mean_candidate_lsd, mean_reference_lsd, context=config["comparison"])
    paired_win_count = sum(1 for row in paired_rows if row["candidate_density_wins"])
    ci_low, ci_high = bootstrap["density_delta_ci95"]
    robust_density_win = bool(mean_density_delta > 0 and paired_win_count == len(paired_rows) and ci_low > 0)
    robust_density_loss = bool(mean_density_delta < 0 and paired_win_count == 0 and ci_high < 0)
    return config | {
        "seed_count": len(paired_rows),
        "paired_win_count": paired_win_count,
        "mean_candidate_lsd": _round(mean_candidate_lsd),
        "mean_reference_lsd": _round(mean_reference_lsd),
        "mean_density_delta": _round(mean_density_delta),
        "mean_density_ratio": _round(mean_density_ratio) if mean_density_ratio is not None else None,
        "mean_density_win": mean_density_delta > 0,
        "robust_density_win": robust_density_win,
        "robust_density_loss": robust_density_loss,
        "sign_test_one_sided_win_p": _sign_test_one_sided_p_value(paired_win_count, len(paired_rows), direction="win"),
        "sign_test_one_sided_loss_p": _sign_test_one_sided_p_value(paired_win_count, len(paired_rows), direction="loss"),
        "bootstrap": bootstrap,
        "paired_rows": paired_rows,
    }


def build_fresh_seed_audit(records: tuple, seeds: tuple[int, ...] = FRESH_SEEDS) -> dict:
    rows_by_family = {
        "active_selection": _run_active_rows(records, seeds),
        "budgeted_acquisition": _run_budgeted_rows(records, seeds),
    }
    comparisons = [_paired_comparison(config, rows_by_family, seeds) for config in COMPARISONS]
    label_counts = Counter(record.label for record in records)
    return {
        "title": "Twenty Newsgroups Fresh-Seed Frontier Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {
            "record_count": len(records),
            "label_count": len(label_counts),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "fresh_seeds": list(seeds),
        "active_runs": [{"train_budget": budget, "condition": condition} for budget, condition in ACTIVE_RUNS],
        "budgeted_runs": [
            {"train_budget": budget, "acquisition_mode": mode, "scan_multiplier": scan_multiplier}
            for budget, mode, scan_multiplier in BUDGETED_RUNS
        ],
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "metadata_stripped": True,
            "fresh_seed_replication": True,
            "preregistered_seed_list": True,
            "paired_seed_audit": True,
            "exact_seed_bootstrap": True,
            "heldout_used_for_selection": False,
            "validation_used_for_selection": False,
            "paper_ready_claim": False,
        },
        "robustness_rule": {
            "robust_density_win": (
                "mean density delta is positive, every fresh paired seed wins, and the exact seed-bootstrap "
                "95% interval for mean density delta is strictly positive"
            ),
            "robust_density_loss": (
                "mean density delta is negative, every fresh paired seed loses, and the exact seed-bootstrap "
                "95% interval for mean density delta is strictly negative"
            ),
        },
        "summary": {
            "comparisons": len(comparisons),
            "fresh_seed_count": len(seeds),
            "mean_density_wins": sum(1 for row in comparisons if row["mean_density_win"]),
            "robust_density_wins": sum(1 for row in comparisons if row["robust_density_win"]),
            "robust_density_losses": sum(1 for row in comparisons if row["robust_density_loss"]),
            "fragile_mean_density_wins": sum(
                1 for row in comparisons if row["mean_density_win"] and not row["robust_density_win"]
            ),
        },
        "comparisons": comparisons,
    }


def render_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        f"Fresh seeds: `{artifact['fresh_seeds']}`",
        "",
        "| Comparison | Mean ratio | Paired wins | Delta CI95 | Sign p(win) | Sign p(loss) | Reading |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in artifact["comparisons"]:
        if row["robust_density_win"]:
            reading = "robust density win"
        elif row["robust_density_loss"]:
            reading = "robust density loss"
        elif row["mean_density_win"]:
            reading = "fragile mean win"
        else:
            reading = "mixed/no win"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    _format_optional_float(row["mean_density_ratio"], 3),
                    f"{row['paired_win_count']}/{row['seed_count']}",
                    f"[{row['bootstrap']['density_delta_ci95'][0]:.6f}, {row['bootstrap']['density_delta_ci95'][1]:.6f}]",
                    f"{row['sign_test_one_sided_win_p']:.6f}",
                    f"{row['sign_test_one_sided_loss_p']:.6f}",
                    reading,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The fresh-seed audit is a replication check, not a policy-search stage.",
            f"- Robust density wins under the fresh-seed rule: {artifact['summary']['robust_density_wins']}.",
            f"- Robust density losses under the fresh-seed rule: {artifact['summary']['robust_density_losses']}.",
            "- The positive class-balanced 80-label mean win remains fragile on untouched seeds.",
            "- Budgeted active-acquisition rows do not replicate as density wins on untouched seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=Path("data/external/twenty_newsgroups.zip"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/twenty_newsgroups_frontier_fresh_seed_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/twenty_newsgroups_frontier_fresh_seed_audit.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FRESH_SEEDS))
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    artifact = build_fresh_seed_audit(records, seeds=tuple(args.seeds))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
