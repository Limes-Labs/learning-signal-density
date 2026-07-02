#!/usr/bin/env python3
"""Run a targeted fresh-seed confirmation for class-balanced Newsgroups sampling."""

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
from learning_signal_density.newsgroups_experiment import (
    TWENTY_NEWSGROUPS_DATASET,
    _round,
    load_twenty_newsgroups_records,
    run_newsgroups_condition,
)


CONFIRMATION_SEEDS = (
    2003,
    2011,
    2017,
    2027,
    2029,
    2039,
    2053,
    2063,
    2069,
    2081,
    2083,
    2087,
    2089,
    2099,
    2111,
    2113,
    2129,
    2131,
    2137,
    2141,
)
TRAIN_BUDGET = 80
CANDIDATE_CONDITION = "class_balanced_sample"
REFERENCE_CONDITION = "random_sample"
QUALITY_KEY = "accuracy_improvement_over_majority"
LSD_KEY = "signed_learning_signal_density_per_1m_event_compute"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260702


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _density_ratio(candidate: float, reference: float) -> float | None:
    if reference <= 0:
        return None
    return candidate / reference


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for i in range(1, k + 1):
        numerator *= n - (k - i)
        denominator *= i
    return numerator // denominator


def _sign_test_one_sided_p_value(win_count: int, seed_count: int) -> float:
    numerator = sum(_binomial(seed_count, k) for k in range(win_count, seed_count + 1))
    return _round(numerator / (2**seed_count))


def _paired_bootstrap(paired_rows: list[dict], resamples: int, seed: int) -> dict:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    density_deltas = []
    density_ratios = []
    quality_deltas = []
    n = len(paired_rows)
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        candidate_density = mean(paired_rows[index]["candidate_lsd"] for index in indices)
        reference_density = mean(paired_rows[index]["reference_lsd"] for index in indices)
        candidate_quality = mean(paired_rows[index]["candidate_quality"] for index in indices)
        reference_quality = mean(paired_rows[index]["reference_quality"] for index in indices)
        density_deltas.append(candidate_density - reference_density)
        density_ratio = _density_ratio(candidate_density, reference_density)
        if density_ratio is not None:
            density_ratios.append(density_ratio)
        quality_deltas.append(candidate_quality - reference_quality)
    return {
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
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
        "quality_delta_ci95": [
            _round(_quantile(quality_deltas, 0.025)),
            _round(_quantile(quality_deltas, 0.975)),
        ],
    }


def _paired_rows(records: tuple, seeds: tuple[int, ...]) -> list[dict]:
    paired_rows = []
    for seed in seeds:
        candidate = run_newsgroups_condition(
            records=records,
            seed=seed,
            condition=CANDIDATE_CONDITION,
            train_budget=TRAIN_BUDGET,
            validation_per_class=20,
            heldout_per_class=20,
            epochs=3,
            proxy_epochs=1,
        )
        reference = run_newsgroups_condition(
            records=records,
            seed=seed,
            condition=REFERENCE_CONDITION,
            train_budget=TRAIN_BUDGET,
            validation_per_class=20,
            heldout_per_class=20,
            epochs=3,
            proxy_epochs=1,
        )
        candidate_lsd = float(candidate[LSD_KEY])
        reference_lsd = float(reference[LSD_KEY])
        candidate_quality = float(candidate[QUALITY_KEY])
        reference_quality = float(reference[QUALITY_KEY])
        density_ratio = _density_ratio(candidate_lsd, reference_lsd)
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
    return paired_rows


def build_confirmation_audit(records: tuple, seeds: tuple[int, ...] = CONFIRMATION_SEEDS) -> dict:
    paired_rows = _paired_rows(records, seeds)
    label_counts = Counter(record.label for record in records)
    mean_candidate_lsd = mean(row["candidate_lsd"] for row in paired_rows)
    mean_reference_lsd = mean(row["reference_lsd"] for row in paired_rows)
    mean_candidate_quality = mean(row["candidate_quality"] for row in paired_rows)
    mean_reference_quality = mean(row["reference_quality"] for row in paired_rows)
    paired_win_count = sum(1 for row in paired_rows if row["candidate_density_wins"])
    bootstrap = _paired_bootstrap(paired_rows, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    ci_low, ci_high = bootstrap["density_delta_ci95"]
    sign_p = _sign_test_one_sided_p_value(paired_win_count, len(paired_rows))
    robust_density_win = bool(mean_candidate_lsd > mean_reference_lsd and ci_low > 0.0 and sign_p <= 0.05)
    return {
        "title": "Twenty Newsgroups Class-Balanced Confirmation Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": TWENTY_NEWSGROUPS_DATASET
        | {
            "record_count": len(records),
            "label_count": len(label_counts),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "confirmation_seeds": list(seeds),
        "candidate_condition": CANDIDATE_CONDITION,
        "reference_condition": REFERENCE_CONDITION,
        "train_budget": TRAIN_BUDGET,
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "metadata_stripped": True,
            "targeted_confirmation": True,
            "predeclared_seed_block": True,
            "introduces_new_policy": False,
            "heldout_used_for_selection": False,
            "validation_used_for_selection": False,
            "paper_ready_claim": False,
        },
        "robustness_rule": {
            "robust_density_win": (
                "mean density delta is positive, paired bootstrap 95% interval for mean density delta "
                "is strictly positive, and one-sided exact sign-test p <= 0.05"
            )
        },
        "summary": {
            "seed_count": len(paired_rows),
            "paired_win_count": paired_win_count,
            "paired_loss_count": len(paired_rows) - paired_win_count,
            "mean_candidate_lsd": _round(mean_candidate_lsd),
            "mean_reference_lsd": _round(mean_reference_lsd),
            "mean_density_delta": _round(mean_candidate_lsd - mean_reference_lsd),
            "mean_density_ratio": _round(_density_ratio(mean_candidate_lsd, mean_reference_lsd)),
            "mean_candidate_quality": _round(mean_candidate_quality),
            "mean_reference_quality": _round(mean_reference_quality),
            "mean_quality_delta": _round(mean_candidate_quality - mean_reference_quality),
            "sign_test_one_sided_win_p": sign_p,
            "robust_density_win": robust_density_win,
            "bootstrap": bootstrap,
        },
        "paired_rows": paired_rows,
    }


def render_markdown(artifact: dict) -> str:
    summary = artifact["summary"]
    return "\n".join(
        [
            f"# {artifact['title']}",
            "",
            f"Generated: `{artifact['generated_at']}`",
            "",
            f"Confirmation seeds: `{artifact['confirmation_seeds']}`",
            "",
            "| Candidate | Reference | Budget | Mean ratio | Paired wins | Delta CI95 | Sign p(win) | Robust win? |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
            "| "
            + " | ".join(
                [
                    artifact["candidate_condition"],
                    artifact["reference_condition"],
                    str(artifact["train_budget"]),
                    f"{summary['mean_density_ratio']:.3f}",
                    f"{summary['paired_win_count']}/{summary['seed_count']}",
                    (
                        f"[{summary['bootstrap']['density_delta_ci95'][0]:.6f}, "
                        f"{summary['bootstrap']['density_delta_ci95'][1]:.6f}]"
                    ),
                    f"{summary['sign_test_one_sided_win_p']:.6f}",
                    str(summary["robust_density_win"]),
                ]
            )
            + " |",
            "",
            "## Interpretation",
            "",
            "- This audit is a targeted confirmation, not a new selector search.",
            "- It tests whether the most plausible positive Newsgroups frontier survives a larger untouched seed block.",
            f"- Robust density win under the confirmation rule: {summary['robust_density_win']}.",
            (
                f"- The candidate wins {summary['paired_win_count']} of {summary['seed_count']} paired seeds "
                f"with one-sided sign-test p={summary['sign_test_one_sided_win_p']:.6f}."
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=Path("data/external/twenty_newsgroups.zip"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/twenty_newsgroups_class_balanced_confirmation_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/twenty_newsgroups_class_balanced_confirmation_audit.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    args = parser.parse_args()
    records = load_twenty_newsgroups_records(cache_path=args.cache_path)
    artifact = build_confirmation_audit(records, seeds=tuple(args.seeds))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
