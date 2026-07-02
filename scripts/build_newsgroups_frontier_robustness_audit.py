#!/usr/bin/env python3
"""Audit seed-level robustness of the Newsgroups density frontier."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import product
import json
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from learning_signal_density.newsgroups_experiment import _round


ACTIVE_SELECTION_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
BUDGETED_ACQUISITION_ARTIFACT = Path("results/twenty_newsgroups_budgeted_acquisition_audit.json")
QUALITY_KEY = "accuracy_improvement_over_majority"
LSD_KEY = "signed_learning_signal_density_per_1m_event_compute"
COMPARISONS = (
    {
        "comparison": "class_balanced_80_vs_random",
        "family": "active_selection",
        "budget": 80,
        "candidate_artifact": "active_selection",
        "candidate_condition": "class_balanced_sample",
        "reference_artifact": "active_selection",
        "reference_condition": "random_sample",
        "reading": "base-grid mean density win",
    },
    {
        "comparison": "prototype_40_vs_random",
        "family": "active_selection",
        "budget": 40,
        "candidate_artifact": "active_selection",
        "candidate_condition": "prototype_retrieval_sample",
        "reference_artifact": "active_selection",
        "reference_condition": "random_sample",
        "reading": "quality win but density loss",
    },
    {
        "comparison": "prototype_160_vs_random",
        "family": "active_selection",
        "budget": 160,
        "candidate_artifact": "active_selection",
        "candidate_condition": "prototype_retrieval_sample",
        "reference_artifact": "active_selection",
        "reference_condition": "random_sample",
        "reading": "quality win but density loss",
    },
    {
        "comparison": "budgeted_margin_1x_160_vs_class_balanced",
        "family": "budgeted_active_acquisition",
        "budget": 160,
        "candidate_artifact": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_1x",
        "reference_artifact": "active_selection",
        "reference_condition": "class_balanced_sample",
        "reading": "budgeted mean density win against class-balanced",
    },
    {
        "comparison": "budgeted_margin_2x_160_vs_class_balanced",
        "family": "budgeted_active_acquisition",
        "budget": 160,
        "candidate_artifact": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_2x",
        "reference_artifact": "active_selection",
        "reference_condition": "class_balanced_sample",
        "reading": "budgeted mean density win against class-balanced",
    },
    {
        "comparison": "budgeted_margin_2x_160_vs_random",
        "family": "budgeted_active_acquisition",
        "budget": 160,
        "candidate_artifact": "budgeted_acquisition",
        "candidate_condition": "budgeted_active_margin_uncertainty_2x",
        "reference_artifact": "active_selection",
        "reference_condition": "random_sample",
        "reading": "budgeted density loss against random",
    },
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _index_per_seed(artifact: dict) -> dict[tuple[int, int, str], dict]:
    return {
        (int(row["seed"]), int(row["train_budget"]), row["condition"]): row
        for row in artifact.get("per_seed", [])
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take quantile of an empty list")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _density_ratio(candidate: float, reference: float, *, context: str) -> float:
    if reference <= 0:
        raise ValueError(f"{context} reference density must be positive, got {reference}")
    return candidate / reference


def _exact_seed_bootstrap(paired_rows: list[dict]) -> dict:
    n = len(paired_rows)
    if n <= 0:
        raise ValueError("paired_rows must not be empty")
    density_deltas = []
    density_ratios = []
    quality_deltas = []
    for indices in product(range(n), repeat=n):
        candidate_density = mean(paired_rows[index]["candidate_lsd"] for index in indices)
        reference_density = mean(paired_rows[index]["reference_lsd"] for index in indices)
        candidate_quality = mean(paired_rows[index]["candidate_quality"] for index in indices)
        reference_quality = mean(paired_rows[index]["reference_quality"] for index in indices)
        density_deltas.append(candidate_density - reference_density)
        density_ratios.append(_density_ratio(candidate_density, reference_density, context="bootstrap sample"))
        quality_deltas.append(candidate_quality - reference_quality)
    return {
        "bootstrap_sample_count": n**n,
        "density_delta_ci95": [
            _round(_quantile(density_deltas, 0.025)),
            _round(_quantile(density_deltas, 0.975)),
        ],
        "density_ratio_ci95": [
            _round(_quantile(density_ratios, 0.025)),
            _round(_quantile(density_ratios, 0.975)),
        ],
        "quality_delta_ci95": [
            _round(_quantile(quality_deltas, 0.025)),
            _round(_quantile(quality_deltas, 0.975)),
        ],
    }


def _paired_comparison(
    config: dict,
    indexes: dict[str, dict[tuple[int, int, str], dict]],
) -> dict:
    budget = int(config["budget"])
    candidate_index = indexes[config["candidate_artifact"]]
    reference_index = indexes[config["reference_artifact"]]
    candidate_keys = {
        key
        for key in candidate_index
        if key[1] == budget and key[2] == config["candidate_condition"]
    }
    reference_keys = {
        key
        for key in reference_index
        if key[1] == budget and key[2] == config["reference_condition"]
    }
    seeds = sorted({key[0] for key in candidate_keys} & {key[0] for key in reference_keys})
    if not seeds:
        raise ValueError(f"no paired seeds for {config['comparison']}")
    paired_rows = []
    for seed in seeds:
        candidate = candidate_index[(seed, budget, config["candidate_condition"])]
        reference = reference_index[(seed, budget, config["reference_condition"])]
        candidate_lsd = float(candidate[LSD_KEY])
        reference_lsd = float(reference[LSD_KEY])
        candidate_quality = float(candidate[QUALITY_KEY])
        reference_quality = float(reference[QUALITY_KEY])
        paired_rows.append(
            {
                "seed": seed,
                "candidate_lsd": _round(candidate_lsd),
                "reference_lsd": _round(reference_lsd),
                "density_delta": _round(candidate_lsd - reference_lsd),
                "density_ratio": _round(
                    _density_ratio(candidate_lsd, reference_lsd, context=f"{config['comparison']} seed {seed}")
                ),
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
    mean_density_win = mean_density_delta > 0
    robust_density_win = bool(mean_density_win and ci_low > 0 and paired_win_count == len(paired_rows))
    robust_density_loss = bool(mean_density_delta < 0 and ci_high < 0 and paired_win_count == 0)
    return config | {
        "seed_count": len(paired_rows),
        "paired_win_count": paired_win_count,
        "all_paired_wins": paired_win_count == len(paired_rows),
        "mean_candidate_lsd": _round(mean_candidate_lsd),
        "mean_reference_lsd": _round(mean_reference_lsd),
        "mean_density_delta": _round(mean_density_delta),
        "mean_density_ratio": _round(mean_density_ratio),
        "mean_density_win": mean_density_win,
        "robust_density_win": robust_density_win,
        "robust_density_loss": robust_density_loss,
        "bootstrap": bootstrap,
        "paired_rows": paired_rows,
    }


def build_frontier_robustness_audit(repo_root: Path) -> dict:
    active = load_json(repo_root / ACTIVE_SELECTION_ARTIFACT)
    budgeted = load_json(repo_root / BUDGETED_ACQUISITION_ARTIFACT)
    indexes = {
        "active_selection": _index_per_seed(active),
        "budgeted_acquisition": _index_per_seed(budgeted),
    }
    comparisons = [_paired_comparison(config, indexes) for config in COMPARISONS]
    summary = {
        "comparisons": len(comparisons),
        "mean_density_wins": sum(1 for row in comparisons if row["mean_density_win"]),
        "robust_density_wins": sum(1 for row in comparisons if row["robust_density_win"]),
        "robust_density_losses": sum(1 for row in comparisons if row["robust_density_loss"]),
        "fragile_mean_density_wins": sum(
            1 for row in comparisons if row["mean_density_win"] and not row["robust_density_win"]
        ),
    }
    return {
        "title": "Twenty Newsgroups Frontier Robustness Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(ACTIVE_SELECTION_ARTIFACT), str(BUDGETED_ACQUISITION_ARTIFACT)],
        "dataset": active["dataset"],
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "metadata_stripped": True,
            "post_hoc_diagnostic": True,
            "paired_seed_audit": True,
            "exact_seed_bootstrap": True,
            "introduces_new_policy": False,
            "heldout_used_for_selection": False,
            "paper_ready_claim": False,
        },
        "robustness_rule": {
            "name": "paired bootstrap density frontier rule",
            "robust_density_win": (
                "mean density delta is positive, every paired seed wins, and the exact seed-bootstrap "
                "95% interval for mean density delta is strictly positive"
            ),
            "interpretation": (
                "With only three paired seeds this is intentionally conservative; fragile mean wins remain useful "
                "hypotheses but should not be promoted as stable frontiers."
            ),
        },
        "summary": summary,
        "comparisons": comparisons,
    }


def render_markdown(artifact: dict) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        "This is a paired-seed robustness audit over committed Twenty Newsgroups artifacts.",
        "It introduces no new policy and tests whether mean density wins survive a conservative seed bootstrap.",
        "",
        "| Comparison | Budget | Mean ratio | Paired wins | Delta CI95 | Robust win? | Robust loss? |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in artifact["comparisons"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    str(row["budget"]),
                    f"{row['mean_density_ratio']:.3f}",
                    f"{row['paired_win_count']}/{row['seed_count']}",
                    f"[{row['bootstrap']['density_delta_ci95'][0]:.6f}, {row['bootstrap']['density_delta_ci95'][1]:.6f}]",
                    str(row["robust_density_win"]),
                    str(row["robust_density_loss"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Mean density wins: {artifact['summary']['mean_density_wins']}.",
            f"- Robust density wins under the paired-bootstrap rule: {artifact['summary']['robust_density_wins']}.",
            f"- Robust density losses under the paired-bootstrap rule: {artifact['summary']['robust_density_losses']}.",
            "- The current positive density frontiers should be read as hypotheses until confirmed on more seeds.",
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/twenty_newsgroups_frontier_robustness_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/twenty_newsgroups_frontier_robustness_audit.md"),
    )
    args = parser.parse_args()
    artifact = build_frontier_robustness_audit(Path("."))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
