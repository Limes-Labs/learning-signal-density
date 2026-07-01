#!/usr/bin/env python3
"""Build break-even selector-cost analysis from Twenty Newsgroups artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from learning_signal_density.break_even import break_even_comparison


SOURCE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
REFERENCE_CONDITION = "random_sample"
CANDIDATE_CONDITIONS = (
    "class_balanced_sample",
    "length_curriculum_sample",
    "prototype_retrieval_sample",
    "validation_selector",
)
QUALITY_KEY = "accuracy_improvement_over_majority_mean"
QUALITY_UPPER_BOUND = 0.95
REUSABLE_COMPUTE_KEYS = ("selection_cost_tokens_mean", "validation_tuning_cost_tokens_mean")


def _round(value: float) -> float:
    return round(value, 6)


def _mean_optional(values: list[int | float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return _round(mean(finite))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_newsgroups_break_even_analysis(repo_root: Path) -> dict:
    artifact = load_json(repo_root / SOURCE_ARTIFACT)
    comparisons: dict[str, dict[str, dict]] = {}
    condition_records: dict[str, list[dict]] = {condition: [] for condition in CANDIDATE_CONDITIONS}

    for budget in artifact["train_budgets"]:
        budget_key = str(budget)
        conditions = artifact["budgets"][budget_key]["conditions"]
        reference = {"condition": REFERENCE_CONDITION} | conditions[REFERENCE_CONDITION]
        comparisons[budget_key] = {}
        for condition in CANDIDATE_CONDITIONS:
            candidate = {"condition": condition} | conditions[condition]
            row = break_even_comparison(
                reference=reference,
                candidate=candidate,
                quality_key=QUALITY_KEY,
                quality_upper_bound=QUALITY_UPPER_BOUND,
                reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
            )
            comparisons[budget_key][condition] = row
            condition_records[condition].append(row)

    summary = {}
    for condition, rows in condition_records.items():
        density_wins = sum(1 for row in rows if row["candidate_density_wins"])
        observed_quality_wins = sum(1 for row in rows if row["candidate_quality"] > row["reference_quality"])
        summary[condition] = {
            "budgets_compared": len(rows),
            "observed_quality_wins": observed_quality_wins,
            "density_wins": density_wins,
            "mean_density_ratio": _round(mean(row["density_ratio"] for row in rows)),
            "mean_event_compute_multiplier": _round(mean(row["event_compute_multiplier"] for row in rows)),
            "mean_quality_multiplier": _round(mean(row["quality_multiplier"] for row in rows)),
            "mean_compute_over_break_even": _round(mean(row["compute_over_break_even"] for row in rows)),
            "mean_amortized_reuses_to_density_win": _mean_optional(
                [row["amortized_reuses_to_density_win"] for row in rows]
            ),
        }

    return {
        "title": "Twenty Newsgroups Break-Even Selection-Cost Analysis",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(SOURCE_ARTIFACT)],
        "reference_condition": REFERENCE_CONDITION,
        "candidate_conditions": list(CANDIDATE_CONDITIONS),
        "quality_metric": QUALITY_KEY,
        "quality_upper_bound": QUALITY_UPPER_BOUND,
        "amortization_model": {
            "reusable_compute_keys": list(REUSABLE_COMPUTE_KEYS),
            "formula": "C_candidate(K) = C_nonreusable + C_reusable / K",
            "interpretation": (
                "K is the number of independent downstream uses over which selector scans "
                "and validation-tuning costs can be reused without changing the observed accuracy gain."
            ),
        },
        "theorem": {
            "name": "same-budget active-selection break-even",
            "statement": (
                "For equal labeled-document budgets and positive random-sampling gain, a candidate "
                "policy has higher learning-signal density than random sampling if and only if its "
                "heldout-accuracy gain multiplier is greater than its event-compute multiplier."
            ),
            "general_inequality": "G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)",
            "bounded_metric_corollary": (
                "If break_even_quality is greater than or equal to the configured accuracy-gain upper bound, "
                "even perfect candidate accuracy cannot strictly beat random density under the charged cost model."
            ),
            "amortized_reuse_corollary": (
                "With reusable selector cost R and nonreusable cost F, same-budget density can win after K "
                "reuses exactly when F + R/K is below the candidate max_affordable_compute_units."
            ),
        },
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "heldout_used_for_selection": False,
            "post_hoc_diagnostic": True,
            "metadata_stripped": True,
            "paper_ready_claim": False,
        },
        "comparisons": comparisons,
        "summary": summary,
    }


def _fmt_optional(value: int | float | None, digits: int = 3) -> str:
    if value is None:
        return "never"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This artifact is a mathematical audit over the committed Twenty Newsgroups active-selection JSON.",
        "It does not introduce a new policy. It asks when each active/curriculum/retrieval policy pays for its charged event-compute multiplier.",
        "",
        "Break-even condition:",
        "",
        "```text",
        result["theorem"]["general_inequality"],
        "```",
        "",
        "| Budget | Candidate | Quality mult. | Event-compute mult. | Density ratio | Break-even gain | Cost over BE | Reuses to win | Fully amortized ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget, candidates in result["comparisons"].items():
        for condition, row in candidates.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        budget,
                        condition,
                        f"{row['quality_multiplier']:.3f}",
                        f"{row['event_compute_multiplier']:.3f}",
                        f"{row['density_ratio']:.3f}",
                        f"{row['break_even_quality']:.3f}",
                        f"{row['compute_over_break_even']:.3f}",
                        _fmt_optional(row["amortized_reuses_to_density_win"]),
                        _fmt_optional(row["fully_amortized_density_ratio"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `density_ratio > 1` means the candidate beats random under learning-signal density.",
            "- `quality_multiplier > 1` means the candidate improves heldout accuracy gain over random before cost is charged.",
            "- `event_compute_multiplier` measures the full external-events times charged-compute bill.",
            "- `compute_over_break_even` is the factor by which charged compute exceeds what the observed quality gain can afford.",
            "- `amortized_reuses_to_density_win` is the minimum independent downstream reuse count needed if selector and validation costs are reusable.",
            "- `fully_amortized_density_ratio` is the limiting ratio after the reusable cost is spread over infinitely many uses.",
            "",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("results/twenty_newsgroups_break_even_analysis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/twenty_newsgroups_break_even_analysis.md"))
    args = parser.parse_args()
    result = build_newsgroups_break_even_analysis(Path("."))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
