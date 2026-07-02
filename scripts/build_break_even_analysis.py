#!/usr/bin/env python3
"""Build break-even selector-cost analysis from real-text SMS artifacts."""

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


SMS_ARTIFACTS = (
    ("SMS Spam v800", Path("results/sms_spam_real_text_selection_cost.json")),
    ("SMS Spam v200", Path("results/sms_spam_real_text_selection_cost_v200.json")),
)
REFERENCE_CONDITION = "random_sample"
CANDIDATE_CONDITIONS = (
    "label_index_balanced_sample",
    "class_balanced_sample",
    "validation_label_index_selector",
    "validation_sample_selector",
)
QUALITY_KEY = "spam_f1_improvement_over_majority_mean"
QUALITY_UPPER_BOUND = 1.0
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


def build_sms_break_even_analysis(repo_root: Path) -> dict:
    comparisons: dict[str, dict[str, dict[str, dict]]] = {}
    summary: dict[str, dict[str, dict]] = {}
    source_artifacts: list[str] = []

    for label, relative_path in SMS_ARTIFACTS:
        artifact = load_json(repo_root / relative_path)
        source_artifacts.append(str(relative_path))
        comparisons[label] = {}
        condition_records: dict[str, list[dict]] = {condition: [] for condition in CANDIDATE_CONDITIONS}

        for budget in artifact["train_budgets"]:
            budget_key = str(budget)
            conditions = artifact["budgets"][budget_key]["conditions"]
            reference = {"condition": REFERENCE_CONDITION} | conditions[REFERENCE_CONDITION]
            comparisons[label][budget_key] = {}
            for condition in CANDIDATE_CONDITIONS:
                candidate = {"condition": condition} | conditions[condition]
                row = break_even_comparison(
                    reference=reference,
                    candidate=candidate,
                    quality_key=QUALITY_KEY,
                    quality_upper_bound=QUALITY_UPPER_BOUND,
                    reusable_compute_keys=REUSABLE_COMPUTE_KEYS,
                )
                comparisons[label][budget_key][condition] = row
                condition_records[condition].append(row)

        summary[label] = {}
        for condition, rows in condition_records.items():
            density_wins = sum(1 for row in rows if row["candidate_density_wins"])
            summary[label][condition] = {
                "budgets_compared": len(rows),
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
        "title": "SMS Spam Break-Even Selection-Cost Analysis",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": source_artifacts,
        "reference_condition": REFERENCE_CONDITION,
        "candidate_conditions": list(CANDIDATE_CONDITIONS),
        "quality_metric": QUALITY_KEY,
        "quality_upper_bound": QUALITY_UPPER_BOUND,
        "amortization_model": {
            "reusable_compute_keys": list(REUSABLE_COMPUTE_KEYS),
            "formula": "C_candidate(K) = C_nonreusable + C_reusable / K",
            "interpretation": (
                "K is the number of independent downstream uses over which selector construction "
                "and validation-tuning costs can be reused without changing the observed quality gain."
            ),
        },
        "theorem": {
            "name": "same-budget selector break-even",
            "statement": (
                "For two policies with the same external-event budget and positive reference gain, "
                "the candidate has higher learning-signal density than the reference if and only if "
                "candidate_quality/reference_quality is greater than candidate_compute/reference_compute."
            ),
            "general_inequality": "G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)",
            "bounded_metric_corollary": (
                "If break_even_quality is greater than or equal to the metric upper bound, even perfect "
                "candidate quality cannot strictly beat the reference density under the charged cost model."
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
            "paper_ready_claim": False,
        },
        "comparisons": comparisons,
        "summary": summary,
    }


def render_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This artifact is a mathematical audit over committed SMS Spam result JSON.",
        "It does not introduce a new policy. It asks whether each non-random policy's quality gain is large enough to pay for its charged event-compute multiplier.",
        "",
        "Break-even condition:",
        "",
        "```text",
        result["theorem"]["general_inequality"],
        "```",
        "",
    ]
    for artifact_label, budgets in result["comparisons"].items():
        lines.extend(
            [
                f"## {artifact_label}",
                "",
                "| Budget | Candidate | Quality mult. | Event-compute mult. | Density ratio | Max possible density | Break-even quality | Compute over break-even | Reuses to win | Fully amortized ratio |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for budget, candidates in budgets.items():
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
                            f"{row['max_possible_density_ratio']:.3f}",
                            f"{row['break_even_quality']:.3f}",
                            f"{row['compute_over_break_even']:.3f}",
                            str(row["amortized_reuses_to_density_win"])
                            if row["amortized_reuses_to_density_win"] is not None
                            else "never",
                            f"{row['fully_amortized_density_ratio']:.3f}"
                            if row["fully_amortized_density_ratio"] is not None
                            else "n/a",
                        ]
                    )
                    + " |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `density_ratio > 1` means the candidate beats random under learning-signal density.",
            "- `break_even_quality >= quality_upper_bound` means the candidate could not strictly beat random even with perfect spam F1 under the charged cost model.",
            "- `max_possible_density_ratio` is the best density ratio reachable at the configured quality upper bound.",
            "- `compute_over_break_even` is the factor by which charged compute exceeds what the observed quality gain can afford.",
            "- `amortized_reuses_to_density_win` is the minimum number of independent downstream uses needed if selector and validation-tuning costs are reusable.",
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
    parser.add_argument("--output-json", type=Path, default=Path("results/sms_spam_break_even_analysis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/sms_spam_break_even_analysis.md"))
    args = parser.parse_args()
    result = build_sms_break_even_analysis(Path("."))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
