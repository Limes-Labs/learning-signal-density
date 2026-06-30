#!/usr/bin/env python3
"""Build a post-hoc regret audit for the high-budget support selectors."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ARTIFACTS: tuple[tuple[str, str, Path], ...] = (
    (
        "train_support_density",
        "Train support-density selector",
        Path("results/tiny_neural_budget_sweep_train_support_density_f1024.json"),
    ),
    (
        "support_probe_window",
        "Support-probe window selector",
        Path("results/tiny_neural_budget_sweep_support_probe_window_f1024.json"),
    ),
    (
        "validation_support_precision",
        "Validation support-precision selector",
        Path("results/tiny_neural_budget_sweep_validation_support_precision_f1024.json"),
    ),
    (
        "validation_support_precision_gate",
        "Validation support-precision gate",
        Path("results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json"),
    ),
    (
        "support_selector_transfer",
        "Support-selector transfer stress",
        Path("results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json"),
    ),
    (
        "validation_support_utility",
        "Validation support-utility selector",
        Path("results/tiny_neural_budget_sweep_validation_support_utility_f1024.json"),
    ),
)
DEFAULT_OUTPUT_JSON = Path("results/support_selector_error_audit_f1024.json")
DEFAULT_OUTPUT_MD = Path("results/support_selector_error_audit_f1024.md")

LSD_METRIC = "signed_learning_signal_density_per_1m_event_compute_mean"
GAIN_METRIC = "accuracy_improvement_over_majority_mean"

SIMPLE_COMPARATOR_CONDITIONS = (
    "raw_text",
    "compact_train_size_gated_induction",
    "support_ramped_compact_induction",
    "density_window_compact_induction",
    "density_capped_compact_induction",
)
SELECTOR_CONDITIONS = (
    "train_support_density_selector",
    "support_probe_window_selector",
    "validation_support_precision_selector",
    "validation_support_precision_gate_selector",
    "validation_support_utility_selector",
)

CONDITION_LABELS = {
    "compact_train_size_gated_induction": "Compact train-size gated",
    "density_capped_compact_induction": "Density-capped compact",
    "density_window_compact_induction": "Density-window compact",
    "raw_text": "Raw text",
    "support_probe_window_selector": "Support-probe window selector",
    "support_ramped_compact_induction": "Support-ramped compact",
    "train_support_density_selector": "Train support-density selector",
    "validation_support_precision_gate_selector": "Validation support-precision gate",
    "validation_support_precision_selector": "Validation support-precision selector",
    "validation_support_utility_selector": "Validation support-utility selector",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(value: float, digits: int = 6) -> float:
    return round(value, digits)


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def average(values: list[float], digits: int = 6) -> float:
    return metric(sum(values) / len(values), digits=digits)


def validate_source_artifact(relative_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{relative_path} must not use heldout for source selection")
    if scope.get("fresh_seed_confirmation") is not True:
        raise ValueError(f"{relative_path} must be a fresh-seed source artifact")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{relative_path} must not be marked paper-ready")

    conditions = set(artifact.get("conditions", []))
    if not conditions.intersection(SIMPLE_COMPARATOR_CONDITIONS):
        raise ValueError(f"{relative_path} has no simple comparator conditions")
    if not conditions.intersection(SELECTOR_CONDITIONS):
        raise ValueError(f"{relative_path} has no selector conditions")
    if not artifact.get("material_counts"):
        raise ValueError(f"{relative_path} has no material counts")


def condition_average_lsd(artifact: dict[str, Any], condition: str) -> float:
    material_keys = [str(material) for material in artifact["material_counts"]]
    return average(
        [
            artifact["budgets"][material][condition][LSD_METRIC]
            for material in material_keys
        ]
    )


def available_conditions(artifact: dict[str, Any], candidates: tuple[str, ...]) -> list[str]:
    source_conditions = set(artifact["conditions"])
    return [condition for condition in candidates if condition in source_conditions]


def best_condition_by_average_lsd(artifact: dict[str, Any], conditions: list[str]) -> str:
    return max(conditions, key=lambda condition: condition_average_lsd(artifact, condition))


def best_simple_by_material(artifact: dict[str, Any], simple_conditions: list[str]) -> dict[str, dict[str, Any]]:
    best_by_material: dict[str, dict[str, Any]] = {}
    for material in artifact["material_counts"]:
        material_key = str(material)
        best_condition = max(
            simple_conditions,
            key=lambda condition: artifact["budgets"][material_key][condition][LSD_METRIC],
        )
        best_row = artifact["budgets"][material_key][best_condition]
        best_by_material[material_key] = {
            "condition": best_condition,
            "condition_label": condition_label(best_condition),
            "gain": best_row[GAIN_METRIC],
            "lsd": best_row[LSD_METRIC],
            "charged_compute_units": best_row["charged_compute_units_mean"],
        }
    return best_by_material


def selection_summary(artifact: dict[str, Any], selector: str) -> dict[str, Any]:
    selected_counts: Counter[str] = Counter()
    total_decisions = 0
    for material in artifact["material_counts"]:
        row_counts = artifact["budgets"][str(material)][selector].get(
            "portfolio_selected_condition_counts",
            {},
        )
        selected_counts.update(row_counts)
        total_decisions += sum(row_counts.values())

    if not total_decisions:
        return {"total_decisions": 0, "selected_condition_counts": {}, "selected_condition_rates": {}}

    return {
        "total_decisions": total_decisions,
        "selected_condition_counts": dict(sorted(selected_counts.items())),
        "selected_condition_rates": {
            condition: metric(count / total_decisions)
            for condition, count in sorted(selected_counts.items())
        },
    }


def selector_diagnostic(
    artifact: dict[str, Any],
    selector: str,
    best_simple_rows: dict[str, dict[str, Any]],
    best_fixed_simple_lsd: float,
) -> dict[str, Any]:
    material_keys = [str(material) for material in artifact["material_counts"]]
    rows = [artifact["budgets"][material][selector] for material in material_keys]
    regrets = [
        best_simple_rows[material]["lsd"]
        - artifact["budgets"][material][selector][LSD_METRIC]
        for material in material_keys
    ]
    worst_index, worst_regret = max(enumerate(regrets), key=lambda item: item[1])
    average_lsd = average([row[LSD_METRIC] for row in rows])
    average_regret = average(regrets)
    average_gain = average([row[GAIN_METRIC] for row in rows])
    average_cost = average([row["portfolio_selection_cost_units_mean"] for row in rows], digits=1)

    diagnostic = {
        "condition": selector,
        "condition_label": condition_label(selector),
        "average_gain": average_gain,
        "average_lsd": average_lsd,
        "average_regret_vs_best_simple_lsd": average_regret,
        "average_delta_vs_best_fixed_simple_lsd": metric(average_lsd - best_fixed_simple_lsd),
        "average_selection_cost_units": average_cost,
        "budgets_beating_best_simple_count": sum(
            1
            for material in material_keys
            if artifact["budgets"][material][selector][LSD_METRIC] > best_simple_rows[material]["lsd"]
        ),
        "budgets_tying_best_simple_count": sum(
            1
            for material in material_keys
            if artifact["budgets"][material][selector][LSD_METRIC] == best_simple_rows[material]["lsd"]
        ),
        "budget_count": len(material_keys),
        "worst_regret_material_count": int(material_keys[worst_index]),
        "worst_regret_vs_best_simple_lsd": metric(worst_regret),
        "beats_best_fixed_simple_average": average_lsd > best_fixed_simple_lsd,
        "local_expected_value_positive": average_regret < 0,
        "regret_by_material": {
            material: metric(regret)
            for material, regret in zip(material_keys, regrets, strict=True)
        },
    }
    diagnostic.update(selection_summary(artifact, selector))
    return diagnostic


def summarize_artifact(
    label: str,
    display_label: str,
    relative_path: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    simple_conditions = available_conditions(artifact, SIMPLE_COMPARATOR_CONDITIONS)
    selector_conditions = available_conditions(artifact, SELECTOR_CONDITIONS)
    best_fixed_simple_condition = best_condition_by_average_lsd(artifact, simple_conditions)
    best_fixed_simple_lsd = condition_average_lsd(artifact, best_fixed_simple_condition)
    best_simple_rows = best_simple_by_material(artifact, simple_conditions)

    selector_diagnostics = {
        selector: selector_diagnostic(artifact, selector, best_simple_rows, best_fixed_simple_lsd)
        for selector in selector_conditions
    }
    best_selector = min(
        selector_diagnostics,
        key=lambda selector: (
            selector_diagnostics[selector]["average_regret_vs_best_simple_lsd"],
            -selector_diagnostics[selector]["average_lsd"],
        ),
    )

    return {
        "label": label,
        "display_label": display_label,
        "source_artifact": str(relative_path),
        "source_generated_at": artifact.get("generated_at"),
        "profile_label": artifact["profile_label"],
        "seeds": artifact["seeds"],
        "material_counts": artifact["material_counts"],
        "simple_comparator_conditions": simple_conditions,
        "selector_conditions": selector_conditions,
        "best_fixed_simple_condition": best_fixed_simple_condition,
        "best_fixed_simple_condition_label": condition_label(best_fixed_simple_condition),
        "best_fixed_simple_lsd": best_fixed_simple_lsd,
        "best_simple_by_material": best_simple_rows,
        "best_selector_by_regret": best_selector,
        "selector_diagnostics": selector_diagnostics,
    }


def cross_artifact_selector_summary(artifact_summaries: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for selector in SELECTOR_CONDITIONS:
        rows = []
        for artifact_label, artifact_summary in artifact_summaries.items():
            diagnostic = artifact_summary["selector_diagnostics"].get(selector)
            if diagnostic is None:
                continue
            rows.append(
                {
                    "artifact": artifact_label,
                    "average_lsd": diagnostic["average_lsd"],
                    "average_regret_vs_best_simple_lsd": diagnostic[
                        "average_regret_vs_best_simple_lsd"
                    ],
                    "budgets_beating_best_simple_count": diagnostic[
                        "budgets_beating_best_simple_count"
                    ],
                    "budget_count": diagnostic["budget_count"],
                }
            )
        if not rows:
            continue
        summary[selector] = {
            "condition_label": condition_label(selector),
            "source_count": len(rows),
            "positive_expected_value_source_count": sum(
                1 for row in rows if row["average_regret_vs_best_simple_lsd"] < 0
            ),
            "mean_average_regret_vs_best_simple_lsd": average(
                [row["average_regret_vs_best_simple_lsd"] for row in rows]
            ),
            "rows": rows,
        }
    return summary


def build_recommendation(artifact_summaries: dict[str, Any]) -> dict[str, Any]:
    transfer = artifact_summaries["support_selector_transfer"]
    selector_diagnostics = transfer["selector_diagnostics"]
    strongest_selector = min(
        selector_diagnostics,
        key=lambda selector: (
            selector_diagnostics[selector]["average_regret_vs_best_simple_lsd"],
            -selector_diagnostics[selector]["average_lsd"],
        ),
    )
    strongest = selector_diagnostics[strongest_selector]
    promote = strongest["average_regret_vs_best_simple_lsd"] < 0 and strongest[
        "beats_best_fixed_simple_average"
    ]

    return {
        "promote_support_selector": promote,
        "strongest_transfer_selector": strongest_selector,
        "strongest_transfer_selector_label": condition_label(strongest_selector),
        "strongest_transfer_selector_average_lsd": strongest["average_lsd"],
        "strongest_transfer_selector_average_regret_vs_best_simple_lsd": strongest[
            "average_regret_vs_best_simple_lsd"
        ],
        "transfer_best_simple_condition": transfer["best_fixed_simple_condition"],
        "transfer_best_simple_lsd": transfer["best_fixed_simple_lsd"],
        "reason": (
            "Do not promote a support selector yet: the least-regret transfer selector "
            "still loses to the best simple comparator after charged selection cost."
        ),
        "next_selector_requirement": (
            "A future selector should be evaluated as an expected-value-of-information "
            "policy and should beat density-capped or raw fallback on fresh transfer seeds."
        ),
    }


def build_support_selector_error_audit(repo_root: Path) -> dict[str, Any]:
    artifact_summaries: dict[str, Any] = {}
    source_artifacts = []
    artifact_order = []

    for label, display_label, relative_path in DEFAULT_SOURCE_ARTIFACTS:
        source_path = repo_root / relative_path
        artifact = load_json(source_path)
        validate_source_artifact(relative_path, artifact)
        source_artifacts.append(str(relative_path))
        artifact_order.append(label)
        artifact_summaries[label] = summarize_artifact(label, display_label, relative_path, artifact)

    return {
        "title": "Learning Signal Density Support-selector Error Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": source_artifacts,
        "artifact_order": artifact_order,
        "metric": LSD_METRIC,
        "simple_comparator_conditions": list(SIMPLE_COMPARATOR_CONDITIONS),
        "selector_conditions": list(SELECTOR_CONDITIONS),
        "artifact_summaries": artifact_summaries,
        "selector_cross_artifact_summary": cross_artifact_selector_summary(artifact_summaries),
        "recommendation": build_recommendation(artifact_summaries),
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "post_hoc_diagnostic": True,
            "uses_committed_fresh_seed_artifacts": True,
            "heldout_used_for_error_analysis": True,
            "heldout_available_to_policies": False,
            "deployable_policy": False,
            "paper_ready_claim": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Learning Signal Density Support-selector Error Audit",
        "",
        "This is a post-hoc support-selector error audit. It reads committed fresh-seed",
        "neural sweeps and asks whether extra selector information has positive",
        "expected value after charged inspection or validation cost.",
        "",
        "## Source Summary",
        "",
        "| Source | Best simple | Best simple LSD | Least-regret selector | Selector LSD | Avg. regret | Wins | Avg. selector cost |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for label in audit["artifact_order"]:
        summary = audit["artifact_summaries"][label]
        best_selector = summary["best_selector_by_regret"]
        diagnostic = summary["selector_diagnostics"][best_selector]
        lines.append(
            " | ".join(
                [
                    f"| {summary['display_label']}",
                    summary["best_fixed_simple_condition"],
                    fmt_float(summary["best_fixed_simple_lsd"]),
                    best_selector,
                    fmt_float(diagnostic["average_lsd"]),
                    fmt_float(diagnostic["average_regret_vs_best_simple_lsd"]),
                    f"{diagnostic['budgets_beating_best_simple_count']}/{diagnostic['budget_count']}",
                    f"{diagnostic['average_selection_cost_units']:.1f} |",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Selector Details",
            "",
            "| Source | Selector | Avg. LSD | Avg. regret | Worst budget | Worst regret | Avg. selector cost |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in audit["artifact_order"]:
        summary = audit["artifact_summaries"][label]
        for selector in summary["selector_conditions"]:
            diagnostic = summary["selector_diagnostics"][selector]
            lines.append(
                " | ".join(
                    [
                        f"| {summary['display_label']}",
                        selector,
                        fmt_float(diagnostic["average_lsd"]),
                        fmt_float(diagnostic["average_regret_vs_best_simple_lsd"]),
                        str(diagnostic["worst_regret_material_count"]),
                        fmt_float(diagnostic["worst_regret_vs_best_simple_lsd"]),
                        f"{diagnostic['average_selection_cost_units']:.1f} |",
                    ]
                )
            )

    recommendation = audit["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Promote support selector: `{str(recommendation['promote_support_selector']).lower()}`.",
            f"- Strongest transfer selector: `{recommendation['strongest_transfer_selector']}`.",
            f"- Transfer best simple comparator: `{recommendation['transfer_best_simple_condition']}`.",
            f"- Reason: {recommendation['reason']}",
            "",
            "## Scope",
            "",
            "- This audit uses completed heldout outcomes after the source sweeps have run.",
            "- The heldout outcomes are not available to any deployable policy.",
            "- Treat this as mechanism evidence and a promotion gate, not as a new selector.",
            "",
            "```json",
            json.dumps(audit["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(audit: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_md.write_text(render_markdown(audit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing committed source artifacts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Output Markdown path, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_json = args.output_json if args.output_json.is_absolute() else repo_root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md
    audit = build_support_selector_error_audit(repo_root)
    write_audit(audit, output_json, output_md)
    print(f"wrote {output_json.relative_to(repo_root)}")
    print(f"wrote {output_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
