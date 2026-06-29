#!/usr/bin/env python3
"""Generate LaTeX tables for the paper from committed result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BUDGET_ARTIFACTS = [
    ("32x8 f1024", Path("results/tiny_neural_budget_sweep_32x8_f1024.json")),
    ("16x8 f1024", Path("results/tiny_neural_budget_sweep_16x8_f1024.json")),
    ("8x8 f1024", Path("results/tiny_neural_budget_sweep_8x8_f1024.json")),
]
FEATURE_FRONTIER_ARTIFACT = Path("results/tiny_neural_feature_sweep_wide.json")
PROFILE_FRONTIER_ARTIFACT = Path("results/tiny_neural_profile_sweep_f1024.json")
VALIDATION_SELECTED_ARTIFACT = Path("results/tiny_neural_budget_sweep_validation_selected_f1024.json")
AGREEMENT_GATED_ARTIFACT = Path("results/tiny_neural_budget_sweep_agreement_gated_f1024.json")

FRONTIER_CONDITIONS = [
    "raw_text",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "counterfactual_expansion",
]
LOW_BUDGET_CONDITIONS = [
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
]
LOW_BUDGET_MATERIALS = ["16", "24", "32"]

CONDITION_LABELS = {
    "agreement_gated_self_ranked_induction": "Agreement-gated",
    "counterfactual_expansion": "Oracle counterfactual",
    "mdl_rule_expansion": "MDL rule expansion",
    "qa_expansion": "QA expansion",
    "raw_text": "Raw text",
    "sample_aware_self_ranked_induction": "Sample-aware self-ranked",
    "self_ranked_induction": "Self-ranked",
    "validation_ranked_induction": "Validation-ranked",
}


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_target(value: int | None) -> str:
    if value is None:
        return "No target"
    return str(value)


def fmt_ops(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return fmt_float(value, 0)


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def validate_claim_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("fresh_seed_confirmation") is not True:
        raise ValueError(f"{artifact_path} must be a fresh-seed confirmation artifact")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")


def load_budget_artifacts(repo_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    loaded = []
    for profile_label, relative_path in BUDGET_ARTIFACTS:
        artifact = load_json(repo_root / relative_path)
        validate_claim_scope(relative_path, artifact)
        loaded.append((profile_label, relative_path, artifact))
    return loaded


def load_supporting_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    artifacts = {
        "feature": load_json(repo_root / FEATURE_FRONTIER_ARTIFACT),
        "profile": load_json(repo_root / PROFILE_FRONTIER_ARTIFACT),
        "validation_selected": load_json(repo_root / VALIDATION_SELECTED_ARTIFACT),
        "agreement_gated": load_json(repo_root / AGREEMENT_GATED_ARTIFACT),
    }
    validate_claim_scope(FEATURE_FRONTIER_ARTIFACT, artifacts["feature"])
    validate_claim_scope(PROFILE_FRONTIER_ARTIFACT, artifacts["profile"])
    validate_claim_scope(VALIDATION_SELECTED_ARTIFACT, artifacts["validation_selected"])
    validate_claim_scope(AGREEMENT_GATED_ARTIFACT, artifacts["agreement_gated"])
    return artifacts


def build_macros(repo_root: Path) -> str:
    budget_artifacts = load_budget_artifacts(repo_root)
    support = load_supporting_artifacts(repo_root)
    target_values = {artifact["target_signed_gain"] for _, _, artifact in budget_artifacts}
    if len(target_values) != 1:
        raise ValueError(f"expected one target signed gain, found {sorted(target_values)}")
    target = next(iter(target_values))

    feature_frontier = support["feature"]["frontier"]["self_ranked_induction"]
    budget_16x8 = budget_artifacts[1][2]
    budget_8x8 = budget_artifacts[2][2]
    ops_ratio = (
        budget_8x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
        / budget_16x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
    )

    lines = [
        "% Generated by scripts/build_paper_tables.py; do not edit by hand.",
        rf"\newcommand{{\LsdPaperArtifactCount}}{{{len(BUDGET_ARTIFACTS) + len(support)}}}",
        rf"\newcommand{{\LsdPaperTargetGain}}{{{fmt_float(target)}}}",
        rf"\newcommand{{\LsdPaperFeatureFrontierSelfRanked}}{{{feature_frontier['best_signed_gain_feature_dimension']}}}",
        rf"\newcommand{{\LsdPaperEightByEightOpsRatio}}{{{fmt_float(ops_ratio)}}}",
        "",
    ]
    return "\n".join(lines)


def build_frontier_table(repo_root: Path) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed f1024 neural frontier. First target is the smallest external-material budget reaching \LsdPaperTargetGain{} signed gain. Best LSD is signed learning-signal density per one million event-compute units.}",
        r"\label{tab:f1024-profile-frontier}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Profile & Condition & First target & Best gain & Best LSD & Ops at 64 \\",
        r"\midrule",
    ]
    for profile_label, _, artifact in load_budget_artifacts(repo_root):
        for condition in FRONTIER_CONDITIONS:
            thresholds = artifact["thresholds"][condition]
            budget64 = artifact["budgets"]["64"][condition]
            row = [
                latex_escape(profile_label),
                latex_escape(condition_label(condition)),
                latex_escape(fmt_target(thresholds["first_material_count_reaching_target"])),
                fmt_float(thresholds["best_signed_gain"]),
                fmt_float(thresholds["best_signed_learning_signal_density_per_1m_event_compute"]),
                latex_escape(fmt_ops(budget64["estimated_neural_training_multiply_adds_mean"])),
            ]
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_validation_selected_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["agreement_gated"]
    conditions = (
        "self_ranked_induction",
        "sample_aware_self_ranked_induction",
        "agreement_gated_self_ranked_induction",
        "validation_ranked_induction",
        "mdl_rule_expansion",
        "counterfactual_expansion",
    )
    materials = ("16", "24", "32", "48")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Reliability-policy probe at the 16x8 f1024 profile. Entries are heldout accuracy improvement over the majority baseline; first target uses \LsdPaperTargetGain{} signed gain.}",
        r"\label{tab:validation-selected-reliability-probe}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & First target & 16 & 24 & 32 & 48 & Best gain \\",
        r"\midrule",
    ]
    for condition in conditions:
        thresholds = artifact["thresholds"][condition]
        row = [
            latex_escape(condition_label(condition)),
            latex_escape(fmt_target(thresholds["first_material_count_reaching_target"])),
            *(
                fmt_float(artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"])
                for material in materials
            ),
            fmt_float(thresholds["best_signed_gain"]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_low_budget_failure_table(repo_root: Path) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Low-budget ranked-induction failure probe. Entries are heldout accuracy improvement over the majority baseline at scarce external-material budgets.}",
        r"\label{tab:low-budget-failure-probe}",
        r"\begin{tabular}{@{}llrrrl@{}}",
        r"\toprule",
        r"Profile & Condition & 16 & 24 & 32 & Reading \\",
        r"\midrule",
    ]
    for profile_label, _, artifact in load_budget_artifacts(repo_root)[1:]:
        for condition in LOW_BUDGET_CONDITIONS:
            gains = [
                artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"]
                for material in LOW_BUDGET_MATERIALS
            ]
            negative_32 = gains[-1] < 0
            reading = "32-material failure" if negative_32 else "positive at 32"
            row = [
                latex_escape(profile_label),
                latex_escape(condition_label(condition)),
                *(fmt_float(gain) for gain in gains),
                latex_escape(reading),
            ]
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_tables(repo_root: Path) -> str:
    return "\n".join(
        [
            build_macros(repo_root),
            build_frontier_table(repo_root),
            build_validation_selected_table(repo_root),
            build_low_budget_failure_table(repo_root),
        ]
    )


def write_tables(repo_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_tables(repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing result artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/results_tables.tex"),
        help="Output TeX file path, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    write_tables(repo_root, output)
    print(f"wrote {output.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
