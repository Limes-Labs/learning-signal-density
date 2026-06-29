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
POLICY_ENVELOPE_ARTIFACT = Path("results/policy_envelope_f1024.json")
VALIDATION_PORTFOLIO_ARTIFACT = Path("results/tiny_neural_budget_sweep_validation_portfolio_f1024.json")
VALIDATION_LINEAR_PROXY_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json"
)
VALIDATION_ABSTAINING_PROXY_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json"
)
VALIDATION_COVERAGE_PROXY_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json"
)
TEMPERED_SAMPLE_AWARE_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json"
)
SELECTOR_TRANSFER_ARTIFACT = Path("results/tiny_neural_budget_sweep_selector_transfer_f1024.json")
TRAIN_SIZE_GATED_ARTIFACT = Path("results/tiny_neural_budget_sweep_train_size_gated_f1024.json")
GENERATED_LABEL_AUDIT_ARTIFACT = Path("results/generated_label_audit_selector_transfer_f1024.json")
GENERATED_COVERAGE_AUDIT_ARTIFACT = Path("results/generated_coverage_audit_selector_transfer_f1024.json")

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
    "tempered_sample_aware_self_ranked_induction": "Tempered sample-aware",
    "train_size_gated_sample_aware_induction": "Train-size gated sample-aware",
    "validation_abstaining_proxy_selector": "Abstaining proxy selector",
    "validation_coverage_proxy_selector": "Validation coverage proxy",
    "validation_linear_proxy_selector": "Linear proxy selector",
    "validation_portfolio_selector": "Validation portfolio selector",
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


def validate_policy_envelope_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("post_hoc_envelope") is not True:
        raise ValueError(f"{artifact_path} must be marked as a post-hoc envelope")
    if scope.get("heldout_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose heldout policy selection")
    if scope.get("deployable_policy") is not False:
        raise ValueError(f"{artifact_path} must not be marked as deployable")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    if artifact.get("oracle_condition") in artifact.get("conditions_considered", []):
        raise ValueError(f"{artifact_path} must exclude the oracle from non-oracle conditions")


def validate_linear_proxy_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_linear_proxy_selector", {})
    if scope.get("low_fidelity_proxy_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the low-fidelity proxy selector")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the proxy selector")


def validate_abstaining_proxy_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_abstaining_proxy_selector", {})
    if scope.get("low_fidelity_proxy_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the low-fidelity proxy selector")
    if scope.get("raw_text_abstention") is not True:
        raise ValueError(f"{artifact_path} must mark raw-text abstention")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the abstaining proxy selector")


def validate_train_size_gated_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("train_size_gated_sample_aware_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only schedule selection")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for train-size schedule selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for train-size transform selection")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the train-size gate")


def validate_tempered_sample_aware_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("tempered_sample_aware_self_ranked_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only tempered selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only tempered induction")
    if scope.get("validation_used_for_threshold") is not False:
        raise ValueError(f"{artifact_path} must not use validation thresholds for the tempered policy")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation transform selection for the tempered policy")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the tempered policy")


def validate_validation_coverage_proxy_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_coverage_proxy_selector", {})
    if scope.get("low_fidelity_coverage_proxy_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the low-fidelity coverage proxy selector")
    if scope.get("validation_motif_distribution_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation motif policy selection")
    if scope.get("validation_labels_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation labels for coverage policy selection")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the coverage proxy selector")


def validate_generated_label_audit_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("uses_hidden_rulebook_for_audit") is not True:
        raise ValueError(f"{artifact_path} must disclose hidden-rulebook audit labels")
    if scope.get("hidden_rulebook_available_to_policies") is not False:
        raise ValueError(f"{artifact_path} must not expose the hidden rulebook to policies")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("deployable_policy") is not False:
        raise ValueError(f"{artifact_path} must be marked non-deployable")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")


def validate_generated_coverage_audit_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("uses_hidden_rulebook_for_label_audit") is not True:
        raise ValueError(f"{artifact_path} must disclose hidden-rulebook label audit")
    if scope.get("uses_heldout_distribution_for_audit") is not True:
        raise ValueError(f"{artifact_path} must disclose heldout distribution audit")
    if scope.get("heldout_distribution_available_to_policies") is not False:
        raise ValueError(f"{artifact_path} must not expose heldout distribution to policies")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("deployable_policy") is not False:
        raise ValueError(f"{artifact_path} must be marked non-deployable")
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
        "policy_envelope": load_json(repo_root / POLICY_ENVELOPE_ARTIFACT),
        "validation_portfolio": load_json(repo_root / VALIDATION_PORTFOLIO_ARTIFACT),
        "validation_linear_proxy": load_json(repo_root / VALIDATION_LINEAR_PROXY_ARTIFACT),
        "validation_abstaining_proxy": load_json(repo_root / VALIDATION_ABSTAINING_PROXY_ARTIFACT),
        "validation_coverage_proxy": load_json(repo_root / VALIDATION_COVERAGE_PROXY_ARTIFACT),
        "tempered_sample_aware": load_json(repo_root / TEMPERED_SAMPLE_AWARE_ARTIFACT),
        "selector_transfer": load_json(repo_root / SELECTOR_TRANSFER_ARTIFACT),
        "train_size_gated": load_json(repo_root / TRAIN_SIZE_GATED_ARTIFACT),
        "generated_label_audit": load_json(repo_root / GENERATED_LABEL_AUDIT_ARTIFACT),
        "generated_coverage_audit": load_json(repo_root / GENERATED_COVERAGE_AUDIT_ARTIFACT),
    }
    validate_claim_scope(FEATURE_FRONTIER_ARTIFACT, artifacts["feature"])
    validate_claim_scope(PROFILE_FRONTIER_ARTIFACT, artifacts["profile"])
    validate_claim_scope(VALIDATION_SELECTED_ARTIFACT, artifacts["validation_selected"])
    validate_claim_scope(AGREEMENT_GATED_ARTIFACT, artifacts["agreement_gated"])
    validate_policy_envelope_scope(POLICY_ENVELOPE_ARTIFACT, artifacts["policy_envelope"])
    validate_claim_scope(VALIDATION_PORTFOLIO_ARTIFACT, artifacts["validation_portfolio"])
    validate_claim_scope(VALIDATION_LINEAR_PROXY_ARTIFACT, artifacts["validation_linear_proxy"])
    validate_linear_proxy_scope(VALIDATION_LINEAR_PROXY_ARTIFACT, artifacts["validation_linear_proxy"])
    validate_claim_scope(VALIDATION_ABSTAINING_PROXY_ARTIFACT, artifacts["validation_abstaining_proxy"])
    validate_abstaining_proxy_scope(
        VALIDATION_ABSTAINING_PROXY_ARTIFACT,
        artifacts["validation_abstaining_proxy"],
    )
    validate_claim_scope(VALIDATION_COVERAGE_PROXY_ARTIFACT, artifacts["validation_coverage_proxy"])
    validate_validation_coverage_proxy_scope(
        VALIDATION_COVERAGE_PROXY_ARTIFACT,
        artifacts["validation_coverage_proxy"],
    )
    validate_claim_scope(TEMPERED_SAMPLE_AWARE_ARTIFACT, artifacts["tempered_sample_aware"])
    validate_tempered_sample_aware_scope(
        TEMPERED_SAMPLE_AWARE_ARTIFACT,
        artifacts["tempered_sample_aware"],
    )
    validate_claim_scope(SELECTOR_TRANSFER_ARTIFACT, artifacts["selector_transfer"])
    validate_abstaining_proxy_scope(SELECTOR_TRANSFER_ARTIFACT, artifacts["selector_transfer"])
    validate_claim_scope(TRAIN_SIZE_GATED_ARTIFACT, artifacts["train_size_gated"])
    validate_train_size_gated_scope(TRAIN_SIZE_GATED_ARTIFACT, artifacts["train_size_gated"])
    validate_generated_label_audit_scope(
        GENERATED_LABEL_AUDIT_ARTIFACT,
        artifacts["generated_label_audit"],
    )
    validate_generated_coverage_audit_scope(
        GENERATED_COVERAGE_AUDIT_ARTIFACT,
        artifacts["generated_coverage_audit"],
    )
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


def build_policy_envelope_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["policy_envelope"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Post-hoc policy envelope for the 16x8 f1024 reliability probe. The envelope is non-deployable because it selects the best non-oracle condition at each budget using completed heldout results; oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:post-hoc-policy-envelope}",
        r"\begin{tabular}{@{}rlrrrr@{}}",
        r"\toprule",
        r"Budget & Envelope condition & Gain & Compute & Oracle gain & Gap \\",
        r"\midrule",
    ]
    for material in artifact["material_counts"]:
        row = artifact["best_by_material"][str(material)]
        lines.append(
            " & ".join(
                [
                    str(material),
                    latex_escape(row["condition_label"]),
                    fmt_float(row["signed_gain"]),
                    latex_escape(fmt_ops(row["charged_compute_units"])),
                    fmt_float(row["oracle_signed_gain"]),
                    fmt_float(row["oracle_gap_signed_gain"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def fmt_selection_counts(counts: dict[str, int]) -> str:
    return "; ".join(
        f"{condition_label(condition)} {count}/5"
        for condition, count in sorted(counts.items(), key=lambda item: (-item[1], condition_label(item[0])))
    )


def build_validation_portfolio_table(repo_root: Path) -> str:
    support = load_supporting_artifacts(repo_root)
    envelope = support["policy_envelope"]
    selectors = [
        (
            "Abstaining proxy selector",
            support["validation_abstaining_proxy"],
            "validation_abstaining_proxy_selector",
        ),
        ("Linear proxy selector", support["validation_linear_proxy"], "validation_linear_proxy_selector"),
        ("Validation portfolio selector", support["validation_portfolio"], "validation_portfolio_selector"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation selector probes versus the post-hoc envelope. The abstaining proxy requires a non-raw candidate to beat raw text by three validation examples before leaving raw text; the linear proxy scores each candidate with a two-epoch linear classifier before training one tiny MLP; the full portfolio selector trains and validates every candidate tiny MLP. All charge selection and evaluate heldout once; the envelope is heldout-selected and non-deployable.}",
        r"\label{tab:validation-portfolio-selector}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rlrrrr>{\raggedright\arraybackslash}p{0.26\linewidth}@{}}",
        r"\toprule",
        r"Budget & Selector & Gain & LSD & Cost & Env. gain & Choices \\",
        r"\midrule",
    ]
    for material in support["validation_abstaining_proxy"]["material_counts"]:
        material_key = str(material)
        envelope_row = envelope["best_by_material"][material_key]
        for selector_label, selector_artifact, condition in selectors:
            row = selector_artifact["budgets"][material_key][condition]
            lines.append(
                " & ".join(
                    [
                        str(material),
                        latex_escape(selector_label),
                        fmt_float(row["accuracy_improvement_over_majority_mean"], digits=6),
                        fmt_float(row["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                        latex_escape(fmt_ops(row["charged_compute_units_mean"])),
                        fmt_float(envelope_row["signed_gain"]),
                        latex_escape(fmt_selection_counts(row["portfolio_selected_condition_counts"])),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_selector_transfer_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["selector_transfer"]
    conditions = (
        "raw_text",
        "sample_aware_self_ranked_induction",
        "validation_abstaining_proxy_selector",
        "validation_linear_proxy_selector",
        "validation_portfolio_selector",
        "counterfactual_expansion",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed selector transfer stress test. The selector-family policies are rerun on seeds 37, 41, 43, 47, and 53 after being developed on the previous selector artifacts. Entries are heldout accuracy improvement over majority; oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:selector-transfer-stress}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Condition & 16 & 24 & 32 & 48 & 64 & Best gain \\",
        r"\midrule",
    ]
    for condition in conditions:
        thresholds = artifact["thresholds"][condition]
        row = [
            latex_escape(condition_label(condition)),
            *(
                fmt_float(
                    artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"],
                    digits=6,
                )
                for material in materials
            ),
            fmt_float(thresholds["best_signed_gain"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_generated_label_audit_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["generated_label_audit"]
    conditions = (
        "sample_aware_self_ranked_induction",
        "agreement_gated_self_ranked_induction",
        "validation_ranked_induction",
        "mdl_rule_expansion",
        "counterfactual_expansion",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Generated-label audit on the selector-transfer seeds. Label precision is computed by comparing generated synthetic labels with the hidden rulebook after the source sweep; the audit is non-deployable and tests whether label correctness alone explains neural gain.}",
        r"\label{tab:generated-label-audit}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Condition & 24 label prec. & 32 label prec. & 32 gain & 64 label prec. & 64 gain \\",
        r"\midrule",
    ]
    for condition in conditions:
        audit24 = artifact["audits"]["24"][condition]
        audit32 = artifact["audits"]["32"][condition]
        audit64 = artifact["audits"]["64"][condition]
        row = [
            latex_escape(condition_label(condition)),
            fmt_float(audit24["label_precision"], digits=6),
            fmt_float(audit32["label_precision"], digits=6),
            fmt_float(audit32["linked_accuracy_improvement_over_majority_mean"], digits=6),
            fmt_float(audit64["label_precision"], digits=6),
            fmt_float(audit64["linked_accuracy_improvement_over_majority_mean"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_generated_coverage_audit_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["generated_coverage_audit"]
    conditions = (
        "sample_aware_self_ranked_induction",
        "agreement_gated_self_ranked_induction",
        "validation_ranked_induction",
        "mdl_rule_expansion",
        "counterfactual_expansion",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Generated-coverage audit on the selector-transfer seeds. Triple L1 distance compares generated synthetic-label motif frequencies with heldout family/stimulus/modifier frequencies after the source sweep; lower is closer. The audit is non-deployable because heldout distribution is used only after the fact.}",
        r"\label{tab:generated-coverage-audit}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Condition & 32 triple L1 & 32 gain & 64 triple L1 & 64 gain & 64 label prec. \\",
        r"\midrule",
    ]
    for condition in conditions:
        audit32 = artifact["audits"]["32"][condition]
        audit64 = artifact["audits"]["64"][condition]
        row = [
            latex_escape(condition_label(condition)),
            fmt_float(audit32["generated_vs_heldout_triple_l1_distance"], digits=6),
            fmt_float(audit32["linked_accuracy_improvement_over_majority_mean"], digits=6),
            fmt_float(audit64["generated_vs_heldout_triple_l1_distance"], digits=6),
            fmt_float(audit64["linked_accuracy_improvement_over_majority_mean"], digits=6),
            fmt_float(audit64["label_precision"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_validation_coverage_proxy_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_coverage_proxy"]
    conditions = (
        "raw_text",
        "sample_aware_self_ranked_induction",
        "train_size_gated_sample_aware_induction",
        "validation_coverage_proxy_selector",
        "validation_abstaining_proxy_selector",
        "validation_portfolio_selector",
        "counterfactual_expansion",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed validation-coverage proxy probe. The deployable proxy chooses one candidate by matching generated synthetic-motif coverage to the validation motif distribution, without using heldout distribution or validation labels for the selector score. Entries are heldout accuracy improvement over majority; oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:validation-coverage-proxy}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Condition & 16 & 24 & 32 & 48 & 64 & First target & Best gain \\",
        r"\midrule",
    ]
    for condition in conditions:
        thresholds = artifact["thresholds"][condition]
        row = [
            latex_escape(condition_label(condition)),
            *(
                fmt_float(
                    artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"],
                    digits=6,
                )
                for material in materials
            ),
            latex_escape(fmt_target(thresholds["first_material_count_reaching_target"])),
            fmt_float(thresholds["best_signed_gain"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_tempered_sample_aware_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["tempered_sample_aware"]
    conditions = (
        "raw_text",
        "qa_expansion",
        "sample_aware_self_ranked_induction",
        "tempered_sample_aware_self_ranked_induction",
        "train_size_gated_sample_aware_induction",
        "validation_coverage_proxy_selector",
        "counterfactual_expansion",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed tempered sample-aware ablation. The tempered policy is train-only and lowers the mid-budget synthetic ratio from 0.75 to 0.50 for train splits below 144 events, testing whether smaller generated-label budgets repair the 24--32 material failure without validation selection. Entries are heldout accuracy improvement over majority; oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:tempered-sample-aware}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Condition & 16 & 24 & 32 & 48 & 64 & First target & Best gain \\",
        r"\midrule",
    ]
    for condition in conditions:
        thresholds = artifact["thresholds"][condition]
        row = [
            latex_escape(condition_label(condition)),
            *(
                fmt_float(
                    artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"],
                    digits=6,
                )
                for material in materials
            ),
            latex_escape(fmt_target(thresholds["first_material_count_reaching_target"])),
            fmt_float(thresholds["best_signed_gain"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_train_size_gated_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["train_size_gated"]
    conditions = (
        "raw_text",
        "sample_aware_self_ranked_induction",
        "train_size_gated_sample_aware_induction",
        "validation_abstaining_proxy_selector",
        "validation_portfolio_selector",
        "counterfactual_expansion",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Unseen-seed train-size gated baseline. Seeds 59, 61, 67, 71, and 73 test a deployable train-only schedule that uses raw text below 144 train events and sample-aware self-ranked induction once the train split is large enough. Entries are heldout accuracy improvement over majority; oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:train-size-gated-baseline}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Condition & 16 & 24 & 32 & 48 & 64 & Best gain & Best LSD \\",
        r"\midrule",
    ]
    for condition in conditions:
        thresholds = artifact["thresholds"][condition]
        row = [
            latex_escape(condition_label(condition)),
            *(
                fmt_float(
                    artifact["budgets"][material][condition]["accuracy_improvement_over_majority_mean"],
                    digits=6,
                )
                for material in materials
            ),
            fmt_float(thresholds["best_signed_gain"], digits=6),
            fmt_float(
                thresholds["best_signed_learning_signal_density_per_1m_event_compute"],
                digits=6,
            ),
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
            build_policy_envelope_table(repo_root),
            build_validation_portfolio_table(repo_root),
            build_selector_transfer_table(repo_root),
            build_generated_label_audit_table(repo_root),
            build_generated_coverage_audit_table(repo_root),
            build_validation_coverage_proxy_table(repo_root),
            build_tempered_sample_aware_table(repo_root),
            build_train_size_gated_table(repo_root),
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
