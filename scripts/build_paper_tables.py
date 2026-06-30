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
VALIDATION_COVERAGE_PRIOR_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.json"
)
TEMPERED_SAMPLE_AWARE_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json"
)
COMPACT_TRAIN_SIZE_GATED_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json"
)
DIVERSITY_INTERACTION_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_diversity_interaction_f1024.json"
)
DENSITY_CAPPED_COMPACT_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_density_capped_compact_f1024.json"
)
SUPPORT_RAMPED_COMPACT_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json"
)
LATE_CONFIDENCE_RAMPED_COMPACT_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json"
)
DENSITY_WINDOW_COMPACT_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_density_window_compact_f1024.json"
)
TRAIN_SUPPORT_DENSITY_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_train_support_density_f1024.json"
)
SUPPORT_PROBE_WINDOW_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_support_probe_window_f1024.json"
)
VALIDATION_SUPPORT_PRECISION_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_support_precision_f1024.json"
)
VALIDATION_SUPPORT_PRECISION_GATE_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json"
)
SUPPORT_SELECTOR_TRANSFER_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json"
)
SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT = Path("results/support_selector_error_audit_f1024.json")
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
    "compact_diverse_train_size_gated_induction": "Compact diverse",
    "compact_train_size_gated_induction": "Compact train-size gated",
    "density_capped_compact_induction": "Density-capped compact",
    "density_window_compact_induction": "Density-window compact",
    "diverse_self_ranked_induction": "Diverse self-ranked",
    "late_confidence_ramped_compact_induction": "Late-confidence compact",
    "mdl_rule_expansion": "MDL rule expansion",
    "qa_expansion": "QA expansion",
    "sample_aware_diverse_self_ranked_induction": "Sample-aware diverse",
    "raw_text": "Raw text",
    "sample_aware_self_ranked_induction": "Sample-aware self-ranked",
    "self_ranked_induction": "Self-ranked",
    "support_ramped_compact_induction": "Support-ramped compact",
    "support_probe_window_selector": "Support-probe window selector",
    "tempered_sample_aware_self_ranked_induction": "Tempered sample-aware",
    "train_support_density_selector": "Train support-density selector",
    "train_size_gated_sample_aware_induction": "Train-size gated sample-aware",
    "validation_abstaining_proxy_selector": "Abstaining proxy selector",
    "validation_coverage_prior_selector": "Coverage-prior selector",
    "validation_coverage_proxy_selector": "Validation coverage proxy",
    "validation_linear_proxy_selector": "Linear proxy selector",
    "validation_portfolio_selector": "Validation portfolio selector",
    "validation_ranked_induction": "Validation-ranked",
    "validation_support_precision_gate_selector": "Validation support-precision gate",
    "validation_support_precision_selector": "Validation support-precision selector",
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


def validate_compact_train_size_gated_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("compact_train_size_gated_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only compact schedule selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only compact induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for compact schedule selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for compact transform selection")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark the compact large-sample encoding")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the compact gate")


def validate_diversity_interaction_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    sample_diverse = artifact.get("condition_scope", {}).get(
        "sample_aware_diverse_self_ranked_induction",
        {},
    )
    if sample_diverse.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only sample-aware diversity selection")
    if sample_diverse.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only sample-aware diversity induction")
    if sample_diverse.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for sample-aware diversity")
    if sample_diverse.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for sample-aware diversity")

    compact_diverse = artifact.get("condition_scope", {}).get(
        "compact_diverse_train_size_gated_induction",
        {},
    )
    if compact_diverse.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only compact-diversity selection")
    if compact_diverse.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only compact-diversity induction")
    if compact_diverse.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for compact-diversity selection")
    if compact_diverse.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for compact-diversity transforms")
    if compact_diverse.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact-diversity large-sample encoding")
    if compact_diverse.get("diversity_penalty_after_compaction") is not True:
        raise ValueError(f"{artifact_path} must mark the post-compaction diversity penalty")
    if compact_diverse.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for compact diversity")


def validate_density_capped_compact_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("density_capped_compact_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only density-capped selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only density-capped induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for density-capped selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for density-capped transforms")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact large-sample encoding")
    if scope.get("abundant_data_raw_fallback") is not True:
        raise ValueError(f"{artifact_path} must mark abundant-data raw fallback")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the density-capped gate")


def validate_support_ramped_compact_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("support_ramped_compact_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-ramped selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-ramped induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-ramped selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-ramped transforms")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact large-sample encoding")
    if scope.get("abundant_data_support_ramp") is not True:
        raise ValueError(f"{artifact_path} must mark abundant-data support ramp")
    if scope.get("abundant_data_min_support") != 4:
        raise ValueError(f"{artifact_path} must disclose the abundant-data support floor")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the support-ramped policy")


def validate_late_confidence_ramped_compact_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("late_confidence_ramped_compact_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only late-confidence selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only late-confidence induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for late-confidence selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for late-confidence transforms")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact large-sample encoding")
    if scope.get("abundant_data_support_ramp") is not True:
        raise ValueError(f"{artifact_path} must mark abundant-data support ramp")
    if scope.get("abundant_data_min_support") != 4:
        raise ValueError(f"{artifact_path} must disclose the abundant-data support floor")
    if scope.get("late_confidence_ramp_min_events") != 432:
        raise ValueError(f"{artifact_path} must disclose the late confidence tier")
    if scope.get("late_confidence_min_confidence") != 0.60:
        raise ValueError(f"{artifact_path} must disclose the late confidence floor")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the late-confidence policy")


def validate_density_window_compact_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("density_window_compact_induction", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only density-window selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only density-window induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for density-window selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for density-window transforms")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact large-sample encoding")
    if scope.get("compact_density_window") is not True:
        raise ValueError(f"{artifact_path} must mark the compact density window")
    if scope.get("compact_density_window_max_events") != 320:
        raise ValueError(f"{artifact_path} must disclose the compact density-window ceiling")
    if scope.get("transition_support_window") is not True:
        raise ValueError(f"{artifact_path} must mark the transition support window")
    if scope.get("transition_support_window_min_events") != 400:
        raise ValueError(f"{artifact_path} must disclose the support-window floor")
    if scope.get("transition_support_window_max_events") != 432:
        raise ValueError(f"{artifact_path} must disclose the support-window ceiling")
    if scope.get("abundant_data_raw_fallback") is not True:
        raise ValueError(f"{artifact_path} must mark abundant-data raw fallback")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the density-window policy")


def validate_train_support_density_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("train_support_density_selector", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-density selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-density induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-density selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-density transforms")
    if scope.get("support_density_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the support-density selector")
    if scope.get("support_density_min_kept_per_compute") != 0.00145:
        raise ValueError(f"{artifact_path} must disclose the support-density threshold")
    if scope.get("compact_original_encoding_at_large_samples") is not True:
        raise ValueError(f"{artifact_path} must mark compact large-sample encoding")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the support-density selector")


def validate_support_probe_window_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("support_probe_window_selector", {})
    if scope.get("train_only_selection") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-probe selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must mark train-only support-probe induction")
    if scope.get("validation_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-probe selection")
    if scope.get("validation_used_for_transform_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for support-probe transforms")
    if scope.get("support_probe_window_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the support-probe window selector")
    if scope.get("support_probe_min_train_events") != 360:
        raise ValueError(f"{artifact_path} must disclose the support-probe lower window")
    if scope.get("support_probe_max_train_events") != 432:
        raise ValueError(f"{artifact_path} must disclose the support-probe upper window")
    if scope.get("reuse_selected_candidate_construction") is not True:
        raise ValueError(f"{artifact_path} must disclose selected-candidate reuse")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the support-probe selector")


def validate_validation_support_precision_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_support_precision_selector", {})
    if scope.get("train_only_selection") is not False:
        raise ValueError(f"{artifact_path} must mark validation-based support selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must keep induction train-only")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("validation_used_for_transform_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation transform selection")
    if scope.get("validation_support_precision_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the validation support-precision selector")
    if scope.get("validation_support_precision_threshold") != 0.825758:
        raise ValueError(f"{artifact_path} must disclose the validation precision threshold")
    if scope.get("validation_support_compact_max_train_events") != 320:
        raise ValueError(f"{artifact_path} must disclose the compact floor")
    if scope.get("validation_support_transition_min_train_events") != 400:
        raise ValueError(f"{artifact_path} must disclose the transition lower window")
    if scope.get("validation_support_transition_max_train_events") != 432:
        raise ValueError(f"{artifact_path} must disclose the transition upper window")
    if scope.get("reuse_selected_candidate_construction") is not True:
        raise ValueError(f"{artifact_path} must disclose selected-candidate reuse")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the validation selector")


def validate_validation_support_precision_gate_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_support_precision_gate_selector", {})
    if scope.get("train_only_selection") is not False:
        raise ValueError(f"{artifact_path} must mark validation-based support selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must keep induction train-only")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("validation_used_for_transform_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation transform selection")
    if scope.get("validation_support_precision_gate_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the no-window validation support gate")
    if scope.get("validation_support_precision_selector") is not False:
        raise ValueError(f"{artifact_path} must distinguish the gate from the fixed-window selector")
    if scope.get("validation_support_precision_threshold") != 0.825758:
        raise ValueError(f"{artifact_path} must disclose the validation precision threshold")
    if scope.get("validation_support_compact_max_train_events") != 320:
        raise ValueError(f"{artifact_path} must disclose the compact floor")
    if scope.get("validation_support_uses_fixed_transition_prior") is not False:
        raise ValueError(f"{artifact_path} must disclose that the fixed transition prior is removed")
    if scope.get("reuse_selected_candidate_construction") is not True:
        raise ValueError(f"{artifact_path} must disclose selected-candidate reuse")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the validation gate")


def validate_support_selector_transfer_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("seeds") != [1459, 1471, 1481, 1483, 1487]:
        raise ValueError(f"{artifact_path} must use the preregistered transfer seed block")
    if artifact.get("confirmation_of") != str(VALIDATION_SUPPORT_PRECISION_GATE_ARTIFACT):
        raise ValueError(f"{artifact_path} must identify the gate artifact it stress-tests")
    fixed_scope = artifact.get("condition_scope", {}).get("validation_support_precision_selector", {})
    gate_scope = artifact.get("condition_scope", {}).get("validation_support_precision_gate_selector", {})
    if fixed_scope.get("validation_support_precision_selector") is not True:
        raise ValueError(f"{artifact_path} must include the fixed-transition validation selector")
    if gate_scope.get("validation_support_precision_gate_selector") is not True:
        raise ValueError(f"{artifact_path} must include the no-window validation gate")
    if gate_scope.get("validation_support_uses_fixed_transition_prior") is not False:
        raise ValueError(f"{artifact_path} must disclose that the gate removes the transition prior")
    if gate_scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the validation gate")


def validate_support_selector_error_audit_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("post_hoc_diagnostic") is not True:
        raise ValueError(f"{artifact_path} must be marked as a post-hoc diagnostic")
    if scope.get("uses_committed_fresh_seed_artifacts") is not True:
        raise ValueError(f"{artifact_path} must use committed fresh-seed artifacts")
    if scope.get("heldout_used_for_error_analysis") is not True:
        raise ValueError(f"{artifact_path} must disclose heldout error analysis")
    if scope.get("heldout_available_to_policies") is not False:
        raise ValueError(f"{artifact_path} must not expose heldout outcomes to policies")
    if scope.get("deployable_policy") is not False:
        raise ValueError(f"{artifact_path} must be marked non-deployable")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    if artifact.get("recommendation", {}).get("promote_support_selector") is not False:
        raise ValueError(f"{artifact_path} must not promote a support selector")
    if str(SUPPORT_SELECTOR_TRANSFER_ARTIFACT) not in artifact.get("source_artifacts", []):
        raise ValueError(f"{artifact_path} must include the support-selector transfer artifact")


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


def validate_validation_coverage_prior_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("condition_scope", {}).get("validation_coverage_prior_selector", {})
    if scope.get("low_fidelity_coverage_proxy_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the low-fidelity coverage proxy selector")
    if scope.get("validation_motif_distribution_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation motif policy selection")
    if scope.get("validation_labels_used_for_policy_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation labels for coverage-prior selection")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("train_size_prior_min_events") != 96:
        raise ValueError(f"{artifact_path} must disclose the train-size prior floor")
    if scope.get("lean_coverage_candidate_set") is not True:
        raise ValueError(f"{artifact_path} must disclose the lean coverage candidate set")
    if scope.get("coverage_utility_compute_penalty") != 0.00001:
        raise ValueError(f"{artifact_path} must disclose the coverage utility compute penalty")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the coverage-prior selector")


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
        "validation_coverage_prior": load_json(repo_root / VALIDATION_COVERAGE_PRIOR_ARTIFACT),
        "tempered_sample_aware": load_json(repo_root / TEMPERED_SAMPLE_AWARE_ARTIFACT),
        "compact_train_size_gated": load_json(repo_root / COMPACT_TRAIN_SIZE_GATED_ARTIFACT),
        "diversity_interaction": load_json(repo_root / DIVERSITY_INTERACTION_ARTIFACT),
        "density_capped_compact": load_json(repo_root / DENSITY_CAPPED_COMPACT_ARTIFACT),
        "support_ramped_compact": load_json(repo_root / SUPPORT_RAMPED_COMPACT_ARTIFACT),
        "late_confidence_ramped_compact": load_json(
            repo_root / LATE_CONFIDENCE_RAMPED_COMPACT_ARTIFACT
        ),
        "density_window_compact": load_json(repo_root / DENSITY_WINDOW_COMPACT_ARTIFACT),
        "train_support_density": load_json(repo_root / TRAIN_SUPPORT_DENSITY_ARTIFACT),
        "support_probe_window": load_json(repo_root / SUPPORT_PROBE_WINDOW_ARTIFACT),
        "validation_support_precision": load_json(repo_root / VALIDATION_SUPPORT_PRECISION_ARTIFACT),
        "validation_support_precision_gate": load_json(
            repo_root / VALIDATION_SUPPORT_PRECISION_GATE_ARTIFACT
        ),
        "support_selector_transfer": load_json(repo_root / SUPPORT_SELECTOR_TRANSFER_ARTIFACT),
        "support_selector_error_audit": load_json(repo_root / SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT),
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
    validate_claim_scope(VALIDATION_COVERAGE_PRIOR_ARTIFACT, artifacts["validation_coverage_prior"])
    validate_validation_coverage_prior_scope(
        VALIDATION_COVERAGE_PRIOR_ARTIFACT,
        artifacts["validation_coverage_prior"],
    )
    validate_claim_scope(TEMPERED_SAMPLE_AWARE_ARTIFACT, artifacts["tempered_sample_aware"])
    validate_tempered_sample_aware_scope(
        TEMPERED_SAMPLE_AWARE_ARTIFACT,
        artifacts["tempered_sample_aware"],
    )
    validate_claim_scope(COMPACT_TRAIN_SIZE_GATED_ARTIFACT, artifacts["compact_train_size_gated"])
    validate_compact_train_size_gated_scope(
        COMPACT_TRAIN_SIZE_GATED_ARTIFACT,
        artifacts["compact_train_size_gated"],
    )
    validate_claim_scope(DIVERSITY_INTERACTION_ARTIFACT, artifacts["diversity_interaction"])
    validate_diversity_interaction_scope(
        DIVERSITY_INTERACTION_ARTIFACT,
        artifacts["diversity_interaction"],
    )
    validate_claim_scope(DENSITY_CAPPED_COMPACT_ARTIFACT, artifacts["density_capped_compact"])
    validate_density_capped_compact_scope(
        DENSITY_CAPPED_COMPACT_ARTIFACT,
        artifacts["density_capped_compact"],
    )
    validate_claim_scope(SUPPORT_RAMPED_COMPACT_ARTIFACT, artifacts["support_ramped_compact"])
    validate_support_ramped_compact_scope(
        SUPPORT_RAMPED_COMPACT_ARTIFACT,
        artifacts["support_ramped_compact"],
    )
    validate_claim_scope(
        LATE_CONFIDENCE_RAMPED_COMPACT_ARTIFACT,
        artifacts["late_confidence_ramped_compact"],
    )
    validate_late_confidence_ramped_compact_scope(
        LATE_CONFIDENCE_RAMPED_COMPACT_ARTIFACT,
        artifacts["late_confidence_ramped_compact"],
    )
    validate_claim_scope(DENSITY_WINDOW_COMPACT_ARTIFACT, artifacts["density_window_compact"])
    validate_density_window_compact_scope(
        DENSITY_WINDOW_COMPACT_ARTIFACT,
        artifacts["density_window_compact"],
    )
    validate_claim_scope(TRAIN_SUPPORT_DENSITY_ARTIFACT, artifacts["train_support_density"])
    validate_train_support_density_scope(
        TRAIN_SUPPORT_DENSITY_ARTIFACT,
        artifacts["train_support_density"],
    )
    validate_claim_scope(SUPPORT_PROBE_WINDOW_ARTIFACT, artifacts["support_probe_window"])
    validate_support_probe_window_scope(
        SUPPORT_PROBE_WINDOW_ARTIFACT,
        artifacts["support_probe_window"],
    )
    validate_claim_scope(
        VALIDATION_SUPPORT_PRECISION_ARTIFACT,
        artifacts["validation_support_precision"],
    )
    validate_validation_support_precision_scope(
        VALIDATION_SUPPORT_PRECISION_ARTIFACT,
        artifacts["validation_support_precision"],
    )
    validate_claim_scope(
        VALIDATION_SUPPORT_PRECISION_GATE_ARTIFACT,
        artifacts["validation_support_precision_gate"],
    )
    validate_validation_support_precision_gate_scope(
        VALIDATION_SUPPORT_PRECISION_GATE_ARTIFACT,
        artifacts["validation_support_precision_gate"],
    )
    validate_claim_scope(SUPPORT_SELECTOR_TRANSFER_ARTIFACT, artifacts["support_selector_transfer"])
    validate_support_selector_transfer_scope(
        SUPPORT_SELECTOR_TRANSFER_ARTIFACT,
        artifacts["support_selector_transfer"],
    )
    validate_support_selector_error_audit_scope(
        SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT,
        artifacts["support_selector_error_audit"],
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


def build_validation_coverage_prior_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_coverage_prior"]
    conditions = (
        "raw_text",
        "sample_aware_self_ranked_induction",
        "train_size_gated_sample_aware_induction",
        "validation_coverage_proxy_selector",
        "validation_coverage_prior_selector",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed coverage-prior selector control. The deployable selector uses raw text below 96 train events; after that floor, it scans only raw, sample-aware self-ranked, and validation-ranked candidates with a validation-motif coverage score plus a small charged-compute penalty. Entries report heldout accuracy improvement over majority, charged compute units, and signed learning-signal density per one million event-compute units.}",
        r"\label{tab:validation-coverage-prior}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 16 & 24 & 32 & 48 & 64 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean", 6),
            ("Compute", "charged_compute_units_mean", 1),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean", 6),
        )
        for index, (metric_label, metric_key, digits) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=digits,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
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


def build_compact_train_size_gated_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["compact_train_size_gated"]
    conditions = (
        "raw_text",
        "train_size_gated_sample_aware_induction",
        "compact_train_size_gated_induction",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed compact train-size gated efficiency probe. The compact policy is deployable and train-only: it matches the raw fallback below 144 train events, matches full sample-aware induction below 224 train events, and drops original QA duplicates only at the large-sample tier. Entries show heldout accuracy improvement over majority, charged compute units, and signed learning-signal density per one million event-compute units.}",
        r"\label{tab:compact-train-size-gated}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 16 & 24 & 32 & 48 & 64 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            (
                "Gain",
                "accuracy_improvement_over_majority_mean",
                6,
            ),
            (
                "Compute",
                "charged_compute_units_mean",
                1,
            ),
            (
                "LSD",
                "signed_learning_signal_density_per_1m_event_compute_mean",
                6,
            ),
        )
        for index, (metric_label, metric_key, digits) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=digits,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_diversity_interaction_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["diversity_interaction"]
    conditions = (
        "self_ranked_induction",
        "diverse_self_ranked_induction",
        "sample_aware_self_ranked_induction",
        "sample_aware_diverse_self_ranked_induction",
        "compact_train_size_gated_induction",
        "compact_diverse_train_size_gated_induction",
    )
    materials = ("16", "24", "32", "48", "64")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Diversity interaction on fresh seeds 701, 709, 719, 727, and 733. All non-raw rows are train-only generated-label policies. The table separates candidate-order diversity from representation cost: sample-aware diversity improves the 64-material non-oracle gain over sample-aware self-ranked induction, while compact diversity loses to the compact train-size gate on the density frontier.}",
        r"\label{tab:diversity-interaction}",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 16 & 24 & 32 & 48 & 64 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean", 6),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean", 6),
        )
        for index, (metric_label, metric_key, digits) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=digits,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_density_capped_compact_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["density_capped_compact"]
    conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "density_capped_compact_induction",
        "counterfactual_expansion",
    )
    materials = ("64", "80", "96", "104", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed high-budget density cap. The density-capped compact policy is train-only: it uses compact generated-label induction through 96 materials, then returns to raw text once the train split reaches the abundant-data tier. Entries report heldout accuracy improvement over majority and signed learning-signal density per one million event-compute units. Oracle counterfactual expansion is shown only as headroom.}",
        r"\label{tab:density-capped-compact}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 64 & 80 & 96 & 104 & 128 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean"),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean"),
        )
        for index, (metric_label, metric_key) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=6,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_support_ramped_compact_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["support_ramped_compact"]
    conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("64", "80", "96", "104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed support-ramped compact probe. The support-ramped policy is train-only: it matches compact induction through 96 materials, then raises the induced-label minimum support from 3 to 4 after the train split reaches the abundant-data tier. Entries report heldout accuracy improvement over majority, charged compute units, and signed learning-signal density per one million event-compute units.}",
        r"\label{tab:support-ramped-compact}",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 64 & 80 & 96 & 104 & 112 & 120 & 128 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean", 6),
            ("Compute", "charged_compute_units_mean", 1),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean", 6),
        )
        for index, (metric_label, metric_key, digits) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=digits,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_late_confidence_ramped_compact_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["late_confidence_ramped_compact"]
    conditions = (
        "raw_text",
        "support_ramped_compact_induction",
        "late_confidence_ramped_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128", "144", "160")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed late-confidence compact control. The policy matches support-ramped compact until the train split reaches 432 events, then keeps support 4 and raises the induced-label confidence floor from 0.55 to 0.60. Entries report heldout accuracy improvement over majority, charged compute units, and signed learning-signal density per one million event-compute units.}",
        r"\label{tab:late-confidence-ramped-compact}",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 96 & 104 & 112 & 120 & 128 & 144 & 160 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean", 6),
            ("Compute", "charged_compute_units_mean", 1),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean", 6),
        )
        for index, (metric_label, metric_key, digits) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=digits,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_density_window_compact_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["density_window_compact"]
    conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fresh-seed density-window compact probe. The policy is train-only: compact generated-label induction is used below 320 train events, raw text from 320 to 400, support-ramped compact induction from 400 to 432, and raw text again after 432. The fixed window improves signed density at 112 materials relative to density-capped raw fallback, preserves raw density at 120, but misses the support-ramped 128-material row.}",
        r"\label{tab:density-window-compact}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Metric & 96 & 104 & 112 & 120 & 128 \\",
        r"\midrule",
    ]
    for condition in conditions:
        rows = (
            ("Gain", "accuracy_improvement_over_majority_mean"),
            ("LSD", "signed_learning_signal_density_per_1m_event_compute_mean"),
        )
        for index, (metric_label, metric_key) in enumerate(rows):
            row = [
                latex_escape(condition_label(condition)) if index == 0 else "",
                metric_label,
                *(
                    fmt_float(
                        artifact["budgets"][material][condition][metric_key],
                        digits=6,
                    )
                    for material in materials
                ),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_train_support_density_selector_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["train_support_density"]
    comparator_conditions = (
        "raw_text",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
    )
    materials = ("104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Train support-density selector control on fresh seeds 1033, 1039, 1049, 1051, and 1061. The selector uses only train-derived support kept per charged compute to choose among raw text, compact train-size gated induction, and support-ramped compact induction, then charges candidate inspection before the final tiny-MLP fit. It often chooses sensible rows, but the inspection overhead erases the local density advantage.}",
        r"\label{tab:train-support-density-selector}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}r>{\raggedright\arraybackslash}p{0.28\linewidth}rrlr@{}}",
        r"\toprule",
        r"Budget & Selector choices & Sel. gain & Sel. LSD & Best comparator & Best LSD \\",
        r"\midrule",
    ]
    for material in materials:
        selector = artifact["budgets"][material]["train_support_density_selector"]
        best_condition = max(
            comparator_conditions,
            key=lambda condition: artifact["budgets"][material][condition][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        best = artifact["budgets"][material][best_condition]
        row = [
            material,
            latex_escape(fmt_selection_counts(selector["portfolio_selected_condition_counts"])),
            fmt_float(selector["accuracy_improvement_over_majority_mean"], digits=6),
            fmt_float(
                selector["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            latex_escape(condition_label(best_condition)),
            fmt_float(
                best["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_support_probe_window_selector_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["support_probe_window"]
    comparator_conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Support-probe window selector on fresh seeds 1063, 1069, 1087, 1091, and 1093. The train-only policy uses compact below 320 train events, raw text outside the 360--432 support-probe window, and inside that window inspects only support-ramped compact induction, reusing the selected candidate construction if support is chosen. Reuse removes the 104-material overhead, but fixed window errors remain at 112 and 120 materials.}",
        r"\label{tab:support-probe-window-selector}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}r>{\raggedright\arraybackslash}p{0.26\linewidth}rrlr@{}}",
        r"\toprule",
        r"Budget & Selector choices & Sel. gain & Sel. LSD & Best comparator & Best LSD \\",
        r"\midrule",
    ]
    for material in materials:
        selector = artifact["budgets"][material]["support_probe_window_selector"]
        best_condition = max(
            comparator_conditions,
            key=lambda condition: artifact["budgets"][material][condition][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        best = artifact["budgets"][material][best_condition]
        row = [
            material,
            latex_escape(fmt_selection_counts(selector["portfolio_selected_condition_counts"])),
            fmt_float(selector["accuracy_improvement_over_majority_mean"], digits=6),
            fmt_float(
                selector["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            latex_escape(condition_label(best_condition)),
            fmt_float(
                best["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_validation_support_precision_selector_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_support_precision"]
    comparator_conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
        "support_probe_window_selector",
        "density_capped_compact_induction",
    )
    materials = ("64", "80", "96", "104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation support-precision selector on fresh seeds 1259, 1277, 1279, 1283, and 1289. The policy keeps the train-only compact floor below 320 train events, keeps the fixed support transition from 400 to 432 train events, and otherwise uses validation labels only to estimate eligible induced-prediction precision before choosing support-ramped compact or raw text. It improves the raw/support boundary at 96 and 104 materials and has the best average LSD in this confirmation artifact, but validation false positives lose density at 120 and 128 materials.}",
        r"\label{tab:validation-support-precision-selector}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}r>{\raggedright\arraybackslash}p{0.22\linewidth}rr>{\raggedright\arraybackslash}p{0.20\linewidth}r@{}}",
        r"\toprule",
        r"Budget & Selector choices & Sel. LSD & Sel. cost & Best comparator & Best LSD \\",
        r"\midrule",
    ]
    for material in materials:
        selector = artifact["budgets"][material]["validation_support_precision_selector"]
        best_condition = max(
            comparator_conditions,
            key=lambda condition: artifact["budgets"][material][condition][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        best = artifact["budgets"][material][best_condition]
        row = [
            material,
            latex_escape(fmt_selection_counts(selector["portfolio_selected_condition_counts"])),
            fmt_float(
                selector["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(selector["portfolio_selection_cost_units_mean"], digits=1),
            latex_escape(condition_label(best_condition)),
            fmt_float(
                best["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_validation_support_precision_gate_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_support_precision_gate"]
    comparator_conditions = (
        "raw_text",
        "support_ramped_compact_induction",
        "support_probe_window_selector",
        "validation_support_precision_selector",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{No-window validation support-precision gate on fresh seeds 1381, 1399, 1409, 1423, and 1427. This control removes the unconditional 400--432 train-event support transition from Table~\ref{tab:validation-support-precision-selector} and applies the same validation precision threshold everywhere above the compact floor. It slightly beats the support-probe average, but falls below the fixed-transition validation selector because it gives up the 112-material support prior.}",
        r"\label{tab:validation-support-precision-gate}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}r>{\raggedright\arraybackslash}p{0.20\linewidth}rr>{\raggedright\arraybackslash}p{0.18\linewidth}r@{}}",
        r"\toprule",
        r"Budget & Gate choices & Gate LSD & Fixed-val. LSD & Best comp. & Best LSD \\",
        r"\midrule",
    ]
    for material in materials:
        gate = artifact["budgets"][material]["validation_support_precision_gate_selector"]
        fixed = artifact["budgets"][material]["validation_support_precision_selector"]
        best_condition = max(
            comparator_conditions,
            key=lambda condition: artifact["budgets"][material][condition][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        best = artifact["budgets"][material][best_condition]
        row = [
            material,
            latex_escape(fmt_selection_counts(gate["portfolio_selected_condition_counts"])),
            fmt_float(
                gate["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(
                fixed["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            latex_escape(condition_label(best_condition)),
            fmt_float(
                best["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_support_selector_transfer_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["support_selector_transfer"]
    conditions = (
        "density_capped_compact_induction",
        "support_probe_window_selector",
        "validation_support_precision_selector",
        "validation_support_precision_gate_selector",
    )
    materials = ("96", "104", "112", "120", "128")

    average = {
        condition: sum(
            artifact["budgets"][str(material)][condition][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in artifact["material_counts"]
        )
        / len(artifact["material_counts"])
        for condition in conditions
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Support-selector transfer stress on fresh seeds 1459, 1471, 1481, 1483, and 1487. The latest validation support selectors are rerun after development on earlier seed blocks. The no-window gate transfers better than the fixed-transition selector on average, but the simple train-only density-capped raw fallback remains the strongest all-budget density baseline in this block.}",
        r"\label{tab:support-selector-transfer}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrr>{\raggedright\arraybackslash}p{0.23\linewidth}@{}}",
        r"\toprule",
        r"Budget & Density cap & Probe & Fixed-val. & Gate & Gate choices \\",
        r"\midrule",
        " & ".join(
            [
                "Avg.",
                fmt_float(average["density_capped_compact_induction"], digits=6),
                fmt_float(average["support_probe_window_selector"], digits=6),
                fmt_float(average["validation_support_precision_selector"], digits=6),
                fmt_float(average["validation_support_precision_gate_selector"], digits=6),
                "--",
            ]
        )
        + r" \\",
    ]
    for material in materials:
        gate = artifact["budgets"][material]["validation_support_precision_gate_selector"]
        row = [
            material,
            fmt_float(
                artifact["budgets"][material]["density_capped_compact_induction"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
                digits=6,
            ),
            fmt_float(
                artifact["budgets"][material]["support_probe_window_selector"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
                digits=6,
            ),
            fmt_float(
                artifact["budgets"][material]["validation_support_precision_selector"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
                digits=6,
            ),
            fmt_float(
                gate["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            latex_escape(fmt_selection_counts(gate["portfolio_selected_condition_counts"])),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_support_selector_error_audit_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["support_selector_error_audit"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Support-selector error audit. This post-hoc diagnostic reads the committed high-budget support-selector artifacts and compares each selector with the best simple comparator in the same seed block. Positive regret means the selector lost density after its charged inspection or validation cost; the audit is non-deployable because completed heldout outcomes define the errors.}",
        r"\label{tab:support-selector-error-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.21\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.20\linewidth}rrrr@{}}",
        r"\toprule",
        r"Source & Best simple & Least-regret selector & Sel. LSD & Regret & Wins & Cost \\",
        r"\midrule",
    ]
    for label in artifact["artifact_order"]:
        summary = artifact["artifact_summaries"][label]
        selector = summary["best_selector_by_regret"]
        diagnostic = summary["selector_diagnostics"][selector]
        row = [
            latex_escape(summary["display_label"]),
            latex_escape(summary["best_fixed_simple_condition_label"]),
            latex_escape(diagnostic["condition_label"]),
            fmt_float(diagnostic["average_lsd"], digits=6),
            fmt_float(diagnostic["average_regret_vs_best_simple_lsd"], digits=6),
            f"{diagnostic['budgets_beating_best_simple_count']}/{diagnostic['budget_count']}",
            fmt_float(diagnostic["average_selection_cost_units"], digits=1),
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
            build_validation_coverage_prior_table(repo_root),
            build_tempered_sample_aware_table(repo_root),
            build_train_size_gated_table(repo_root),
            build_compact_train_size_gated_table(repo_root),
            build_diversity_interaction_table(repo_root),
            build_density_capped_compact_table(repo_root),
            build_support_ramped_compact_table(repo_root),
            build_late_confidence_ramped_compact_table(repo_root),
            build_density_window_compact_table(repo_root),
            build_train_support_density_selector_table(repo_root),
            build_support_probe_window_selector_table(repo_root),
            build_validation_support_precision_selector_table(repo_root),
            build_validation_support_precision_gate_table(repo_root),
            build_support_selector_transfer_table(repo_root),
            build_support_selector_error_audit_table(repo_root),
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
