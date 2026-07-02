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
VALIDATION_SUPPORT_UTILITY_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_support_utility_f1024.json"
)
VALIDATION_SUPPORT_GAIN_GATE_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_validation_support_gain_gate_f1024.json"
)
SUPPORT_SELECTOR_TRANSFER_ARTIFACT = Path(
    "results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json"
)
SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT = Path("results/support_selector_error_audit_f1024.json")
SUPPORT_MECHANISM_AUDIT_ARTIFACT = Path("results/support_mechanism_audit_f1024.json")
SELECTOR_TRANSFER_ARTIFACT = Path("results/tiny_neural_budget_sweep_selector_transfer_f1024.json")
TRAIN_SIZE_GATED_ARTIFACT = Path("results/tiny_neural_budget_sweep_train_size_gated_f1024.json")
GENERATED_LABEL_AUDIT_ARTIFACT = Path("results/generated_label_audit_selector_transfer_f1024.json")
GENERATED_COVERAGE_AUDIT_ARTIFACT = Path("results/generated_coverage_audit_selector_transfer_f1024.json")
REAL_TEXT_ARTIFACTS = [
    ("SMS Spam v800", Path("results/sms_spam_real_text_selection_cost.json")),
    ("SMS Spam v200", Path("results/sms_spam_real_text_selection_cost_v200.json")),
]
SMS_BREAK_EVEN_ARTIFACT = Path("results/sms_spam_break_even_analysis.json")
TWENTY_NEWSGROUPS_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT = Path("results/twenty_newsgroups_break_even_analysis.json")
TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_retrieval_cost_audit.json"
)
TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_self_training_audit.json"
)
TWENTY_NEWSGROUPS_ACTIVE_ACQUISITION_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_active_acquisition_audit.json"
)
TWENTY_NEWSGROUPS_BUDGETED_ACQUISITION_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_budgeted_acquisition_audit.json"
)
REAL_TEXT_BREAK_EVEN_CERTIFICATE_ARTIFACT = Path("results/real_text_break_even_certificate.json")

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
    "class_balanced_sample": "Class-balanced",
    "length_curriculum_sample": "Length curriculum",
    "prototype_retrieval_sample": "Prototype retrieval",
    "validation_selector": "Validation selector",
    "validation_abstaining_proxy_selector": "Abstaining proxy selector",
    "validation_coverage_prior_selector": "Coverage-prior selector",
    "validation_coverage_proxy_selector": "Validation coverage proxy",
    "validation_linear_proxy_selector": "Linear proxy selector",
    "validation_portfolio_selector": "Validation portfolio selector",
    "validation_ranked_induction": "Validation-ranked",
    "validation_support_precision_gate_selector": "Validation support-precision gate",
    "validation_support_precision_selector": "Validation support-precision selector",
    "validation_support_utility_selector": "Validation support-utility selector",
    "validation_support_gain_gate_selector": "Validation support-gain gate",
    "label_index_balanced_sample": "Label-index balanced",
    "class_balanced_sample": "Full-scan balanced",
    "random_sample": "Random sample",
    "validation_label_index_selector": "Validation label-index selector",
    "validation_sample_selector": "Validation full-scan selector",
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


def fmt_reuse_count(value: int | None) -> str:
    if value is None:
        return "Never"
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


def validate_validation_support_utility_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("seeds") != [1601, 1607, 1609, 1613, 1619]:
        raise ValueError(f"{artifact_path} must use the preregistered utility fresh-seed block")
    if artifact.get("confirmation_of") != str(SUPPORT_MECHANISM_AUDIT_ARTIFACT):
        raise ValueError(f"{artifact_path} must identify the mechanism audit it follows")
    if artifact.get("comparison_of") != str(SUPPORT_SELECTOR_TRANSFER_ARTIFACT):
        raise ValueError(f"{artifact_path} must identify the transfer comparison artifact")
    scope = artifact.get("condition_scope", {}).get("validation_support_utility_selector", {})
    if scope.get("train_only_selection") is not False:
        raise ValueError(f"{artifact_path} must mark validation-based support selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must keep induction train-only")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("validation_used_for_transform_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation transform selection")
    if scope.get("validation_labels_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation-label use")
    if scope.get("validation_motif_distribution_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation-motif use")
    if scope.get("validation_support_utility_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the validation support-utility selector")
    if scope.get("validation_support_utility_min_score") != 0.0:
        raise ValueError(f"{artifact_path} must disclose the utility score floor")
    if scope.get("validation_support_utility_pair_coverage_weight") != 0.25:
        raise ValueError(f"{artifact_path} must disclose the pair-coverage weight")
    if scope.get("validation_support_utility_triple_l1_weight") != 0.20:
        raise ValueError(f"{artifact_path} must disclose the triple-L1 weight")
    if scope.get("validation_support_utility_compute_penalty") != 0.000001:
        raise ValueError(f"{artifact_path} must disclose the compute penalty")
    if scope.get("reuse_selected_candidate_construction") is not True:
        raise ValueError(f"{artifact_path} must disclose selected-candidate reuse")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the utility selector")


def validate_validation_support_gain_gate_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("seeds") != [1667, 1669, 1693, 1697, 1699]:
        raise ValueError(f"{artifact_path} must use the gain-gate fresh-seed block")
    if artifact.get("confirmation_of") != str(VALIDATION_SUPPORT_UTILITY_ARTIFACT):
        raise ValueError(f"{artifact_path} must identify the utility artifact it follows")
    if artifact.get("comparison_of") != str(SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT):
        raise ValueError(f"{artifact_path} must identify the support-selector audit comparison")
    scope = artifact.get("condition_scope", {}).get("validation_support_gain_gate_selector", {})
    if scope.get("train_only_selection") is not False:
        raise ValueError(f"{artifact_path} must mark validation-based support selection")
    if scope.get("train_only_induction") is not True:
        raise ValueError(f"{artifact_path} must keep induction train-only")
    if scope.get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")
    if scope.get("validation_used_for_transform_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation transform selection")
    if scope.get("validation_labels_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation-label use")
    if scope.get("validation_support_gain_gate_selector") is not True:
        raise ValueError(f"{artifact_path} must mark the validation support-gain gate")
    if scope.get("validation_support_gain_precision_prefilter") is not True:
        raise ValueError(f"{artifact_path} must disclose the precision prefilter")
    if scope.get("validation_support_gain_proxy_epochs") != 2:
        raise ValueError(f"{artifact_path} must disclose gain-gate proxy epochs")
    if scope.get("validation_support_gain_min_score") != 0.0:
        raise ValueError(f"{artifact_path} must disclose the gain-gate score floor")
    if scope.get("validation_support_gain_compute_penalty") != 0.0000005:
        raise ValueError(f"{artifact_path} must disclose the gain-gate compute penalty")
    if scope.get("validation_support_precision_threshold") != 0.825758:
        raise ValueError(f"{artifact_path} must disclose the precision prefilter threshold")
    if scope.get("reuse_selected_candidate_construction") is not True:
        raise ValueError(f"{artifact_path} must disclose selected-candidate reuse")
    if scope.get("oracle_generated_labels") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for the gain gate")


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


def validate_support_mechanism_audit_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("uses_hidden_rulebook_for_label_audit") is not True:
        raise ValueError(f"{artifact_path} must disclose hidden-rulebook label audit")
    if scope.get("uses_heldout_distribution_for_audit") is not True:
        raise ValueError(f"{artifact_path} must disclose heldout-distribution audit")
    if scope.get("hidden_rulebook_available_to_policies") is not False:
        raise ValueError(f"{artifact_path} must not expose the hidden rulebook to policies")
    if scope.get("heldout_available_to_policies") is not False:
        raise ValueError(f"{artifact_path} must not expose heldout distribution to policies")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("deployable_policy") is not False:
        raise ValueError(f"{artifact_path} must be marked non-deployable")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    if artifact.get("mechanism_summary", {}).get("promote_support_ramp_mechanism") is not False:
        raise ValueError(f"{artifact_path} must not promote the support-ramp mechanism")
    if str(SUPPORT_SELECTOR_TRANSFER_ARTIFACT) not in artifact.get("source_artifacts", []):
        raise ValueError(f"{artifact_path} must audit the support-selector transfer artifact")


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


def validate_real_text_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "UCI SMS Spam Collection":
        raise ValueError(f"{artifact_path} must identify the UCI SMS Spam Collection")
    if dataset.get("sha256") != "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3":
        raise ValueError(f"{artifact_path} must record the checked UCI SMS Spam sha256")
    if dataset.get("license") != "CC BY 4.0":
        raise ValueError(f"{artifact_path} must record the UCI license")
    scope = artifact.get("claim_scope", {})
    if scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must not be marked synthetic")
    if scope.get("real_dataset") is not True:
        raise ValueError(f"{artifact_path} must be marked as a real-dataset artifact")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    condition_scope = artifact.get("condition_scope", {})
    if condition_scope.get("label_index_balanced_sample", {}).get("selection_cost_model") != (
        "one_unit_per_train_pool_label"
    ):
        raise ValueError(f"{artifact_path} must disclose label-index selection cost")
    if condition_scope.get("validation_label_index_selector", {}).get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation use for the label-index selector")
    if condition_scope.get("validation_label_index_selector", {}).get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for the label-index selector")


def load_real_text_artifacts(repo_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    loaded = []
    for label, relative_path in REAL_TEXT_ARTIFACTS:
        artifact = load_json(repo_root / relative_path)
        validate_real_text_scope(relative_path, artifact)
        loaded.append((label, relative_path, artifact))
    return loaded


def validate_break_even_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(path) for _, path in REAL_TEXT_ARTIFACTS]:
        raise ValueError(f"{artifact_path} must analyze the committed SMS real-text artifacts")
    if artifact.get("reference_condition") != "random_sample":
        raise ValueError(f"{artifact_path} must use random sampling as the break-even reference")
    if artifact.get("quality_upper_bound") != 1.0:
        raise ValueError(f"{artifact_path} must record the spam F1 gain upper bound")
    amortization = artifact.get("amortization_model", {})
    if amortization.get("reusable_compute_keys") != [
        "selection_cost_tokens_mean",
        "validation_tuning_cost_tokens_mean",
    ]:
        raise ValueError(f"{artifact_path} must record the selector-cost amortization model")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True:
        raise ValueError(f"{artifact_path} must be marked as a real-dataset audit")
    if scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must not be marked synthetic")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use heldout data for selection")
    if scope.get("post_hoc_diagnostic") is not True:
        raise ValueError(f"{artifact_path} must be marked as a post-hoc diagnostic")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    if "G_candidate / G_reference" not in artifact.get("theorem", {}).get("general_inequality", ""):
        raise ValueError(f"{artifact_path} must state the break-even inequality")
    for by_budget in artifact.get("comparisons", {}).values():
        for by_condition in by_budget.values():
            for row in by_condition.values():
                if row.get("candidate_density_wins") is not False:
                    raise ValueError(f"{artifact_path} must preserve the current no-selector-density-win finding")
                if "max_possible_density_ratio" not in row or "perfect_quality_can_beat" not in row:
                    raise ValueError(f"{artifact_path} must record bounded-metric break-even fields")
                if "amortized_reuses_to_density_win" not in row or "fully_amortized_density_ratio" not in row:
                    raise ValueError(f"{artifact_path} must record amortized-reuse break-even fields")


def load_break_even_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / SMS_BREAK_EVEN_ARTIFACT)
    validate_break_even_scope(SMS_BREAK_EVEN_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "Twenty Newsgroups":
        raise ValueError(f"{artifact_path} must use Twenty Newsgroups")
    if dataset.get("record_count") != 1998 or dataset.get("label_count") != 20:
        raise ValueError(f"{artifact_path} must use the committed mini Twenty Newsgroups artifact")
    if dataset.get("license") != "CC BY 4.0":
        raise ValueError(f"{artifact_path} must record the UCI license")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic dataset")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must strip newsgroup metadata before training")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    conditions = artifact.get("condition_scope", {})
    for condition in (
        "random_sample",
        "class_balanced_sample",
        "length_curriculum_sample",
        "prototype_retrieval_sample",
        "validation_selector",
    ):
        if condition not in conditions:
            raise ValueError(f"{artifact_path} missing condition {condition}")
    if conditions["validation_selector"].get("validation_used_for_policy_selection") is not True:
        raise ValueError(f"{artifact_path} must disclose validation policy selection")


def load_newsgroups_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_ARTIFACT)
    validate_newsgroups_scope(TWENTY_NEWSGROUPS_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_break_even_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(TWENTY_NEWSGROUPS_ARTIFACT)]:
        raise ValueError(f"{artifact_path} must analyze the committed Twenty Newsgroups artifact")
    if artifact.get("reference_condition") != "random_sample":
        raise ValueError(f"{artifact_path} must use random sampling as the break-even reference")
    if artifact.get("quality_metric") != "accuracy_improvement_over_majority_mean":
        raise ValueError(f"{artifact_path} must audit heldout accuracy improvement")
    if artifact.get("quality_upper_bound") != 0.95:
        raise ValueError(f"{artifact_path} must record the 20-class perfect accuracy-gain upper bound")
    amortization = artifact.get("amortization_model", {})
    if amortization.get("reusable_compute_keys") != [
        "selection_cost_tokens_mean",
        "validation_tuning_cost_tokens_mean",
    ]:
        raise ValueError(f"{artifact_path} must record the selector-cost amortization model")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic audit")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must inherit the metadata stripping constraint")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    if scope.get("post_hoc_diagnostic") is not True:
        raise ValueError(f"{artifact_path} must be marked as a post-hoc diagnostic")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    if "G_candidate / G_reference" not in artifact.get("theorem", {}).get("general_inequality", ""):
        raise ValueError(f"{artifact_path} must state the break-even inequality")

    comparisons = artifact.get("comparisons", {})
    for budget in ("40", "80", "160"):
        by_condition = comparisons.get(budget, {})
        for condition in (
            "class_balanced_sample",
            "length_curriculum_sample",
            "prototype_retrieval_sample",
            "validation_selector",
        ):
            if condition not in by_condition:
                raise ValueError(f"{artifact_path} missing {condition} at budget {budget}")
            row = by_condition[condition]
            for key in (
                "quality_multiplier",
                "event_compute_multiplier",
                "density_ratio",
                "break_even_quality",
                "compute_over_break_even",
                "amortized_reuses_to_density_win",
                "fully_amortized_density_ratio",
            ):
                if key not in row:
                    raise ValueError(f"{artifact_path} missing {key} for {condition} at budget {budget}")
    if comparisons["80"]["class_balanced_sample"].get("candidate_density_wins") is not True:
        raise ValueError(f"{artifact_path} must preserve the class-balanced 80-document density win")
    if comparisons["40"]["prototype_retrieval_sample"].get("candidate_density_wins") is not False:
        raise ValueError(f"{artifact_path} must preserve the prototype retrieval density failure")


def load_newsgroups_break_even_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT)
    validate_newsgroups_break_even_scope(TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_retrieval_cost_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(TWENTY_NEWSGROUPS_ARTIFACT)]:
        raise ValueError(f"{artifact_path} must analyze the committed Twenty Newsgroups artifact")
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "Twenty Newsgroups" or dataset.get("record_count") != 1998:
        raise ValueError(f"{artifact_path} must use the committed Twenty Newsgroups mini corpus")
    if artifact.get("alphas") != [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        raise ValueError(f"{artifact_path} must record the alpha sweep grid")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic audit")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must inherit metadata stripping")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    if scope.get("post_hoc_optimization_attempt") is not True:
        raise ValueError(f"{artifact_path} must disclose the post-hoc optimization attempt")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    for budget in ("40", "80", "160"):
        budget_row = artifact.get("budgets", {}).get(budget, {})
        if "random_reference" not in budget_row or "alpha_results" not in budget_row:
            raise ValueError(f"{artifact_path} missing retrieval audit fields at budget {budget}")
        for alpha in ("0", "0.25", "0.5", "0.75", "1", "1.25"):
            row = budget_row["alpha_results"].get(alpha)
            if not row or "break_even_vs_random" not in row:
                raise ValueError(f"{artifact_path} missing alpha {alpha} break-even row at budget {budget}")
            if row["break_even_vs_random"].get("candidate_density_wins") is not False:
                raise ValueError(f"{artifact_path} must preserve the no-retrieval-density-win finding")
    if artifact["budgets"]["80"].get("best_accuracy_alpha") != "0.25":
        raise ValueError(f"{artifact_path} must preserve the 80-document length-penalty accuracy improvement")
    if artifact["budgets"]["160"].get("best_density_alpha") != "0.5":
        raise ValueError(f"{artifact_path} must preserve the 160-document retrieval-density improvement")


def load_newsgroups_retrieval_cost_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT)
    validate_newsgroups_retrieval_cost_scope(TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_self_training_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(TWENTY_NEWSGROUPS_ARTIFACT)]:
        raise ValueError(f"{artifact_path} must analyze the committed Twenty Newsgroups artifact")
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "Twenty Newsgroups" or dataset.get("record_count") != 1998:
        raise ValueError(f"{artifact_path} must use the committed Twenty Newsgroups mini corpus")
    if artifact.get("pseudo_multipliers") != [1, 2]:
        raise ValueError(f"{artifact_path} must record the pseudo-label multiplier grid")
    if artifact.get("filter_modes") != ["global_margin", "balanced_margin"]:
        raise ValueError(f"{artifact_path} must record the pseudo-label filter modes")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic audit")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must inherit metadata stripping")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    if scope.get("pseudo_labels_use_teacher_predictions") is not True:
        raise ValueError(f"{artifact_path} must use teacher pseudo-labels")
    if scope.get("oracle_train_labels_used_for_pseudo_label_selection") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for pseudo-label selection")
    if scope.get("post_hoc_optimization_attempt") is not True:
        raise ValueError(f"{artifact_path} must disclose the post-hoc optimization attempt")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    for budget in ("40", "80", "160"):
        budget_row = artifact.get("budgets", {}).get(budget, {})
        if "random_reference" not in budget_row or "class_balanced_reference" not in budget_row:
            raise ValueError(f"{artifact_path} missing references at budget {budget}")
        for condition, row in budget_row.get("condition_results", {}).items():
            if row.get("pseudo_label_agreement_mean", 1.0) > 0.25:
                raise ValueError(f"{artifact_path} must preserve the low pseudo-label agreement finding")
            if (
                row.get("signed_learning_signal_density_per_1m_event_compute_mean", 1.0)
                >= budget_row["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"]
            ):
                raise ValueError(f"{artifact_path} must preserve the no self-training density win finding")


def load_newsgroups_self_training_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT)
    validate_newsgroups_self_training_scope(TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_active_acquisition_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(TWENTY_NEWSGROUPS_ARTIFACT)]:
        raise ValueError(f"{artifact_path} must analyze the committed Twenty Newsgroups artifact")
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "Twenty Newsgroups" or dataset.get("record_count") != 1998:
        raise ValueError(f"{artifact_path} must use the committed Twenty Newsgroups mini corpus")
    if artifact.get("acquisition_modes") != [
        "margin_uncertainty",
        "balanced_margin_uncertainty",
        "short_margin_uncertainty",
        "confidence_curriculum",
    ]:
        raise ValueError(f"{artifact_path} must record the active acquisition mode grid")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic audit")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must inherit metadata stripping")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    if scope.get("validation_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for active acquisition")
    if scope.get("oracle_train_labels_used_for_acquisition") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for acquisition")
    if scope.get("true_labels_acquired_after_selection") is not True:
        raise ValueError(f"{artifact_path} must acquire true labels only after selection")
    if scope.get("post_hoc_optimization_attempt") is not True:
        raise ValueError(f"{artifact_path} must disclose the post-hoc optimization attempt")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    for budget in ("40", "80", "160"):
        budget_row = artifact.get("budgets", {}).get(budget, {})
        if "random_reference" not in budget_row or "class_balanced_reference" not in budget_row:
            raise ValueError(f"{artifact_path} missing references at budget {budget}")
        for condition, row in budget_row.get("condition_results", {}).items():
            if row.get("break_even_vs_random", {}).get("candidate_density_wins") is not False:
                raise ValueError(f"{artifact_path} must preserve the no random-density-win finding")
            if row.get("break_even_vs_class_balanced", {}).get("candidate_density_wins") is not False:
                raise ValueError(f"{artifact_path} must preserve the no class-density-win finding")
    if artifact["budgets"]["160"].get("best_density_condition") != "class_balanced_seed_active_short_margin_uncertainty":
        raise ValueError(f"{artifact_path} must preserve the best active-acquisition density row")
    best_160 = artifact["budgets"]["160"]["condition_results"]["class_balanced_seed_active_short_margin_uncertainty"]
    if best_160.get("break_even_vs_class_balanced", {}).get("amortized_reuses_to_density_win") != 4:
        raise ValueError(f"{artifact_path} must preserve the four-use active-acquisition frontier")


def load_newsgroups_active_acquisition_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_ACTIVE_ACQUISITION_AUDIT_ARTIFACT)
    validate_newsgroups_active_acquisition_scope(TWENTY_NEWSGROUPS_ACTIVE_ACQUISITION_AUDIT_ARTIFACT, artifact)
    return artifact


def validate_newsgroups_budgeted_acquisition_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [str(TWENTY_NEWSGROUPS_ARTIFACT)]:
        raise ValueError(f"{artifact_path} must analyze the committed Twenty Newsgroups artifact")
    dataset = artifact.get("dataset", {})
    if dataset.get("name") != "Twenty Newsgroups" or dataset.get("record_count") != 1998:
        raise ValueError(f"{artifact_path} must use the committed Twenty Newsgroups mini corpus")
    if artifact.get("acquisition_modes") != [
        "margin_uncertainty",
        "balanced_margin_uncertainty",
        "short_margin_uncertainty",
    ]:
        raise ValueError(f"{artifact_path} must record the budgeted acquisition mode grid")
    if artifact.get("scan_multipliers") != [1, 2, 4]:
        raise ValueError(f"{artifact_path} must record the scan-window multiplier grid")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic audit")
    if scope.get("metadata_stripped") is not True:
        raise ValueError(f"{artifact_path} must inherit metadata stripping")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must keep heldout closed for selection")
    if scope.get("validation_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must not use validation for budgeted acquisition")
    if scope.get("oracle_train_labels_used_for_acquisition") is not False:
        raise ValueError(f"{artifact_path} must not use oracle labels for acquisition")
    if scope.get("true_labels_acquired_after_selection") is not True:
        raise ValueError(f"{artifact_path} must acquire true labels only after selection")
    if scope.get("scan_window_sampled_without_text_scoring") is not True:
        raise ValueError(f"{artifact_path} must sample scan windows before teacher scoring")
    if scope.get("post_hoc_optimization_attempt") is not True:
        raise ValueError(f"{artifact_path} must disclose the post-hoc optimization attempt")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{artifact_path} must not mark the current result as paper-ready")
    for budget in ("40", "80", "160"):
        budget_row = artifact.get("budgets", {}).get(budget, {})
        if "random_reference" not in budget_row or "class_balanced_reference" not in budget_row:
            raise ValueError(f"{artifact_path} missing references at budget {budget}")
        for condition, row in budget_row.get("condition_results", {}).items():
            if row.get("break_even_vs_random", {}).get("candidate_density_wins") is not False:
                raise ValueError(f"{artifact_path} must preserve the no random-density-win finding")
            if row.get("external_events_mean") != float(budget):
                raise ValueError(f"{artifact_path} must preserve full label-budget use for {condition}")
    if artifact["budgets"]["160"].get("best_density_condition") != "budgeted_active_margin_uncertainty_2x":
        raise ValueError(f"{artifact_path} must preserve the budgeted-window density frontier")
    best_160 = artifact["budgets"]["160"]["condition_results"]["budgeted_active_margin_uncertainty_2x"]
    if best_160.get("break_even_vs_class_balanced", {}).get("candidate_density_wins") is not True:
        raise ValueError(f"{artifact_path} must preserve the class-balanced density win")
    if best_160.get("break_even_vs_random", {}).get("candidate_density_wins") is not False:
        raise ValueError(f"{artifact_path} must preserve the no-random-density-win finding")


def load_newsgroups_budgeted_acquisition_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / TWENTY_NEWSGROUPS_BUDGETED_ACQUISITION_AUDIT_ARTIFACT)
    validate_newsgroups_budgeted_acquisition_scope(
        TWENTY_NEWSGROUPS_BUDGETED_ACQUISITION_AUDIT_ARTIFACT,
        artifact,
    )
    return artifact


def validate_real_text_break_even_certificate_scope(artifact_path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("source_artifacts") != [
        str(TWENTY_NEWSGROUPS_ARTIFACT),
        str(TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT),
        str(TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT),
        str(TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT),
        str(TWENTY_NEWSGROUPS_ACTIVE_ACQUISITION_AUDIT_ARTIFACT),
        str(TWENTY_NEWSGROUPS_BUDGETED_ACQUISITION_AUDIT_ARTIFACT),
        str(SMS_BREAK_EVEN_ARTIFACT),
    ]:
        raise ValueError(f"{artifact_path} must certify the committed real-text break-even artifacts")
    scope = artifact.get("claim_scope", {})
    if scope.get("real_dataset") is not True or scope.get("synthetic_domain") is not False:
        raise ValueError(f"{artifact_path} must be marked as a real non-synthetic certificate")
    if scope.get("mathematical_certificate") is not True:
        raise ValueError(f"{artifact_path} must identify itself as a mathematical certificate")
    if scope.get("introduces_new_policy") is not False:
        raise ValueError(f"{artifact_path} must not introduce a new policy")
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{artifact_path} must preserve heldout isolation")
    if "G_candidate / G_reference" not in artifact.get("theorem", {}).get("density_condition", ""):
        raise ValueError(f"{artifact_path} must state the density break-even inequality")
    summary = artifact.get("summary", {})
    if summary.get("rows") != 172:
        raise ValueError(f"{artifact_path} must certify the full 172-row real-text comparison set")
    if summary.get("observed_quality_wins") != 38 or summary.get("density_wins") != 3:
        raise ValueError(f"{artifact_path} must preserve the quality-win versus density-win gap")
    if summary.get("finite_reuse_needed") != 13:
        raise ValueError(f"{artifact_path} must preserve the finite-reuse frontier count")
    families = summary.get("families", {})
    if families.get("twenty_newsgroups_self_training", {}).get("density_wins") != 0:
        raise ValueError(f"{artifact_path} must preserve the self-training no-density-win result")
    if families.get("twenty_newsgroups_active_acquisition", {}).get("density_wins") != 0:
        raise ValueError(f"{artifact_path} must preserve the active-acquisition no-density-win result")
    if families.get("twenty_newsgroups_budgeted_active_acquisition", {}).get("density_wins") != 2:
        raise ValueError(f"{artifact_path} must preserve the budgeted active-acquisition density win")
    strongest = summary.get("strongest_observed_density_win", {})
    if strongest.get("candidate_condition") != "class_balanced_sample" or strongest.get("budget") != "80":
        raise ValueError(f"{artifact_path} must preserve the 80-document class-balanced density frontier")
    cheapest_reuse = summary.get("cheapest_finite_reuse_frontier", {})
    if cheapest_reuse.get("candidate_condition") != "budgeted_active_balanced_margin_uncertainty_1x":
        raise ValueError(f"{artifact_path} must preserve the budgeted finite-reuse frontier")
    if cheapest_reuse.get("amortized_reuses_to_density_win") != 2:
        raise ValueError(f"{artifact_path} must preserve the two-use amortization threshold")


def load_real_text_break_even_certificate_artifact(repo_root: Path) -> dict[str, Any]:
    artifact = load_json(repo_root / REAL_TEXT_BREAK_EVEN_CERTIFICATE_ARTIFACT)
    validate_real_text_break_even_certificate_scope(REAL_TEXT_BREAK_EVEN_CERTIFICATE_ARTIFACT, artifact)
    return artifact


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
        "validation_support_utility": load_json(repo_root / VALIDATION_SUPPORT_UTILITY_ARTIFACT),
        "validation_support_gain_gate": load_json(repo_root / VALIDATION_SUPPORT_GAIN_GATE_ARTIFACT),
        "support_selector_transfer": load_json(repo_root / SUPPORT_SELECTOR_TRANSFER_ARTIFACT),
        "support_selector_error_audit": load_json(repo_root / SUPPORT_SELECTOR_ERROR_AUDIT_ARTIFACT),
        "support_mechanism_audit": load_json(repo_root / SUPPORT_MECHANISM_AUDIT_ARTIFACT),
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
    validate_claim_scope(
        VALIDATION_SUPPORT_UTILITY_ARTIFACT,
        artifacts["validation_support_utility"],
    )
    validate_validation_support_utility_scope(
        VALIDATION_SUPPORT_UTILITY_ARTIFACT,
        artifacts["validation_support_utility"],
    )
    validate_claim_scope(
        VALIDATION_SUPPORT_GAIN_GATE_ARTIFACT,
        artifacts["validation_support_gain_gate"],
    )
    validate_validation_support_gain_gate_scope(
        VALIDATION_SUPPORT_GAIN_GATE_ARTIFACT,
        artifacts["validation_support_gain_gate"],
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
    validate_support_mechanism_audit_scope(
        SUPPORT_MECHANISM_AUDIT_ARTIFACT,
        artifacts["support_mechanism_audit"],
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
    real_text_artifacts = load_real_text_artifacts(repo_root)
    load_newsgroups_artifact(repo_root)
    load_newsgroups_break_even_artifact(repo_root)
    load_newsgroups_retrieval_cost_artifact(repo_root)
    load_newsgroups_self_training_artifact(repo_root)
    load_newsgroups_active_acquisition_artifact(repo_root)
    load_newsgroups_budgeted_acquisition_artifact(repo_root)
    load_break_even_artifact(repo_root)
    load_real_text_break_even_certificate_artifact(repo_root)
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
        rf"\newcommand{{\LsdPaperArtifactCount}}{{{len(BUDGET_ARTIFACTS) + len(support) + len(real_text_artifacts) + 8}}}",
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


def build_validation_support_utility_selector_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_support_utility"]
    simple_conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128")
    all_materials = [str(material) for material in artifact["material_counts"]]

    def lsd(material: str, condition: str) -> float:
        return artifact["budgets"][material][condition][
            "signed_learning_signal_density_per_1m_event_compute_mean"
        ]

    utility_average = sum(lsd(material, "validation_support_utility_selector") for material in all_materials) / len(
        all_materials
    )
    gate_average = sum(lsd(material, "validation_support_precision_gate_selector") for material in all_materials) / len(
        all_materials
    )
    density_average = sum(lsd(material, "density_capped_compact_induction") for material in all_materials) / len(
        all_materials
    )
    utility_regrets = [
        max(lsd(material, condition) for condition in simple_conditions)
        - lsd(material, "validation_support_utility_selector")
        for material in all_materials
    ]
    utility_average_regret = sum(utility_regrets) / len(utility_regrets)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation support-utility selector on fresh seeds 1601, 1607, 1609, 1613, and 1619. This follow-up adds a cheap validation-precision prefilter before a validation precision, motif coverage, and compute-cost utility score for support-ramped compact induction. The prefilter lowers raw-abstention cost, but the utility selector still loses average signed density to the no-window precision gate and the density-capped simple fallback.}",
        r"\label{tab:validation-support-utility-selector}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.22\linewidth}rrrr@{}}",
        r"\toprule",
        r"Budget & Utility choices & Utility LSD & Gate LSD & Density cap & Utility regret \\",
        r"\midrule",
        " & ".join(
            [
                "Avg.",
                "--",
                fmt_float(utility_average, digits=6),
                fmt_float(gate_average, digits=6),
                fmt_float(density_average, digits=6),
                fmt_float(utility_average_regret, digits=6),
            ]
        )
        + r" \\",
    ]
    for material in materials:
        utility = artifact["budgets"][material]["validation_support_utility_selector"]
        utility_regret = (
            max(lsd(material, condition) for condition in simple_conditions)
            - lsd(material, "validation_support_utility_selector")
        )
        row = [
            material,
            latex_escape(fmt_selection_counts(utility["portfolio_selected_condition_counts"])),
            fmt_float(
                utility["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(
                lsd(material, "validation_support_precision_gate_selector"),
                digits=6,
            ),
            fmt_float(
                lsd(material, "density_capped_compact_induction"),
                digits=6,
            ),
            fmt_float(utility_regret, digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_validation_support_gain_gate_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["validation_support_gain_gate"]
    simple_conditions = (
        "raw_text",
        "compact_train_size_gated_induction",
        "support_ramped_compact_induction",
        "density_window_compact_induction",
        "density_capped_compact_induction",
    )
    materials = ("96", "104", "112", "120", "128")
    all_materials = [str(material) for material in artifact["material_counts"]]

    def lsd(material: str, condition: str) -> float:
        return artifact["budgets"][material][condition][
            "signed_learning_signal_density_per_1m_event_compute_mean"
        ]

    def avg(condition: str) -> float:
        return sum(lsd(material, condition) for material in all_materials) / len(all_materials)

    gain_gate_average = avg("validation_support_gain_gate_selector")
    utility_average = avg("validation_support_utility_selector")
    precision_gate_average = avg("validation_support_precision_gate_selector")
    best_simple_by_average = max(simple_conditions, key=avg)
    best_simple_average = avg(best_simple_by_average)
    gain_gate_regrets = [
        max(lsd(material, condition) for condition in simple_conditions)
        - lsd(material, "validation_support_gain_gate_selector")
        for material in all_materials
    ]
    gain_gate_average_regret = sum(gain_gate_regrets) / len(gain_gate_regrets)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation support-gain gate on fresh seeds 1667, 1669, 1693, 1697, and 1699. This expected-gain follow-up first applies the cheap validation support-precision prefilter, then trains two-epoch linear proxies for raw text and support-ramped compact only when support is eligible. The prefilter reduces wasted proxy work, but the direct validation-gain estimate still loses signed density to simpler support policies after charged selection cost.}",
        r"\label{tab:validation-support-gain-gate}",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.20\linewidth}rrrrr@{}}",
        r"\toprule",
        r"Budget & Gain-gate choices & Gain gate & Utility & Precision gate & Best simple & Regret \\",
        r"\midrule",
        " & ".join(
            [
                "Avg.",
                "--",
                fmt_float(gain_gate_average, digits=6),
                fmt_float(utility_average, digits=6),
                fmt_float(precision_gate_average, digits=6),
                fmt_float(best_simple_average, digits=6),
                fmt_float(gain_gate_average_regret, digits=6),
            ]
        )
        + r" \\",
    ]
    for material in materials:
        gain_gate = artifact["budgets"][material]["validation_support_gain_gate_selector"]
        best_simple_lsd = max(lsd(material, condition) for condition in simple_conditions)
        gain_gate_regret = best_simple_lsd - lsd(material, "validation_support_gain_gate_selector")
        row = [
            material,
            latex_escape(fmt_selection_counts(gain_gate["portfolio_selected_condition_counts"])),
            fmt_float(
                gain_gate["signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(
                lsd(material, "validation_support_utility_selector"),
                digits=6,
            ),
            fmt_float(
                lsd(material, "validation_support_precision_gate_selector"),
                digits=6,
            ),
            fmt_float(best_simple_lsd, digits=6),
            fmt_float(gain_gate_regret, digits=6),
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


def build_support_mechanism_audit_table(repo_root: Path) -> str:
    artifact = load_supporting_artifacts(repo_root)["support_mechanism_audit"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Support-ramp mechanism audit on the support-selector transfer seeds. The audit reconstructs candidate pipelines after the neural sweep and compares generated labels with the hidden rulebook and heldout motif distribution. The support ramp reduces generated volume and cost, but in the transition region it does not improve label precision or motif coverage versus compact induction, and it beats the density-capped raw fallback on signed LSD at only one of four transition budgets.}",
        r"\label{tab:support-mechanism-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"Budget & Supp. prec. & Compact prec. & Supp. cov. & Compact cov. & Supp. LSD & Cap LSD & $\Delta$LSD \\",
        r"\midrule",
    ]
    for material in artifact["mechanism_summary"]["transition_material_counts"]:
        material_key = str(material)
        support = artifact["audits"][material_key]["support_ramped_compact_induction"]
        compact = artifact["audits"][material_key]["compact_train_size_gated_induction"]
        density = artifact["audits"][material_key]["density_capped_compact_induction"]
        diagnostic = artifact["transition_diagnostics"][material_key]
        row = [
            str(material),
            fmt_float(support["label_precision"], digits=6),
            fmt_float(compact["label_precision"], digits=6),
            fmt_float(support["heldout_triple_coverage"], digits=6),
            fmt_float(compact["heldout_triple_coverage"], digits=6),
            fmt_float(
                support["linked_signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(
                density["linked_signed_learning_signal_density_per_1m_event_compute_mean"],
                digits=6,
            ),
            fmt_float(diagnostic["support_minus_density_capped_lsd"], digits=6),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_real_text_selection_cost_table(repo_root: Path) -> str:
    artifact = load_real_text_artifacts(repo_root)[0][2]
    conditions = (
        "random_sample",
        "label_index_balanced_sample",
        "class_balanced_sample",
        "validation_label_index_selector",
        "validation_sample_selector",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Real-text SMS Spam selection-cost pilot. The dataset is UCI SMS Spam Collection (5,574 labeled SMS messages, CC BY 4.0). Entries are spam-class F1 and signed learning-signal density per one million event-compute units; validation selectors use 800 validation examples and never inspect heldout examples.}",
        r"\label{tab:sms-spam-real-text-selection-cost}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rlllll@{}}",
        r"\toprule",
        r"Budget & Random & Label-index bal. & Full-scan bal. & Val. label-index & Val. full-scan \\",
        r" & F1/LSD & F1/LSD & F1/LSD & F1/LSD & F1/LSD \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_key = str(budget)
        cells = [str(budget)]
        for condition in conditions:
            row = artifact["budgets"][budget_key]["conditions"][condition]
            cells.append(
                latex_escape(
                    f"{fmt_float(row['heldout_spam_f1_mean'])}/"
                    f"{fmt_float(row['signed_learning_signal_density_per_1m_event_compute_mean'], digits=6)}"
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_selection_table(repo_root: Path) -> str:
    artifact = load_newsgroups_artifact(repo_root)
    conditions = (
        "random_sample",
        "class_balanced_sample",
        "length_curriculum_sample",
        "prototype_retrieval_sample",
        "validation_selector",
    )
    labels = {
        "random_sample": "Random",
        "class_balanced_sample": "Class-balanced",
        "length_curriculum_sample": "Length curriculum",
        "prototype_retrieval_sample": "Prototype retrieval",
        "validation_selector": "Validation selector",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups active selection-cost pilot. The dataset is UCI Twenty Newsgroups mini (1,998 cleaned documents, 20 classes, CC BY 4.0). Entries are heldout accuracy and signed learning-signal density per one million event-compute units; headers, quote lines, and reply boilerplate are stripped before splitting.}",
        r"\label{tab:twenty-newsgroups-active-selection}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rlllll@{}}",
        r"\toprule",
        "Budget & " + " & ".join(latex_escape(labels[condition]) for condition in conditions) + r" \\",
        r" & Acc/LSD & Acc/LSD & Acc/LSD & Acc/LSD & Acc/LSD \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_key = str(budget)
        cells = [budget_key]
        for condition in conditions:
            row = artifact["budgets"][budget_key]["conditions"][condition]
            cells.append(
                latex_escape(
                    f"{fmt_float(row['heldout_accuracy_mean'])}/"
                    f"{fmt_float(row['signed_learning_signal_density_per_1m_event_compute_mean'], digits=6)}"
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_break_even_table(repo_root: Path) -> str:
    artifact = load_newsgroups_break_even_artifact(repo_root)
    labels = {
        "class_balanced_sample": "Class-balanced",
        "length_curriculum_sample": "Length curriculum",
        "prototype_retrieval_sample": "Prototype retrieval",
        "validation_selector": "Validation selector",
    }
    rows_to_show = (
        ("40", "prototype_retrieval_sample"),
        ("40", "validation_selector"),
        ("80", "class_balanced_sample"),
        ("80", "prototype_retrieval_sample"),
        ("160", "prototype_retrieval_sample"),
        ("160", "length_curriculum_sample"),
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups break-even audit. A candidate beats random sampling on density only when its heldout-accuracy gain multiplier exceeds its event-compute multiplier. The table shows that prototype retrieval can improve quality while still missing the density break-even condition.}",
        r"\label{tab:twenty-newsgroups-break-even}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rlrrrrrr@{}}",
        r"\toprule",
        r"Budget & Candidate & Q mult. & EC mult. & Density ratio & BE gain & Cost/BE & Reuses \\",
        r"\midrule",
    ]
    for budget, condition in rows_to_show:
        row = artifact["comparisons"][budget][condition]
        lines.append(
            " & ".join(
                [
                    budget,
                    latex_escape(labels[condition]),
                    fmt_float(row["quality_multiplier"], digits=3),
                    fmt_float(row["event_compute_multiplier"], digits=3),
                    fmt_float(row["density_ratio"], digits=3),
                    fmt_float(row["break_even_quality"], digits=3),
                    fmt_float(row["compute_over_break_even"], digits=3),
                    fmt_reuse_count(row["amortized_reuses_to_density_win"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_retrieval_cost_table(repo_root: Path) -> str:
    artifact = load_newsgroups_retrieval_cost_artifact(repo_root)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups length-penalized prototype-retrieval audit. Alpha penalizes selected-document length in the prototype score. Length penalties improve some retrieval rows, but no tested alpha beats random sampling on learning-signal density.}",
        r"\label{tab:twenty-newsgroups-retrieval-cost-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"Budget & Best acc $\alpha$ & Acc & LSD & Best LSD $\alpha$ & Acc & LSD vs random \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_acc = budget_row["alpha_results"][budget_row["best_accuracy_alpha"]]
        best_density = budget_row["alpha_results"][budget_row["best_density_alpha"]]
        random_ref = budget_row["random_reference"]
        density_cell = (
            f"{fmt_float(best_density['signed_learning_signal_density_per_1m_event_compute_mean'], digits=6)}/"
            f"{fmt_float(random_ref['signed_learning_signal_density_per_1m_event_compute_mean'], digits=6)}"
        )
        lines.append(
            " & ".join(
                [
                    str(budget),
                    latex_escape(budget_row["best_accuracy_alpha"]),
                    fmt_float(best_acc["heldout_accuracy_mean"]),
                    fmt_float(best_acc["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    latex_escape(budget_row["best_density_alpha"]),
                    fmt_float(best_density["heldout_accuracy_mean"]),
                    latex_escape(density_cell),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_self_training_table(repo_root: Path) -> str:
    artifact = load_newsgroups_self_training_artifact(repo_root)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups self-training pseudo-label audit. A class-balanced teacher pseudo-labels the train-only pool, then a student trains on labeled plus pseudo-labeled examples. Pseudo-label agreement is diagnostic only and is not used for filtering or student labels.}",
        r"\label{tab:twenty-newsgroups-self-training-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rlrrrrr@{}}",
        r"\toprule",
        r"Budget & Best pseudo filter & Acc & Pseudo agree & LSD & Class LSD & Random LSD \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_condition = budget_row["best_self_training_condition"]
        best = budget_row["condition_results"][best_condition]
        class_ref = budget_row["class_balanced_reference"]
        random_ref = budget_row["random_reference"]
        label = best_condition.replace("class_balanced_self_training_", "").replace("_", " ")
        lines.append(
            " & ".join(
                [
                    str(budget),
                    latex_escape(label),
                    fmt_float(best["heldout_accuracy_mean"]),
                    fmt_float(best["pseudo_label_agreement_mean"], digits=3),
                    fmt_float(best["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(class_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(random_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_active_acquisition_table(repo_root: Path) -> str:
    artifact = load_newsgroups_active_acquisition_artifact(repo_root)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups active label-acquisition audit. A class-balanced seed trains a teacher, the teacher selects train-pool records by margin signals, and true labels are acquired only after selection. No tested acquisition mode beats random or class-balanced density without explicit reuse.}",
        r"\label{tab:twenty-newsgroups-active-acquisition-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rlrrrrrr@{}}",
        r"\toprule",
        r"Budget & Best density mode & Acc & Teacher agree & LSD & Random LSD & Class LSD & Reuses vs class \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_condition = budget_row["best_density_condition"]
        best = budget_row["condition_results"][best_condition]
        random_ref = budget_row["random_reference"]
        class_ref = budget_row["class_balanced_reference"]
        label = best_condition.replace("class_balanced_seed_active_", "").replace("_", " ")
        lines.append(
            " & ".join(
                [
                    str(budget),
                    latex_escape(label),
                    fmt_float(best["heldout_accuracy_mean"]),
                    fmt_float(best["acquired_teacher_agreement_mean"], digits=3),
                    fmt_float(best["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(random_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(class_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_reuse_count(best["break_even_vs_class_balanced"]["amortized_reuses_to_density_win"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_twenty_newsgroups_budgeted_acquisition_table(repo_root: Path) -> str:
    artifact = load_newsgroups_budgeted_acquisition_artifact(repo_root)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Twenty Newsgroups budgeted-window active acquisition audit. The teacher scores only a sampled train-pool window before true labels are acquired. Reducing scan cost produces density wins against class-balanced sampling at 160 labels, but no tested row beats random sampling.}",
        r"\label{tab:twenty-newsgroups-budgeted-acquisition-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rlrrrrrr@{}}",
        r"\toprule",
        r"Budget & Best density mode & Acc & Scan & LSD & Random LSD & Class LSD & Class win \\",
        r"\midrule",
    ]
    for budget in artifact["train_budgets"]:
        budget_row = artifact["budgets"][str(budget)]
        best_condition = budget_row["best_density_condition"]
        best = budget_row["condition_results"][best_condition]
        random_ref = budget_row["random_reference"]
        class_ref = budget_row["class_balanced_reference"]
        label = best_condition.replace("budgeted_active_", "").replace("_", " ")
        lines.append(
            " & ".join(
                [
                    str(budget),
                    latex_escape(label),
                    fmt_float(best["heldout_accuracy_mean"]),
                    fmt_float(best["scan_window_size_mean"], digits=1),
                    fmt_float(best["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(random_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    fmt_float(class_ref["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
                    latex_escape(str(best["break_even_vs_class_balanced"]["candidate_density_wins"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_real_text_break_even_certificate_table(repo_root: Path) -> str:
    artifact = load_real_text_break_even_certificate_artifact(repo_root)
    summary = artifact["summary"]
    family_labels = {
        "sms_spam_selection": "SMS Spam selectors",
        "twenty_newsgroups_active_acquisition": "Newsgroups active acquisition",
        "twenty_newsgroups_active_selection": "Newsgroups active",
        "twenty_newsgroups_budgeted_active_acquisition": "Newsgroups budgeted acquisition",
        "twenty_newsgroups_retrieval_alpha": "Newsgroups retrieval alpha",
        "twenty_newsgroups_self_training": "Newsgroups self-training",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Real-text break-even frontier certificate. The certificate applies the same density break-even inequality across committed SMS Spam and Twenty Newsgroups audits. Quality wins count rows with higher heldout gain than the reference; density wins count rows that also clear charged event-compute.}",
        r"\label{tab:real-text-break-even-certificate}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Family & Rows & Q wins & Density wins & Q-win losses & Finite reuse & K=1 impossible & Mean ratio \\",
        r"\midrule",
    ]
    for family, family_summary in summary["families"].items():
        lines.append(
            " & ".join(
                [
                    latex_escape(family_labels[family]),
                    str(family_summary["rows"]),
                    str(family_summary["observed_quality_wins"]),
                    str(family_summary["density_wins"]),
                    str(family_summary["quality_win_density_losses"]),
                    str(family_summary["finite_reuse_needed"]),
                    str(family_summary["bounded_quality_impossible_at_k1"]),
                    fmt_float(family_summary["mean_density_ratio"], digits=3),
                ]
            )
            + r" \\"
        )
    lines.append(
        " & ".join(
            [
                "Total",
                str(summary["rows"]),
                str(summary["observed_quality_wins"]),
                str(summary["density_wins"]),
                str(summary["quality_win_density_losses"]),
                str(summary["finite_reuse_needed"]),
                str(summary["bounded_quality_impossible_at_k1"]),
                fmt_float(summary["mean_density_ratio"], digits=3),
            ]
        )
        + r" \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_real_text_validation_size_table(repo_root: Path) -> str:
    artifacts = {label: artifact for label, _, artifact in load_real_text_artifacts(repo_root)}
    default_artifact = artifacts["SMS Spam v800"]
    small_artifact = artifacts["SMS Spam v200"]
    condition = "validation_label_index_selector"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation-set size ablation for the real-text label-index selector. Both rows compare the same random and label-index balanced candidates; the selector cost includes proxy training, validation scoring, and unreused candidate construction.}",
        r"\label{tab:sms-spam-validation-size}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"Budget & F1 v800 & LSD v800 & Cost v800 & F1 v200 & LSD v200 & Cost v200 \\",
        r"\midrule",
    ]
    for budget in default_artifact["train_budgets"]:
        budget_key = str(budget)
        default_row = default_artifact["budgets"][budget_key]["conditions"][condition]
        small_row = small_artifact["budgets"][budget_key]["conditions"][condition]
        row = [
            budget_key,
            fmt_float(default_row["heldout_spam_f1_mean"]),
            fmt_float(default_row["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
            latex_escape(fmt_ops(default_row["charged_compute_units_mean"])),
            fmt_float(small_row["heldout_spam_f1_mean"]),
            fmt_float(small_row["signed_learning_signal_density_per_1m_event_compute_mean"], digits=6),
            latex_escape(fmt_ops(small_row["charged_compute_units_mean"])),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_sms_break_even_table(repo_root: Path) -> str:
    artifact = load_break_even_artifact(repo_root)
    rows_to_show = (
        ("SMS Spam v800", "32", "label_index_balanced_sample"),
        ("SMS Spam v800", "32", "validation_label_index_selector"),
        ("SMS Spam v800", "64", "class_balanced_sample"),
        ("SMS Spam v800", "128", "validation_label_index_selector"),
        ("SMS Spam v200", "32", "validation_label_index_selector"),
        ("SMS Spam v200", "256", "label_index_balanced_sample"),
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Break-even audit for SMS Spam selection policies. For positive random-sampling gain, a candidate beats random density only when its quality multiplier exceeds its event-compute multiplier. Max ratio is the largest density ratio reachable at perfect spam F1 gain.}",
        r"\label{tab:sms-spam-break-even}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rlrrrrrrr@{}}",
        r"\toprule",
        r"Artifact & Candidate & Budget & Q mult. & EC mult. & Density ratio & Max ratio & BE quality & Cost/BE \\",
        r"\midrule",
    ]
    for artifact_label, budget, condition in rows_to_show:
        row = artifact["comparisons"][artifact_label][budget][condition]
        lines.append(
            " & ".join(
                [
                    latex_escape(artifact_label),
                    latex_escape(condition_label(condition)),
                    budget,
                    fmt_float(row["quality_multiplier"], digits=3),
                    fmt_float(row["event_compute_multiplier"], digits=3),
                    fmt_float(row["density_ratio"], digits=3),
                    fmt_float(row["max_possible_density_ratio"], digits=3),
                    fmt_float(row["break_even_quality"], digits=3),
                    fmt_float(row["compute_over_break_even"], digits=3),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_sms_amortized_reuse_table(repo_root: Path) -> str:
    artifact = load_break_even_artifact(repo_root)
    rows_to_show = (
        ("SMS Spam v800", "32", "label_index_balanced_sample"),
        ("SMS Spam v800", "32", "validation_label_index_selector"),
        ("SMS Spam v800", "64", "label_index_balanced_sample"),
        ("SMS Spam v800", "512", "validation_label_index_selector"),
        ("SMS Spam v200", "32", "validation_label_index_selector"),
        ("SMS Spam v200", "256", "validation_label_index_selector"),
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Amortized selector-cost audit for SMS Spam. Reuses is the minimum number of independent downstream uses needed to beat random density if selector construction and validation-tuning costs are reusable; Never means the nonreusable compute already exceeds the observed-gain affordability bound. Limit ratio is the density ratio after spreading reusable cost over infinitely many uses.}",
        r"\label{tab:sms-spam-amortized-reuse}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}rlrrrrrr@{}}",
        r"\toprule",
        r"Artifact & Candidate & Budget & Reuse frac. & Fixed cost & Affordable & Reuses & Limit ratio \\",
        r"\midrule",
    ]
    for artifact_label, budget, condition in rows_to_show:
        row = artifact["comparisons"][artifact_label][budget][condition]
        lines.append(
            " & ".join(
                [
                    latex_escape(artifact_label),
                    latex_escape(condition_label(condition)),
                    budget,
                    fmt_float(row["reusable_compute_fraction"], digits=3),
                    latex_escape(fmt_ops(row["candidate_nonreusable_compute_units"])),
                    latex_escape(fmt_ops(row["max_affordable_compute_units"])),
                    fmt_reuse_count(row["amortized_reuses_to_density_win"]),
                    fmt_float(row["fully_amortized_density_ratio"], digits=3),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_tables(repo_root: Path) -> str:
    return "\n".join(
        [
            build_macros(repo_root),
            build_twenty_newsgroups_selection_table(repo_root),
            build_twenty_newsgroups_break_even_table(repo_root),
            build_twenty_newsgroups_retrieval_cost_table(repo_root),
            build_twenty_newsgroups_self_training_table(repo_root),
            build_twenty_newsgroups_active_acquisition_table(repo_root),
            build_twenty_newsgroups_budgeted_acquisition_table(repo_root),
            build_real_text_break_even_certificate_table(repo_root),
            build_real_text_selection_cost_table(repo_root),
            build_real_text_validation_size_table(repo_root),
            build_sms_break_even_table(repo_root),
            build_sms_amortized_reuse_table(repo_root),
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
            build_validation_support_utility_selector_table(repo_root),
            build_validation_support_gain_gate_table(repo_root),
            build_support_selector_transfer_table(repo_root),
            build_support_selector_error_audit_table(repo_root),
            build_support_mechanism_audit_table(repo_root),
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
