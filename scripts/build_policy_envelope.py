#!/usr/bin/env python3
"""Build a post-hoc policy-envelope artifact from committed sweep results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ARTIFACT = Path("results/tiny_neural_budget_sweep_agreement_gated_f1024.json")
DEFAULT_OUTPUT_JSON = Path("results/policy_envelope_f1024.json")
DEFAULT_OUTPUT_MD = Path("results/policy_envelope_f1024.md")
ORACLE_CONDITION = "counterfactual_expansion"
SELECTION_METRIC = "accuracy_improvement_over_majority_mean"
LSD_METRIC = "signed_learning_signal_density_per_1m_event_compute_mean"

CONDITION_LABELS = {
    "agreement_gated_self_ranked_induction": "Agreement-gated",
    "counterfactual_expansion": "Oracle counterfactual",
    "mdl_rule_expansion": "MDL rule expansion",
    "raw_text": "Raw text",
    "sample_aware_self_ranked_induction": "Sample-aware self-ranked",
    "self_ranked_induction": "Self-ranked",
    "validation_ranked_induction": "Validation-ranked",
}

CONDITION_POLICY_SCOPE = {
    "agreement_gated_self_ranked_induction": {
        "uses_validation_labels": False,
        "uses_hidden_rulebook": False,
        "candidate_scope": "train-only agreement gate",
    },
    "mdl_rule_expansion": {
        "uses_validation_labels": True,
        "uses_hidden_rulebook": False,
        "candidate_scope": "charged validation-selected MDL rules",
    },
    "raw_text": {
        "uses_validation_labels": False,
        "uses_hidden_rulebook": False,
        "candidate_scope": "no generated-label transform",
    },
    "sample_aware_self_ranked_induction": {
        "uses_validation_labels": False,
        "uses_hidden_rulebook": False,
        "candidate_scope": "train-only sample-aware ranking",
    },
    "self_ranked_induction": {
        "uses_validation_labels": False,
        "uses_hidden_rulebook": False,
        "candidate_scope": "train-only ranking",
    },
    "validation_ranked_induction": {
        "uses_validation_labels": True,
        "uses_hidden_rulebook": False,
        "candidate_scope": "charged validation-ranked candidates",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def metric(value: float) -> float:
    return round(value, 6)


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def validate_source_artifact(relative_path: Path, artifact: dict[str, Any]) -> None:
    if ORACLE_CONDITION not in artifact.get("conditions", []):
        raise ValueError(f"{relative_path} must include {ORACLE_CONDITION}")
    scope = artifact.get("claim_scope", {})
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{relative_path} must not use heldout data for source selection")
    if scope.get("fresh_seed_confirmation") is not True:
        raise ValueError(f"{relative_path} must be a fresh-seed confirmation artifact")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{relative_path} must not mark the result as paper-ready")


def non_oracle_conditions(artifact: dict[str, Any]) -> list[str]:
    return [
        condition
        for condition in artifact["conditions"]
        if condition != ORACLE_CONDITION and condition in CONDITION_POLICY_SCOPE
    ]


def best_candidate_for_budget(budget: dict[str, dict[str, float]], conditions: list[str]) -> str:
    return min(
        conditions,
        key=lambda condition: (
            -budget[condition][SELECTION_METRIC],
            budget[condition]["charged_compute_units_mean"],
            condition,
        ),
    )


def build_policy_envelope(
    repo_root: Path,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    source_path = source_artifact if source_artifact.is_absolute() else repo_root / source_artifact
    source = load_json(source_path)
    relative_source = source_artifact if not source_artifact.is_absolute() else source_artifact.relative_to(repo_root)
    validate_source_artifact(relative_source, source)

    conditions = non_oracle_conditions(source)
    material_counts = [str(material) for material in source["material_counts"]]
    best_by_material: dict[str, dict[str, Any]] = {}

    for material in material_counts:
        budget = source["budgets"][material]
        best_condition = best_candidate_for_budget(budget, conditions)
        best = budget[best_condition]
        oracle = budget[ORACLE_CONDITION]
        best_by_material[material] = {
            "condition": best_condition,
            "condition_label": condition_label(best_condition),
            "signed_gain": metric(best[SELECTION_METRIC]),
            "charged_compute_units": metric(best["charged_compute_units_mean"]),
            "signed_learning_signal_density_per_1m_event_compute": metric(best[LSD_METRIC]),
            "heldout_accuracy": metric(best["heldout_accuracy_mean"]),
            "oracle_condition": ORACLE_CONDITION,
            "oracle_signed_gain": metric(oracle[SELECTION_METRIC]),
            "oracle_charged_compute_units": metric(oracle["charged_compute_units_mean"]),
            "oracle_gap_signed_gain": metric(oracle[SELECTION_METRIC] - best[SELECTION_METRIC]),
        }

    target = source["target_signed_gain"]
    first_target = next(
        (
            int(material)
            for material in material_counts
            if best_by_material[material]["signed_gain"] >= target
        ),
        None,
    )
    best_material = min(
        material_counts,
        key=lambda material: (
            -best_by_material[material]["signed_gain"],
            best_by_material[material]["charged_compute_units"],
            int(material),
        ),
    )
    best_condition_counts = Counter(row["condition"] for row in best_by_material.values())

    return {
        "title": "Learning Signal Density Post-hoc Policy Envelope",
        "source_artifacts": [str(relative_source)],
        "source_generated_at": source.get("generated_at"),
        "profile_label": source["profile_label"],
        "learner_backend": source["learner_backend"],
        "epochs": source["epochs"],
        "hidden_units": source["hidden_units"],
        "feature_dimension": source["feature_dimension"],
        "learning_rate": source["learning_rate"],
        "seeds": source["seeds"],
        "material_counts": source["material_counts"],
        "target_signed_gain": target,
        "conditions_considered": conditions,
        "condition_policy_scope": {
            condition: CONDITION_POLICY_SCOPE[condition] for condition in conditions
        },
        "oracle_condition": ORACLE_CONDITION,
        "selection_metric": SELECTION_METRIC,
        "tie_break": "lower charged_compute_units_mean, then condition name",
        "best_by_material": best_by_material,
        "best_condition_counts": dict(sorted(best_condition_counts.items())),
        "first_material_count_reaching_target": first_target,
        "best_material_count": int(best_material),
        "best_condition": best_by_material[best_material]["condition"],
        "best_signed_gain": best_by_material[best_material]["signed_gain"],
        "non_deployable_reason": (
            "The envelope selects the best non-oracle condition separately at each "
            "material budget using heldout accuracy from the completed sweep. It is "
            "a diagnostic upper bound for policy selection, not a deployable policy."
        ),
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "fresh_seed_confirmation": True,
            "post_hoc_envelope": True,
            "heldout_used_for_policy_selection": True,
            "deployable_policy": False,
            "paper_ready_claim": False,
        },
    }


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        f"# {artifact['title']}",
        "",
        f"Source artifact: `{artifact['source_artifacts'][0]}`",
        f"Source generated: `{artifact['source_generated_at']}`",
        f"Profile label: `{artifact['profile_label']}`",
        "",
        (
            "This is a post-hoc policy envelope. It uses heldout sweep results to "
            "choose the best non-oracle condition at each material budget, so it is "
            "not deployable. Its purpose is to measure the selection problem left "
            "for a future adaptive policy."
        ),
        "",
        f"Target signed gain: `{artifact['target_signed_gain']}`",
        f"First envelope budget reaching target: `{artifact['first_material_count_reaching_target']}`",
        f"Best envelope condition: `{artifact['best_condition']}` at "
        f"`{artifact['best_material_count']}` materials",
        "",
        "| Materials | Envelope condition | Signed gain | Charged compute | Oracle gain | Oracle gap |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for material in artifact["material_counts"]:
        row = artifact["best_by_material"][str(material)]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(material),
                    row["condition"],
                    fmt_float(row["signed_gain"]),
                    fmt_float(row["charged_compute_units"]),
                    fmt_float(row["oracle_signed_gain"]),
                    fmt_float(row["oracle_gap_signed_gain"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            f"- Non-deployable reason: {artifact['non_deployable_reason']}",
            "- Oracle counterfactual expansion is excluded from the envelope and reported only as headroom.",
            "- Validation-ranked and MDL rows use validation labels and charge that selection work in the source artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_policy_envelope(
    repo_root: Path,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    artifact = build_policy_envelope(repo_root, source_artifact)
    json_path = output_json if output_json.is_absolute() else repo_root / output_json
    md_path = output_md if output_md.is_absolute() else repo_root / output_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(artifact))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing result artifacts.",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT,
        help="Committed sweep JSON to derive the envelope from.",
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
    write_policy_envelope(repo_root, output_json, output_md, args.source_artifact)
    print(f"wrote {output_json.relative_to(repo_root)} and {output_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
