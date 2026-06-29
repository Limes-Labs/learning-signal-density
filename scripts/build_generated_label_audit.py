#!/usr/bin/env python3
"""Audit generated-label correctness against the synthetic hidden rulebook."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import mean
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning_signal_density.domain import build_world, split_observations
from learning_signal_density.experiment import _round
from learning_signal_density.neural_experiment import VALIDATION_PORTFOLIO_CANDIDATES
from learning_signal_density.pipelines import PipelineExamples, TrainingExample, build_pipeline_examples


DEFAULT_SOURCE_ARTIFACT = Path("results/tiny_neural_budget_sweep_selector_transfer_f1024.json")
DEFAULT_OUTPUT_JSON = Path("results/generated_label_audit_selector_transfer_f1024.json")
DEFAULT_OUTPUT_MD = Path("results/generated_label_audit_selector_transfer_f1024.md")

AUDIT_CONDITIONS = (
    "raw_text",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "agreement_gated_self_ranked_induction",
    "validation_ranked_induction",
    "mdl_rule_expansion",
    "counterfactual_expansion",
)
GENERATED_SOURCE_KINDS = frozenset({
    "agreement_gated_self_ranked_induced",
    "counterfactual",
    "induced_counterfactual",
    "mdl_counterfactual",
    "sample_aware_self_ranked_induced",
    "self_ranked_induced",
    "validation_ranked_induced",
})
FIELD_PATTERN = re.compile(r"\b(family|modifier|stimulus)=([A-Za-z0-9_]+)")


CONDITION_LABELS = {
    "agreement_gated_self_ranked_induction": "Agreement-gated",
    "counterfactual_expansion": "Oracle counterfactual",
    "mdl_rule_expansion": "MDL rule expansion",
    "raw_text": "Raw text",
    "sample_aware_self_ranked_induction": "Sample-aware self-ranked",
    "self_ranked_induction": "Self-ranked",
    "validation_ranked_induction": "Validation-ranked",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def metric(value: float) -> float:
    return round(value, 6)


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def validate_source_artifact(relative_path: Path, artifact: dict[str, Any]) -> None:
    scope = artifact.get("claim_scope", {})
    if scope.get("heldout_used_for_selection") is not False:
        raise ValueError(f"{relative_path} must not use heldout data for source selection")
    if scope.get("fresh_seed_confirmation") is not True:
        raise ValueError(f"{relative_path} must be a fresh-seed confirmation artifact")
    if scope.get("paper_ready_claim") is not False:
        raise ValueError(f"{relative_path} must not mark the result as paper-ready")

    missing = sorted(set(AUDIT_CONDITIONS) - set(artifact.get("conditions", [])))
    if missing:
        raise ValueError(f"{relative_path} is missing audit conditions: {missing}")


def parse_generated_example_fields(example: TrainingExample) -> dict[str, str]:
    fields = dict(FIELD_PATTERN.findall(example.text))
    missing = sorted({"family", "stimulus", "modifier"} - set(fields))
    if missing:
        raise ValueError(f"generated example is missing fields {missing}: {example.text}")
    return {
        "family": fields["family"],
        "stimulus": fields["stimulus"],
        "modifier": fields["modifier"],
    }


def is_generated_label_example(example: TrainingExample) -> bool:
    return example.source_kind in GENERATED_SOURCE_KINDS


def build_pipeline_for_audit(condition: str, seed: int, material_count: int) -> tuple[PipelineExamples, Any]:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    if condition in {"validation_ranked_induction", "mdl_rule_expansion"}:
        pipeline = build_pipeline_examples(
            condition,
            split.train,
            world.rules,
            validation_observations=split.validation,
        )
    else:
        pipeline = build_pipeline_examples(condition, split.train, world.rules)
    return pipeline, world.rules


def audit_pipeline_labels(pipeline: PipelineExamples, rules: Any) -> dict[str, Any]:
    generated = [example for example in pipeline.examples if is_generated_label_example(example)]
    correct = 0
    for example in generated:
        fields = parse_generated_example_fields(example)
        rulebook_label = rules.label(
            fields["family"],
            fields["stimulus"],
            fields["modifier"],
        )
        if example.label == rulebook_label:
            correct += 1
    total = len(generated)
    return {
        "synthetic_example_count": total,
        "correct_synthetic_label_count": correct,
        "label_precision": metric(correct / total) if total else None,
        "pipeline_internal_example_count": pipeline.internal_example_count,
        "pipeline_internal_token_count": pipeline.internal_token_count,
        "pipeline_ranked_candidate_count": pipeline.ranked_candidate_count,
        "pipeline_ranked_kept_candidate_count": pipeline.ranked_kept_candidate_count,
    }


def audit_seed_condition(seed: int, material_count: int, condition: str) -> dict[str, Any]:
    pipeline, rules = build_pipeline_for_audit(condition, seed, material_count)
    row = audit_pipeline_labels(pipeline, rules)
    row.update(
        {
            "seed": seed,
            "material_count": material_count,
            "condition": condition,
        }
    )
    return row


def aggregate_audit_rows(rows: list[dict[str, Any]], linked_stats: dict[str, Any]) -> dict[str, Any]:
    synthetic_count = sum(row["synthetic_example_count"] for row in rows)
    correct_count = sum(row["correct_synthetic_label_count"] for row in rows)
    precision = metric(correct_count / synthetic_count) if synthetic_count else None
    return {
        "synthetic_example_count": synthetic_count,
        "correct_synthetic_label_count": correct_count,
        "label_precision": precision,
        "mean_synthetic_examples_per_seed": metric(mean(row["synthetic_example_count"] for row in rows)),
        "mean_pipeline_internal_examples_per_seed": metric(
            mean(row["pipeline_internal_example_count"] for row in rows)
        ),
        "mean_pipeline_internal_tokens_per_seed": metric(
            mean(row["pipeline_internal_token_count"] for row in rows)
        ),
        "mean_ranked_candidate_count_per_seed": metric(
            mean(row["pipeline_ranked_candidate_count"] for row in rows)
        ),
        "mean_ranked_kept_candidate_count_per_seed": metric(
            mean(row["pipeline_ranked_kept_candidate_count"] for row in rows)
        ),
        "linked_accuracy_improvement_over_majority_mean": linked_stats[
            "accuracy_improvement_over_majority_mean"
        ],
        "linked_signed_learning_signal_density_per_1m_event_compute_mean": linked_stats[
            "signed_learning_signal_density_per_1m_event_compute_mean"
        ],
        "linked_charged_compute_units_mean": linked_stats["charged_compute_units_mean"],
    }


def skipped_source_conditions(source_conditions: list[str]) -> list[str]:
    return [
        condition
        for condition in source_conditions
        if condition not in AUDIT_CONDITIONS and condition not in VALIDATION_PORTFOLIO_CANDIDATES
    ]


def build_generated_label_audit(
    repo_root: Path,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    source_path = source_artifact if source_artifact.is_absolute() else repo_root / source_artifact
    source = load_json(source_path)
    relative_source = source_artifact if not source_artifact.is_absolute() else source_artifact.relative_to(repo_root)
    validate_source_artifact(relative_source, source)

    seeds = source["seeds"]
    material_counts = source["material_counts"]
    per_seed: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}

    for material in material_counts:
        material_key = str(material)
        audits[material_key] = {}
        for condition in AUDIT_CONDITIONS:
            rows = [
                audit_seed_condition(
                    seed=seed,
                    material_count=material,
                    condition=condition,
                )
                for seed in seeds
            ]
            per_seed.extend(rows)
            audits[material_key][condition] = aggregate_audit_rows(
                rows,
                source["budgets"][material_key][condition],
            )

    return {
        "title": "Learning Signal Density Generated-label Audit",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [str(relative_source)],
        "source_generated_at": source.get("generated_at"),
        "profile_label": source["profile_label"],
        "learner_backend": source["learner_backend"],
        "epochs": source["epochs"],
        "hidden_units": source["hidden_units"],
        "feature_dimension": source["feature_dimension"],
        "learning_rate": source["learning_rate"],
        "seeds": seeds,
        "material_counts": material_counts,
        "conditions_audited": list(AUDIT_CONDITIONS),
        "source_conditions_not_audited": skipped_source_conditions(list(source["conditions"])),
        "label_precision_definition": (
            "correct generated synthetic labels divided by all generated synthetic labels, "
            "using the synthetic hidden rulebook only for this diagnostic audit"
        ),
        "audits": audits,
        "per_seed": per_seed,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "fresh_seed_confirmation": source["claim_scope"]["fresh_seed_confirmation"],
            "uses_hidden_rulebook_for_audit": True,
            "hidden_rulebook_available_to_policies": False,
            "audit_only": True,
            "heldout_used_for_selection": False,
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
            "This hidden-rulebook audit is non-deployable. It compares generated synthetic "
            "labels with the synthetic world's true rulebook after the source sweep has "
            "already been run. The purpose is diagnostic: separate generated-label "
            "correctness from the tiny learner's downstream gain."
        ),
        "",
        "| Materials | Condition | Synthetic labels | Label precision | Linked gain | Linked LSD/1M |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for material in artifact["material_counts"]:
        for condition in artifact["conditions_audited"]:
            row = artifact["audits"][str(material)][condition]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(material),
                        condition,
                        str(row["synthetic_example_count"]),
                        fmt_float(row["label_precision"]),
                        fmt_float(row["linked_accuracy_improvement_over_majority_mean"]),
                        fmt_float(row["linked_signed_learning_signal_density_per_1m_event_compute_mean"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The audit uses the hidden rulebook, so it cannot be used as a deployable policy selector.",
            "- The source sweep remains the neural experiment; this artifact explains a mechanism candidate.",
            "- High generated-label precision is not sufficient evidence of neural gain at scarce budgets.",
            "",
            "```json",
            json.dumps(artifact["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_generated_label_audit(
    repo_root: Path,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    artifact = build_generated_label_audit(repo_root, source_artifact)
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
        help="Committed sweep JSON to audit.",
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
    write_generated_label_audit(repo_root, output_json, output_md, args.source_artifact)
    print(f"wrote {output_json.relative_to(repo_root)} and {output_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
