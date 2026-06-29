#!/usr/bin/env python3
"""Audit generated-label coverage against heldout motif distributions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from math import log2
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning_signal_density.domain import Observation, build_world, split_observations
from learning_signal_density.neural_experiment import VALIDATION_PORTFOLIO_CANDIDATES
from learning_signal_density.pipelines import PipelineExamples, TrainingExample, build_pipeline_examples


DEFAULT_SOURCE_ARTIFACT = Path("results/tiny_neural_budget_sweep_selector_transfer_f1024.json")
DEFAULT_OUTPUT_JSON = Path("results/generated_coverage_audit_selector_transfer_f1024.json")
DEFAULT_OUTPUT_MD = Path("results/generated_coverage_audit_selector_transfer_f1024.md")

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


def metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def parse_example_fields(example: TrainingExample) -> tuple[str, str, str]:
    fields = dict(FIELD_PATTERN.findall(example.text))
    missing = sorted({"family", "stimulus", "modifier"} - set(fields))
    if missing:
        raise ValueError(f"example is missing fields {missing}: {example.text}")
    return fields["family"], fields["stimulus"], fields["modifier"]


def generated_examples(pipeline: PipelineExamples) -> tuple[TrainingExample, ...]:
    return tuple(example for example in pipeline.examples if example.source_kind in GENERATED_SOURCE_KINDS)


def distribution_l1_distance(generated_counts: dict[Any, int], target_counts: dict[Any, int]) -> float | None:
    generated_total = sum(generated_counts.values())
    target_total = sum(target_counts.values())
    if generated_total == 0 or target_total == 0:
        return None
    keys = set(generated_counts) | set(target_counts)
    distance = sum(
        abs(generated_counts.get(key, 0) / generated_total - target_counts.get(key, 0) / target_total)
        for key in keys
    )
    return metric(distance / 2)


def entropy_bits(counts: dict[Any, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return metric(-sum((count / total) * log2(count / total) for count in counts.values())) or 0.0


def heldout_counters(heldout: tuple[Observation, ...]) -> tuple[Counter[tuple[str, str, str]], Counter[tuple[str, str]]]:
    triple_counts: Counter[tuple[str, str, str]] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for item in heldout:
        triple_counts[(item.family, item.stimulus, item.modifier)] += 1
        pair_counts[(item.family, item.stimulus)] += 1
    return triple_counts, pair_counts


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


def build_pipeline_for_audit(
    condition: str,
    seed: int,
    material_count: int,
) -> tuple[PipelineExamples, tuple[Observation, ...], Any]:
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
    return pipeline, split.heldout, world.rules


def seed_condition_counters(seed: int, material_count: int, condition: str) -> dict[str, Any]:
    pipeline, heldout, rules = build_pipeline_for_audit(condition, seed, material_count)
    generated_triples: Counter[tuple[str, str, str]] = Counter()
    modifier_counts: Counter[str] = Counter()
    correct = 0

    for example in generated_examples(pipeline):
        family, stimulus, modifier = parse_example_fields(example)
        triple = (family, stimulus, modifier)
        generated_triples[triple] += 1
        modifier_counts[modifier] += 1
        if example.label == rules.label(family, stimulus, modifier):
            correct += 1

    heldout_triples, heldout_pairs = heldout_counters(heldout)
    generated_pairs = {(family, stimulus) for family, stimulus, _ in generated_triples}
    synthetic_count = sum(generated_triples.values())
    heldout_count = sum(heldout_triples.values())
    heldout_pair_count = sum(heldout_pairs.values())

    return {
        "seed": seed,
        "material_count": material_count,
        "condition": condition,
        "generated_triples": generated_triples,
        "heldout_triples": heldout_triples,
        "heldout_pairs": heldout_pairs,
        "generated_pairs": generated_pairs,
        "modifier_counts": modifier_counts,
        "synthetic_example_count": synthetic_count,
        "correct_synthetic_label_count": correct,
        "heldout_triple_coverage": metric(
            sum(count for triple, count in heldout_triples.items() if triple in generated_triples)
            / heldout_count
        )
        if heldout_count
        else 0.0,
        "heldout_pair_coverage": metric(
            sum(count for pair, count in heldout_pairs.items() if pair in generated_pairs)
            / heldout_pair_count
        )
        if heldout_pair_count
        else 0.0,
        "generated_vs_heldout_triple_l1_distance": distribution_l1_distance(
            generated_triples,
            heldout_triples,
        ),
        "generated_unique_triple_count": len(generated_triples),
        "modifier_entropy_bits": entropy_bits(modifier_counts),
        "max_modifier_share": metric(max(modifier_counts.values(), default=0) / max(1, synthetic_count)),
    }


def public_seed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": row["seed"],
        "material_count": row["material_count"],
        "condition": row["condition"],
        "synthetic_example_count": row["synthetic_example_count"],
        "correct_synthetic_label_count": row["correct_synthetic_label_count"],
        "label_precision": metric(row["correct_synthetic_label_count"] / row["synthetic_example_count"])
        if row["synthetic_example_count"]
        else None,
        "heldout_triple_coverage": row["heldout_triple_coverage"],
        "heldout_pair_coverage": row["heldout_pair_coverage"],
        "generated_vs_heldout_triple_l1_distance": row["generated_vs_heldout_triple_l1_distance"],
        "generated_unique_triple_count": row["generated_unique_triple_count"],
        "modifier_entropy_bits": row["modifier_entropy_bits"],
        "max_modifier_share": row["max_modifier_share"],
    }


def aggregate_condition_rows(rows: list[dict[str, Any]], linked_stats: dict[str, Any]) -> dict[str, Any]:
    generated_triples: Counter[tuple[str, str, str]] = Counter()
    heldout_triples: Counter[tuple[str, str, str]] = Counter()
    heldout_pairs: Counter[tuple[str, str]] = Counter()
    generated_pairs: set[tuple[str, str]] = set()
    modifier_counts: Counter[str] = Counter()
    synthetic_count = 0
    correct_count = 0

    for row in rows:
        generated_triples.update(row["generated_triples"])
        heldout_triples.update(row["heldout_triples"])
        heldout_pairs.update(row["heldout_pairs"])
        generated_pairs.update(row["generated_pairs"])
        modifier_counts.update(row["modifier_counts"])
        synthetic_count += row["synthetic_example_count"]
        correct_count += row["correct_synthetic_label_count"]

    heldout_count = sum(heldout_triples.values())
    heldout_pair_count = sum(heldout_pairs.values())
    return {
        "synthetic_example_count": synthetic_count,
        "correct_synthetic_label_count": correct_count,
        "label_precision": metric(correct_count / synthetic_count) if synthetic_count else None,
        "heldout_triple_coverage": metric(
            sum(count for triple, count in heldout_triples.items() if triple in generated_triples)
            / heldout_count
        )
        if heldout_count
        else 0.0,
        "heldout_pair_coverage": metric(
            sum(count for pair, count in heldout_pairs.items() if pair in generated_pairs)
            / heldout_pair_count
        )
        if heldout_pair_count
        else 0.0,
        "generated_vs_heldout_triple_l1_distance": distribution_l1_distance(
            generated_triples,
            heldout_triples,
        ),
        "generated_unique_triple_count": len(generated_triples),
        "modifier_entropy_bits": entropy_bits(modifier_counts),
        "max_modifier_share": metric(max(modifier_counts.values(), default=0) / max(1, synthetic_count)),
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


def build_generated_coverage_audit(
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
                seed_condition_counters(seed=seed, material_count=material, condition=condition)
                for seed in seeds
            ]
            per_seed.extend(public_seed_row(row) for row in rows)
            audits[material_key][condition] = aggregate_condition_rows(
                rows,
                source["budgets"][material_key][condition],
            )

    return {
        "title": "Learning Signal Density Generated-coverage Audit",
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
        "coverage_definition": (
            "motifs are family/stimulus/modifier triples; L1 distance compares generated "
            "synthetic-label motif frequencies with heldout motif frequencies after the source sweep"
        ),
        "audits": audits,
        "per_seed": per_seed,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "fresh_seed_confirmation": source["claim_scope"]["fresh_seed_confirmation"],
            "uses_hidden_rulebook_for_label_audit": True,
            "uses_heldout_distribution_for_audit": True,
            "hidden_rulebook_available_to_policies": False,
            "heldout_distribution_available_to_policies": False,
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
            "This heldout-distribution audit is non-deployable. It compares where "
            "generated synthetic labels land in motif space against the heldout motif "
            "distribution after the source sweep has already been run."
        ),
        "",
        "| Materials | Condition | Label precision | Heldout triple coverage | Triple L1 distance | Linked gain |",
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
                        fmt_float(row["label_precision"]),
                        fmt_float(row["heldout_triple_coverage"]),
                        fmt_float(row["generated_vs_heldout_triple_l1_distance"]),
                        fmt_float(row["linked_accuracy_improvement_over_majority_mean"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The audit uses the hidden rulebook and heldout motif distribution, so it cannot be used as a deployable selector.",
            "- Lower triple L1 distance means generated labels are distributed more like heldout motifs.",
            "- The source sweep remains the neural experiment; this artifact tests a mechanism hypothesis.",
            "",
            "```json",
            json.dumps(artifact["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_generated_coverage_audit(
    repo_root: Path,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    artifact = build_generated_coverage_audit(repo_root, source_artifact)
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
    write_generated_coverage_audit(repo_root, output_json, output_md, args.source_artifact)
    print(f"wrote {output_json.relative_to(repo_root)} and {output_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
