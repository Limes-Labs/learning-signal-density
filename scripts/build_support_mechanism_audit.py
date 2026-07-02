#!/usr/bin/env python3
"""Audit high-budget support-ramp mechanisms on the selector-transfer seeds."""

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
from learning_signal_density.pipelines import PipelineExamples, TrainingExample, build_pipeline_examples


DEFAULT_SOURCE_ARTIFACT = Path("results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json")
DEFAULT_OUTPUT_JSON = Path("results/support_mechanism_audit_f1024.json")
DEFAULT_OUTPUT_MD = Path("results/support_mechanism_audit_f1024.md")

AUDIT_CONDITIONS = (
    "raw_text",
    "compact_train_size_gated_induction",
    "support_ramped_compact_induction",
    "density_window_compact_induction",
    "density_capped_compact_induction",
)
GENERATED_SOURCE_KINDS = frozenset({
    "agreement_gated_self_ranked_induced",
    "compact_diverse_sample_aware_self_ranked_induced",
    "compact_sample_aware_self_ranked_induced",
    "counterfactual",
    "diverse_self_ranked_induced",
    "induced_counterfactual",
    "late_confidence_ramped_compact_sample_aware_self_ranked_induced",
    "mdl_counterfactual",
    "sample_aware_diverse_self_ranked_induced",
    "sample_aware_self_ranked_induced",
    "self_ranked_induced",
    "support_ramped_compact_sample_aware_self_ranked_induced",
    "validation_ranked_induced",
})
TRANSITION_MATERIAL_MIN = 104
FIELD_PATTERN = re.compile(r"\b(family|modifier|stimulus)=([A-Za-z0-9_]+)")

CONDITION_LABELS = {
    "compact_train_size_gated_induction": "Compact train-size gated",
    "density_capped_compact_induction": "Density-capped compact",
    "density_window_compact_induction": "Density-window compact",
    "raw_text": "Raw text",
    "support_ramped_compact_induction": "Support-ramped compact",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


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
    return metric(
        sum(
            abs(generated_counts.get(key, 0) / generated_total - target_counts.get(key, 0) / target_total)
            for key in keys
        )
        / 2
    )


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


def seed_condition_counters(seed: int, material_count: int, condition: str) -> dict[str, Any]:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    pipeline = build_pipeline_examples(condition, split.train, world.rules)

    generated_triples: Counter[tuple[str, str, str]] = Counter()
    generated_pairs: set[tuple[str, str]] = set()
    modifier_counts: Counter[str] = Counter()
    correct = 0

    for example in generated_examples(pipeline):
        family, stimulus, modifier = parse_example_fields(example)
        triple = (family, stimulus, modifier)
        generated_triples[triple] += 1
        generated_pairs.add((family, stimulus))
        modifier_counts[modifier] += 1
        if example.label == world.rules.label(family, stimulus, modifier):
            correct += 1

    heldout_triples, heldout_pairs = heldout_counters(split.heldout)
    synthetic_count = sum(generated_triples.values())
    heldout_count = sum(heldout_triples.values())
    heldout_pair_count = sum(heldout_pairs.values())

    return {
        "seed": seed,
        "material_count": material_count,
        "condition": condition,
        "generated_triples": generated_triples,
        "generated_pairs": generated_pairs,
        "heldout_triples": heldout_triples,
        "heldout_pairs": heldout_pairs,
        "modifier_counts": modifier_counts,
        "synthetic_example_count": synthetic_count,
        "correct_synthetic_label_count": correct,
        "pipeline_internal_example_count": pipeline.internal_example_count,
        "pipeline_internal_token_count": pipeline.internal_token_count,
        "ranked_candidate_count": pipeline.ranked_candidate_count,
        "ranked_kept_candidate_count": pipeline.ranked_kept_candidate_count,
        "ranked_induction_min_support": pipeline.ranked_induction_min_support,
        "ranked_induction_min_confidence": pipeline.ranked_induction_min_confidence,
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
        "ranked_kept_candidate_count": row["ranked_kept_candidate_count"],
        "ranked_induction_min_support": row["ranked_induction_min_support"],
        "ranked_induction_min_confidence": row["ranked_induction_min_confidence"],
    }


def aggregate_condition_rows(rows: list[dict[str, Any]], linked_stats: dict[str, Any]) -> dict[str, Any]:
    generated_triples: Counter[tuple[str, str, str]] = Counter()
    generated_pairs: set[tuple[str, str]] = set()
    heldout_triples: Counter[tuple[str, str, str]] = Counter()
    heldout_pairs: Counter[tuple[str, str]] = Counter()
    modifier_counts: Counter[str] = Counter()
    synthetic_count = 0
    correct_count = 0

    for row in rows:
        generated_triples.update(row["generated_triples"])
        generated_pairs.update(row["generated_pairs"])
        heldout_triples.update(row["heldout_triples"])
        heldout_pairs.update(row["heldout_pairs"])
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
        "mean_pipeline_internal_examples_per_seed": metric(
            sum(row["pipeline_internal_example_count"] for row in rows) / len(rows)
        ),
        "mean_pipeline_internal_tokens_per_seed": metric(
            sum(row["pipeline_internal_token_count"] for row in rows) / len(rows)
        ),
        "mean_ranked_candidate_count_per_seed": metric(
            sum(row["ranked_candidate_count"] for row in rows) / len(rows)
        ),
        "mean_ranked_kept_candidate_count_per_seed": metric(
            sum(row["ranked_kept_candidate_count"] for row in rows) / len(rows)
        ),
        "mean_ranked_induction_min_support": metric(
            sum(row["ranked_induction_min_support"] for row in rows) / len(rows)
        ),
        "mean_ranked_induction_min_confidence": metric(
            sum(row["ranked_induction_min_confidence"] for row in rows) / len(rows)
        ),
        "linked_accuracy_improvement_over_majority_mean": linked_stats[
            "accuracy_improvement_over_majority_mean"
        ],
        "linked_signed_learning_signal_density_per_1m_event_compute_mean": linked_stats[
            "signed_learning_signal_density_per_1m_event_compute_mean"
        ],
        "linked_charged_compute_units_mean": linked_stats["charged_compute_units_mean"],
    }


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return metric(left - right)


def build_transition_diagnostics(audits: dict[str, dict[str, Any]], material_counts: list[int]) -> dict[str, Any]:
    transition: dict[str, Any] = {}
    for material in material_counts:
        if material < TRANSITION_MATERIAL_MIN:
            continue
        material_key = str(material)
        compact = audits[material_key]["compact_train_size_gated_induction"]
        support = audits[material_key]["support_ramped_compact_induction"]
        density = audits[material_key]["density_capped_compact_induction"]
        transition[material_key] = {
            "support_minus_compact_label_precision": delta(
                support["label_precision"],
                compact["label_precision"],
            ),
            "support_minus_compact_heldout_triple_coverage": delta(
                support["heldout_triple_coverage"],
                compact["heldout_triple_coverage"],
            ),
            "support_minus_compact_synthetic_example_count": (
                support["synthetic_example_count"] - compact["synthetic_example_count"]
            ),
            "support_minus_density_capped_gain": metric(
                support["linked_accuracy_improvement_over_majority_mean"]
                - density["linked_accuracy_improvement_over_majority_mean"]
            ),
            "support_minus_density_capped_lsd": metric(
                support["linked_signed_learning_signal_density_per_1m_event_compute_mean"]
                - density["linked_signed_learning_signal_density_per_1m_event_compute_mean"]
            ),
            "support_minus_density_capped_cost": metric(
                support["linked_charged_compute_units_mean"]
                - density["linked_charged_compute_units_mean"],
                digits=1,
            ),
            "support_precision_beats_compact": (
                support["label_precision"] is not None
                and compact["label_precision"] is not None
                and support["label_precision"] > compact["label_precision"]
            ),
            "support_coverage_beats_compact": (
                support["heldout_triple_coverage"] > compact["heldout_triple_coverage"]
            ),
            "support_lsd_beats_density_capped": (
                support["linked_signed_learning_signal_density_per_1m_event_compute_mean"]
                > density["linked_signed_learning_signal_density_per_1m_event_compute_mean"]
            ),
            "support_gain_beats_density_capped": (
                support["linked_accuracy_improvement_over_majority_mean"]
                > density["linked_accuracy_improvement_over_majority_mean"]
            ),
        }
    return transition


def average_deltas(rows: list[float | None]) -> float | None:
    values = [row for row in rows if row is not None]
    if not values:
        return None
    return metric(sum(values) / len(values))


def summarize_mechanism(transition: dict[str, Any]) -> dict[str, Any]:
    material_counts = [int(material) for material in transition]
    precision_count = sum(1 for row in transition.values() if row["support_precision_beats_compact"])
    coverage_loss_count = sum(1 for row in transition.values() if not row["support_coverage_beats_compact"])
    density_win_count = sum(1 for row in transition.values() if row["support_lsd_beats_density_capped"])
    gain_win_count = sum(1 for row in transition.values() if row["support_gain_beats_density_capped"])
    return {
        "transition_material_counts": material_counts,
        "support_ramp_precision_improvement_count": precision_count,
        "support_ramp_coverage_loss_count": coverage_loss_count,
        "support_ramp_density_win_over_density_cap_count": density_win_count,
        "support_ramp_gain_win_over_density_cap_count": gain_win_count,
        "mean_support_minus_compact_label_precision": average_deltas(
            [row["support_minus_compact_label_precision"] for row in transition.values()]
        ),
        "mean_support_minus_compact_heldout_triple_coverage": average_deltas(
            [row["support_minus_compact_heldout_triple_coverage"] for row in transition.values()]
        ),
        "mean_support_minus_density_capped_lsd": average_deltas(
            [row["support_minus_density_capped_lsd"] for row in transition.values()]
        ),
        "promote_support_ramp_mechanism": (
            precision_count == len(transition)
            and density_win_count == len(transition)
            and coverage_loss_count == 0
        ),
        "interpretation": (
            "The support ramp is a cost-reduction mechanism, not a reliability improvement: "
            "on transition budgets it lowers synthetic volume and usually loses heldout motif "
            "coverage and signed density versus the density-capped raw fallback."
        ),
    }


def build_support_mechanism_audit(
    repo_root: Path,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    source_path = source_artifact if source_artifact.is_absolute() else repo_root / source_artifact
    source = load_json(source_path)
    relative_source = source_artifact if not source_artifact.is_absolute() else source_artifact.relative_to(repo_root)
    validate_source_artifact(relative_source, source)

    seeds = source["seeds"]
    material_counts = source["material_counts"]
    audits: dict[str, dict[str, Any]] = {}
    per_seed: list[dict[str, Any]] = []

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

    transition = build_transition_diagnostics(audits, material_counts)
    return {
        "title": "Learning Signal Density Support-ramp Mechanism Audit",
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
        "mechanism_question": (
            "Does the abundant-data support ramp improve generated-label reliability "
            "or heldout motif coverage enough to justify extra representation cost?"
        ),
        "audits": audits,
        "transition_diagnostics": transition,
        "mechanism_summary": summarize_mechanism(transition),
        "per_seed": per_seed,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "fresh_seed_confirmation": source["claim_scope"]["fresh_seed_confirmation"],
            "uses_hidden_rulebook_for_label_audit": True,
            "uses_heldout_distribution_for_audit": True,
            "hidden_rulebook_available_to_policies": False,
            "heldout_available_to_policies": False,
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
            "This support-ramp mechanism audit is non-deployable. It reconstructs "
            "candidate pipelines on the support-selector transfer seeds and compares "
            "generated labels with the hidden rulebook and heldout motif distribution "
            "after the neural sweep has already run."
        ),
        "",
        "## Transition Diagnostics",
        "",
        "| Materials | Support precision | Compact precision | Support coverage | Compact coverage | Support LSD | Density-cap LSD | Support minus density-cap LSD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for material in artifact["mechanism_summary"]["transition_material_counts"]:
        material_key = str(material)
        support = artifact["audits"][material_key]["support_ramped_compact_induction"]
        compact = artifact["audits"][material_key]["compact_train_size_gated_induction"]
        density = artifact["audits"][material_key]["density_capped_compact_induction"]
        diagnostic = artifact["transition_diagnostics"][material_key]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(material),
                    fmt_float(support["label_precision"]),
                    fmt_float(compact["label_precision"]),
                    fmt_float(support["heldout_triple_coverage"]),
                    fmt_float(compact["heldout_triple_coverage"]),
                    fmt_float(support["linked_signed_learning_signal_density_per_1m_event_compute_mean"]),
                    fmt_float(density["linked_signed_learning_signal_density_per_1m_event_compute_mean"]),
                    fmt_float(diagnostic["support_minus_density_capped_lsd"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Mechanism Summary",
            "",
            f"- Support precision improvements over compact: `{artifact['mechanism_summary']['support_ramp_precision_improvement_count']}`.",
            f"- Support coverage losses versus compact: `{artifact['mechanism_summary']['support_ramp_coverage_loss_count']}`.",
            f"- Support density wins over density cap: `{artifact['mechanism_summary']['support_ramp_density_win_over_density_cap_count']}`.",
            f"- Promote support-ramp mechanism: `{str(artifact['mechanism_summary']['promote_support_ramp_mechanism']).lower()}`.",
            f"- Interpretation: {artifact['mechanism_summary']['interpretation']}",
            "",
            "## Scope",
            "",
            "- The hidden rulebook and heldout motif distribution are used only after the source sweep.",
            "- The source neural artifact remains the downstream experiment; this artifact explains mechanism failures.",
            "- Treat the result as a design constraint for future expected-value selectors.",
            "",
            "```json",
            json.dumps(artifact["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_support_mechanism_audit(
    repo_root: Path,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
) -> dict[str, Any]:
    artifact = build_support_mechanism_audit(repo_root, source_artifact)
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
        help="Committed support-selector sweep JSON to audit.",
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
    write_support_mechanism_audit(repo_root, output_json, output_md, args.source_artifact)
    print(f"wrote {output_json.relative_to(repo_root)} and {output_md.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
