from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .domain import build_world, split_observations
from .learner import PerceptronClassifier, majority_baseline
from .pipelines import AVAILABLE_CONDITIONS, CONDITION_SCOPE, build_evaluation_examples, build_pipeline_examples


DEFAULT_SEEDS = (3, 5, 7, 11, 13)
DEFAULT_CONDITIONS = (
    "raw_text",
    "selected_text",
    "qa_expansion",
    "induced_rule_expansion",
    "counterfactual_expansion",
    "prioritized_replay",
    "selected_counterfactual_replay",
)


def _round(value: float) -> float:
    return round(value, 6)


def run_condition(seed: int, condition: str, material_count: int, epochs: int) -> dict:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    pipeline = build_pipeline_examples(condition, split.train, world.rules)
    heldout_examples = build_evaluation_examples(split.heldout)
    validation_examples = build_evaluation_examples(split.validation)

    learner = PerceptronClassifier()
    update_count = learner.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = learner.evaluate(heldout_examples)
    validation = learner.evaluate(validation_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)

    charged_compute_units = (
        pipeline.internal_token_count * epochs
        + pipeline.selection_cost_tokens
        + pipeline.modeling_cost_tokens
        + pipeline.transform_cost_tokens
    )
    improvement = heldout.accuracy - baseline.accuracy
    positive_improvement = max(0.0, improvement)
    signed_external_sample_efficiency = improvement / max(1, pipeline.external_event_count)
    clipped_external_sample_efficiency = positive_improvement / max(1, pipeline.external_event_count)
    signed_compute_efficiency = 10000.0 * improvement / max(1, charged_compute_units)
    clipped_compute_efficiency = 10000.0 * positive_improvement / max(1, charged_compute_units)
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)
    clipped_lsd = 1_000_000.0 * positive_improvement / max(1, pipeline.external_event_count * charged_compute_units)

    return {
        "seed": seed,
        "condition": condition,
        "external_events": pipeline.external_event_count,
        "internal_examples": pipeline.internal_example_count,
        "internal_tokens": pipeline.internal_token_count,
        "selection_cost_tokens": pipeline.selection_cost_tokens,
        "modeling_cost_tokens": pipeline.modeling_cost_tokens,
        "transform_cost_tokens": pipeline.transform_cost_tokens,
        "charged_compute_units": charged_compute_units,
        "perceptron_updates": update_count,
        "heldout_accuracy": _round(heldout.accuracy),
        "validation_accuracy": _round(validation.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(improvement),
        "signed_external_sample_efficiency": _round(signed_external_sample_efficiency),
        "clipped_external_sample_efficiency": _round(clipped_external_sample_efficiency),
        "signed_compute_efficiency_per_10k_units": _round(signed_compute_efficiency),
        "clipped_compute_efficiency_per_10k_units": _round(clipped_compute_efficiency),
        "signed_learning_signal_density_per_1m_event_compute": _round(signed_lsd),
        "clipped_learning_signal_density_per_1m_event_compute": _round(clipped_lsd),
        "external_sample_efficiency": _round(clipped_external_sample_efficiency),
        "compute_efficiency_per_10k_units": _round(clipped_compute_efficiency),
        "learning_signal_density_per_1m_event_compute": _round(clipped_lsd),
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "external_events",
        "internal_examples",
        "internal_tokens",
        "selection_cost_tokens",
        "modeling_cost_tokens",
        "transform_cost_tokens",
        "charged_compute_units",
        "perceptron_updates",
        "heldout_accuracy",
        "validation_accuracy",
        "majority_baseline_accuracy",
        "accuracy_improvement_over_majority",
        "signed_external_sample_efficiency",
        "clipped_external_sample_efficiency",
        "signed_compute_efficiency_per_10k_units",
        "clipped_compute_efficiency_per_10k_units",
        "signed_learning_signal_density_per_1m_event_compute",
        "clipped_learning_signal_density_per_1m_event_compute",
        "external_sample_efficiency",
        "compute_efficiency_per_10k_units",
        "learning_signal_density_per_1m_event_compute",
    )
    return {f"{key}_mean": _round(mean(row[key] for row in rows)) for key in keys}


def _pareto_frontier(conditions: dict[str, dict]) -> list[str]:
    frontier: list[str] = []
    for name, stats in conditions.items():
        dominated = False
        for other_name, other in conditions.items():
            if other_name == name:
                continue
            at_least_as_accurate = other["heldout_accuracy_mean"] >= stats["heldout_accuracy_mean"]
            no_more_external = other["external_events_mean"] <= stats["external_events_mean"]
            no_more_compute = other["charged_compute_units_mean"] <= stats["charged_compute_units_mean"]
            strictly_better = (
                other["heldout_accuracy_mean"] > stats["heldout_accuracy_mean"]
                or other["external_events_mean"] < stats["external_events_mean"]
                or other["charged_compute_units_mean"] < stats["charged_compute_units_mean"]
            )
            if at_least_as_accurate and no_more_external and no_more_compute and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def run_seedset(
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    conditions: list[str] | tuple[str, ...] = DEFAULT_CONDITIONS,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    material_count: int = 48,
    epochs: int = 5,
) -> dict:
    unknown = sorted(set(conditions) - set(AVAILABLE_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    per_seed: list[dict] = []
    for seed in seeds:
        for condition in conditions:
            per_seed.append(run_condition(seed=seed, condition=condition, material_count=material_count, epochs=epochs))

    grouped: dict[str, list[dict]] = {condition: [] for condition in conditions}
    for row in per_seed:
        grouped[row["condition"]].append(row)

    condition_stats = {condition: _aggregate(rows) for condition, rows in grouped.items()}
    result = {
        "title": "Learning Signal Density Pilot",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seeds": list(seeds),
        "material_count": material_count,
        "epochs": epochs,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": False,
            "heldout_used_for_selection": False,
            "oracle_transform": any(CONDITION_SCOPE[condition]["oracle_generated_labels"] for condition in conditions),
            "paper_ready_claim": False,
        },
        "condition_scope": CONDITION_SCOPE,
        "conditions": condition_stats,
        "pareto_frontier_conditions": _pareto_frontier(condition_stats),
        "per_seed": per_seed,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_markdown(result))
    return result


def render_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This is a controlled pilot on a synthetic causal-text domain. It is not a neural-language-model result.",
        "The heldout split is not used for selection or transformation. Counterfactual expansion is oracle-generated inside the synthetic world.",
        "",
        "| Condition | Heldout acc. | Majority acc. | Signed gain | External events | Internal tokens | Compute units | Signed LSD/1M | Clipped LSD/1M |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, stats in result["conditions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    f"{stats['heldout_accuracy_mean']:.3f}",
                    f"{stats['majority_baseline_accuracy_mean']:.3f}",
                    f"{stats['accuracy_improvement_over_majority_mean']:.3f}",
                    f"{stats['external_events_mean']:.1f}",
                    f"{stats['internal_tokens_mean']:.1f}",
                    f"{stats['charged_compute_units_mean']:.1f}",
                    f"{stats['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    f"{stats['clipped_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            ", ".join(f"`{name}`" for name in result["pareto_frontier_conditions"]),
            "",
            "## Scope Flags",
            "",
            "```json",
            json.dumps(result["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
            "## Condition Scope",
            "",
            "| Condition | Oracle labels | Train-only selection | Train-only induction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition in result["conditions"]:
        scope = result["condition_scope"][condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(scope["oracle_generated_labels"]).lower(),
                    str(scope["train_only_selection"]).lower(),
                    str(scope["train_only_induction"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- External sample efficiency charges the original observations only.",
            "- Compute efficiency charges training tokens, train-only selection cost, and synthetic transform tokens.",
            "- Signed metrics preserve negative results; clipped metrics count only per-seed positive improvements.",
            "- Learning-signal density is reported as heldout improvement per external event per charged internal unit, scaled by 1M.",
            "- The first useful scientific question is the Pareto frontier, not a single winning condition.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the learning-signal density causal-domain pilot.")
    parser.add_argument("--output-json", type=Path, default=Path("results/causal_world_pilot_seedset.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/causal_world_pilot_seedset.md"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--material-count", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    run_seedset(
        seeds=args.seeds,
        conditions=args.conditions,
        output_json=args.output_json,
        output_markdown=args.output_md,
        material_count=args.material_count,
        epochs=args.epochs,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
