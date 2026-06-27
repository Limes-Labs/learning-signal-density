from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .domain import build_world, split_observations
from .experiment import DEFAULT_SEEDS, _pipeline_compute_units, _round
from .learner import majority_baseline
from .neural import TinyMlpClassifier
from .pipelines import CONDITION_SCOPE, build_evaluation_examples, build_pipeline_examples


DEFAULT_NEURAL_CONDITIONS = (
    "raw_text",
    "qa_expansion",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "counterfactual_expansion",
)

VALIDATION_SELECTED_NEURAL_CONDITIONS = frozenset({
    "validation_ranked_induction",
    "mdl_rule_expansion",
})

UNSUPPORTED_NEURAL_CONDITIONS = frozenset({
    "validation_gated_induction",
    "direct_validation_gated_induction",
})


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "external_events",
        "internal_examples",
        "internal_tokens",
        "charged_compute_units",
        "heldout_accuracy",
        "majority_baseline_accuracy",
        "accuracy_improvement_over_majority",
        "signed_learning_signal_density_per_1m_event_compute",
        "neural_parameter_count",
        "neural_training_step_count",
        "estimated_neural_training_multiply_adds",
    )
    return {f"{key}_mean": _round(mean(row[key] for row in rows)) for key in keys}


def run_neural_condition(
    seed: int,
    condition: str,
    material_count: int,
    epochs: int,
    hidden_units: int,
    feature_dimension: int,
    learning_rate: float,
) -> dict:
    if condition in UNSUPPORTED_NEURAL_CONDITIONS:
        raise ValueError(f"{condition} requires a neural validation gate before it can be run honestly")

    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    if condition in VALIDATION_SELECTED_NEURAL_CONDITIONS:
        pipeline = build_pipeline_examples(
            condition,
            split.train,
            world.rules,
            validation_observations=split.validation,
        )
    else:
        pipeline = build_pipeline_examples(condition, split.train, world.rules)

    heldout_examples = build_evaluation_examples(split.heldout)
    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)

    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs)
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": condition,
        "external_events": pipeline.external_event_count,
        "internal_examples": pipeline.internal_example_count,
        "internal_tokens": pipeline.internal_token_count,
        "charged_compute_units": charged_compute_units,
        "heldout_accuracy": _round(heldout.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(improvement),
        "signed_learning_signal_density_per_1m_event_compute": _round(signed_lsd),
        "neural_parameter_count": profile.parameter_count,
        "neural_training_step_count": profile.training_step_count,
        "estimated_neural_training_multiply_adds": profile.estimated_training_multiply_adds,
    }


def run_neural_seedset(
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    conditions: list[str] | tuple[str, ...] = DEFAULT_NEURAL_CONDITIONS,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    material_count: int = 48,
    epochs: int = 8,
    hidden_units: int = 16,
    feature_dimension: int = 128,
    learning_rate: float = 0.03,
) -> dict:
    per_seed: list[dict] = []
    for seed in seeds:
        for condition in conditions:
            per_seed.append(
                run_neural_condition(
                    seed=seed,
                    condition=condition,
                    material_count=material_count,
                    epochs=epochs,
                    hidden_units=hidden_units,
                    feature_dimension=feature_dimension,
                    learning_rate=learning_rate,
                )
            )

    grouped: dict[str, list[dict]] = {condition: [] for condition in conditions}
    for row in per_seed:
        grouped[row["condition"]].append(row)

    result = {
        "title": "Learning Signal Density Tiny Neural Replication",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "learner_backend": "tiny_mlp",
        "seeds": list(seeds),
        "conditions": {condition: _aggregate(rows) for condition, rows in grouped.items()},
        "condition_scope": {condition: CONDITION_SCOPE[condition] for condition in conditions},
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "heldout_used_for_selection": False,
            "oracle_transform": any(CONDITION_SCOPE[condition]["oracle_generated_labels"] for condition in conditions),
            "paper_ready_claim": False,
        },
        "material_count": material_count,
        "epochs": epochs,
        "hidden_units": hidden_units,
        "feature_dimension": feature_dimension,
        "learning_rate": learning_rate,
        "per_seed": per_seed,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_neural_markdown(result))
    return result


def render_neural_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This is a deterministic CPU tiny-MLP replication of the causal-domain pilot.",
        "It is not a language-model result and not a paper-ready frontier claim.",
        "",
        f"Backend: `{result['learner_backend']}`",
        f"Hidden units: `{result['hidden_units']}`",
        f"Feature dimension: `{result['feature_dimension']}`",
        "",
        "| Condition | Heldout acc. | Signed gain | Compute units | Neural params | Neural train ops | Signed LSD/1M |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, stats in result["conditions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    f"{stats['heldout_accuracy_mean']:.3f}",
                    f"{stats['accuracy_improvement_over_majority_mean']:.3f}",
                    f"{stats['charged_compute_units_mean']:.1f}",
                    f"{stats['neural_parameter_count_mean']:.0f}",
                    f"{stats['estimated_neural_training_multiply_adds_mean']:.0f}",
                    f"{stats['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tiny neural learning-signal density replication.")
    parser.add_argument("--output-json", type=Path, default=Path("results/tiny_neural_replication.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/tiny_neural_replication.md"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_NEURAL_CONDITIONS))
    parser.add_argument("--material-count", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--feature-dimension", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()
    run_neural_seedset(
        seeds=args.seeds,
        conditions=args.conditions,
        output_json=args.output_json,
        output_markdown=args.output_md,
        material_count=args.material_count,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        feature_dimension=args.feature_dimension,
        learning_rate=args.learning_rate,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
