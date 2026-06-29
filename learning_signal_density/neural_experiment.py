from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean

from .domain import build_world, split_observations
from .experiment import DEFAULT_SEEDS, _pipeline_compute_units, _round
from .learner import PerceptronClassifier, majority_baseline
from .neural import TinyMlpClassifier
from .pipelines import CONDITION_SCOPE, PipelineExamples, build_evaluation_examples, build_pipeline_examples


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
VALIDATION_PORTFOLIO_SELECTOR = "validation_portfolio_selector"
VALIDATION_LINEAR_PROXY_SELECTOR = "validation_linear_proxy_selector"
VALIDATION_PORTFOLIO_CANDIDATES = (
    "raw_text",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "agreement_gated_self_ranked_induction",
    "validation_ranked_induction",
    "mdl_rule_expansion",
)
VALIDATION_LINEAR_PROXY_EPOCHS = 2
VALIDATION_PORTFOLIO_SCOPE = {
    "oracle_generated_labels": False,
    "train_only_selection": False,
    "train_only_induction": True,
    "validation_used_for_threshold": False,
    "validation_used_for_transform_selection": True,
    "validation_used_for_policy_selection": True,
}
VALIDATION_LINEAR_PROXY_SCOPE = {
    **VALIDATION_PORTFOLIO_SCOPE,
    "low_fidelity_proxy_selector": True,
}
VALIDATION_SELECTOR_CONDITIONS = frozenset({
    VALIDATION_PORTFOLIO_SELECTOR,
    VALIDATION_LINEAR_PROXY_SELECTOR,
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
        "portfolio_candidate_count",
        "portfolio_selection_cost_units",
        "portfolio_validation_score",
        "portfolio_proxy_epochs",
    )
    return {f"{key}_mean": _round(mean(row[key] for row in rows)) for key in keys}


def _confirmation_summary(conditions: dict[str, dict], target_signed_gain: float) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for condition, stats in conditions.items():
        signed_gain = stats["accuracy_improvement_over_majority_mean"]
        summary[condition] = {
            "signed_gain": signed_gain,
            "reaches_target": signed_gain >= target_signed_gain,
        }
    return summary


def neural_condition_scope(condition: str) -> dict:
    if condition == VALIDATION_PORTFOLIO_SELECTOR:
        return VALIDATION_PORTFOLIO_SCOPE
    if condition == VALIDATION_LINEAR_PROXY_SELECTOR:
        return VALIDATION_LINEAR_PROXY_SCOPE
    return CONDITION_SCOPE[condition]


def _build_neural_pipeline(condition: str, split, rules) -> PipelineExamples:
    if condition in VALIDATION_SELECTED_NEURAL_CONDITIONS:
        return build_pipeline_examples(
            condition,
            split.train,
            rules,
            validation_observations=split.validation,
        )
    return build_pipeline_examples(condition, split.train, rules)


def _empty_portfolio_fields() -> dict:
    return {
        "portfolio_candidate_count": 0,
        "portfolio_selection_cost_units": 0,
        "portfolio_validation_score": 0.0,
        "portfolio_selected_condition": None,
        "portfolio_candidate_conditions": [],
        "portfolio_selection_metric": "none",
        "portfolio_proxy_epochs": 0,
        "portfolio_candidate_summaries": [],
    }


def _run_validation_portfolio_selector(
    seed: int,
    material_count: int,
    epochs: int,
    hidden_units: int,
    feature_dimension: int,
    learning_rate: float,
) -> dict:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    heldout_examples = build_evaluation_examples(split.heldout)
    validation_examples = build_evaluation_examples(split.validation)
    validation_eval_cost = sum(example.token_count for example in validation_examples)

    candidate_records: list[dict] = []
    for candidate_condition in VALIDATION_PORTFOLIO_CANDIDATES:
        pipeline = _build_neural_pipeline(candidate_condition, split, world.rules)
        model = TinyMlpClassifier(
            feature_dimension=feature_dimension,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
        )
        model.fit(pipeline.examples, epochs=epochs, seed=seed)
        validation = model.evaluate(validation_examples)
        validation_baseline = majority_baseline(pipeline.examples, validation_examples)
        validation_score = validation.accuracy - validation_baseline.accuracy
        candidate_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + validation_eval_cost
        profile = model.training_profile
        candidate_records.append(
            {
                "condition": candidate_condition,
                "pipeline": pipeline,
                "model": model,
                "validation_accuracy": validation.accuracy,
                "validation_score": validation_score,
                "candidate_compute_units": candidate_compute_units,
                "neural_training_step_count": profile.training_step_count,
                "estimated_neural_training_multiply_adds": profile.estimated_training_multiply_adds,
                "neural_parameter_count": profile.parameter_count,
            }
        )

    best = min(
        candidate_records,
        key=lambda record: (
            -record["validation_score"],
            record["candidate_compute_units"],
            VALIDATION_PORTFOLIO_CANDIDATES.index(record["condition"]),
        ),
    )
    pipeline = best["pipeline"]
    model = best["model"]
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    charged_compute_units = sum(record["candidate_compute_units"] for record in candidate_records)
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    return {
        "seed": seed,
        "condition": VALIDATION_PORTFOLIO_SELECTOR,
        "external_events": pipeline.external_event_count,
        "internal_examples": pipeline.internal_example_count,
        "internal_tokens": pipeline.internal_token_count,
        "charged_compute_units": charged_compute_units,
        "heldout_accuracy": _round(heldout.accuracy),
        "majority_baseline_accuracy": _round(baseline.accuracy),
        "accuracy_improvement_over_majority": _round(improvement),
        "signed_learning_signal_density_per_1m_event_compute": _round(signed_lsd),
        "neural_parameter_count": best["neural_parameter_count"],
        "neural_training_step_count": sum(record["neural_training_step_count"] for record in candidate_records),
        "estimated_neural_training_multiply_adds": sum(
            record["estimated_neural_training_multiply_adds"]
            for record in candidate_records
        ),
        "portfolio_candidate_count": len(candidate_records),
        "portfolio_selection_cost_units": charged_compute_units,
        "portfolio_validation_score": _round(best["validation_score"]),
        "portfolio_selected_condition": best["condition"],
        "portfolio_candidate_conditions": list(VALIDATION_PORTFOLIO_CANDIDATES),
        "portfolio_selection_metric": "validation_accuracy_improvement_over_majority",
        "portfolio_proxy_epochs": 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "validation_accuracy": _round(record["validation_accuracy"]),
                "validation_score": _round(record["validation_score"]),
                "candidate_compute_units": record["candidate_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _run_validation_linear_proxy_selector(
    seed: int,
    material_count: int,
    epochs: int,
    hidden_units: int,
    feature_dimension: int,
    learning_rate: float,
) -> dict:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    heldout_examples = build_evaluation_examples(split.heldout)
    validation_examples = build_evaluation_examples(split.validation)
    validation_eval_cost = sum(example.token_count for example in validation_examples)

    candidate_records: list[dict] = []
    for candidate_condition in VALIDATION_PORTFOLIO_CANDIDATES:
        pipeline = _build_neural_pipeline(candidate_condition, split, world.rules)
        proxy = PerceptronClassifier()
        proxy.fit(pipeline.examples, epochs=VALIDATION_LINEAR_PROXY_EPOCHS, seed=seed)
        validation = proxy.evaluate(validation_examples)
        validation_baseline = majority_baseline(pipeline.examples, validation_examples)
        validation_score = validation.accuracy - validation_baseline.accuracy
        proxy_compute_units = (
            _pipeline_compute_units(pipeline=pipeline, epochs=VALIDATION_LINEAR_PROXY_EPOCHS)
            + validation_eval_cost
        )
        candidate_records.append(
            {
                "condition": candidate_condition,
                "pipeline": pipeline,
                "validation_accuracy": validation.accuracy,
                "validation_score": validation_score,
                "candidate_compute_units": proxy_compute_units,
            }
        )

    best = min(
        candidate_records,
        key=lambda record: (
            -record["validation_score"],
            record["candidate_compute_units"],
            VALIDATION_PORTFOLIO_CANDIDATES.index(record["condition"]),
        ),
    )
    pipeline = best["pipeline"]
    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    proxy_selection_cost_units = sum(record["candidate_compute_units"] for record in candidate_records)
    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + proxy_selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": VALIDATION_LINEAR_PROXY_SELECTOR,
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
        "portfolio_candidate_count": len(candidate_records),
        "portfolio_selection_cost_units": proxy_selection_cost_units,
        "portfolio_validation_score": _round(best["validation_score"]),
        "portfolio_selected_condition": best["condition"],
        "portfolio_candidate_conditions": list(VALIDATION_PORTFOLIO_CANDIDATES),
        "portfolio_selection_metric": "linear_proxy_validation_accuracy_improvement_over_majority",
        "portfolio_proxy_epochs": VALIDATION_LINEAR_PROXY_EPOCHS,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "validation_accuracy": _round(record["validation_accuracy"]),
                "validation_score": _round(record["validation_score"]),
                "candidate_compute_units": record["candidate_compute_units"],
            }
            for record in candidate_records
        ],
    }


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
    if condition == VALIDATION_PORTFOLIO_SELECTOR:
        return _run_validation_portfolio_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_LINEAR_PROXY_SELECTOR:
        return _run_validation_linear_proxy_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )

    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    pipeline = _build_neural_pipeline(condition, split, world.rules)

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
        **_empty_portfolio_fields(),
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
    target_signed_gain: float = 0.03,
    confirmation_of: str | None = None,
    fresh_seed_confirmation: bool = False,
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

    conditions_summary = {condition: _aggregate(rows) for condition, rows in grouped.items()}
    portfolio_selection_counts = {}
    for condition, rows in grouped.items():
        if condition not in VALIDATION_SELECTOR_CONDITIONS:
            continue
        counts = Counter(row["portfolio_selected_condition"] for row in rows)
        conditions_summary[condition]["portfolio_selected_condition_counts"] = dict(sorted(counts.items()))
        conditions_summary[condition]["portfolio_candidate_conditions"] = list(VALIDATION_PORTFOLIO_CANDIDATES)
        portfolio_selection_counts[condition] = dict(sorted(counts.items()))

    result = {
        "title": (
            "Learning Signal Density Tiny Neural Confirmation"
            if fresh_seed_confirmation
            else "Learning Signal Density Tiny Neural Replication"
        ),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "learner_backend": "tiny_mlp",
        "seeds": list(seeds),
        "conditions": conditions_summary,
        "condition_scope": {condition: neural_condition_scope(condition) for condition in conditions},
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "heldout_used_for_selection": False,
            "oracle_transform": any(
                neural_condition_scope(condition)["oracle_generated_labels"]
                for condition in conditions
            ),
            "paper_ready_claim": False,
            "fresh_seed_confirmation": fresh_seed_confirmation,
        },
        "material_count": material_count,
        "epochs": epochs,
        "hidden_units": hidden_units,
        "feature_dimension": feature_dimension,
        "learning_rate": learning_rate,
        "target_signed_gain": target_signed_gain,
        "confirmation": _confirmation_summary(conditions_summary, target_signed_gain),
        "portfolio_selection_counts": portfolio_selection_counts,
        "per_seed": per_seed,
    }
    if confirmation_of is not None:
        result["confirmation_of"] = confirmation_of

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
    ]
    if result["claim_scope"].get("fresh_seed_confirmation"):
        lines.extend(
            [
                "Fresh-seed confirmation: `true`",
                f"Confirmation target: `{result.get('confirmation_of', 'unspecified')}`",
                f"Target signed gain over majority: `{result['target_signed_gain']}`",
                "",
            ]
        )
    lines.extend(
        [
            f"Backend: `{result['learner_backend']}`",
            f"Hidden units: `{result['hidden_units']}`",
            f"Feature dimension: `{result['feature_dimension']}`",
            "",
            "| Condition | Heldout acc. | Signed gain | Reaches target | Compute units | Neural params | Neural train ops | Signed LSD/1M |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition, stats in result["conditions"].items():
        confirmation = result["confirmation"][condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    f"{stats['heldout_accuracy_mean']:.3f}",
                    f"{stats['accuracy_improvement_over_majority_mean']:.3f}",
                    "yes" if confirmation["reaches_target"] else "no",
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
    parser.add_argument("--target-signed-gain", type=float, default=0.03)
    parser.add_argument("--confirmation-of", default=None)
    parser.add_argument("--fresh-seed-confirmation", action="store_true")
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
        target_signed_gain=args.target_signed_gain,
        confirmation_of=args.confirmation_of,
        fresh_seed_confirmation=args.fresh_seed_confirmation,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
