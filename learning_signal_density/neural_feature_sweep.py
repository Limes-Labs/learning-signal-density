from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from .experiment import DEFAULT_SEEDS
from .neural_experiment import DEFAULT_NEURAL_CONDITIONS, run_neural_seedset


DEFAULT_FEATURE_DIMENSIONS = (16, 32, 64, 128, 256)


def _dimension_id(feature_dimension: int) -> str:
    return f"features={feature_dimension}"


def _normalize_feature_dimensions(feature_dimensions: list[int] | tuple[int, ...]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for feature_dimension in feature_dimensions:
        value = int(feature_dimension)
        if value <= 0:
            raise ValueError("feature dimensions must be positive")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError("at least one feature dimension is required")
    return normalized


def _frontier_summary(
    dimension_results: dict[str, dict],
    conditions: list[str] | tuple[str, ...],
    target_signed_gain: float,
) -> dict:
    frontier: dict[str, dict] = {}
    for condition in conditions:
        best_gain_dimension = None
        best_gain = None
        best_lsd_dimension = None
        best_lsd = None
        lowest_ops_dimension = None
        lowest_ops = None
        lowest_ops_gain = None
        for dimension_key, result in dimension_results.items():
            stats = result["conditions"][condition]
            gain = stats["accuracy_improvement_over_majority_mean"]
            lsd = stats["signed_learning_signal_density_per_1m_event_compute_mean"]
            ops = stats["estimated_neural_training_multiply_adds_mean"]
            feature_dimension = result["feature_dimension"]
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_gain_dimension = feature_dimension
            if best_lsd is None or lsd > best_lsd:
                best_lsd = lsd
                best_lsd_dimension = feature_dimension
            if gain >= target_signed_gain and (lowest_ops is None or ops < lowest_ops):
                lowest_ops = ops
                lowest_ops_dimension = feature_dimension
                lowest_ops_gain = gain
        frontier[condition] = {
            "best_signed_gain_feature_dimension": best_gain_dimension,
            "best_signed_gain": round(best_gain or 0.0, 6),
            "best_signed_lsd_feature_dimension": best_lsd_dimension,
            "best_signed_learning_signal_density_per_1m_event_compute": round(best_lsd or 0.0, 6),
            "lowest_neural_ops_reaching_target_feature_dimension": lowest_ops_dimension,
            "lowest_neural_ops_reaching_target": round(lowest_ops or 0.0, 6),
            "lowest_neural_ops_reaching_target_signed_gain": round(lowest_ops_gain or 0.0, 6),
        }
    return frontier


def run_neural_feature_sweep(
    feature_dimensions: list[int] | tuple[int, ...] = DEFAULT_FEATURE_DIMENSIONS,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    conditions: list[str] | tuple[str, ...] = DEFAULT_NEURAL_CONDITIONS,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    material_count: int = 64,
    epochs: int = 32,
    hidden_units: int = 8,
    learning_rate: float = 0.03,
    target_signed_gain: float = 0.03,
    confirmation_of: str | None = None,
    comparison_of: str | None = None,
    profile_label: str | None = None,
    fresh_seed_confirmation: bool = False,
) -> dict:
    normalized_dimensions = _normalize_feature_dimensions(feature_dimensions)
    dimension_results: dict[str, dict] = {}
    for feature_dimension in normalized_dimensions:
        result = run_neural_seedset(
            seeds=seeds,
            conditions=conditions,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
            target_signed_gain=target_signed_gain,
            confirmation_of=confirmation_of,
            fresh_seed_confirmation=fresh_seed_confirmation,
        )
        dimension_results[_dimension_id(feature_dimension)] = {
            "feature_dimension": feature_dimension,
            "conditions": result["conditions"],
        }

    result = {
        "title": "Learning Signal Density Tiny Neural Feature-Dimension Sweep",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "learner_backend": "tiny_mlp",
        "material_count": material_count,
        "seeds": list(seeds),
        "conditions": list(conditions),
        "feature_dimensions": normalized_dimensions,
        "epochs": epochs,
        "hidden_units": hidden_units,
        "learning_rate": learning_rate,
        "profile_label": profile_label or f"epochs={epochs}_hidden={hidden_units}",
        "target_signed_gain": target_signed_gain,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "heldout_used_for_selection": False,
            "paper_ready_claim": False,
            "fresh_seed_confirmation": fresh_seed_confirmation,
        },
        "frontier": _frontier_summary(dimension_results, conditions, target_signed_gain),
        "dimension_results": dimension_results,
    }
    if confirmation_of is not None:
        result["confirmation_of"] = confirmation_of
    if comparison_of is not None:
        result["comparison_of"] = comparison_of

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_neural_feature_markdown(result))
    return result


def render_neural_feature_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This sweep reruns the deterministic CPU tiny-MLP profile across hashed feature dimensions.",
        "It is still synthetic and not a language-model result.",
        "",
    ]
    if result["claim_scope"].get("fresh_seed_confirmation"):
        lines.extend(
            [
                "Fresh-seed confirmation sweep: `true`",
                f"Confirmation target: `{result.get('confirmation_of', 'unspecified')}`",
                "",
            ]
        )
    lines.extend(
        [
            f"Backend: `{result['learner_backend']}`",
            f"Profile label: `{result['profile_label']}`",
            f"Material count: `{result['material_count']}`",
            f"Hidden units: `{result['hidden_units']}`",
            f"Feature dimensions: `{', '.join(str(value) for value in result['feature_dimensions'])}`",
            f"Target signed gain over majority: `{result['target_signed_gain']}`",
        ]
    )
    if result.get("comparison_of"):
        lines.append(f"Comparison target: `{result['comparison_of']}`")
    lines.extend(
        [
            "",
            "| Condition | Best gain dimension | Best signed gain | Best LSD dimension | Best signed LSD/1M | Lowest-op target dimension | Target-dimension ops |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in result["conditions"]:
        summary = result["frontier"][condition]
        target_dimension = summary["lowest_neural_ops_reaching_target_feature_dimension"]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(summary["best_signed_gain_feature_dimension"]),
                    f"{summary['best_signed_gain']:.3f}",
                    str(summary["best_signed_lsd_feature_dimension"]),
                    f"{summary['best_signed_learning_signal_density_per_1m_event_compute']:.6f}",
                    "not reached" if target_dimension is None else str(target_dimension),
                    f"{summary['lowest_neural_ops_reaching_target']:.0f}",
                ]
            )
            + " |"
        )

    for feature_dimension in result["feature_dimensions"]:
        dimension_key = _dimension_id(feature_dimension)
        lines.extend(
            [
                "",
                f"## Feature Dimension {feature_dimension}",
                "",
                "| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in result["conditions"]:
            stats = result["dimension_results"][dimension_key]["conditions"][condition]
            lines.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        f"{stats['heldout_accuracy_mean']:.3f}",
                        f"{stats['accuracy_improvement_over_majority_mean']:.3f}",
                        f"{stats['charged_compute_units_mean']:.1f}",
                        f"{stats['estimated_neural_training_multiply_adds_mean']:.0f}",
                        f"{stats['signed_learning_signal_density_per_1m_event_compute_mean']:.6f}",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny neural feature-dimension sweep.")
    parser.add_argument("--output-json", type=Path, default=Path("results/tiny_neural_feature_sweep.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/tiny_neural_feature_sweep.md"))
    parser.add_argument("--feature-dimensions", nargs="+", type=int, default=list(DEFAULT_FEATURE_DIMENSIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_NEURAL_CONDITIONS))
    parser.add_argument("--material-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--target-signed-gain", type=float, default=0.03)
    parser.add_argument("--confirmation-of", default=None)
    parser.add_argument("--comparison-of", default=None)
    parser.add_argument("--profile-label", default=None)
    parser.add_argument("--fresh-seed-confirmation", action="store_true")
    args = parser.parse_args()
    run_neural_feature_sweep(
        feature_dimensions=args.feature_dimensions,
        seeds=args.seeds,
        conditions=args.conditions,
        output_json=args.output_json,
        output_markdown=args.output_md,
        material_count=args.material_count,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        target_signed_gain=args.target_signed_gain,
        confirmation_of=args.confirmation_of,
        comparison_of=args.comparison_of,
        profile_label=args.profile_label,
        fresh_seed_confirmation=args.fresh_seed_confirmation,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
