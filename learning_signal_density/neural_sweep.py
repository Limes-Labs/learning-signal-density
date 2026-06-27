from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from .experiment import DEFAULT_SEEDS
from .neural_experiment import DEFAULT_NEURAL_CONDITIONS, run_neural_seedset
from .sweep import DEFAULT_MATERIAL_COUNTS


def _threshold_summary(
    budget_results: dict[str, dict],
    material_counts: list[int] | tuple[int, ...],
    conditions: list[str] | tuple[str, ...],
    target_signed_gain: float,
) -> dict:
    thresholds: dict[str, dict] = {}
    for condition in conditions:
        first_count = None
        best_count = None
        best_gain = None
        best_lsd = None
        for material_count in material_counts:
            stats = budget_results[str(material_count)][condition]
            gain = stats["accuracy_improvement_over_majority_mean"]
            lsd = stats["signed_learning_signal_density_per_1m_event_compute_mean"]
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_count = material_count
                best_lsd = lsd
            if first_count is None and gain >= target_signed_gain:
                first_count = material_count
        thresholds[condition] = {
            "first_material_count_reaching_target": first_count,
            "best_material_count": best_count,
            "best_signed_gain": round(best_gain or 0.0, 6),
            "best_signed_learning_signal_density_per_1m_event_compute": round(best_lsd or 0.0, 6),
        }
    return thresholds


def run_neural_budget_sweep(
    material_counts: list[int] | tuple[int, ...] = DEFAULT_MATERIAL_COUNTS,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    conditions: list[str] | tuple[str, ...] = DEFAULT_NEURAL_CONDITIONS,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    epochs: int = 32,
    hidden_units: int = 32,
    feature_dimension: int = 128,
    learning_rate: float = 0.03,
    target_signed_gain: float = 0.03,
    confirmation_of: str | None = None,
    comparison_of: str | None = None,
    profile_label: str | None = None,
    fresh_seed_confirmation: bool = False,
) -> dict:
    budget_results: dict[str, dict] = {}
    for material_count in material_counts:
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
        budget_results[str(material_count)] = result["conditions"]

    thresholds = _threshold_summary(
        budget_results=budget_results,
        material_counts=material_counts,
        conditions=conditions,
        target_signed_gain=target_signed_gain,
    )
    result = {
        "title": "Learning Signal Density Tiny Neural Budget Sweep",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "learner_backend": "tiny_mlp",
        "material_counts": list(material_counts),
        "seeds": list(seeds),
        "conditions": list(conditions),
        "epochs": epochs,
        "hidden_units": hidden_units,
        "feature_dimension": feature_dimension,
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
        "thresholds": thresholds,
        "budgets": budget_results,
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
        output_markdown.write_text(render_neural_sweep_markdown(result))
    return result


def render_neural_sweep_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.",
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
            f"Hidden units: `{result['hidden_units']}`",
            f"Feature dimension: `{result['feature_dimension']}`",
            f"Target signed gain over majority: `{result['target_signed_gain']}`",
            "",
            "| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in result["conditions"]:
        summary = result["thresholds"][condition]
        first = summary["first_material_count_reaching_target"]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    "not reached" if first is None else str(first),
                    str(summary["best_material_count"]),
                    f"{summary['best_signed_gain']:.3f}",
                    f"{summary['best_signed_learning_signal_density_per_1m_event_compute']:.6f}",
                ]
            )
            + " |"
        )
    if result.get("comparison_of"):
        lines.extend(["", f"Comparison target: `{result['comparison_of']}`"])

    for material_count in result["material_counts"]:
        lines.extend(
            [
                "",
                f"## Material Count {material_count}",
                "",
                "| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in result["conditions"]:
            stats = result["budgets"][str(material_count)][condition]
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
    parser = argparse.ArgumentParser(description="Run a tiny neural learning-signal density sample-budget sweep.")
    parser.add_argument("--output-json", type=Path, default=Path("results/tiny_neural_budget_sweep.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/tiny_neural_budget_sweep.md"))
    parser.add_argument("--material-counts", nargs="+", type=int, default=list(DEFAULT_MATERIAL_COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_NEURAL_CONDITIONS))
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--feature-dimension", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--target-signed-gain", type=float, default=0.03)
    parser.add_argument("--confirmation-of", default=None)
    parser.add_argument("--comparison-of", default=None)
    parser.add_argument("--profile-label", default=None)
    parser.add_argument("--fresh-seed-confirmation", action="store_true")
    args = parser.parse_args()
    run_neural_budget_sweep(
        material_counts=args.material_counts,
        seeds=args.seeds,
        conditions=args.conditions,
        output_json=args.output_json,
        output_markdown=args.output_md,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        feature_dimension=args.feature_dimension,
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
