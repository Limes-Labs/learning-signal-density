from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Iterable

from .experiment import DEFAULT_SEEDS
from .neural_experiment import DEFAULT_NEURAL_CONDITIONS, run_neural_seedset


DEFAULT_PROFILES = (
    {"epochs": 8, "hidden_units": 8},
    {"epochs": 16, "hidden_units": 8},
    {"epochs": 32, "hidden_units": 8},
    {"epochs": 8, "hidden_units": 16},
    {"epochs": 16, "hidden_units": 16},
    {"epochs": 32, "hidden_units": 16},
    {"epochs": 8, "hidden_units": 32},
    {"epochs": 16, "hidden_units": 32},
    {"epochs": 32, "hidden_units": 32},
)


def _profile_id(profile: dict) -> str:
    return f"epochs={profile['epochs']}_hidden={profile['hidden_units']}"


def _normalize_profiles(profiles: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for profile in profiles:
        epochs = int(profile["epochs"])
        hidden_units = int(profile["hidden_units"])
        if epochs <= 0:
            raise ValueError("profile epochs must be positive")
        if hidden_units <= 0:
            raise ValueError("profile hidden_units must be positive")
        normalized_profile = {"epochs": epochs, "hidden_units": hidden_units}
        key = _profile_id(normalized_profile)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_profile)
    if not normalized:
        raise ValueError("at least one profile is required")
    return normalized


def _frontier_summary(
    profile_results: dict[str, dict],
    conditions: list[str] | tuple[str, ...],
    target_signed_gain: float,
) -> dict:
    frontier: dict[str, dict] = {}
    for condition in conditions:
        best_gain_profile = None
        best_gain = None
        best_lsd_profile = None
        best_lsd = None
        lowest_ops_profile = None
        lowest_ops = None
        lowest_ops_gain = None
        for profile_id, result in profile_results.items():
            stats = result["conditions"][condition]
            gain = stats["accuracy_improvement_over_majority_mean"]
            lsd = stats["signed_learning_signal_density_per_1m_event_compute_mean"]
            ops = stats["estimated_neural_training_multiply_adds_mean"]
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_gain_profile = profile_id
            if best_lsd is None or lsd > best_lsd:
                best_lsd = lsd
                best_lsd_profile = profile_id
            if gain >= target_signed_gain and (lowest_ops is None or ops < lowest_ops):
                lowest_ops = ops
                lowest_ops_profile = profile_id
                lowest_ops_gain = gain
        frontier[condition] = {
            "best_signed_gain_profile": best_gain_profile,
            "best_signed_gain": round(best_gain or 0.0, 6),
            "best_signed_lsd_profile": best_lsd_profile,
            "best_signed_learning_signal_density_per_1m_event_compute": round(best_lsd or 0.0, 6),
            "lowest_neural_ops_reaching_target_profile": lowest_ops_profile,
            "lowest_neural_ops_reaching_target": round(lowest_ops or 0.0, 6),
            "lowest_neural_ops_reaching_target_signed_gain": round(lowest_ops_gain or 0.0, 6),
        }
    return frontier


def run_neural_profile_sweep(
    profiles: list[dict] | tuple[dict, ...] = DEFAULT_PROFILES,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    conditions: list[str] | tuple[str, ...] = DEFAULT_NEURAL_CONDITIONS,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    material_count: int = 64,
    feature_dimension: int = 128,
    learning_rate: float = 0.03,
    target_signed_gain: float = 0.03,
    confirmation_of: str | None = None,
    fresh_seed_confirmation: bool = False,
) -> dict:
    normalized_profiles = _normalize_profiles(profiles)
    profile_results: dict[str, dict] = {}
    for profile in normalized_profiles:
        profile_id = _profile_id(profile)
        result = run_neural_seedset(
            seeds=seeds,
            conditions=conditions,
            material_count=material_count,
            epochs=profile["epochs"],
            hidden_units=profile["hidden_units"],
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
            target_signed_gain=target_signed_gain,
            confirmation_of=confirmation_of,
            fresh_seed_confirmation=fresh_seed_confirmation,
        )
        profile_results[profile_id] = {
            "profile": profile,
            "conditions": result["conditions"],
        }

    result = {
        "title": "Learning Signal Density Tiny Neural Profile Sweep",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "learner_backend": "tiny_mlp",
        "material_count": material_count,
        "seeds": list(seeds),
        "conditions": list(conditions),
        "profiles": normalized_profiles,
        "feature_dimension": feature_dimension,
        "learning_rate": learning_rate,
        "target_signed_gain": target_signed_gain,
        "claim_scope": {
            "synthetic_domain": True,
            "neural_model": True,
            "heldout_used_for_selection": False,
            "paper_ready_claim": False,
            "fresh_seed_confirmation": fresh_seed_confirmation,
        },
        "frontier": _frontier_summary(profile_results, conditions, target_signed_gain),
        "profile_results": profile_results,
    }
    if confirmation_of is not None:
        result["confirmation_of"] = confirmation_of

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_neural_profile_markdown(result))
    return result


def render_neural_profile_markdown(result: dict) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "This sweep reruns the deterministic CPU tiny-MLP profile grid at one external sample budget.",
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
            f"Material count: `{result['material_count']}`",
            f"Feature dimension: `{result['feature_dimension']}`",
            f"Learning rate: `{result['learning_rate']}`",
            f"Target signed gain over majority: `{result['target_signed_gain']}`",
            "",
            "| Condition | Best gain profile | Best signed gain | Best LSD profile | Best signed LSD/1M | Lowest-op target profile | Target-profile ops |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in result["conditions"]:
        summary = result["frontier"][condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(summary["best_signed_gain_profile"]),
                    f"{summary['best_signed_gain']:.3f}",
                    str(summary["best_signed_lsd_profile"]),
                    f"{summary['best_signed_learning_signal_density_per_1m_event_compute']:.6f}",
                    str(summary["lowest_neural_ops_reaching_target_profile"]),
                    f"{summary['lowest_neural_ops_reaching_target']:.0f}",
                ]
            )
            + " |"
        )

    for profile in result["profiles"]:
        profile_id = _profile_id(profile)
        lines.extend(
            [
                "",
                f"## Profile {profile_id}",
                "",
                "| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in result["conditions"]:
            stats = result["profile_results"][profile_id]["conditions"][condition]
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


def _parse_profile(value: str) -> dict:
    parts = value.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("profiles must use EPOCHSxHIDDEN_UNITS format")
    return {"epochs": int(parts[0]), "hidden_units": int(parts[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny neural learning-signal density profile sweep.")
    parser.add_argument("--output-json", type=Path, default=Path("results/tiny_neural_profile_sweep.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/tiny_neural_profile_sweep.md"))
    parser.add_argument("--profiles", nargs="+", type=_parse_profile, default=list(DEFAULT_PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_NEURAL_CONDITIONS))
    parser.add_argument("--material-count", type=int, default=64)
    parser.add_argument("--feature-dimension", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--target-signed-gain", type=float, default=0.03)
    parser.add_argument("--confirmation-of", default=None)
    parser.add_argument("--fresh-seed-confirmation", action="store_true")
    args = parser.parse_args()
    run_neural_profile_sweep(
        profiles=args.profiles,
        seeds=args.seeds,
        conditions=args.conditions,
        output_json=args.output_json,
        output_markdown=args.output_md,
        material_count=args.material_count,
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
