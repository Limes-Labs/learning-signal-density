#!/usr/bin/env python3
"""Build a cross-artifact break-even certificate for real-text audits."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from learning_signal_density.break_even import break_even_comparison


SMS_BREAK_EVEN_ARTIFACT = Path("results/sms_spam_break_even_analysis.json")
TWENTY_NEWSGROUPS_ACTIVE_ARTIFACT = Path("results/twenty_newsgroups_active_selection.json")
TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT = Path("results/twenty_newsgroups_break_even_analysis.json")
TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_retrieval_cost_audit.json"
)
TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT = Path(
    "results/twenty_newsgroups_self_training_audit.json"
)
SELF_TRAINING_REUSABLE_KEYS = (
    "selection_cost_tokens_mean",
    "teacher_training_cost_tokens_mean",
    "pseudo_scoring_cost_tokens_mean",
)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _regime(row: dict) -> str:
    if row["candidate_density_wins"]:
        return "observed_density_win"
    if row["candidate_quality"] <= row["reference_quality"]:
        return "quality_not_improved"
    if row.get("amortized_reuses_to_density_win") is not None:
        return "finite_reuse_needed"
    if row.get("perfect_quality_can_beat") is False:
        return "bounded_quality_impossible_at_k1"
    if (row.get("fully_amortized_density_ratio") or 0.0) > 1.0:
        return "asymptotic_reuse_only"
    return "nonreusable_cost_too_high"


def _copy_row(
    *,
    family: str,
    artifact_label: str,
    source_artifact: str,
    budget: str,
    reference: str,
    candidate: str,
    row: dict,
    note: str,
) -> dict:
    regime = _regime(row)
    return {
        "family": family,
        "artifact_label": artifact_label,
        "source_artifact": source_artifact,
        "budget": budget,
        "reference_condition": reference,
        "candidate_condition": candidate,
        "regime": regime,
        "note": note,
        "reference_quality": row["reference_quality"],
        "candidate_quality": row["candidate_quality"],
        "quality_multiplier": row["quality_multiplier"],
        "event_compute_multiplier": row["event_compute_multiplier"],
        "density_ratio": row["density_ratio"],
        "break_even_quality": row["break_even_quality"],
        "compute_over_break_even": row["compute_over_break_even"],
        "perfect_quality_can_beat": row.get("perfect_quality_can_beat"),
        "amortized_reuses_to_density_win": row.get("amortized_reuses_to_density_win"),
        "fully_amortized_density_ratio": row.get("fully_amortized_density_ratio"),
        "candidate_density_wins": row["candidate_density_wins"],
        "observed_quality_wins": row["candidate_quality"] > row["reference_quality"],
    }


def _add_newsgroups_break_even_rows(rows: list[dict], artifact: dict) -> None:
    for budget, comparisons in artifact["comparisons"].items():
        for candidate, row in comparisons.items():
            rows.append(
                _copy_row(
                    family="twenty_newsgroups_active_selection",
                    artifact_label="Twenty Newsgroups active",
                    source_artifact=str(TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT),
                    budget=budget,
                    reference=artifact["reference_condition"],
                    candidate=candidate,
                    row=row,
                    note="active selection against random sampling",
                )
            )


def _add_sms_break_even_rows(rows: list[dict], artifact: dict) -> None:
    for artifact_label, budgets in artifact["comparisons"].items():
        for budget, comparisons in budgets.items():
            for candidate, row in comparisons.items():
                rows.append(
                    _copy_row(
                        family="sms_spam_selection",
                        artifact_label=artifact_label,
                        source_artifact=str(SMS_BREAK_EVEN_ARTIFACT),
                        budget=budget,
                        reference=artifact["reference_condition"],
                        candidate=candidate,
                        row=row,
                        note="SMS selector-cost break-even",
                    )
                )


def _add_retrieval_cost_rows(rows: list[dict], artifact: dict) -> None:
    for budget in artifact["train_budgets"]:
        budget_key = str(budget)
        for alpha, alpha_row in artifact["budgets"][budget_key]["alpha_results"].items():
            row = alpha_row["break_even_vs_random"]
            rows.append(
                _copy_row(
                    family="twenty_newsgroups_retrieval_alpha",
                    artifact_label="Twenty Newsgroups retrieval alpha",
                    source_artifact=str(TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT),
                    budget=budget_key,
                    reference=row["reference_condition"],
                    candidate=f"prototype_retrieval_alpha_{alpha}",
                    row=row,
                    note="length-penalized prototype retrieval against random sampling",
                )
            )


def _add_self_training_rows(rows: list[dict], active: dict, artifact: dict) -> None:
    for budget in artifact["train_budgets"]:
        budget_key = str(budget)
        references = {
            "random_sample": active["budgets"][budget_key]["conditions"]["random_sample"],
            "class_balanced_sample": active["budgets"][budget_key]["conditions"]["class_balanced_sample"],
        }
        for reference_name, reference_stats in references.items():
            reference = {"condition": reference_name} | reference_stats
            for condition, stats in artifact["budgets"][budget_key]["condition_results"].items():
                candidate = {"condition": condition} | stats
                comparison = break_even_comparison(
                    reference=reference,
                    candidate=candidate,
                    quality_key="accuracy_improvement_over_majority_mean",
                    quality_upper_bound=0.95,
                    reusable_compute_keys=SELF_TRAINING_REUSABLE_KEYS,
                )
                rows.append(
                    _copy_row(
                        family="twenty_newsgroups_self_training",
                        artifact_label="Twenty Newsgroups self-training",
                        source_artifact=str(TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT),
                        budget=budget_key,
                        reference=reference_name,
                        candidate=condition,
                        row=comparison,
                        note="pseudo-label self-training against real-label sampling",
                    )
                )


def _summarize_family(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "observed_quality_wins": sum(1 for row in rows if row["observed_quality_wins"]),
        "density_wins": sum(1 for row in rows if row["candidate_density_wins"]),
        "quality_win_density_losses": sum(
            1 for row in rows if row["observed_quality_wins"] and not row["candidate_density_wins"]
        ),
        "bounded_quality_impossible_at_k1": sum(1 for row in rows if row.get("perfect_quality_can_beat") is False),
        "finite_reuse_needed": sum(1 for row in rows if row["regime"] == "finite_reuse_needed"),
        "nonreusable_cost_too_high": sum(1 for row in rows if row["regime"] == "nonreusable_cost_too_high"),
        "mean_density_ratio": _round(
            sum(row["density_ratio"] for row in rows if row["density_ratio"] is not None) / max(1, len(rows))
        ),
    }


def _best(rows: list[dict], key: str, predicate) -> dict | None:
    candidates = [row for row in rows if predicate(row) and row.get(key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[key])


def _smallest(rows: list[dict], key: str, predicate) -> dict | None:
    candidates = [row for row in rows if predicate(row) and row.get(key) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row[key])


def build_real_text_break_even_certificate(repo_root: Path) -> dict:
    active = load_json(repo_root / TWENTY_NEWSGROUPS_ACTIVE_ARTIFACT)
    newsgroups_break_even = load_json(repo_root / TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT)
    retrieval_cost = load_json(repo_root / TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT)
    self_training = load_json(repo_root / TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT)
    sms_break_even = load_json(repo_root / SMS_BREAK_EVEN_ARTIFACT)

    rows: list[dict] = []
    _add_newsgroups_break_even_rows(rows, newsgroups_break_even)
    _add_retrieval_cost_rows(rows, retrieval_cost)
    _add_self_training_rows(rows, active, self_training)
    _add_sms_break_even_rows(rows, sms_break_even)

    families = sorted({row["family"] for row in rows})
    by_family = {family: _summarize_family([row for row in rows if row["family"] == family]) for family in families}
    summary = _summarize_family(rows)
    summary["families"] = by_family
    summary["strongest_observed_density_win"] = _best(rows, "density_ratio", lambda row: row["candidate_density_wins"])
    summary["largest_quality_win_without_density_win"] = _best(
        rows,
        "quality_multiplier",
        lambda row: row["observed_quality_wins"] and not row["candidate_density_wins"],
    )
    summary["cheapest_finite_reuse_frontier"] = _smallest(
        rows,
        "amortized_reuses_to_density_win",
        lambda row: row["regime"] == "finite_reuse_needed",
    )
    summary["worst_cost_over_break_even"] = _best(
        rows,
        "compute_over_break_even",
        lambda row: row["observed_quality_wins"] and not row["candidate_density_wins"],
    )

    return {
        "title": "Real-Text Break-Even Frontier Certificate",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_artifacts": [
            str(TWENTY_NEWSGROUPS_ACTIVE_ARTIFACT),
            str(TWENTY_NEWSGROUPS_BREAK_EVEN_ARTIFACT),
            str(TWENTY_NEWSGROUPS_RETRIEVAL_COST_AUDIT_ARTIFACT),
            str(TWENTY_NEWSGROUPS_SELF_TRAINING_AUDIT_ARTIFACT),
            str(SMS_BREAK_EVEN_ARTIFACT),
        ],
        "theorem": {
            "name": "cross-artifact break-even frontier",
            "density_condition": "G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)",
            "interpretation": (
                "Observed quality wins are not sufficient: a candidate must clear the event-compute "
                "multiplier, or have enough explicitly reusable selector work, to improve learning-signal density."
            ),
        },
        "claim_scope": {
            "real_dataset": True,
            "synthetic_domain": False,
            "mathematical_certificate": True,
            "post_hoc_diagnostic": True,
            "introduces_new_policy": False,
            "heldout_used_for_selection": False,
            "paper_ready_claim": False,
        },
        "summary": summary,
        "rows": rows,
    }


def _fmt_optional(value: int | float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _row_id(row: dict | None) -> str:
    if row is None:
        return "n/a"
    return f"{row['artifact_label']} b={row['budget']} {row['candidate_condition']} vs {row['reference_condition']}"


def render_markdown(artifact: dict) -> str:
    summary = artifact["summary"]
    lines = [
        f"# {artifact['title']}",
        "",
        f"Generated: `{artifact['generated_at']}`",
        "",
        "This is a mathematical certificate over committed real-text result artifacts.",
        "It introduces no new policy; it classifies whether observed quality gains clear the charged event-compute break-even condition.",
        "",
        "Break-even condition:",
        "",
        "```text",
        artifact["theorem"]["density_condition"],
        "```",
        "",
        "| Family | Rows | Quality wins | Density wins | Q-win density losses | Finite reuse | Nonreusable too high | K=1 bound impossible | Mean density ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, family_summary in summary["families"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(family_summary["rows"]),
                    str(family_summary["observed_quality_wins"]),
                    str(family_summary["density_wins"]),
                    str(family_summary["quality_win_density_losses"]),
                    str(family_summary["finite_reuse_needed"]),
                    str(family_summary["nonreusable_cost_too_high"]),
                    str(family_summary["bounded_quality_impossible_at_k1"]),
                    f"{family_summary['mean_density_ratio']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Frontier Witnesses",
            "",
            f"- Strongest observed density win: `{_row_id(summary['strongest_observed_density_win'])}`.",
            f"- Largest quality win without density win: `{_row_id(summary['largest_quality_win_without_density_win'])}`.",
            f"- Cheapest finite reuse frontier: `{_row_id(summary['cheapest_finite_reuse_frontier'])}`.",
            f"- Worst cost over break-even among quality wins: `{_row_id(summary['worst_cost_over_break_even'])}`.",
            "",
            "## Scope Flags",
            "",
            "```json",
            json.dumps(artifact["claim_scope"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/real_text_break_even_certificate.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/real_text_break_even_certificate.md"),
    )
    args = parser.parse_args()
    artifact = build_real_text_break_even_certificate(Path("."))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
