from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import mean

from .domain import Observation, build_world, split_observations
from .experiment import DEFAULT_SEEDS, _pipeline_compute_units, _round
from .induction import fit_induced_rule_model
from .learner import PerceptronClassifier, majority_baseline
from .neural import TinyMlpClassifier
from .pipelines import (
    CONDITION_SCOPE,
    PipelineExamples,
    TrainingExample,
    _support_ramped_compact_ranked_policy,
    build_evaluation_examples,
    build_pipeline_examples,
    raw_observation_example,
)


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
VALIDATION_ABSTAINING_PROXY_SELECTOR = "validation_abstaining_proxy_selector"
VALIDATION_COVERAGE_PROXY_SELECTOR = "validation_coverage_proxy_selector"
VALIDATION_COVERAGE_PRIOR_SELECTOR = "validation_coverage_prior_selector"
TRAIN_SUPPORT_DENSITY_SELECTOR = "train_support_density_selector"
SUPPORT_PROBE_WINDOW_SELECTOR = "support_probe_window_selector"
VALIDATION_SUPPORT_PRECISION_SELECTOR = "validation_support_precision_selector"
VALIDATION_SUPPORT_PRECISION_GATE_SELECTOR = "validation_support_precision_gate_selector"
VALIDATION_SUPPORT_UTILITY_SELECTOR = "validation_support_utility_selector"
VALIDATION_SUPPORT_GAIN_GATE_SELECTOR = "validation_support_gain_gate_selector"
VALIDATION_PORTFOLIO_CANDIDATES = (
    "raw_text",
    "self_ranked_induction",
    "sample_aware_self_ranked_induction",
    "agreement_gated_self_ranked_induction",
    "validation_ranked_induction",
    "mdl_rule_expansion",
)
VALIDATION_COVERAGE_PRIOR_CANDIDATES = (
    "raw_text",
    "sample_aware_self_ranked_induction",
    "validation_ranked_induction",
)
TRAIN_SUPPORT_DENSITY_CANDIDATES = (
    "raw_text",
    "compact_train_size_gated_induction",
    "support_ramped_compact_induction",
)
VALIDATION_LINEAR_PROXY_EPOCHS = 2
VALIDATION_ABSTAINING_PROXY_EXTRA_CORRECT = 3
VALIDATION_COVERAGE_PAIR_BONUS = 0.15
VALIDATION_COVERAGE_MIN_PAIR_COVERAGE = 0.5
VALIDATION_COVERAGE_PRIOR_MIN_TRAIN_EVENTS = 96
VALIDATION_COVERAGE_PRIOR_COMPUTE_PENALTY = 0.00001
TRAIN_SUPPORT_DENSITY_COMPACT_MAX_EVENTS = 320
TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE = 0.00145
SUPPORT_PROBE_RAW_MIN_EVENTS = 320
SUPPORT_PROBE_MIN_EVENTS = 360
SUPPORT_PROBE_MAX_EVENTS = 432
VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS = 320
VALIDATION_SUPPORT_PRECISION_TRANSITION_MIN_EVENTS = 400
VALIDATION_SUPPORT_PRECISION_TRANSITION_MAX_EVENTS = 432
VALIDATION_SUPPORT_PRECISION_THRESHOLD = 0.825758
VALIDATION_SUPPORT_UTILITY_PAIR_COVERAGE_WEIGHT = 0.25
VALIDATION_SUPPORT_UTILITY_TRIPLE_L1_WEIGHT = 0.20
VALIDATION_SUPPORT_UTILITY_COMPUTE_PENALTY = 0.000001
VALIDATION_SUPPORT_UTILITY_MIN_SCORE = 0.0
VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS = 2
VALIDATION_SUPPORT_GAIN_COMPUTE_PENALTY = 0.0000005
VALIDATION_SUPPORT_GAIN_MIN_SCORE = 0.0
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
FIELD_PATTERN = re.compile(r"\b(family|modifier|stimulus)=([A-Za-z0-9_]+)")
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
VALIDATION_ABSTAINING_PROXY_SCOPE = {
    **VALIDATION_LINEAR_PROXY_SCOPE,
    "raw_text_abstention": True,
    "validation_abstention_extra_correct": VALIDATION_ABSTAINING_PROXY_EXTRA_CORRECT,
}
VALIDATION_COVERAGE_PROXY_SCOPE = {
    **VALIDATION_PORTFOLIO_SCOPE,
    "low_fidelity_coverage_proxy_selector": True,
    "validation_motif_distribution_used_for_policy_selection": True,
    "validation_labels_used_for_policy_selection": False,
}
VALIDATION_COVERAGE_PRIOR_SCOPE = {
    **VALIDATION_COVERAGE_PROXY_SCOPE,
    "train_size_prior": True,
    "train_size_prior_min_events": VALIDATION_COVERAGE_PRIOR_MIN_TRAIN_EVENTS,
    "lean_coverage_candidate_set": True,
    "coverage_utility_compute_penalty": VALIDATION_COVERAGE_PRIOR_COMPUTE_PENALTY,
}
TRAIN_SUPPORT_DENSITY_SCOPE = {
    "oracle_generated_labels": False,
    "train_only_selection": True,
    "train_only_induction": True,
    "validation_used_for_threshold": False,
    "validation_used_for_transform_selection": False,
    "validation_used_for_policy_selection": False,
    "compact_original_encoding_at_large_samples": True,
    "support_density_selector": True,
    "support_density_min_kept_per_compute": TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE,
    "support_density_compact_max_events": TRAIN_SUPPORT_DENSITY_COMPACT_MAX_EVENTS,
}
SUPPORT_PROBE_WINDOW_SCOPE = {
    "oracle_generated_labels": False,
    "train_only_selection": True,
    "train_only_induction": True,
    "validation_used_for_threshold": False,
    "validation_used_for_transform_selection": False,
    "validation_used_for_policy_selection": False,
    "compact_original_encoding_at_large_samples": True,
    "support_probe_window_selector": True,
    "support_probe_raw_min_train_events": SUPPORT_PROBE_RAW_MIN_EVENTS,
    "support_probe_min_train_events": SUPPORT_PROBE_MIN_EVENTS,
    "support_probe_max_train_events": SUPPORT_PROBE_MAX_EVENTS,
    "support_density_min_kept_per_compute": TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE,
    "reuse_selected_candidate_construction": True,
}
VALIDATION_SUPPORT_PRECISION_SCOPE = {
    "oracle_generated_labels": False,
    "train_only_selection": False,
    "train_only_induction": True,
    "validation_used_for_threshold": True,
    "validation_used_for_transform_selection": True,
    "validation_used_for_policy_selection": True,
    "compact_original_encoding_at_large_samples": True,
    "validation_support_precision_selector": True,
    "validation_support_precision_threshold": VALIDATION_SUPPORT_PRECISION_THRESHOLD,
    "validation_support_compact_max_train_events": VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS,
    "validation_support_transition_min_train_events": VALIDATION_SUPPORT_PRECISION_TRANSITION_MIN_EVENTS,
    "validation_support_transition_max_train_events": VALIDATION_SUPPORT_PRECISION_TRANSITION_MAX_EVENTS,
    "support_density_min_kept_per_compute": TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE,
    "reuse_selected_candidate_construction": True,
}
VALIDATION_SUPPORT_PRECISION_GATE_SCOPE = {
    **VALIDATION_SUPPORT_PRECISION_SCOPE,
    "validation_support_precision_selector": False,
    "validation_support_precision_gate_selector": True,
    "validation_support_uses_fixed_transition_prior": False,
}
VALIDATION_SUPPORT_UTILITY_SCOPE = {
    **VALIDATION_SUPPORT_PRECISION_SCOPE,
    "validation_support_precision_selector": False,
    "validation_support_utility_selector": True,
    "validation_support_uses_fixed_transition_prior": False,
    "validation_labels_used_for_policy_selection": True,
    "validation_motif_distribution_used_for_policy_selection": True,
    "validation_support_utility_min_score": VALIDATION_SUPPORT_UTILITY_MIN_SCORE,
    "validation_support_utility_pair_coverage_weight": VALIDATION_SUPPORT_UTILITY_PAIR_COVERAGE_WEIGHT,
    "validation_support_utility_triple_l1_weight": VALIDATION_SUPPORT_UTILITY_TRIPLE_L1_WEIGHT,
    "validation_support_utility_compute_penalty": VALIDATION_SUPPORT_UTILITY_COMPUTE_PENALTY,
}
VALIDATION_SUPPORT_GAIN_GATE_SCOPE = {
    "oracle_generated_labels": False,
    "train_only_selection": False,
    "train_only_induction": True,
    "validation_used_for_threshold": True,
    "validation_used_for_transform_selection": True,
    "validation_used_for_policy_selection": True,
    "compact_original_encoding_at_large_samples": True,
    "validation_labels_used_for_policy_selection": True,
    "low_fidelity_proxy_selector": True,
    "validation_support_gain_gate_selector": True,
    "validation_support_gain_precision_prefilter": True,
    "validation_support_precision_threshold": VALIDATION_SUPPORT_PRECISION_THRESHOLD,
    "validation_support_gain_proxy_epochs": VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS,
    "validation_support_gain_min_score": VALIDATION_SUPPORT_GAIN_MIN_SCORE,
    "validation_support_gain_compute_penalty": VALIDATION_SUPPORT_GAIN_COMPUTE_PENALTY,
    "validation_support_compact_max_train_events": VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS,
    "reuse_selected_candidate_construction": True,
}
VALIDATION_SELECTOR_CONDITIONS = frozenset({
    VALIDATION_ABSTAINING_PROXY_SELECTOR,
    VALIDATION_COVERAGE_PRIOR_SELECTOR,
    VALIDATION_COVERAGE_PROXY_SELECTOR,
    VALIDATION_PORTFOLIO_SELECTOR,
    VALIDATION_LINEAR_PROXY_SELECTOR,
    TRAIN_SUPPORT_DENSITY_SELECTOR,
    SUPPORT_PROBE_WINDOW_SELECTOR,
    VALIDATION_SUPPORT_PRECISION_SELECTOR,
    VALIDATION_SUPPORT_PRECISION_GATE_SELECTOR,
    VALIDATION_SUPPORT_UTILITY_SELECTOR,
    VALIDATION_SUPPORT_GAIN_GATE_SELECTOR,
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
        "portfolio_abstention_extra_correct",
        "portfolio_abstention_margin",
        "portfolio_raw_text_abstention",
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
    if condition == VALIDATION_ABSTAINING_PROXY_SELECTOR:
        return VALIDATION_ABSTAINING_PROXY_SCOPE
    if condition == VALIDATION_COVERAGE_PROXY_SELECTOR:
        return VALIDATION_COVERAGE_PROXY_SCOPE
    if condition == VALIDATION_COVERAGE_PRIOR_SELECTOR:
        return VALIDATION_COVERAGE_PRIOR_SCOPE
    if condition == TRAIN_SUPPORT_DENSITY_SELECTOR:
        return TRAIN_SUPPORT_DENSITY_SCOPE
    if condition == SUPPORT_PROBE_WINDOW_SELECTOR:
        return SUPPORT_PROBE_WINDOW_SCOPE
    if condition == VALIDATION_SUPPORT_PRECISION_SELECTOR:
        return VALIDATION_SUPPORT_PRECISION_SCOPE
    if condition == VALIDATION_SUPPORT_PRECISION_GATE_SELECTOR:
        return VALIDATION_SUPPORT_PRECISION_GATE_SCOPE
    if condition == VALIDATION_SUPPORT_UTILITY_SELECTOR:
        return VALIDATION_SUPPORT_UTILITY_SCOPE
    if condition == VALIDATION_SUPPORT_GAIN_GATE_SELECTOR:
        return VALIDATION_SUPPORT_GAIN_GATE_SCOPE
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
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": 0.0,
        "portfolio_raw_text_abstention": 0,
        "portfolio_candidate_summaries": [],
    }


def _example_motif(example: TrainingExample) -> tuple[str, str, str]:
    fields = dict(FIELD_PATTERN.findall(example.text))
    missing = sorted({"family", "stimulus", "modifier"} - set(fields))
    if missing:
        raise ValueError(f"example is missing fields {missing}: {example.text}")
    return fields["family"], fields["stimulus"], fields["modifier"]


def _generated_motif_counts(pipeline: PipelineExamples) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for example in pipeline.examples:
        if example.source_kind in GENERATED_SOURCE_KINDS:
            counts[_example_motif(example)] += 1
    return counts


def _observation_motif_counts(observations: tuple[Observation, ...]) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for observation in observations:
        counts[(observation.family, observation.stimulus, observation.modifier)] += 1
    return counts


def _observation_pair_counts(observations: tuple[Observation, ...]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for observation in observations:
        counts[(observation.family, observation.stimulus)] += 1
    return counts


def _pair_counts_from_motifs(motifs: Counter[tuple[str, str, str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for (family, stimulus, _modifier), count in motifs.items():
        counts[(family, stimulus)] += count
    return counts


def _distribution_l1_distance(
    generated_counts: Counter[tuple[str, str, str]],
    target_counts: Counter[tuple[str, str, str]],
) -> float | None:
    generated_total = sum(generated_counts.values())
    target_total = sum(target_counts.values())
    if generated_total == 0 or target_total == 0:
        return None
    keys = set(generated_counts) | set(target_counts)
    return sum(
        abs(generated_counts.get(key, 0) / generated_total - target_counts.get(key, 0) / target_total)
        for key in keys
    ) / 2


def _candidate_coverage_record(
    condition: str,
    pipeline: PipelineExamples,
    validation_motifs: Counter[tuple[str, str, str]],
    validation_pairs: Counter[tuple[str, str]],
    validation_eval_cost: int,
) -> dict:
    generated_motifs = _generated_motif_counts(pipeline)
    generated_pairs = _pair_counts_from_motifs(generated_motifs)
    triple_l1 = _distribution_l1_distance(generated_motifs, validation_motifs)
    validation_pair_total = sum(validation_pairs.values())
    pair_coverage = (
        sum(count for pair, count in validation_pairs.items() if pair in generated_pairs)
        / validation_pair_total
        if validation_pair_total
        else 0.0
    )
    synthetic_example_count = sum(generated_motifs.values())
    score = (triple_l1 if triple_l1 is not None else 1.0) - (
        VALIDATION_COVERAGE_PAIR_BONUS * pair_coverage
    )
    return {
        "condition": condition,
        "pipeline": pipeline,
        "coverage_score": score,
        "coverage_triple_l1_distance": triple_l1,
        "coverage_pair_coverage": pair_coverage,
        "synthetic_example_count": synthetic_example_count,
        "candidate_compute_units": _pipeline_compute_units(pipeline=pipeline, epochs=0)
        + validation_eval_cost,
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
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": 0.0,
        "portfolio_raw_text_abstention": 0,
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
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": 0.0,
        "portfolio_raw_text_abstention": 0,
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


def _run_validation_abstaining_proxy_selector(
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

    raw = next(record for record in candidate_records if record["condition"] == "raw_text")
    best_non_raw = min(
        (record for record in candidate_records if record["condition"] != "raw_text"),
        key=lambda record: (
            -record["validation_score"],
            record["candidate_compute_units"],
            VALIDATION_PORTFOLIO_CANDIDATES.index(record["condition"]),
        ),
    )
    abstention_margin = VALIDATION_ABSTAINING_PROXY_EXTRA_CORRECT / max(1, len(validation_examples))
    raw_text_abstention = best_non_raw["validation_score"] < raw["validation_score"] + abstention_margin
    best = raw if raw_text_abstention else best_non_raw

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
        "condition": VALIDATION_ABSTAINING_PROXY_SELECTOR,
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
        "portfolio_selection_metric": (
            "linear_proxy_validation_accuracy_with_raw_text_abstention_"
            f"{VALIDATION_ABSTAINING_PROXY_EXTRA_CORRECT}_extra_correct"
        ),
        "portfolio_proxy_epochs": VALIDATION_LINEAR_PROXY_EPOCHS,
        "portfolio_abstention_extra_correct": VALIDATION_ABSTAINING_PROXY_EXTRA_CORRECT,
        "portfolio_abstention_margin": _round(abstention_margin),
        "portfolio_raw_text_abstention": 1 if raw_text_abstention else 0,
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


def _run_validation_coverage_proxy_selector(
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
    validation_motifs = _observation_motif_counts(split.validation)
    validation_pairs = _observation_pair_counts(split.validation)

    candidate_records: list[dict] = []
    for candidate_condition in VALIDATION_PORTFOLIO_CANDIDATES:
        pipeline = _build_neural_pipeline(candidate_condition, split, world.rules)
        candidate_records.append(
            _candidate_coverage_record(
                condition=candidate_condition,
                pipeline=pipeline,
                validation_motifs=validation_motifs,
                validation_pairs=validation_pairs,
                validation_eval_cost=validation_eval_cost,
            )
        )

    raw = next(record for record in candidate_records if record["condition"] == "raw_text")
    eligible_non_raw = [
        record
        for record in candidate_records
        if record["condition"] != "raw_text" and record["synthetic_example_count"] > 0
    ]
    best_non_raw = min(
        eligible_non_raw,
        key=lambda record: (
            record["coverage_score"],
            record["coverage_triple_l1_distance"]
            if record["coverage_triple_l1_distance"] is not None
            else 1.0,
            _pipeline_compute_units(pipeline=record["pipeline"], epochs=epochs),
            VALIDATION_PORTFOLIO_CANDIDATES.index(record["condition"]),
        ),
    )
    raw_text_abstention = best_non_raw["coverage_pair_coverage"] < VALIDATION_COVERAGE_MIN_PAIR_COVERAGE
    best = raw if raw_text_abstention else best_non_raw

    pipeline = best["pipeline"]
    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    coverage_selection_cost_units = sum(record["candidate_compute_units"] for record in candidate_records)
    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + coverage_selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": VALIDATION_COVERAGE_PROXY_SELECTOR,
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
        "portfolio_selection_cost_units": coverage_selection_cost_units,
        "portfolio_validation_score": _round(best_non_raw["coverage_score"]),
        "portfolio_selected_condition": best["condition"],
        "portfolio_candidate_conditions": list(VALIDATION_PORTFOLIO_CANDIDATES),
        "portfolio_selection_metric": "validation_motif_coverage_l1_with_pair_coverage_bonus",
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(VALIDATION_COVERAGE_MIN_PAIR_COVERAGE),
        "portfolio_raw_text_abstention": 1 if raw_text_abstention else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "coverage_score": _round(record["coverage_score"]),
                "coverage_triple_l1_distance": _round(record["coverage_triple_l1_distance"])
                if record["coverage_triple_l1_distance"] is not None
                else None,
                "coverage_pair_coverage": _round(record["coverage_pair_coverage"]),
                "synthetic_example_count": record["synthetic_example_count"],
                "candidate_compute_units": record["candidate_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _run_validation_coverage_prior_selector(
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

    if len(split.train) < VALIDATION_COVERAGE_PRIOR_MIN_TRAIN_EVENTS:
        candidate_records: list[dict] = []
        pipeline = _build_neural_pipeline("raw_text", split, world.rules)
        selected_condition = "raw_text"
        selection_cost_units = 0
        selection_score = 0.0
        raw_text_abstention = 1
    else:
        validation_motifs = _observation_motif_counts(split.validation)
        validation_pairs = _observation_pair_counts(split.validation)
        candidate_records = []
        for candidate_condition in VALIDATION_COVERAGE_PRIOR_CANDIDATES:
            pipeline = _build_neural_pipeline(candidate_condition, split, world.rules)
            record = _candidate_coverage_record(
                condition=candidate_condition,
                pipeline=pipeline,
                validation_motifs=validation_motifs,
                validation_pairs=validation_pairs,
                validation_eval_cost=validation_eval_cost,
            )
            final_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs)
            record["final_compute_units"] = final_compute_units
            record["coverage_utility_score"] = (
                record["coverage_score"]
                + VALIDATION_COVERAGE_PRIOR_COMPUTE_PENALTY * final_compute_units
            )
            candidate_records.append(record)

        raw = next(record for record in candidate_records if record["condition"] == "raw_text")
        eligible_non_raw = [
            record
            for record in candidate_records
            if record["condition"] != "raw_text" and record["synthetic_example_count"] > 0
        ]
        best_non_raw = min(
            eligible_non_raw,
            key=lambda record: (
                record["coverage_utility_score"],
                record["coverage_score"],
                record["coverage_triple_l1_distance"]
                if record["coverage_triple_l1_distance"] is not None
                else 1.0,
                record["final_compute_units"],
                VALIDATION_COVERAGE_PRIOR_CANDIDATES.index(record["condition"]),
            ),
        )
        raw_text_abstention = (
            1
            if best_non_raw["coverage_pair_coverage"] < VALIDATION_COVERAGE_MIN_PAIR_COVERAGE
            else 0
        )
        best = raw if raw_text_abstention else best_non_raw
        pipeline = best["pipeline"]
        selected_condition = best["condition"]
        selection_cost_units = sum(record["candidate_compute_units"] for record in candidate_records)
        selection_score = best_non_raw["coverage_utility_score"]

    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": VALIDATION_COVERAGE_PRIOR_SELECTOR,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(selection_score),
        "portfolio_selected_condition": selected_condition,
        "portfolio_candidate_conditions": list(VALIDATION_COVERAGE_PRIOR_CANDIDATES),
        "portfolio_selection_metric": (
            "validation_motif_coverage_l1_with_train_size_prior_"
            f"{VALIDATION_COVERAGE_PRIOR_MIN_TRAIN_EVENTS}_and_lean_cost_penalty"
        ),
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(VALIDATION_COVERAGE_MIN_PAIR_COVERAGE),
        "portfolio_raw_text_abstention": raw_text_abstention,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "coverage_score": _round(record["coverage_score"]),
                "coverage_utility_score": _round(record["coverage_utility_score"]),
                "coverage_triple_l1_distance": _round(record["coverage_triple_l1_distance"])
                if record["coverage_triple_l1_distance"] is not None
                else None,
                "coverage_pair_coverage": _round(record["coverage_pair_coverage"]),
                "synthetic_example_count": record["synthetic_example_count"],
                "candidate_compute_units": record["candidate_compute_units"],
                "final_compute_units": record["final_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _run_train_support_density_selector(
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

    candidate_records: list[dict] = []
    for candidate_condition in TRAIN_SUPPORT_DENSITY_CANDIDATES:
        pipeline = _build_neural_pipeline(candidate_condition, split, world.rules)
        final_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs)
        support_density = (
            pipeline.ranked_kept_candidate_count / max(1, final_compute_units)
            if candidate_condition == "support_ramped_compact_induction"
            else 0.0
        )
        candidate_records.append(
            {
                "condition": candidate_condition,
                "pipeline": pipeline,
                "candidate_compute_units": _pipeline_compute_units(pipeline=pipeline, epochs=0),
                "final_compute_units": final_compute_units,
                "support_density": support_density,
                "ranked_kept_candidate_count": pipeline.ranked_kept_candidate_count,
                "ranked_candidate_count": pipeline.ranked_candidate_count,
            }
        )

    raw = next(record for record in candidate_records if record["condition"] == "raw_text")
    compact = next(
        record
        for record in candidate_records
        if record["condition"] == "compact_train_size_gated_induction"
    )
    support = next(
        record
        for record in candidate_records
        if record["condition"] == "support_ramped_compact_induction"
    )

    if len(split.train) < TRAIN_SUPPORT_DENSITY_COMPACT_MAX_EVENTS:
        best = compact
    elif support["support_density"] >= TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE:
        best = support
    else:
        best = raw

    pipeline = best["pipeline"]
    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    selection_cost_units = sum(record["candidate_compute_units"] for record in candidate_records)
    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": TRAIN_SUPPORT_DENSITY_SELECTOR,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(support["support_density"]),
        "portfolio_selected_condition": best["condition"],
        "portfolio_candidate_conditions": list(TRAIN_SUPPORT_DENSITY_CANDIDATES),
        "portfolio_selection_metric": (
            "train_support_density_min_"
            f"{TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE}_kept_per_compute"
        ),
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE),
        "portfolio_raw_text_abstention": 1 if best["condition"] == "raw_text" else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "support_density": _round(record["support_density"]),
                "ranked_kept_candidate_count": record["ranked_kept_candidate_count"],
                "ranked_candidate_count": record["ranked_candidate_count"],
                "candidate_compute_units": record["candidate_compute_units"],
                "final_compute_units": record["final_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _run_support_probe_window_selector(
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

    train_event_count = len(split.train)
    candidate_records: list[dict] = []
    selection_cost_units = 0
    support_density = 0.0
    if train_event_count < SUPPORT_PROBE_RAW_MIN_EVENTS:
        selected_condition = "compact_train_size_gated_induction"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "train_support_probe_window_compact_below_raw_floor"
    elif train_event_count >= SUPPORT_PROBE_MAX_EVENTS:
        selected_condition = "raw_text"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "train_support_probe_window_raw_above_probe_ceiling"
    elif train_event_count < SUPPORT_PROBE_MIN_EVENTS:
        selected_condition = "raw_text"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "train_support_probe_window_raw_below_probe_floor"
    else:
        support_pipeline = _build_neural_pipeline("support_ramped_compact_induction", split, world.rules)
        support_final_compute_units = _pipeline_compute_units(pipeline=support_pipeline, epochs=epochs)
        selection_cost_units = _pipeline_compute_units(pipeline=support_pipeline, epochs=0)
        support_density = support_pipeline.ranked_kept_candidate_count / max(1, support_final_compute_units)
        candidate_records.append(
            {
                "condition": "support_ramped_compact_induction",
                "support_density": support_density,
                "ranked_kept_candidate_count": support_pipeline.ranked_kept_candidate_count,
                "ranked_candidate_count": support_pipeline.ranked_candidate_count,
                "candidate_compute_units": selection_cost_units,
                "final_compute_units": support_final_compute_units,
            }
        )
        if support_density >= TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE:
            selected_condition = "support_ramped_compact_induction"
            pipeline = support_pipeline
        else:
            selected_condition = "raw_text"
            pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = (
            "train_support_probe_window_min_"
            f"{TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE}_kept_per_compute"
        )

    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    charged_compute_units = selection_cost_units + pipeline.internal_token_count * epochs
    if selected_condition != "support_ramped_compact_induction" or not candidate_records:
        charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": SUPPORT_PROBE_WINDOW_SELECTOR,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(support_density),
        "portfolio_selected_condition": selected_condition,
        "portfolio_candidate_conditions": [
            record["condition"] for record in candidate_records
        ],
        "portfolio_selection_metric": selection_metric,
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(TRAIN_SUPPORT_DENSITY_MIN_KEPT_PER_COMPUTE),
        "portfolio_raw_text_abstention": 1 if selected_condition == "raw_text" else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "support_density": _round(record["support_density"]),
                "ranked_kept_candidate_count": record["ranked_kept_candidate_count"],
                "ranked_candidate_count": record["ranked_candidate_count"],
                "candidate_compute_units": record["candidate_compute_units"],
                "final_compute_units": record["final_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _validation_support_precision_calibration(split) -> dict:
    min_support, min_confidence, _ = _support_ramped_compact_ranked_policy(split.train)
    model = fit_induced_rule_model(split.train)
    modeling_cost_tokens = sum(raw_observation_example(item).token_count for item in split.train)
    validation_scoring_cost_tokens = sum(
        raw_observation_example(item).token_count for item in split.validation
    )

    covered = 0
    correct = 0
    for item in split.validation:
        prediction = model.predict(item.family, item.stimulus, item.modifier)
        if prediction.support < min_support or prediction.confidence < min_confidence:
            continue
        covered += 1
        correct += int(prediction.label == item.label)

    precision = (correct + 1) / (covered + 2) if covered else 0.5
    return {
        "precision": precision,
        "covered": covered,
        "correct": correct,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "modeling_cost_tokens": modeling_cost_tokens,
        "validation_scoring_cost_tokens": validation_scoring_cost_tokens,
    }


def _support_utility_score(
    calibration: dict,
    coverage_record: dict,
    final_compute_units: int,
) -> float:
    triple_l1 = coverage_record["coverage_triple_l1_distance"]
    return (
        calibration["precision"]
        - VALIDATION_SUPPORT_PRECISION_THRESHOLD
        + VALIDATION_SUPPORT_UTILITY_PAIR_COVERAGE_WEIGHT * coverage_record["coverage_pair_coverage"]
        - VALIDATION_SUPPORT_UTILITY_TRIPLE_L1_WEIGHT * (triple_l1 if triple_l1 is not None else 1.0)
        - VALIDATION_SUPPORT_UTILITY_COMPUTE_PENALTY * final_compute_units
    )


def _run_validation_support_utility_selector(
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

    train_event_count = len(split.train)
    candidate_records: list[dict] = []
    selected_condition: str
    selection_metric: str
    selection_cost_units = 0
    utility_score = 0.0

    if train_event_count < VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS:
        selected_condition = "compact_train_size_gated_induction"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "validation_support_expected_utility_compact_below_calibration_floor"
    else:
        calibration = _validation_support_precision_calibration(split)
        precision_margin = calibration["precision"] - VALIDATION_SUPPORT_PRECISION_THRESHOLD
        if precision_margin < VALIDATION_SUPPORT_UTILITY_MIN_SCORE:
            utility_score = precision_margin
            selection_cost_units = (
                calibration["modeling_cost_tokens"]
                + calibration["validation_scoring_cost_tokens"]
            )
            selected_condition = "raw_text"
            pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
            candidate_records.append(
                {
                    "condition": "support_ramped_compact_induction",
                    "pipeline": None,
                    "support_utility_score": utility_score,
                    "validation_precision": calibration["precision"],
                    "validation_covered_count": calibration["covered"],
                    "validation_correct_count": calibration["correct"],
                    "validation_pair_coverage": None,
                    "validation_triple_l1_distance": None,
                    "synthetic_example_count": 0,
                    "candidate_compute_units": selection_cost_units,
                    "final_compute_units": 0,
                    "min_support": calibration["min_support"],
                    "min_confidence": calibration["min_confidence"],
                }
            )
        else:
            support_pipeline = _build_neural_pipeline("support_ramped_compact_induction", split, world.rules)
            validation_examples = build_evaluation_examples(split.validation)
            validation_eval_cost = sum(example.token_count for example in validation_examples)
            coverage_record = _candidate_coverage_record(
                condition="support_ramped_compact_induction",
                pipeline=support_pipeline,
                validation_motifs=_observation_motif_counts(split.validation),
                validation_pairs=_observation_pair_counts(split.validation),
                validation_eval_cost=validation_eval_cost,
            )
            support_final_compute_units = _pipeline_compute_units(pipeline=support_pipeline, epochs=epochs)
            utility_score = _support_utility_score(
                calibration=calibration,
                coverage_record=coverage_record,
                final_compute_units=support_final_compute_units,
            )
            selection_cost_units = (
                _pipeline_compute_units(pipeline=support_pipeline, epochs=0)
                + validation_eval_cost
                + calibration["validation_scoring_cost_tokens"]
            )
            candidate_records.append(
                {
                    "condition": "support_ramped_compact_induction",
                    "pipeline": support_pipeline,
                    "support_utility_score": utility_score,
                    "validation_precision": calibration["precision"],
                    "validation_covered_count": calibration["covered"],
                    "validation_correct_count": calibration["correct"],
                    "validation_pair_coverage": coverage_record["coverage_pair_coverage"],
                    "validation_triple_l1_distance": coverage_record["coverage_triple_l1_distance"],
                    "synthetic_example_count": coverage_record["synthetic_example_count"],
                    "candidate_compute_units": selection_cost_units,
                    "final_compute_units": support_final_compute_units,
                    "min_support": calibration["min_support"],
                    "min_confidence": calibration["min_confidence"],
                }
            )
            if utility_score >= VALIDATION_SUPPORT_UTILITY_MIN_SCORE:
                selected_condition = "support_ramped_compact_induction"
                pipeline = support_pipeline
            else:
                selected_condition = "raw_text"
                pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = (
            "validation_support_expected_utility_precision_coverage_cost_min_"
            f"{VALIDATION_SUPPORT_UTILITY_MIN_SCORE}"
        )

    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    if selected_condition == "support_ramped_compact_induction" and candidate_records:
        charged_compute_units = pipeline.internal_token_count * epochs + selection_cost_units
    else:
        charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": VALIDATION_SUPPORT_UTILITY_SELECTOR,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(utility_score),
        "portfolio_selected_condition": selected_condition,
        "portfolio_candidate_conditions": [
            record["condition"] for record in candidate_records
        ],
        "portfolio_selection_metric": selection_metric,
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(VALIDATION_SUPPORT_UTILITY_MIN_SCORE),
        "portfolio_raw_text_abstention": 1 if selected_condition == "raw_text" else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "support_utility_score": _round(record["support_utility_score"]),
                "validation_precision": _round(record["validation_precision"]),
                "validation_covered_count": record["validation_covered_count"],
                "validation_correct_count": record["validation_correct_count"],
                "validation_pair_coverage": _round(record["validation_pair_coverage"])
                if record["validation_pair_coverage"] is not None
                else None,
                "validation_triple_l1_distance": _round(record["validation_triple_l1_distance"])
                if record["validation_triple_l1_distance"] is not None
                else None,
                "synthetic_example_count": record["synthetic_example_count"],
                "candidate_compute_units": record["candidate_compute_units"],
                "final_compute_units": record["final_compute_units"],
                "min_support": record["min_support"],
                "min_confidence": _round(record["min_confidence"]),
            }
            for record in candidate_records
        ],
    }


def _validation_gain_proxy_candidate_record(
    condition: str,
    pipeline: PipelineExamples,
    validation_examples: tuple[TrainingExample, ...],
    validation_eval_cost: int,
    seed: int,
    final_epochs: int,
) -> dict:
    proxy = PerceptronClassifier()
    proxy.fit(
        pipeline.examples,
        epochs=VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS,
        seed=seed,
    )
    validation = proxy.evaluate(validation_examples)
    validation_baseline = majority_baseline(pipeline.examples, validation_examples)
    validation_gain = validation.accuracy - validation_baseline.accuracy
    return {
        "condition": condition,
        "pipeline": pipeline,
        "validation_accuracy": validation.accuracy,
        "validation_baseline_accuracy": validation_baseline.accuracy,
        "validation_gain": validation_gain,
        "candidate_compute_units": (
            _pipeline_compute_units(
                pipeline=pipeline,
                epochs=VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS,
            )
            + validation_eval_cost
        ),
        "final_compute_units": _pipeline_compute_units(
            pipeline=pipeline,
            epochs=final_epochs,
        ),
    }


def _run_validation_support_gain_gate_selector(
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

    train_event_count = len(split.train)
    candidate_records: list[dict] = []
    selected_condition: str
    selection_metric: str
    selection_cost_units = 0
    support_score = 0.0

    if train_event_count < VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS:
        selected_condition = "compact_train_size_gated_induction"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "validation_support_gain_gate_compact_below_proxy_floor"
        proxy_epochs = 0
    else:
        raw_pipeline = _build_neural_pipeline("raw_text", split, world.rules)
        calibration = _validation_support_precision_calibration(split)
        precision_margin = calibration["precision"] - VALIDATION_SUPPORT_PRECISION_THRESHOLD
        if precision_margin < 0:
            selected_condition = "raw_text"
            pipeline = raw_pipeline
            support_score = precision_margin
            selection_cost_units = (
                calibration["modeling_cost_tokens"]
                + calibration["validation_scoring_cost_tokens"]
            )
            candidate_records.append(
                {
                    "condition": "support_ramped_compact_induction",
                    "pipeline": None,
                    "validation_accuracy": None,
                    "validation_baseline_accuracy": None,
                    "validation_gain": None,
                    "validation_gain_delta_vs_raw": None,
                    "validation_gain_gate_score": support_score,
                    "extra_final_compute_vs_raw": 0,
                    "validation_precision": calibration["precision"],
                    "validation_covered_count": calibration["covered"],
                    "validation_correct_count": calibration["correct"],
                    "candidate_compute_units": selection_cost_units,
                    "final_compute_units": 0,
                }
            )
            selection_metric = (
                "validation_support_gain_gate_precision_prefilter_min_"
                f"{VALIDATION_SUPPORT_PRECISION_THRESHOLD}"
            )
            proxy_epochs = 0
        else:
            validation_examples = build_evaluation_examples(split.validation)
            validation_eval_cost = sum(example.token_count for example in validation_examples)
            support_pipeline = _build_neural_pipeline("support_ramped_compact_induction", split, world.rules)
            raw_record = _validation_gain_proxy_candidate_record(
                condition="raw_text",
                pipeline=raw_pipeline,
                validation_examples=validation_examples,
                validation_eval_cost=validation_eval_cost,
                seed=seed,
                final_epochs=epochs,
            )
            support_record = _validation_gain_proxy_candidate_record(
                condition="support_ramped_compact_induction",
                pipeline=support_pipeline,
                validation_examples=validation_examples,
                validation_eval_cost=validation_eval_cost,
                seed=seed,
                final_epochs=epochs,
            )
            support_record["validation_precision"] = calibration["precision"]
            support_record["validation_covered_count"] = calibration["covered"]
            support_record["validation_correct_count"] = calibration["correct"]
            raw_validation_gain = raw_record["validation_gain"]
            for record in (raw_record, support_record):
                validation_gain_delta = record["validation_gain"] - raw_validation_gain
                extra_final_compute = max(
                    0,
                    record["final_compute_units"] - raw_record["final_compute_units"],
                )
                record["validation_gain_delta_vs_raw"] = validation_gain_delta
                record["validation_gain_gate_score"] = (
                    validation_gain_delta
                    - VALIDATION_SUPPORT_GAIN_COMPUTE_PENALTY * extra_final_compute
                )
                record["extra_final_compute_vs_raw"] = extra_final_compute
                candidate_records.append(record)

            support_score = support_record["validation_gain_gate_score"]
            if support_score >= VALIDATION_SUPPORT_GAIN_MIN_SCORE:
                selected_condition = "support_ramped_compact_induction"
                pipeline = support_pipeline
            else:
                selected_condition = "raw_text"
                pipeline = raw_pipeline
            selection_cost_units = sum(record["candidate_compute_units"] for record in candidate_records)
            selection_metric = (
                "validation_support_gain_gate_precision_prefilter_then_proxy_epochs_"
                f"{VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS}_min_"
                f"{VALIDATION_SUPPORT_GAIN_MIN_SCORE}_compute_penalty_"
                f"{VALIDATION_SUPPORT_GAIN_COMPUTE_PENALTY}"
            )
            proxy_epochs = VALIDATION_SUPPORT_GAIN_PROXY_EPOCHS

    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    if selected_condition == "support_ramped_compact_induction" and candidate_records:
        charged_compute_units = pipeline.internal_token_count * epochs + selection_cost_units
    else:
        charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": VALIDATION_SUPPORT_GAIN_GATE_SELECTOR,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(support_score),
        "portfolio_selected_condition": selected_condition,
        "portfolio_candidate_conditions": [
            record["condition"] for record in candidate_records
        ],
        "portfolio_selection_metric": selection_metric,
        "portfolio_proxy_epochs": proxy_epochs,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(VALIDATION_SUPPORT_GAIN_MIN_SCORE),
        "portfolio_raw_text_abstention": 1 if selected_condition == "raw_text" else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "validation_accuracy": _round(record["validation_accuracy"])
                if record.get("validation_accuracy") is not None
                else None,
                "validation_baseline_accuracy": _round(record["validation_baseline_accuracy"])
                if record.get("validation_baseline_accuracy") is not None
                else None,
                "validation_gain": _round(record["validation_gain"])
                if record.get("validation_gain") is not None
                else None,
                "validation_gain_delta_vs_raw": _round(record["validation_gain_delta_vs_raw"])
                if record.get("validation_gain_delta_vs_raw") is not None
                else None,
                "validation_gain_gate_score": _round(record["validation_gain_gate_score"]),
                "validation_precision": _round(record["validation_precision"])
                if record.get("validation_precision") is not None
                else None,
                "validation_covered_count": record.get("validation_covered_count"),
                "validation_correct_count": record.get("validation_correct_count"),
                "extra_final_compute_vs_raw": record["extra_final_compute_vs_raw"],
                "candidate_compute_units": record["candidate_compute_units"],
                "final_compute_units": record["final_compute_units"],
            }
            for record in candidate_records
        ],
    }


def _run_validation_support_precision_selector(
    seed: int,
    material_count: int,
    epochs: int,
    hidden_units: int,
    feature_dimension: int,
    learning_rate: float,
    use_transition_prior: bool = True,
    selector_condition: str = VALIDATION_SUPPORT_PRECISION_SELECTOR,
) -> dict:
    world = build_world(seed=seed, material_count=material_count)
    split = split_observations(world.observations)
    heldout_examples = build_evaluation_examples(split.heldout)

    train_event_count = len(split.train)
    candidate_records: list[dict] = []
    validation_score = 0.0
    selection_cost_units = 0
    selected_condition: str
    selection_metric: str

    if train_event_count < VALIDATION_SUPPORT_PRECISION_COMPACT_MAX_EVENTS:
        selected_condition = "compact_train_size_gated_induction"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "validation_support_precision_compact_below_calibration_floor"
    elif (
        use_transition_prior
        and
        VALIDATION_SUPPORT_PRECISION_TRANSITION_MIN_EVENTS
        <= train_event_count
        < VALIDATION_SUPPORT_PRECISION_TRANSITION_MAX_EVENTS
    ):
        selected_condition = "support_ramped_compact_induction"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        selection_metric = "validation_support_precision_transition_prior_support"
    else:
        calibration = _validation_support_precision_calibration(split)
        validation_score = calibration["precision"]
        support_by_precision = validation_score >= VALIDATION_SUPPORT_PRECISION_THRESHOLD
        selected_condition = "support_ramped_compact_induction" if support_by_precision else "raw_text"
        pipeline = _build_neural_pipeline(selected_condition, split, world.rules)
        if selected_condition == "support_ramped_compact_induction":
            selection_cost_units = calibration["validation_scoring_cost_tokens"]
        else:
            selection_cost_units = (
                calibration["modeling_cost_tokens"]
                + calibration["validation_scoring_cost_tokens"]
            )
        selection_metric = (
            (
                "validation_support_precision_min_"
                if use_transition_prior
                else "validation_support_precision_gate_min_"
            )
            + f"{VALIDATION_SUPPORT_PRECISION_THRESHOLD}_eligible_prediction_precision"
        )
        candidate_records.append(
            {
                "condition": "support_ramped_compact_induction",
                "validation_precision": validation_score,
                "validation_covered_count": calibration["covered"],
                "validation_correct_count": calibration["correct"],
                "candidate_compute_units": selection_cost_units,
                "min_support": calibration["min_support"],
                "min_confidence": calibration["min_confidence"],
            }
        )

    model = TinyMlpClassifier(
        feature_dimension=feature_dimension,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    model.fit(pipeline.examples, epochs=epochs, seed=seed)
    heldout = model.evaluate(heldout_examples)
    baseline = majority_baseline(pipeline.examples, heldout_examples)
    charged_compute_units = _pipeline_compute_units(pipeline=pipeline, epochs=epochs) + selection_cost_units
    improvement = heldout.accuracy - baseline.accuracy
    signed_lsd = 1_000_000.0 * improvement / max(1, pipeline.external_event_count * charged_compute_units)

    profile = model.training_profile
    return {
        "seed": seed,
        "condition": selector_condition,
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
        "portfolio_selection_cost_units": selection_cost_units,
        "portfolio_validation_score": _round(validation_score),
        "portfolio_selected_condition": selected_condition,
        "portfolio_candidate_conditions": [
            record["condition"] for record in candidate_records
        ],
        "portfolio_selection_metric": selection_metric,
        "portfolio_proxy_epochs": 0,
        "portfolio_abstention_extra_correct": 0,
        "portfolio_abstention_margin": _round(VALIDATION_SUPPORT_PRECISION_THRESHOLD),
        "portfolio_raw_text_abstention": 1 if selected_condition == "raw_text" else 0,
        "portfolio_candidate_summaries": [
            {
                "condition": record["condition"],
                "validation_precision": _round(record["validation_precision"]),
                "validation_covered_count": record["validation_covered_count"],
                "validation_correct_count": record["validation_correct_count"],
                "candidate_compute_units": record["candidate_compute_units"],
                "min_support": record["min_support"],
                "min_confidence": _round(record["min_confidence"]),
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
    if condition == VALIDATION_ABSTAINING_PROXY_SELECTOR:
        return _run_validation_abstaining_proxy_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_COVERAGE_PROXY_SELECTOR:
        return _run_validation_coverage_proxy_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_COVERAGE_PRIOR_SELECTOR:
        return _run_validation_coverage_prior_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == TRAIN_SUPPORT_DENSITY_SELECTOR:
        return _run_train_support_density_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == SUPPORT_PROBE_WINDOW_SELECTOR:
        return _run_support_probe_window_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_SUPPORT_PRECISION_SELECTOR:
        return _run_validation_support_precision_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_SUPPORT_PRECISION_GATE_SELECTOR:
        return _run_validation_support_precision_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
            use_transition_prior=False,
            selector_condition=VALIDATION_SUPPORT_PRECISION_GATE_SELECTOR,
        )
    if condition == VALIDATION_SUPPORT_UTILITY_SELECTOR:
        return _run_validation_support_utility_selector(
            seed=seed,
            material_count=material_count,
            epochs=epochs,
            hidden_units=hidden_units,
            feature_dimension=feature_dimension,
            learning_rate=learning_rate,
        )
    if condition == VALIDATION_SUPPORT_GAIN_GATE_SELECTOR:
        return _run_validation_support_gain_gate_selector(
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
        conditions_summary[condition]["portfolio_candidate_conditions"] = rows[0][
            "portfolio_candidate_conditions"
        ]
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
