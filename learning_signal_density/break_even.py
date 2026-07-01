from __future__ import annotations


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def break_even_comparison(
    reference: dict,
    candidate: dict,
    quality_key: str,
    compute_key: str = "charged_compute_units_mean",
    external_key: str = "external_events_mean",
    quality_upper_bound: float | None = None,
) -> dict:
    """Compare a candidate policy against a reference under LSD accounting.

    For positive reference quality, candidate density wins exactly when

        candidate_quality / reference_quality
        >
        candidate_event_compute / reference_event_compute.

    The inequality is only algebra on the learning-signal-density definition,
    but making it explicit prevents absolute-quality gains from being mistaken
    for efficiency gains.
    """

    reference_quality = float(reference[quality_key])
    candidate_quality = float(candidate[quality_key])
    reference_compute = float(reference[compute_key])
    candidate_compute = float(candidate[compute_key])
    reference_external = float(reference[external_key])
    candidate_external = float(candidate[external_key])
    quality_upper_bound_value = float(quality_upper_bound) if quality_upper_bound is not None else None

    reference_event_compute = reference_external * reference_compute
    candidate_event_compute = candidate_external * candidate_compute
    if reference_quality <= 0 or reference_event_compute <= 0:
        quality_multiplier = None
        event_compute_multiplier = None
        density_ratio = None
        break_even_quality = None
        required_quality_delta = None
        max_affordable_compute = None
        compute_over_break_even = None
        max_possible_density_ratio = None
        required_fraction_of_quality_bound = None
        perfect_quality_can_beat = None
        candidate_density_wins = candidate_quality > 0
    else:
        quality_multiplier = candidate_quality / reference_quality
        event_compute_multiplier = candidate_event_compute / reference_event_compute
        density_ratio = quality_multiplier / event_compute_multiplier
        break_even_quality = reference_quality * event_compute_multiplier
        required_quality_delta = max(0.0, break_even_quality - candidate_quality)
        max_affordable_compute = (
            reference_event_compute * candidate_quality / reference_quality / candidate_external
            if candidate_external > 0
            else None
        )
        compute_over_break_even = (
            candidate_compute / max_affordable_compute if max_affordable_compute and max_affordable_compute > 0 else None
        )
        if quality_upper_bound_value is not None and quality_upper_bound_value > 0 and break_even_quality > 0:
            max_possible_density_ratio = quality_upper_bound_value / break_even_quality
            required_fraction_of_quality_bound = break_even_quality / quality_upper_bound_value
            perfect_quality_can_beat = quality_upper_bound_value > break_even_quality
        else:
            max_possible_density_ratio = None
            required_fraction_of_quality_bound = None
            perfect_quality_can_beat = None
        candidate_density_wins = density_ratio > 1.0

    return {
        "reference_condition": reference["condition"],
        "candidate_condition": candidate["condition"],
        "reference_quality": _round(reference_quality),
        "candidate_quality": _round(candidate_quality),
        "reference_compute_units": _round(reference_compute),
        "candidate_compute_units": _round(candidate_compute),
        "reference_external_events": _round(reference_external),
        "candidate_external_events": _round(candidate_external),
        "quality_multiplier": _round(quality_multiplier),
        "event_compute_multiplier": _round(event_compute_multiplier),
        "density_ratio": _round(density_ratio),
        "break_even_quality": _round(break_even_quality),
        "required_quality_delta_to_break_even": _round(required_quality_delta),
        "max_affordable_compute_units": _round(max_affordable_compute),
        "compute_over_break_even": _round(compute_over_break_even),
        "quality_upper_bound": _round(quality_upper_bound_value),
        "max_possible_density_ratio": _round(max_possible_density_ratio),
        "required_fraction_of_quality_bound": _round(required_fraction_of_quality_bound),
        "perfect_quality_can_beat": perfect_quality_can_beat,
        "candidate_density_wins": candidate_density_wins,
    }
