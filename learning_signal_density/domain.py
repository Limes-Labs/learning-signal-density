from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


FAMILIES = (
    "auren",
    "belith",
    "cyron",
    "darsen",
    "elion",
    "ferran",
    "gald",
    "hyron",
)

STIMULI = (
    "blue_radiation",
    "amber_heat",
    "silver_pressure",
    "violet_sound",
    "green_current",
    "white_vacuum",
)

MODIFIERS = (
    "plain",
    "l7_crystal",
    "q9_salt",
    "m3_coating",
    "r2_dopant",
)


@dataclass(frozen=True)
class RuleBook:
    """Deterministic rules for the synthetic causal domain."""

    seed: int
    base_effects: dict[tuple[str, str], bool]
    inhibitors: dict[tuple[str, str], str]
    amplifiers: dict[tuple[str, str], str]

    def label(self, family: str, stimulus: str, modifier: str) -> bool:
        base = self.base_effects[(family, stimulus)]
        if modifier == self.inhibitors[(family, stimulus)]:
            return False
        if modifier == self.amplifiers[(family, stimulus)]:
            return True
        return base


@dataclass(frozen=True)
class Observation:
    observation_id: str
    material: str
    family: str
    stimulus: str
    modifier: str
    label: bool

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.material, self.stimulus)

    @property
    def outcome(self) -> str:
        return "brittle" if self.label else "stable"


@dataclass(frozen=True)
class World:
    seed: int
    rules: RuleBook
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class ObservationSplit:
    train: tuple[Observation, ...]
    validation: tuple[Observation, ...]
    heldout: tuple[Observation, ...]


def _build_rules(seed: int) -> RuleBook:
    base_effects: dict[tuple[str, str], bool] = {}
    inhibitors: dict[tuple[str, str], str] = {}
    amplifiers: dict[tuple[str, str], str] = {}
    for family_index, family in enumerate(FAMILIES):
        for stimulus_index, stimulus in enumerate(STIMULI):
            key = (family, stimulus)
            base_effects[key] = ((family_index * 11 + stimulus_index * 7 + seed) % 5) in {0, 2}
            inhibitors[key] = MODIFIERS[1 + ((family_index + 2 * stimulus_index + seed) % (len(MODIFIERS) - 1))]
            amplifiers[key] = MODIFIERS[1 + ((2 * family_index + stimulus_index + seed + 1) % (len(MODIFIERS) - 1))]
    return RuleBook(seed=seed, base_effects=base_effects, inhibitors=inhibitors, amplifiers=amplifiers)


def build_world(seed: int, material_count: int = 48) -> World:
    if material_count < len(FAMILIES) * 2:
        raise ValueError("material_count must cover each family at least twice")

    rules = _build_rules(seed)
    observations: list[Observation] = []
    for material_index in range(material_count):
        family = FAMILIES[material_index % len(FAMILIES)]
        material = f"{family}_alloy_{material_index:03d}"
        for stimulus_index, stimulus in enumerate(STIMULI):
            modifier = MODIFIERS[(material_index * 3 + stimulus_index * 5 + seed) % len(MODIFIERS)]
            label = rules.label(family, stimulus, modifier)
            observations.append(
                Observation(
                    observation_id=f"obs-{seed}-{material_index:03d}-{stimulus_index:02d}",
                    material=material,
                    family=family,
                    stimulus=stimulus,
                    modifier=modifier,
                    label=label,
                )
            )
    return World(seed=seed, rules=rules, observations=tuple(observations))


def split_observations(
    observations: Iterable[Observation],
    validation_fraction: float = 0.2,
    heldout_fraction: float = 0.2,
    split_seed: int = 20260627,
) -> ObservationSplit:
    observations = tuple(observations)
    pairs = sorted({item.pair_key for item in observations})
    rng = random.Random(split_seed)
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)

    heldout_count = max(1, round(len(shuffled_pairs) * heldout_fraction))
    validation_count = max(1, round(len(shuffled_pairs) * validation_fraction))
    heldout_pairs = set(shuffled_pairs[:heldout_count])
    validation_pairs = set(shuffled_pairs[heldout_count : heldout_count + validation_count])

    train: list[Observation] = []
    validation: list[Observation] = []
    heldout: list[Observation] = []
    for item in observations:
        if item.pair_key in heldout_pairs:
            heldout.append(item)
        elif item.pair_key in validation_pairs:
            validation.append(item)
        else:
            train.append(item)

    return ObservationSplit(train=tuple(train), validation=tuple(validation), heldout=tuple(heldout))
