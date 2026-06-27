# Preregistration: Causal-Domain Learning-Loop Pilot

Date: 2026-06-27

Author: Francesco Giannicola, Limes Labs

## Question

Can fixed external observations produce different heldout capability depending
on how the learning loop selects, transforms, and replays them, after charging
internal processing cost?

## Hypothesis

Selected, transformed, or replayed training material will improve external
sample efficiency relative to raw observations on a synthetic causal-text
domain, but the same conditions will not necessarily improve compute efficiency.
The expected scientific object is a Pareto frontier between external data and
internal processing cost.

## Locked Conditions

- `raw_text`: one training example per external observation.
- `selected_text`: train-only heuristic selection of higher-value examples.
- `qa_expansion`: each observation becomes an observation plus a question-answer
  training view.
- `counterfactual_expansion`: each train observation generates same-pair
  modifier counterfactuals using the synthetic world's rules.
- `prioritized_replay`: high-value train observations are replayed more often.
- `selected_counterfactual_replay`: selection plus same-pair counterfactuals
  and replay.

## Splits

The unit of split is `(material, stimulus)`. A pair assigned to heldout is never
used by selection, transformation, replay, or training. Counterfactual expansion
is allowed only for train pairs.

## Metrics

- Heldout accuracy.
- Majority-label baseline accuracy.
- Improvement over majority.
- External events.
- Internal examples.
- Internal tokens.
- Charged compute units: training tokens over epochs plus train-only selection
  and transform token costs.
- External sample efficiency.
- Compute efficiency per 10,000 charged units.
- Learning-signal density per 1,000,000 event-compute units.

## Anti-Cheat Rules

- No heldout example may be used to select examples, fit transforms, tune
  conditions, or set replay weights.
- All transform and replay overhead must be charged in the artifact.
- Negative or mixed results remain publishable.
- The current pilot must mark `neural_model=false`.
- The current pilot must mark `oracle_transform=true` because the synthetic
  world supplies ground-truth counterfactual labels.
- A paper-level claim requires a neural replication and at least one non-oracle
  transformation condition.

## Promotion Gate

This first slice is promoted only if:

1. Unit tests pass.
2. The seedset artifact is regenerated from code.
3. The artifact reports scope flags.
4. At least one non-raw condition differs in cost from the raw condition.
5. Documentation states the limitations plainly.
