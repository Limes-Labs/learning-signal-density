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
- `induced_rule_expansion`: train-only empirical rule induction generates
  counterfactual labels without querying the hidden synthetic rulebook.
- `validation_gated_induction`: train-only empirical rule induction with
  confidence/support thresholds selected on validation, then charged as tuning
  overhead.
- `direct_validation_gated_induction`: threshold selection by direct
  induced-rule precision/coverage on validation, without retraining a learner
  for each threshold candidate.
- `validation_ranked_induction`: train-only induced counterfactual candidates
  are ranked by validation-estimated source reliability, then a fixed budgeted
  subset is used for training.
- `train_calibrated_ranked_induction`: source reliability is estimated from a
  held-out slice of the train split rather than from validation.
- `self_ranked_induction`: induced counterfactual candidates are ranked only by
  train-only confidence, support, and salience signals, with no calibration
  labels.
- `sample_aware_self_ranked_induction`: the self-ranked candidate list uses
  train-size-only rules to lower the synthetic budget for tiny train splits and
  raise minimum support when the train split is large.
- `diverse_self_ranked_induction`: the self-ranked candidate list is selected
  with an explicit diversity penalty over modifier, stimulus, and family
  coverage.
- `mdl_rule_expansion`: train-only empirical rules are scored on validation
  with a description-length penalty, then only selected compact rules are used
  for counterfactual generation.
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
- Signed and clipped versions of the efficiency metrics. Signed metrics are the
  primary scientific readout; clipped metrics are diagnostic win-only summaries.
- Sample-budget thresholds for reaching a target signed gain over the majority
  baseline.

## Anti-Cheat Rules

- No heldout example may be used to select examples, fit transforms, tune
  conditions, or set replay weights.
- Validation examples may select induction thresholds only in explicitly marked
  validation-gated conditions.
- All transform and replay overhead must be charged in the artifact.
- Validation-gated conditions must charge threshold-search overhead.
- Direct validation gating must still charge train-modeling and validation
  scoring overhead, even though it avoids per-candidate learner retraining.
- Validation-ranked induction must charge validation scoring and candidate
  ranking overhead, and the committed pilot must use a fixed budget ratio rather
  than a heldout-selected ratio.
- Train-calibrated ranked induction must charge the extra train-calibration
  model and scoring pass.
- Self-ranked induction must not use validation or calibration labels for
  transform selection, and must still charge candidate-ranking overhead.
- Sample-aware self-ranked induction must record its effective budget ratio,
  minimum support, and minimum confidence, and must use only train split size to
  choose those policy knobs.
- Diverse self-ranked induction must record its diversity penalty and modifier
  concentration metrics, and must not use validation, calibration, heldout, or
  oracle labels for transform selection.
- MDL rule expansion must charge rule-search cost, validation-scoring cost,
  and selected-rule description length.
- Negative or mixed results remain publishable.
- The current pilot must mark `neural_model=false`.
- The current pilot must mark `oracle_transform=true` because the synthetic
  world supplies ground-truth counterfactual labels for at least one condition.
- Condition-level scope must identify which transforms use oracle-generated
  labels and which transforms use validation-gated threshold selection.
- A paper-level claim requires a neural replication and at least one non-oracle
  transformation condition.

## Promotion Gate

This first slice is promoted only if:

1. Unit tests pass.
2. The seedset artifact is regenerated from code.
3. The artifact reports scope flags.
4. At least one non-raw condition differs in cost from the raw condition.
5. Documentation states the limitations plainly.
