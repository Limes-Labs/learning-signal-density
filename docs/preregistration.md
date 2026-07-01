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
- `sample_aware_diverse_self_ranked_induction`: the same diversity penalty is
  applied inside the train-size-aware synthetic budget.
- `compact_diverse_train_size_gated_induction`: the train-size gate remains raw
  below the scarcity floor and matches sample-aware induction before the
  compact tier; the diversity penalty is applied only after compact encoding is
  active.
- `density_window_compact_induction`: a fixed train-only transition schedule
  that uses compact induction below 320 train events, raw text from 320 to 400,
  support-ramped compact induction from 400 to 432, and raw text again after
  432.
- `train_support_density_selector`: a train-only selector-cost control that
  chooses among raw text, compact train-size gated induction, and
  support-ramped compact induction using support kept per charged compute, then
  charges candidate inspection before training the selected final tiny MLP.
- `support_probe_window_selector`: a reuse-aware train-only selector-cost
  control that uses compact induction below 320 train events, raw text outside
  the 360--432 train-event support-probe window, and inside the window inspects
  only support-ramped compact induction before training the selected final tiny
  MLP.
- `validation_support_precision_selector`: a validation-calibrated high-budget
  support/raw selector. It keeps compact induction below 320 train events, keeps
  support-ramped compact in the fixed 400--432 train-event transition, and
  otherwise uses validation labels only to estimate eligible induced-prediction
  precision before selecting support-ramped compact or raw text.
- `validation_support_precision_gate_selector`: the no-window control for the
  validation support-precision selector. It removes the fixed support transition
  prior and applies the same validation precision threshold everywhere above 320
  train events.
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
- Any post-hoc frontier improvement should receive a fresh-seed confirmation
  sweep before being promoted beyond exploratory status.
- Post-hoc selector-error audits may use completed heldout outcomes only to
  diagnose regret and define future promotion gates; they must be marked
  non-deployable and cannot be used as policy selection results.
- Post-hoc mechanism audits may compare generated labels with the hidden
  rulebook or heldout motif distribution only after the source sweep has been
  committed; they must be marked non-deployable and cannot be used to select or
  tune policies.
- Train-only selector controls must charge candidate construction or inspection
  even when the final selected condition is cheap raw text.
- Reuse-aware selector controls may avoid double-charging selected candidate
  construction only when the artifact explicitly marks that reuse policy and
  still charges unselected candidate inspection.
- Validation utility selectors must disclose their utility weights, validation
  label use, validation motif-distribution use, prefilter behavior, and
  candidate-construction reuse; any validation scan or rejected support
  candidate must be charged before the final heldout evaluation.
- Validation gain-gate selectors must disclose their precision prefilter,
  validation-label use, proxy model class/epochs, gain threshold, compute
  penalty, and selected-candidate reuse. They must charge cheap rejected
  prefilter scans and any raw/support proxy training before final heldout
  evaluation.
- The current pilot must mark `neural_model=false`.
- Tiny neural replication artifacts must mark `neural_model=true`, keep the
  same heldout isolation rules, and report neural parameter count, training
  step count, and estimated training multiply-adds separately from pipeline
  token costs.
- Neural hyperparameter changes after seeing discovery seeds are exploratory
  until checked on fresh seeds.
- Neural confirmation artifacts must mark `fresh_seed_confirmation=true`,
  record the artifact being confirmed, and report whether each condition reaches
  the target signed gain.
- Selector policies are not promoted from their development seed block alone;
  they need a later transfer seed block that compares them against the simplest
  fixed schedule or no-scan fallback they aim to replace.
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

## Addendum: Real-Text SMS Spam Probe

Date: 2026-07-01

The first non-synthetic sanity check uses UCI SMS Spam Collection as a small
public binary text-classification dataset. This addendum does not retroactively
change the synthetic-domain preregistration above. It adds a separate artifact
class with these rules:

- The dataset archive must be downloaded from UCI, cached outside git, and
  verified by SHA-256 before parsing.
- The split must be train-pool/validation/heldout with stratification and
  disjoint record IDs.
- The primary quality metric is spam-class F1, not accuracy, because the task
  is class-imbalanced.
- Random sampling, full-pool class-balanced sampling, label-index balanced
  sampling, and validation selectors must report their selection-cost model.
- Validation selectors may use validation labels for policy selection only when
  the condition scope discloses that use and charges proxy/scoring overhead.
- Heldout examples may not be used for sampling-policy selection.
- Break-even analysis may be run after the SMS artifacts are committed. It is a
  mathematical diagnostic, not a new policy, and must use random sampling as the
  reference unless a later addendum states otherwise.
- The artifact remains exploratory and must mark `paper_ready_claim=false`.
