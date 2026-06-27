# Learning Signal Density

Learning Signal Density is a public Limes Labs research workstream on a narrow
question:

> How much useful learning signal can a system extract from each external
> observation after accounting for selection, transformation, replay, feedback,
> and internal compute?

The project reframes "LLMs versus brains" as a measurable learning-loop problem.
The claim is not that brains are magic and not that current language models are
secretly equivalent to brains. The claim under test is smaller and falsifiable:
external sample efficiency belongs to the whole loop that selects, transforms,
replays, rewards, and consolidates experience, not only to the frozen neural
network being updated.

## Current Status

This first repo slice is a controlled causal-domain audit. It generates a hidden
synthetic rule system, freezes train/validation/heldout splits, compares several
learning pipelines under fixed external observations, and writes costed result
artifacts.

The current pilot is intentionally modest:

- It is not a neural language-model result.
- It uses an online linear learner as the first audit instrument.
- Counterfactual transforms are oracle-generated inside the synthetic world.
- `induced_rule_expansion` is the first non-oracle transform: it fits simple
  train-only empirical rules, then generates counterfactual labels from those
  induced rules rather than from the hidden rulebook.
- `validation_gated_induction` spends validation compute to choose the
  confidence/support thresholds for induced counterfactual generation, then
  charges that search cost in the final metric.
- `direct_validation_gated_induction` chooses the same thresholds by direct
  induced-rule precision/coverage on validation, avoiding per-candidate learner
  retraining and charging the cheaper gate search.
- `validation_ranked_induction` scores train-only induced counterfactual
  candidates by validation-estimated reliability, keeps a fixed budgeted subset,
  and charges validation scoring plus candidate-ranking cost.
- `train_calibrated_ranked_induction` estimates the same reliability from a
  held-out slice of the train split, testing whether validation labels are
  necessary for candidate ranking.
- `self_ranked_induction` removes calibration entirely and ranks induced
  candidates by train-only confidence, support, and salience signals.
- `sample_aware_self_ranked_induction` uses the same train-only ranking, but
  adapts the synthetic budget and minimum support to the size of the train
  split so scarce external data is not overwhelmed by generated labels.
- `diverse_self_ranked_induction` applies a diversity penalty to the same
  train-only ranking to test whether balancing modifier/stimulus/family coverage
  improves the fixed synthetic budget.
- `mdl_rule_expansion` learns a compact set of train-only empirical rules,
  selects them on validation with a description-length penalty, and charges
  rule search, validation scoring, and rule-description costs.
- Heldout examples are not used for selection, transformation, or replay.
- The artifact reports cost and scope flags so readers do not mistake the pilot
  for the final paper claim.

The next scientific step is a tiny transformer or nanoGPT-scale replication
using the same split and accounting discipline.

## What Is Here

- `learning_signal_density/` - deterministic causal-domain generator, pipeline
  transforms, simple learner, metrics, and artifact writer.
- `tests/` - unit tests for split isolation, pipeline accounting, and artifact
  honesty flags.
- `docs/preregistration.md` - first locked hypothesis, metrics, anti-cheat
  boundaries, and promotion gate.
- `docs/literature.md` - source-backed research map for data selection,
  transformation, dense feedback, replay, and world-model imagination.
- `results/` - checked-in result cards and JSON artifacts.
- `paper/` - paper skeleton and BibTeX file for the eventual technical report.
- `autoresearch/` - Limes AutoResearch config for ledgered reruns.
- `UPSTREAMS.md` - inspected inspirations and reuse boundary.

## Quickstart

Requirements:

- Python 3.11 or newer
- No external Python packages for the current tests or pilot

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the smoke experiment:

```bash
./scripts/run_smoke.sh
```

Regenerate the committed pilot artifact:

```bash
python3 -m learning_signal_density \
  --output-json results/causal_world_pilot_seedset.json \
  --output-md results/causal_world_pilot_seedset.md \
  --seeds 3 5 7 11 13 \
  --material-count 48 \
  --epochs 5
```

Run the sample-budget sweep:

```bash
python3 -m learning_signal_density.sweep \
  --output-json results/sample_budget_sweep.json \
  --output-md results/sample_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 3 5 7 11 13 \
  --epochs 5 \
  --target-signed-gain 0.03
```

## Metrics

The repo reports three families of measurements:

- External sample efficiency: heldout improvement per original external
  observation.
- Compute efficiency: heldout improvement per charged internal unit, including
  training tokens, train-only selection cost, and transform tokens.
- Learning-signal density: heldout improvement per external event per charged
  internal unit.
- Signed metrics preserve losses; clipped metrics count only positive per-seed
  improvements. Public interpretation should prefer signed metrics unless the
  question is explicitly "how often did this condition win?"
- Validation-gated transforms are allowed to use validation labels for threshold
  choice, never heldout labels, and their search overhead is charged.
- Direct validation gating is an explicit attempt to trade a slightly weaker
  selection signal for much lower tuning cost.
- Validation-ranked induction tests whether a fixed-budget non-oracle transform
  can keep the most reliable induced counterfactuals without using heldout
  labels or the hidden rulebook.
- Train-calibrated and self-ranked induction ablate whether validation
  reliability estimates are actually needed, or whether confidence/support
  ranking already captures most of the useful signal.
- Sample-aware self-ranked induction tests whether the fixed ranked budget
  should be reduced at tiny sample counts and made stricter when support is
  plentiful, without using validation, heldout, calibration, or oracle labels.
- Diverse self-ranked induction tests whether the ranked budget should be
  spread across feature regions, rather than concentrated where induced
  confidence/support is strongest.
- MDL rule expansion tests whether compressing the transform policy itself can
  reduce synthetic-example cost without quietly using the hidden rulebook.

The aim is to map a Pareto frontier, not to crown one universal pipeline.

## Current Empirical Readout

The current artifacts show a useful split:

- Oracle counterfactuals and selected oracle replay are still the strongest
  external-sample interventions.
- Train-only induced rules improve heldout accuracy once the external sample
  budget is large enough.
- Validation-gated induction improves accuracy, but full per-candidate
  retraining is too expensive under charged learning-signal density.
- Direct validation gating keeps much of that accuracy lift at lower cost.
- Self-ranked induction improves over unconstrained induced rules at the
  48-material pilot, beats the validation-ranked variants at 48 and 64
  materials, and uses no validation labels for transform selection. It is still
  weak at very low sample budgets.
- Sample-aware self-ranked induction is the strongest current non-oracle sweep
  result: it ties self-ranked induction at the 48-material pilot, improves the
  16-material budget from -0.011 to 0.032 signed gain, and improves the
  64-material budget from 0.140 to 0.151 signed gain while reducing charged
  compute from 48520.8 to 45806.4. It remains negative at 24 and 32 materials,
  so the result is a budget-sensitive frontier improvement, not a universal fix.
- A fresh-seed confirmation sweep keeps the sample-aware advantage at 24, 32,
  48, and 64 materials, but does not confirm the 16-material gain. On
  confirmation seeds, sample-aware improves best signed gain from 0.122 to
  0.135 versus self-ranked induction and reaches the target at 48 materials,
  not 16.
- Diverse self-ranked induction is a negative ablation: it lowers modifier
  concentration, but loses heldout gain at the medium/high budgets where
  self-ranked induction works. In the 48-material pilot, max modifier
  concentration falls from 49.4 to 35.0 while signed gain falls from 0.055 to
  0.041; in the sweep, its best signed gain is 0.070 versus 0.140 for
  self-ranked induction.
- MDL rule compression recovers some heldout signal while reducing synthetic
  examples, but its validation-scored rule search is too expensive to reach
  the current sample-budget target or dominate the frontier.

## Research Thesis

Apparent sample efficiency is determined less by raw external exposure than by
the density of useful learning signal extracted per exposure. Selection,
transformation, replay, and dense feedback can improve external sample
efficiency, but often by spending more internal compute. That tradeoff is the
object of measurement.

## Authorship

Francesco Giannicola, Limes Labs.
