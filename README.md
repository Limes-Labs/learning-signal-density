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

The aim is to map a Pareto frontier, not to crown one universal pipeline.

## Research Thesis

Apparent sample efficiency is determined less by raw external exposure than by
the density of useful learning signal extracted per exposure. Selection,
transformation, replay, and dense feedback can improve external sample
efficiency, but often by spending more internal compute. That tradeoff is the
object of measurement.

## Authorship

Francesco Giannicola, Limes Labs.
