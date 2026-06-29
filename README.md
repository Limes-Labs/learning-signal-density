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
- A first tiny neural replication now uses a deterministic CPU MLP with the
  same split discipline and cost accounting; it is still not a language-model
  or frontier-model result.
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
  train-only ranking to test whether balancing modifier/stimulus/family
  coverage improves the fixed synthetic budget.
- `sample_aware_diverse_self_ranked_induction` combines the sample-aware
  budget policy with that diversity penalty, testing whether coverage balance
  helps after the budget is made train-size aware.
- `tempered_sample_aware_self_ranked_induction` is a train-only ablation of
  sample-aware ranking. It lowers the mid-budget synthetic ratio from `0.75` to
  `0.50` below 144 train events, testing whether smaller generated-label
  budgets repair the 24--32 material failure without validation selection.
- `train_size_gated_sample_aware_induction` uses raw text below 144 train
  events, then switches to sample-aware self-ranked induction after the train
  split is large enough.
- `compact_train_size_gated_induction` keeps that train-size schedule but drops
  original QA duplicates at the large-sample tier, testing whether generated
  labels can be made denser without changing their labels.
- `compact_diverse_train_size_gated_induction` keeps compact train-size gating
  but applies diversity only after the compact large-sample tier, separating
  candidate-order coverage from representation cost.
- `density_capped_compact_induction` keeps compact induction only while it
  remains density-efficient, then returns to raw text after the abundant-data
  tier.
- `support_ramped_compact_induction` is a train-only high-budget tradeoff: it
  matches compact induction through 96 materials, then raises induced-label
  minimum support from `3` to `4` after the abundant-data tier.
- `density_window_compact_induction` is a train-only fixed-window policy for
  the high-budget transition: compact below 320 train events, raw from 320 to
  400, support-ramped compact from 400 to 432, and raw again after 432.
- `late_confidence_ramped_compact_induction` is a train-only negative/mixed
  control for that tradeoff. It matches support-ramped compact until the train
  split reaches 432 events, then raises induced-label confidence from `0.55` to
  `0.60`.
- `agreement_gated_self_ranked_induction` keeps train-only generated labels
  only when independent induced-rule projections agree, testing whether source
  agreement is enough reliability signal without validation labels.
- A post-hoc policy envelope selects the best non-oracle condition at each
  budget from completed heldout results. It is explicitly not deployable; it is
  a diagnostic upper bound for the adaptive policy-selection problem.
- `validation_portfolio_selector` is the first deployable neural selector
  probe: it trains six non-oracle candidate policies, chooses by validation
  improvement over majority, charges the full portfolio search, and evaluates
  heldout only after selection.
- `validation_linear_proxy_selector` is the cheaper deployable selector probe:
  it scores the same candidate policies with a two-epoch linear proxy, charges
  those proxy fits, trains one final tiny MLP, and evaluates heldout only after
  selection.
- `validation_abstaining_proxy_selector` adds a raw-text fallback to the
  linear-proxy selector: a non-raw policy must beat raw text by three
  validation examples before the selector leaves raw text.
- A hidden-rulebook generated-label audit is included as a non-deployable
  diagnostic. It checks whether generated synthetic labels are correct after
  the selector-transfer sweep has run, then links that precision to the neural
  gains in the source artifact.
- `validation_coverage_proxy_selector` is a deployable follow-up to the
  generated-coverage audit: it scores candidate policies by how closely their
  generated synthetic-label motifs match the validation motif distribution,
  without using heldout distribution or validation labels for the selector
  score. It improves the 32-material confirmation row but exposes a new
  24-material MDL over-selection failure.
- `validation_coverage_prior_selector` adds a low-budget train-size prior and
  lean candidate set to that coverage proxy. It uses raw text below 96 train
  events, then scans only raw, sample-aware self-ranked, and validation-ranked
  candidates with a small charged-compute penalty in the coverage score.
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
- `results/tiny_neural_replication.*` - first deterministic tiny-MLP
  replication artifact with neural parameter, step, and estimated operation
  accounting.
- `results/tiny_neural_confirmation.*` - fresh-seed confirmation of the same
  tiny-MLP profile.
- `results/tiny_neural_budget_sweep.*` - fresh-seed tiny-MLP sample-budget
  frontier across 16, 24, 32, 48, and 64 materials.
- `results/tiny_neural_budget_sweep_32x8.*` - same fresh-seed neural
  sample-budget frontier rerun with the more efficient `epochs=32_hidden=8`
  profile and compared against the original `32x32` artifact.
- `results/tiny_neural_feature_sweep.*` - fresh-seed `32x8` feature-hash
  dimension frontier at the 64-material budget.
- `results/tiny_neural_budget_sweep_32x8_f256.*` - full sample-budget rerun
  of the `32x8` profile with 256 hashed features, compared against the
  128-feature `32x8` artifact.
- `results/tiny_neural_feature_sweep_wide.*` - wider 128/256/512/1024
  feature-hash sweep at the 64-material budget.
- `results/tiny_neural_budget_sweep_32x8_f1024.*` - full sample-budget rerun
  of the `32x8` profile with 1024 hashed features, compared against the
  256-feature `32x8` artifact.
- `results/tiny_neural_profile_sweep_f1024.*` - epoch/width profile sweep at
  the 64-material budget with 1024 hashed features.
- `results/tiny_neural_budget_sweep_16x8_f1024.*` - full sample-budget rerun
  of the lower-epoch `16x8` profile with 1024 hashed features, compared
  against the `32x8_features=1024` artifact.
- `results/tiny_neural_budget_sweep_8x8_f1024.*` - full sample-budget
  ablation of an even lower-epoch `8x8` profile with 1024 hashed features,
  compared against the `16x8_features=1024` artifact.
- `results/tiny_neural_budget_sweep_validation_selected_f1024.*` - charged
  validation-ranked and MDL rule-selection probe for the `16x8` 1024-feature
  profile.
- `results/tiny_neural_budget_sweep_agreement_gated_f1024.*` - train-only
  source-agreement reliability probe for the same `16x8` 1024-feature profile.
- `results/policy_envelope_f1024.*` - derived post-hoc non-oracle policy
  envelope over the same reliability probe. This artifact uses heldout results
  after the fact, marks itself non-deployable, and exists to quantify the
  selector problem for the paper.
- `results/tiny_neural_budget_sweep_validation_portfolio_f1024.*` - deployable
  validation-portfolio selector probe for the same `16x8` 1024-feature profile,
  with portfolio training and validation selection costs charged.
- `results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.*` -
  low-fidelity validation selector probe for the same profile, using a
  two-epoch linear proxy to choose one final tiny-MLP policy before heldout
  evaluation.
- `results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.*` -
  raw-abstaining low-fidelity selector probe for the same profile, testing
  whether a three-validation-example margin reduces scarce-budget downside.
- `results/tiny_neural_budget_sweep_selector_transfer_f1024.*` - fresh-seed
  selector-family stress test on seeds `37 41 43 47 53`, checking whether the
  deployable selector results transfer beyond the development selector
  artifacts.
- `results/generated_label_audit_selector_transfer_f1024.*` - hidden-rulebook
  generated-label audit for the selector-transfer seeds. This is explicitly
  non-deployable and exists to test whether label precision alone explains
  neural gain.
- `results/generated_coverage_audit_selector_transfer_f1024.*` -
  heldout-distribution coverage audit for the selector-transfer seeds. This is
  explicitly non-deployable and tests whether generated-label motif coverage
  tracks gain better than label precision alone.
- `results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.*` -
  fresh-seed validation-coverage proxy probe on seeds `103 107 109 113 127`.
  This is deployable, uses validation motif distribution rather than heldout
  distribution for policy selection, and records a mixed result: positive at
  32 and 64 materials, but harmful at 24 materials.
- `results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.*` -
  fresh-seed coverage-prior selector control on seeds `601 607 613 617 619`.
  It fixes the old coverage proxy's 24-material over-selection and cuts
  selector cost at 48/64, but the train-size gate still has better density at
  those high-budget rows.
- `results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.*` -
  fresh-seed train-only synthetic-budget ablation on seeds `157 163 167 173
  179`. It improves 24/32-material gain relative to fixed sample-aware ranking,
  but still loses to the train-size raw fallback at those scarce budgets.
- `results/tiny_neural_budget_sweep_train_size_gated_f1024.*` - second
  unseen-seed baseline on seeds `59 61 67 71 73`, testing a deployable
  train-size-only schedule that stays raw below 144 train events and switches
  to sample-aware self-ranked induction once the train split is large enough.
- `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.*` -
  fresh-seed train-only efficiency probe on seeds `181 191 193 197 199`. It
  matches the train-size gate through 48 materials, then drops original QA
  duplicates at the large-sample tier and improves 64-material signed LSD.
- `results/tiny_neural_budget_sweep_diversity_interaction_f1024.*` -
  fresh-seed train-only diversity interaction probe on seeds `701 709 719 727
  733`. It shows that diversity improves the 64-material sample-aware row, but
  compact diversity does not beat the compact density frontier.
- `results/tiny_neural_budget_sweep_density_capped_compact_f1024.*` -
  fresh-seed high-budget density probe on seeds `293 307 311 313 317`. It
  matches compact train-size gating through 96 materials, then returns to raw
  text once the train split is abundant, improving density while giving up
  some absolute gain.
- `results/tiny_neural_budget_sweep_support_ramped_compact_f1024.*` -
  fresh-seed high-budget support-ramp probe on seeds `401 409 419 421 431`.
  It matches compact and density-capped compact through 96 materials, then
  raises induced-label support at the abundant-data tier. The result is mixed:
  it improves compact density after 104 materials and recovers more gain than
  raw fallback at several high-budget points, but raw fallback still has higher
  signed LSD at 128.
- `results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.*` -
  fresh-seed late-confidence control on seeds `499 503 509 521 523`. It raises
  the abundant-tier confidence floor from `0.55` to `0.60` only after 432 train
  events. The result is mostly negative: it improves the support-ramped row at
  120 materials, but raw/density-capped fallback still dominates that row and
  the best signed gain remains below plain compact.
- `results/tiny_neural_budget_sweep_density_window_compact_f1024.*` -
  fresh-seed fixed-window high-budget probe on seeds `929 937 941 947 953`.
  It preserves compact density at 64/80, returns to raw around 96/104/120, and
  uses support-ramped compact only in the 400--432 train-event transition. The
  result is mixed but useful: it improves signed LSD at 112 materials over
  density-capped/raw fallback, preserves raw density at 120, and misses the
  support-ramped 128-material row.
- `results/tiny_neural_profile_sweep.*` - fresh-seed tiny-MLP epoch/width
  frontier at the 64-material budget.
- `paper/` - working paper draft, generated result tables, BibTeX file, and
  release-readiness checklist for the eventual technical report.
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

Regenerate the manuscript result tables from checked-in JSON artifacts:

```bash
python3 scripts/build_policy_envelope.py
python3 scripts/build_generated_label_audit.py
python3 scripts/build_generated_coverage_audit.py
python3 scripts/build_paper_tables.py
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

Run the tiny neural replication:

```bash
python3 -m learning_signal_density.neural_experiment \
  --output-json results/tiny_neural_replication.json \
  --output-md results/tiny_neural_replication.md \
  --seeds 3 5 7 11 13 \
  --material-count 48 \
  --epochs 32 \
  --hidden-units 32 \
  --feature-dimension 128 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03
```

Run the tiny neural fresh-seed confirmation:

```bash
python3 -m learning_signal_density.neural_experiment \
  --output-json results/tiny_neural_confirmation.json \
  --output-md results/tiny_neural_confirmation.md \
  --seeds 17 19 23 29 31 \
  --material-count 48 \
  --epochs 32 \
  --hidden-units 32 \
  --feature-dimension 128 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_replication.json
```

Run the tiny neural budget sweep:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep.json \
  --output-md results/tiny_neural_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 32 \
  --hidden-units 32 \
  --feature-dimension 128 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_confirmation.json
```

Run the tiny neural profile sweep:

```bash
python3 -m learning_signal_density.neural_profile_sweep \
  --output-json results/tiny_neural_profile_sweep.json \
  --output-md results/tiny_neural_profile_sweep.md \
  --profiles 8x8 16x8 32x8 8x16 16x16 32x16 8x32 16x32 32x32 \
  --seeds 17 19 23 29 31 \
  --material-count 64 \
  --feature-dimension 128 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep.json
```

Run the efficient tiny neural budget sweep:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_32x8.json \
  --output-md results/tiny_neural_budget_sweep_32x8.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 32 \
  --hidden-units 8 \
  --feature-dimension 128 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_profile_sweep.json \
  --comparison-of results/tiny_neural_budget_sweep.json \
  --profile-label epochs=32_hidden=8
```

Run the tiny neural feature-dimension sweep:

```bash
python3 -m learning_signal_density.neural_feature_sweep \
  --output-json results/tiny_neural_feature_sweep.json \
  --output-md results/tiny_neural_feature_sweep.md \
  --feature-dimensions 16 32 64 128 256 \
  --seeds 17 19 23 29 31 \
  --material-count 64 \
  --epochs 32 \
  --hidden-units 8 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_32x8.json \
  --comparison-of results/tiny_neural_budget_sweep_32x8.json \
  --profile-label epochs=32_hidden=8
```

Run the 256-feature efficient tiny neural budget sweep:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_32x8_f256.json \
  --output-md results/tiny_neural_budget_sweep_32x8_f256.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 32 \
  --hidden-units 8 \
  --feature-dimension 256 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_feature_sweep.json \
  --comparison-of results/tiny_neural_budget_sweep_32x8.json \
  --profile-label epochs=32_hidden=8_features=256
```

Run the wide tiny neural feature-dimension sweep:

```bash
python3 -m learning_signal_density.neural_feature_sweep \
  --output-json results/tiny_neural_feature_sweep_wide.json \
  --output-md results/tiny_neural_feature_sweep_wide.md \
  --feature-dimensions 128 256 512 1024 \
  --seeds 17 19 23 29 31 \
  --material-count 64 \
  --epochs 32 \
  --hidden-units 8 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_feature_sweep.json \
  --comparison-of results/tiny_neural_feature_sweep.json \
  --profile-label epochs=32_hidden=8
```

Run the 1024-feature efficient tiny neural budget sweep:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_32x8_f1024.json \
  --output-md results/tiny_neural_budget_sweep_32x8_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 32 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_feature_sweep_wide.json \
  --comparison-of results/tiny_neural_budget_sweep_32x8_f256.json \
  --profile-label epochs=32_hidden=8_features=1024
```

Run the 1024-feature tiny neural profile sweep:

```bash
python3 -m learning_signal_density.neural_profile_sweep \
  --output-json results/tiny_neural_profile_sweep_f1024.json \
  --output-md results/tiny_neural_profile_sweep_f1024.md \
  --profiles 8x8 16x8 32x8 64x8 8x16 16x16 32x16 64x16 32x32 \
  --seeds 17 19 23 29 31 \
  --material-count 64 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_32x8_f1024.json
```

Run the 16x8 1024-feature efficient tiny neural budget sweep:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_16x8_f1024.json \
  --output-md results/tiny_neural_budget_sweep_16x8_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_profile_sweep_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_32x8_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024
```

Run the 8x8 1024-feature low-epoch ablation:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_8x8_f1024.json \
  --output-md results/tiny_neural_budget_sweep_8x8_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 8 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_profile_sweep_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_16x8_f1024.json \
  --profile-label epochs=8_hidden=8_features=1024
```

Run the 16x8 1024-feature validation-selected reliability probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_selected_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_selected_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction validation_ranked_induction mdl_rule_expansion counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_16x8_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_16x8_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_selected
```

Run the 16x8 1024-feature agreement-gated reliability probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_agreement_gated_f1024.json \
  --output-md results/tiny_neural_budget_sweep_agreement_gated_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction agreement_gated_self_ranked_induction validation_ranked_induction mdl_rule_expansion counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_selected_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_selected_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_agreement_gated
```

Build the post-hoc policy envelope used by the paper tables:

```bash
python3 scripts/build_policy_envelope.py
```

Run the 16x8 1024-feature validation-portfolio selector probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_portfolio_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_portfolio_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction agreement_gated_self_ranked_induction validation_ranked_induction mdl_rule_expansion validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_agreement_gated_f1024.json \
  --comparison-of results/policy_envelope_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_portfolio
```

Run the 16x8 1024-feature linear-proxy validation selector probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction agreement_gated_self_ranked_induction validation_ranked_induction mdl_rule_expansion validation_linear_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_portfolio_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_portfolio_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_linear_proxy
```

Run the 16x8 1024-feature abstaining-proxy validation selector probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction agreement_gated_self_ranked_induction validation_ranked_induction mdl_rule_expansion validation_abstaining_proxy_selector validation_linear_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_abstaining_proxy
```

Run the 16x8 1024-feature fresh-seed selector-transfer stress test:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_selector_transfer_f1024.json \
  --output-md results/tiny_neural_budget_sweep_selector_transfer_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 37 41 43 47 53 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction agreement_gated_self_ranked_induction validation_ranked_induction mdl_rule_expansion validation_abstaining_proxy_selector validation_linear_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_selector_transfer
```

Run the 16x8 1024-feature train-size gated baseline on a second unseen seed set:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_train_size_gated_f1024.json \
  --output-md results/tiny_neural_budget_sweep_train_size_gated_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 59 61 67 71 73 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction train_size_gated_sample_aware_induction validation_ranked_induction mdl_rule_expansion validation_abstaining_proxy_selector validation_linear_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_selector_transfer_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_selector_transfer_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_train_size_gated
```

Run the 16x8 1024-feature validation-coverage proxy on a fresh confirmation
seed set:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 103 107 109 113 127 \
  --conditions raw_text sample_aware_self_ranked_induction train_size_gated_sample_aware_induction validation_coverage_proxy_selector validation_abstaining_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_train_size_gated_f1024.json \
  --comparison-of results/generated_coverage_audit_selector_transfer_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_coverage_proxy
```

Run the 16x8 1024-feature coverage-prior selector control:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 601 607 613 617 619 \
  --conditions raw_text sample_aware_self_ranked_induction train_size_gated_sample_aware_induction validation_coverage_proxy_selector validation_coverage_prior_selector validation_abstaining_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_validation_coverage_prior
```

Run the 16x8 1024-feature tempered sample-aware ablation:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json \
  --output-md results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 157 163 167 173 179 \
  --conditions raw_text qa_expansion sample_aware_self_ranked_induction tempered_sample_aware_self_ranked_induction train_size_gated_sample_aware_induction validation_coverage_proxy_selector validation_abstaining_proxy_selector validation_portfolio_selector counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_train_size_gated_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_tempered_sample_aware
```

Run the 16x8 1024-feature compact train-size gated efficiency probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json \
  --output-md results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 181 191 193 197 199 \
  --conditions raw_text train_size_gated_sample_aware_induction compact_train_size_gated_induction \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_train_size_gated_f1024.json \
  --profile-label f1024_16x8_compact_train_size_gated
```

Run the 16x8 1024-feature diversity interaction probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_diversity_interaction_f1024.json \
  --output-md results/tiny_neural_budget_sweep_diversity_interaction_f1024.md \
  --material-counts 16 24 32 48 64 \
  --seeds 701 709 719 727 733 \
  --conditions raw_text self_ranked_induction sample_aware_self_ranked_induction diverse_self_ranked_induction sample_aware_diverse_self_ranked_induction train_size_gated_sample_aware_induction compact_train_size_gated_induction compact_diverse_train_size_gated_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json \
  --profile-label epochs=16_hidden=8_features=1024_diversity_interaction
```

The diversity interaction probe is train-only. On seeds `701 709 719 727
733`, sample-aware diversity improves the 64-material sample-aware row from
`0.158441` gain and `0.005544` signed LSD to `0.168831` and `0.005908`, but
compact diversity lowers the compact 64-material row from `0.135065` gain and
`0.007594` signed LSD to `0.116883` and `0.006573`. Diversity is useful inside
the full sample-aware view, but it is not the compact density frontier.

Run the 16x8 1024-feature density-capped compact high-budget probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_density_capped_compact_f1024.json \
  --output-md results/tiny_neural_budget_sweep_density_capped_compact_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 293 307 311 313 317 \
  --conditions raw_text train_size_gated_sample_aware_induction compact_train_size_gated_induction density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json \
  --profile-label f1024_16x8_density_capped_compact
```

Run the 16x8 1024-feature support-ramped compact high-budget probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json \
  --output-md results/tiny_neural_budget_sweep_support_ramped_compact_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 401 409 419 421 431 \
  --conditions raw_text train_size_gated_sample_aware_induction compact_train_size_gated_induction support_ramped_compact_induction density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_density_capped_compact_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_density_capped_compact_f1024.json \
  --profile-label f1024_16x8_support_ramped_compact
```

Run the 16x8 1024-feature late-confidence compact control:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json \
  --output-md results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.md \
  --material-counts 96 104 112 120 128 144 160 \
  --seeds 499 503 509 521 523 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction late_confidence_ramped_compact_induction density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json \
  --profile-label f1024_16x8_late_confidence_ramped_compact
```

Run the 16x8 1024-feature density-window compact transition probe:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_density_window_compact_f1024.json \
  --output-md results/tiny_neural_budget_sweep_density_window_compact_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 929 937 941 947 953 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json \
  --profile-label f1024_16x8_density_window_compact
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
- The post-hoc policy envelope is an analysis artifact, not a method: it uses
  completed heldout results to choose the best non-oracle condition at each
  budget, marks that heldout policy selection explicitly, and should only be
  used to define the next adaptive selector target.
- The validation portfolio selector is a deployable selector probe, but it is
  expensive: it trains all candidate policies for validation scoring and
  charges that search before heldout evaluation.
- The linear-proxy validation selector tests whether a cheaper model hierarchy
  can preserve selector signal while charging the proxy search and one selected
  final MLP training run.
- The abstaining-proxy validation selector tests whether requiring a
  three-validation-example margin over raw text reduces downside before paying
  for generated-label policies.
- The validation-coverage proxy selector tests whether the non-deployable
  heldout-coverage audit can be approximated by validation motif distribution
  before heldout is opened. It uses no validation labels for its selector score,
  but still charges candidate construction and validation-motif scanning.
- The tempered sample-aware policy tests whether lowering the mid-budget
  generated-label ratio reduces scarce-budget damage without adding validation
  selection, heldout access, or selector-search cost.
- The train-size gated sample-aware policy tests a cheaper schedule baseline:
  raw text below 144 train events and sample-aware self-ranked induction once
  the train split is large enough. It uses no validation labels or selector
  search, so it is a baseline future adaptive selectors must beat.
- The compact train-size gated policy keeps that same train-only schedule, but
  at 224 or more train events drops original QA duplicates while retaining raw
  originals and ranked generated examples. It tests whether density can improve
  by removing redundant transformed originals rather than by changing labels.
- The density-capped compact policy extends that schedule to abundant-data
  budgets: after the train split reaches 360 events, it returns to raw text
  because generated-label transforms can lose the signed-density frontier even
  when they still improve absolute gain.

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
- The first tiny-MLP replication changes the learner-dependent readout:
  raw text and QA expansion are negative, self-ranked and sample-aware
  self-ranked induction reach a positive `0.041` signed gain, and oracle
  counterfactual expansion reaches `0.055`. Because counterfactual expansion
  spends more internal compute, its signed learning-signal density is lower
  than the train-only ranked conditions (`0.000851` versus `0.001343`). This
  result is exploratory and neural, but not a language-model result.
- A fresh-seed tiny-MLP confirmation keeps the ranked train-only conditions
  above the `0.03` target (`0.038` signed gain, `0.001230` signed LSD), turns
  QA expansion positive (`0.034` signed gain), and strengthens oracle
  counterfactual expansion (`0.121` signed gain, `0.001861` signed LSD). Raw
  text remains negative. This confirms a small neural learning-loop effect
  under the synthetic audit, while still leaving the language-model claim open.
- The tiny-MLP budget sweep shows where that effect starts. Raw text never
  reaches the target. Oracle counterfactual expansion first reaches target at
  32 materials and peaks at 48 (`0.121` signed gain). QA, self-ranked
  induction, and sample-aware self-ranked induction first reach target at 48.
  At 64 materials, self-ranked induction is the strongest non-oracle condition
  (`0.073` signed gain), ahead of sample-aware self-ranked induction (`0.062`)
  and QA expansion (`0.036`).
- The tiny-MLP profile sweep shows that bigger is not automatically better.
  At 64 materials, `epochs=32_hidden=8` improves self-ranked induction to
  `0.078` signed gain while using about one quarter of the neural train ops of
  the original `32x32` profile. Sample-aware self-ranked induction can clear
  the target with the cheaper `16x8` profile, and oracle counterfactual
  expansion already clears the target with `8x8`.
- The efficient-profile neural budget sweep confirms that the `32x8` profile is
  not only a single-budget accident. Relative to the original `32x32` budget
  sweep, it uses exactly one quarter of the estimated neural training
  multiply-adds at every matched condition and budget. Thresholds are unchanged:
  oracle counterfactual expansion first reaches target at 32 materials, while
  QA expansion, self-ranked induction, and sample-aware self-ranked induction
  first reach target at 48. Best signed gains improve slightly for QA
  (`0.036` to `0.041`), self-ranked induction (`0.073` to `0.078`),
  sample-aware self-ranked induction (`0.062` to `0.065`), and oracle
  counterfactual expansion (`0.121` to `0.124`). The artifact also keeps the
  negative case visible: raw text stays below target, and oracle
  counterfactual expansion at 64 materials dips from `0.106` to `0.101` even
  though its best budget improves.
- The feature-dimension sweep shows that the cheaper `32x8` profile was
  under-resolved at 128 hashed features, not over-resolved. At the 64-material
  budget, 16, 32, and 64 features fail the ranked train-only conditions, while
  256 features improves every condition over the 128-feature run. Self-ranked
  induction rises from `0.078` to `0.132` signed gain and sample-aware
  self-ranked induction rises from `0.065` to `0.130`; the neural train-op
  increase at 64 materials is about `1.03x` for those ranked conditions.
- A full 256-feature budget rerun confirms that the feature result is not only
  a 64-material profile artifact. Against the 128-feature `32x8` sweep, target
  thresholds stay the same for QA and ranked train-only conditions (`48`
  materials) and oracle counterfactual expansion (`32` materials), while raw
  text first reaches target at `64`. Best signed gains improve from `0.041` to
  `0.078` for QA, `0.078` to `0.132` for self-ranked induction, `0.065` to
  `0.130` for sample-aware self-ranked induction, and `0.124` to `0.197` for
  oracle counterfactual expansion. The result still has a clear low-budget
  limit: train-only ranked conditions remain negative at 24 and 32 materials.
- The wider feature sweep shows that the hashed feature frontier had not yet
  saturated at 256. At 64 materials, 1024 features improves signed gain and
  signed LSD over 256 for every condition while increasing estimated neural
  train ops by only about `1.01x` to `1.02x`. Self-ranked induction rises from
  `0.132` to `0.145` signed gain, sample-aware self-ranked induction rises
  from `0.130` to `0.156`, and oracle counterfactual expansion rises from
  `0.174` to `0.247`.
- A full 1024-feature budget rerun confirms the wider-feature result across the
  sample-budget frontier. Against the 256-feature sweep, raw text first reaches
  target at `48` rather than `64`, oracle counterfactual expansion first reaches
  target at `24` rather than `32`, and the transformed train-only conditions
  keep the same `48`-material threshold while improving best signed gain. The
  low-budget caveat remains: self-ranked and sample-aware self-ranked induction
  are still negative at 24 and 32 materials, so feature capacity helps but does
  not solve scarce-data generated-label noise.
- The 1024-feature profile sweep shows that the old `32x8` profile is no
  longer the only efficient point. At 64 materials, `16x8` is best for
  self-ranked induction (`0.153` signed gain) and signed LSD (`0.005247`),
  using exactly half the neural train ops of `32x8`. The full `16x8`
  1024-feature budget rerun confirms this tradeoff: self-ranked induction keeps
  the `48`-material target threshold, improves best signed gain from `0.145` to
  `0.153`, and halves neural train ops at every matched budget. It is not a
  universal replacement: raw text no longer reaches target, QA best gain falls
  from `0.101` to `0.052`, oracle counterfactual best gain falls from `0.252`
  to `0.239`, and ranked train-only conditions are still negative at 24 and 32
  materials.
- The `8x8` low-epoch ablation shows where undertraining starts to bite. It
  halves neural train ops again and improves sample-aware self-ranked induction
  at the very smallest budgets (`-0.084` to `0.011` at 16 materials and
  `-0.041` to `0.007` at 24), but it loses the self-ranked target entirely:
  best self-ranked gain falls from `0.153` to `0.021`, and sample-aware first
  reaches target only at 64 materials. This suggests the 24/32-material failure
  is not solved by simply reducing training steps; it likely needs a better
  generated-label policy or reliability gate.
- The post-hoc non-oracle policy envelope confirms that this is a selector
  problem, not only a condition-design problem. It chooses MDL rule expansion at
  16 materials, raw text at 24, MDL again at 32, validation-ranked induction at
  48, and self-ranked induction at 64. Because this selection uses completed
  heldout results, it is not deployable; even so, the target is first reached
  only at 48 materials. The next useful experiment should learn that switching
  rule from train/validation signals before heldout is opened.
- The validation-portfolio selector is the first deployable attempt at that
  switching rule. It reaches target at 48 materials and matches the post-hoc
  envelope at 16 materials, but it remains negative at 24 and 32, peaks at
  `0.109` signed gain versus `0.153` for fixed self-ranked induction, and its
  64-material signed LSD falls to `0.000627` because six candidate trainings
  are charged. The result is useful because it rules out a naive "try every
  policy on validation" fix.
- The linear-proxy selector improves that deployable selector frontier without
  pretending the low-budget problem is solved. It reaches `0.103` signed gain
  at 48 materials and `0.153` at 64, improves 64-material signed LSD from the
  portfolio selector's `0.000627` to `0.002091`, and cuts charged compute at 64
  from `734.1k` to `316.9k`. It is still negative at 16, 24, and 32 materials
  and remains less density-efficient than the best fixed train-only policies.
- The abstaining-proxy selector improves downside control but not the core
  frontier. It moves the proxy result from `-0.105` to `0.000` at 16 materials
  and from `-0.041` to `0.006897` at 24 materials, while keeping the same
  `0.153` gain and `0.002091` signed LSD at 64. It still fails at 32
  materials and lowers the 48-material gain from `0.103` to `0.082759`, so the
  useful next step is a better reliability signal rather than stricter
  abstention alone.
- The fresh-seed selector-transfer stress test is a negative control for that
  story. On seeds `37 41 43 47 53`, the abstaining and linear proxy selectors
  fall to `0.114286` best signed gain, the full portfolio reaches `0.127273`
  at much lower density, and fixed sample-aware self-ranked induction reaches
  `0.142857` with much better 64-material signed LSD. At 32 materials, raw text
  is less negative than every deployable selector. The selector direction is
  therefore not paper-ready without a stronger reliability signal.
- The generated-label audit narrows the selector-transfer failure. At 32
  materials, agreement-gated generated labels are `0.917241` correct against
  the hidden rulebook but still have `-0.131579` signed gain; oracle
  counterfactual labels are perfectly correct and only reach `0.015790` signed
  gain at the same budget. At 64 materials, agreement-gated labels are more
  precise than sample-aware labels (`0.954783` versus `0.829565`) but have
  lower gain (`0.080519` versus `0.142857`). Label correctness is therefore
  necessary but not sufficient; coverage, distribution, learner interaction,
  and charged selection cost remain live bottlenecks.
- The generated-coverage audit makes that bottleneck measurable. At 64
  materials, sample-aware induction has lower generated-versus-heldout motif
  L1 distance than agreement-gated induction (`0.477843` versus `0.576443`)
  and higher gain (`0.142857` versus `0.080519`) despite lower label
  precision. At 32 materials, validation-ranked induction is much less
  negative than agreement-gated induction and also has lower motif L1 distance
  (`0.683666` versus `0.764428`). The next deployable policy needs a
  train/validation-side proxy for this coverage signal.
- The validation-coverage proxy is the first deployable test of that mechanism
  on fresh seeds `103 107 109 113 127`. It uses validation motif coverage, not
  heldout distribution or validation labels, to choose one candidate. The result
  is useful but mixed: it moves the 32-material row above fixed sample-aware
  induction (`0.010526` versus `-0.036842`) and reaches `0.171428` at 64, but
  it selects MDL at every 24-material seed and falls to `-0.082759`. Coverage
  is therefore a real signal, but this proxy is not yet a robust selector.
- The coverage-prior selector tests the next mechanistic fix on fresh seeds
  `601 607 613 617 619`: raw text below 96 train events, then a lean
  validation-coverage scan over raw, sample-aware, and validation-ranked
  candidates. It removes the old 24-material coverage failure (`0.000000`
  versus `-0.062069`) and improves coverage-proxy density at 48/64
  (`0.004756` versus `0.003147`, and `0.005001` versus `0.003171`). It still
  does not beat the train-size gate at 48/64 (`0.005552` and `0.006089`), so
  the result is a cost-control improvement, not a final selector.
- The tempered sample-aware ablation lowers the mid-budget synthetic ratio from
  `0.75` to `0.50` on fresh seeds `157 163 167 173 179`. It reduces
  sample-aware damage at 24 materials (`-0.096552` versus `-0.137931`) and 32
  materials (`-0.078947` versus `-0.152632`) while preserving the 48/64
  high-budget gains. It still loses to the train-size raw fallback at 24 and
  32, so budget tempering is useful but not enough.
- The train-size gated baseline adds a second unseen seed check on seeds
  `59 61 67 71 73`. It reaches the same `0.145454` best gain and `0.005090`
  best signed LSD as fixed sample-aware induction while using raw text at 16,
  24, and 32 materials. It is still negative at 16 and 32, so it is not a
  solution; it is a cheap schedule baseline that beats the abstaining proxy on
  64-material gain and density and exposes how much selector search cost must
  be justified by any future adaptive rule.
- The compact train-size gated efficiency probe adds fresh seeds
  `181 191 193 197 199`. It is identical to the train-size gate through 48
  materials, then improves the 64-material row from `0.132467` gain and
  `0.004634` signed LSD to `0.140260` gain and `0.007883` signed LSD by
  dropping original QA duplicates while keeping train-only generated labels.
- The diversity interaction probe adds fresh seeds `701 709 719 727 733`. It
  improves the 64-material sample-aware row from `0.158441` gain and
  `0.005544` signed LSD to `0.168831` and `0.005908`, but compact diversity
  loses to compact train-size gating (`0.116883` gain and `0.006573` signed
  LSD versus `0.135065` and `0.007594`). Diversity is a candidate-ordering
  help inside the full sample-aware view, not a replacement for compact
  representation savings.
- The density-capped compact probe extends the budget range to 128 materials
  on fresh seeds `293 307 311 313 317`. It keeps compact induction through 96
  materials, then returns to raw text from 104 onward. At 128 materials this
  trades gain (`0.132468` to `0.081818`) for higher signed LSD (`0.001860` to
  `0.003452`), showing that the density frontier can switch back to raw data
  when external evidence is abundant.
- The support-ramped compact probe tests the middle of that tradeoff on fresh
  seeds `401 409 419 421 431`. It matches compact through 96 materials, then
  raises generated-label minimum support from `3` to `4`. At 104 materials it
  improves signed LSD over both plain compact and raw fallback (`0.003735`
  versus `0.002857` and `0.002145`). At 128 materials it recovers more gain
  than raw fallback (`0.184416` versus `0.154545`) and much more than plain
  compact (`0.119481`), but raw fallback remains denser (`0.006521` versus
  `0.005153`). The useful claim is a Pareto tradeoff, not a universal
  improvement.
- The late-confidence compact control tests whether that support-ramped policy
  only needed a stricter confidence floor at later abundant budgets. On fresh
  seeds `499 503 509 521 523`, it improves the 120-material support-ramped row
  from `0.134722` gain and `0.004119` signed LSD to `0.138889` and `0.004322`,
  but raw/density-capped fallback is still better at that same row (`0.145833`
  gain and `0.006977` signed LSD). Its best signed gain, `0.171098`, also
  remains below plain compact's `0.196875`. Confidence tightening alone is
  therefore a useful negative control, not the next frontier.
- The density-window compact probe tests whether a fixed transition window is
  enough. On fresh seeds `929 937 941 947 953`, it matches compact at 64/80,
  raw at 96/104/120, support-ramped compact at 112, and raw at 128. The window
  improves the 112-material density over density-capped/raw fallback
  (`0.004269` versus `0.004001`) and keeps the 120-material raw density
  (`0.005648` versus support-ramped `0.004899`), but it misses the
  support-ramped 128-material row (`0.003726` versus `0.004290`). Fixed windows
  are therefore a sharper control, not the solved selector.

## Research Thesis

Apparent sample efficiency is determined less by raw external exposure than by
the density of useful learning signal extracted per exposure. Selection,
transformation, replay, and dense feedback can improve external sample
efficiency, but often by spending more internal compute. That tradeoff is the
object of measurement.

## Authorship

Francesco Giannicola, Limes Labs.
