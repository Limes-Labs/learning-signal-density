# Results

This directory stores checked-in result artifacts that can be regenerated from
the repository.

Current canonical pilot:

```bash
python3 -m learning_signal_density \
  --output-json results/causal_world_pilot_seedset.json \
  --output-md results/causal_world_pilot_seedset.md \
  --seeds 3 5 7 11 13 \
  --material-count 48 \
  --epochs 5
```

Current sample-budget sweep:

```bash
python3 -m learning_signal_density.sweep \
  --output-json results/sample_budget_sweep.json \
  --output-md results/sample_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 3 5 7 11 13 \
  --epochs 5 \
  --target-signed-gain 0.03
```

Fresh-seed confirmation sweep:

```bash
python3 -m learning_signal_density.sweep \
  --output-json results/confirmation_budget_sweep.json \
  --output-md results/confirmation_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 5 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/sample_budget_sweep.json
```

Tiny neural replication:

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

Tiny neural fresh-seed confirmation:

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

Tiny neural budget sweep:

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

Tiny neural profile sweep:

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

Efficient tiny neural budget sweep:

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

Tiny neural feature-dimension sweep:

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

256-feature efficient tiny neural budget sweep:

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

Wide tiny neural feature-dimension sweep:

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

1024-feature efficient tiny neural budget sweep:

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

1024-feature tiny neural profile sweep:

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

16x8 1024-feature efficient tiny neural budget sweep:

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

8x8 1024-feature low-epoch ablation:

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

16x8 1024-feature validation-selected reliability probe:

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

16x8 1024-feature agreement-gated reliability probe:

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

Post-hoc policy envelope used by the paper tables:

```bash
python3 scripts/build_policy_envelope.py
```

This derived artifact uses completed heldout results to choose the best
non-oracle condition at each material budget. It is intentionally marked
non-deployable and should be interpreted as a selector diagnostic, not a
candidate learning policy.

Build the generated-label audit used by the paper tables:

```bash
python3 scripts/build_generated_label_audit.py
```

This derived artifact uses the synthetic hidden rulebook after the
selector-transfer sweep has run. It is intentionally non-deployable and should
be interpreted as a mechanism diagnostic, not a policy selector.

Build the generated-coverage audit used by the paper tables:

```bash
python3 scripts/build_generated_coverage_audit.py
```

This derived artifact compares generated-label motif frequencies with the
heldout motif distribution after the selector-transfer sweep has run. It is
non-deployable and should be read as a coverage mechanism diagnostic.

16x8 1024-feature validation-portfolio selector probe:

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

The portfolio selector trains and validates six non-oracle candidate policies,
charges that whole selection search, and only then evaluates the selected model
on heldout examples. It does not use heldout labels for selection.

16x8 1024-feature linear-proxy validation selector probe:

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

The linear-proxy selector charges a two-epoch linear fit for each candidate
policy, then trains only the selected tiny MLP before heldout evaluation. It
improves the 64-material deployable selector result from `0.109` gain and
`0.000627` signed LSD to `0.153` gain and `0.002091` signed LSD, but it still
fails at 16, 24, and 32 materials.

16x8 1024-feature abstaining-proxy validation selector probe:

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

The abstaining-proxy selector uses the same two-epoch proxy fits, but falls
back to raw text unless the best non-raw policy beats raw text by three
validation examples. It removes the 16-material proxy loss and makes 24
materials slightly positive, but it still fails at 32 materials and gives up
some 48-material gain.

16x8 1024-feature fresh-seed selector-transfer stress test:

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

This stress test uses fresh seeds after the selector policies were developed.
It is intentionally uncomfortable: fixed sample-aware self-ranked induction
beats the deployable selector family at 64 materials, and raw text is less
negative than the selector family at 32 materials.

16x8 1024-feature train-size gated baseline on a second unseen seed set:

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

The train-size gate uses raw text below 144 train events and switches to
sample-aware self-ranked induction at larger train splits. On seeds `59 61 67
71 73`, it reaches target at 48 materials and matches sample-aware induction at
64 materials (`0.145454` gain, `0.005090` signed LSD), without selector search
cost. It remains negative at 16 and 32 materials, so it should be read as a
cheap baseline and promotion gate rather than a scarce-budget fix.

16x8 1024-feature validation-coverage proxy on a fresh confirmation seed set:

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

The coverage proxy is deployable: it approximates the non-deployable
heldout-coverage audit with validation motif distribution and does not use
validation labels for its selector score. On seeds `103 107 109 113 127`, it
turns the 32-material row positive (`0.010526`) and reaches `0.171428` at 64
materials, but it overselects MDL rule expansion at 24 materials and falls to
`-0.082759`. Treat it as mechanism evidence, not as a robust selector.

16x8 1024-feature coverage-prior selector control:

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

The coverage-prior selector keeps the validation motif signal but adds a raw
floor below 96 train events, prunes the candidate set, and uses a small
compute penalty in the coverage score. On seeds `601 607 613 617 619`, it
removes the 24-material coverage-proxy failure (`0.000000` versus
`-0.062069`) and improves signed LSD at 48/64 versus the full coverage proxy.
It still trails the train-size gate on 48/64 density, so it is cost-control
evidence rather than a promoted selector.

16x8 1024-feature tempered sample-aware ablation:

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

The tempered policy is train-only and lowers the mid-budget synthetic ratio
from `0.75` to `0.50`. On seeds `157 163 167 173 179`, it improves over fixed
sample-aware induction at 24 materials (`-0.096552` versus `-0.137931`) and 32
materials (`-0.078947` versus `-0.152632`), but it remains worse than the raw
fallback used by the train-size gate at both scarce budgets.

16x8 1024-feature compact train-size gated efficiency probe:

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

The compact train-size gate is train-only. It matches the raw fallback below
144 train events, matches full sample-aware induction below 224 train events,
and at the large-sample tier keeps raw originals while dropping original QA
duplicates. On seeds `181 191 193 197 199`, it is identical to the train-size
gate through 48 materials, then improves the 64-material row from `0.132467`
gain and `0.004634` signed LSD to `0.140260` gain and `0.007883` signed LSD.

16x8 1024-feature diversity interaction probe:

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

The diversity interaction probe is train-only. It shows that modifier/stimulus
coverage balancing can help the full sample-aware view at 64 materials
(`0.168831` gain and `0.005908` signed LSD versus `0.158441` and `0.005544`),
but it hurts the compact density frontier (`0.116883` gain and `0.006573`
signed LSD versus compact's `0.135065` and `0.007594`).

16x8 1024-feature density-capped compact high-budget probe:

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

The density-capped compact policy is train-only. It matches compact
train-size gating through 96 materials, then returns to raw text once the train
split reaches the abundant-data tier. On seeds `293 307 311 313 317`, it
keeps the compact 64-material signed LSD (`0.007887`) and improves density
over compact at 104, 112, 120, and 128 materials. The tradeoff is explicit:
at 128 materials it gives up gain (`0.132468` to `0.081818`) to improve signed
LSD (`0.001860` to `0.003452`).

Do not edit generated result JSON by hand. If the code changes, regenerate the
artifact and rerun tests.
