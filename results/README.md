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

16x8 1024-feature density-window compact transition probe:

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

The density-window compact policy is train-only. It uses compact induction
below 320 train events, raw text from 320 to 400, support-ramped compact from
400 to 432, and raw text again after 432. On seeds `929 937 941 947 953`, the
fixed window improves the 112-material signed LSD over density-capped/raw
fallback (`0.004269` versus `0.004001`) and keeps the 120-material raw density
(`0.005648` versus support-ramped `0.004899`). It also misses the
support-ramped 128-material row (`0.003726` versus `0.004290`), so the artifact
is evidence for a local transition-window tradeoff, not a solved selector.

16x8 1024-feature train-support-density selector-cost control:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_train_support_density_f1024.json \
  --output-md results/tiny_neural_budget_sweep_train_support_density_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1033 1039 1049 1051 1061 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction train_support_density_selector density_window_compact_induction density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_density_window_compact_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_density_window_compact_f1024.json \
  --profile-label f1024_16x8_train_support_density
```

The train-support-density selector is train-only and chooses among raw text,
compact train-size gated induction, and support-ramped compact induction by
checking support kept per charged compute. Candidate inspection is charged. On
seeds `1033 1039 1049 1051 1061`, it chooses support-ramped compact at 104 and
112 materials, raw text at all 128-material runs, and a raw/support mix at 120.
That is directionally sensible, but the overhead is too expensive: at 104
materials it matches support-ramped gain while reducing signed LSD from
`0.003994` to `0.003262`; at 120 materials it reaches `0.003844` while
raw/density-window reaches `0.005515`; at 128 materials it reaches `0.004142`
while raw reaches `0.005809`. This artifact is a selector-cost failure, not a
frontier improvement.

16x8 1024-feature reuse-aware support-probe window control:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_support_probe_window_f1024.json \
  --output-md results/tiny_neural_budget_sweep_support_probe_window_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1063 1069 1087 1091 1093 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction train_support_density_selector support_probe_window_selector density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_train_support_density_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_train_support_density_f1024.json \
  --profile-label f1024_16x8_support_probe_window
```

The support-probe window selector is train-only and reuse-aware. It uses compact
induction below 320 train events, raw text outside the 360--432 support-probe
window, and inside that window inspects only support-ramped compact induction.
If support is selected, the candidate construction is reused rather than
charged again. On seeds `1063 1069 1087 1091 1093`, this removes the 104-material
selector overhead: signed LSD is `0.004655` versus `0.003805` for the full
train-support-density selector. The result is still mixed. At 112 materials the
selector chooses support, but raw/density-capped is denser (`0.007103` versus
`0.005161`); at 120 materials the selector chooses raw, but support-ramped is
denser (`0.004240` versus `0.003588`). The artifact isolates accounting reuse
from the harder policy-selection problem.

16x8 1024-feature validation support-precision selector:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_support_precision_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_support_precision_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1259 1277 1279 1283 1289 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction support_probe_window_selector validation_support_precision_selector train_support_density_selector density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_support_probe_window_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_support_probe_window_f1024.json \
  --profile-label f1024_16x8_validation_support_precision
```

The validation support-precision selector keeps the compact floor below 320
train events and the fixed 400--432 support transition, then uses validation
labels only outside that fixed transition to estimate eligible induced-prediction
precision before choosing support-ramped compact or raw text. On seeds `1259
1277 1279 1283 1289`, it improves the raw/support boundary at 96 materials
(`0.004198` signed LSD versus `0.002075` for support-probe/raw) and at 104
materials (`0.004455` versus `0.004331` for support-probe/support-ramped). The
average signed LSD across 64--128 materials is `0.006138`, above the
support-probe window's `0.005941`, but the misses remain visible: at 120
materials raw/support-probe reaches `0.006246` while the selector reaches
`0.005723`, and at 128 raw reaches `0.005206` while the selector reaches
`0.004866`.

16x8 1024-feature no-window validation support-precision gate:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1381 1399 1409 1423 1427 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction support_probe_window_selector validation_support_precision_selector validation_support_precision_gate_selector train_support_density_selector density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_support_precision_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_support_precision_f1024.json \
  --profile-label f1024_16x8_validation_support_precision_gate
```

The no-window gate removes the fixed 400--432 train-event support prior and
applies the same validation precision threshold everywhere above the compact
floor. On seeds `1381 1399 1409 1423 1427`, it slightly improves over the
support-probe average (`0.006104` versus `0.006074`) but falls below the
fixed-transition validation selector (`0.006223`). The main miss is at 112
materials: the gate reaches `0.005031` signed LSD while the fixed-transition
selector and support-ramped compact reach `0.005864`. This is a negative control
for removing the transition prior, not a promoted selector.

16x8 1024-feature support-selector transfer stress:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json \
  --output-md results/tiny_neural_budget_sweep_support_selector_transfer_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1459 1471 1481 1483 1487 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction support_probe_window_selector validation_support_precision_selector validation_support_precision_gate_selector train_support_density_selector density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json \
  --profile-label f1024_16x8_support_selector_transfer
```

The transfer stress reruns the latest support-selector family on fresh seeds
`1459 1471 1481 1483 1487` after the fixed selector and no-window gate were
developed. The no-window gate transfers better than the fixed-transition
validation selector on average (`0.005936` versus `0.005601` signed LSD), mainly
because it avoids the fixed 112-material support choice (`0.006517` versus
`0.004175`). The result is still not a promoted selector: the simple train-only
density-capped raw fallback has the strongest all-budget average in this block
(`0.006115`), and raw/probe still beat the validation gate at 120 and 128
materials.

Build the post-hoc support-selector error audit:

```bash
python3 scripts/build_support_selector_error_audit.py
```

The audit reads the committed high-budget support-selector JSON artifacts and
computes signed-LSD regret against the best simple comparator in each source
seed block. It is explicitly non-deployable because completed heldout outcomes
define the error ledger. The transfer block remains negative for promotion: the
least-regret gate has `0.000496` average regret versus the best simple
comparator and wins only `1/7` budgets.

Run the validation support-utility selector follow-up:

```bash
python3 -m learning_signal_density.neural_sweep \
  --output-json results/tiny_neural_budget_sweep_validation_support_utility_f1024.json \
  --output-md results/tiny_neural_budget_sweep_validation_support_utility_f1024.md \
  --material-counts 64 80 96 104 112 120 128 \
  --seeds 1601 1607 1609 1613 1619 \
  --conditions raw_text compact_train_size_gated_induction support_ramped_compact_induction density_window_compact_induction support_probe_window_selector validation_support_precision_selector validation_support_precision_gate_selector validation_support_utility_selector train_support_density_selector density_capped_compact_induction counterfactual_expansion \
  --epochs 16 \
  --hidden-units 8 \
  --feature-dimension 1024 \
  --learning-rate 0.03 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/support_mechanism_audit_f1024.json \
  --comparison-of results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json \
  --profile-label f1024_16x8_validation_support_utility
```

The utility follow-up uses a validation-precision prefilter before paying for a
validation-label, validation-motif-coverage, and compute-penalized support
score. The prefilter improves accounting versus the first utility attempt, but
the fresh-seed result remains negative for promotion: `0.005473` average signed
LSD versus `0.005746` for the no-window precision gate and `0.005721` for the
density-capped fallback.

Build the post-hoc support-ramp mechanism audit:

```bash
python3 scripts/build_support_mechanism_audit.py
```

The mechanism audit reconstructs the support-selector transfer candidates after
the neural sweep and compares generated labels with the hidden rulebook and
heldout motif distribution. It is non-deployable and should be read as a
failure-mechanism diagnostic: support-ramped compact has zero precision
improvements over compact induction, loses heldout motif coverage on all four
transition budgets, and beats the density-capped fallback on signed LSD only at
104 materials.

Do not edit generated result JSON by hand. If the code changes, regenerate the
artifact and rerun tests.
