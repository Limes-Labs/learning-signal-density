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

Do not edit generated result JSON by hand. If the code changes, regenerate the
artifact and rerun tests.
