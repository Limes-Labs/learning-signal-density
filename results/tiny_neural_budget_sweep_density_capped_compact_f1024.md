# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T15:39:37Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_density_capped_compact`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 96 | 96 | 0.099 | 0.007393 |
| train_size_gated_sample_aware_induction | 64 | 96 | 0.176 | 0.002955 |
| compact_train_size_gated_induction | 64 | 96 | 0.163 | 0.004668 |
| density_capped_compact_induction | 64 | 96 | 0.163 | 0.004668 |
| counterfactual_expansion | 64 | 80 | 0.235 | 0.002534 |

Comparison target: `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.592 | -0.021 | 25760.0 | 1583360 | -0.003507 |
| train_size_gated_sample_aware_induction | 0.769 | 0.156 | 124236.4 | 6221696 | 0.005454 |
| compact_train_size_gated_induction | 0.753 | 0.140 | 77316.4 | 3902336 | 0.007887 |
| density_capped_compact_induction | 0.753 | 0.140 | 77316.4 | 3902336 | 0.007887 |
| counterfactual_expansion | 0.792 | 0.179 | 257600.0 | 13183360 | 0.003025 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.633 | 0.013 | 32256.0 | 1984128 | 0.001346 |
| train_size_gated_sample_aware_induction | 0.769 | 0.148 | 138988.8 | 7038362 | 0.003696 |
| compact_train_size_gated_induction | 0.775 | 0.154 | 80236.8 | 4132634 | 0.006706 |
| density_capped_compact_induction | 0.775 | 0.154 | 80236.8 | 4132634 | 0.006706 |
| counterfactual_expansion | 0.856 | 0.235 | 322560.0 | 16514688 | 0.002534 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.685 | 0.099 | 38752.0 | 2380032 | 0.007393 |
| train_size_gated_sample_aware_induction | 0.762 | 0.176 | 171806.0 | 8668416 | 0.002955 |
| compact_train_size_gated_induction | 0.750 | 0.163 | 101222.0 | 5181184 | 0.004668 |
| density_capped_compact_induction | 0.750 | 0.163 | 101222.0 | 5181184 | 0.004668 |
| counterfactual_expansion | 0.812 | 0.226 | 387520.0 | 19820032 | 0.001686 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.670 | 0.064 | 41888.0 | 2572544 | 0.004085 |
| train_size_gated_sample_aware_induction | 0.725 | 0.118 | 202099.6 | 10115712 | 0.001567 |
| compact_train_size_gated_induction | 0.734 | 0.128 | 125803.6 | 6346368 | 0.002720 |
| density_capped_compact_induction | 0.670 | 0.064 | 41888.0 | 2572544 | 0.004085 |
| counterfactual_expansion | 0.821 | 0.214 | 418880.0 | 21421184 | 0.001369 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.651 | 0.055 | 45248.0 | 2780544 | 0.003021 |
| train_size_gated_sample_aware_induction | 0.734 | 0.139 | 216892.0 | 10880256 | 0.001584 |
| compact_train_size_gated_induction | 0.690 | 0.094 | 134476.0 | 6806912 | 0.001731 |
| density_capped_compact_induction | 0.651 | 0.055 | 45248.0 | 2780544 | 0.003021 |
| counterfactual_expansion | 0.801 | 0.206 | 452480.0 | 23145984 | 0.001127 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.668 | 0.064 | 48384.0 | 2976256 | 0.003056 |
| train_size_gated_sample_aware_induction | 0.722 | 0.118 | 233426.4 | 11685888 | 0.001171 |
| compact_train_size_gated_induction | 0.707 | 0.103 | 145298.4 | 7327232 | 0.001637 |
| density_capped_compact_induction | 0.668 | 0.064 | 48384.0 | 2976256 | 0.003056 |
| counterfactual_expansion | 0.797 | 0.193 | 483840.0 | 24769536 | 0.000924 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.690 | 0.082 | 51520.0 | 3162112 | 0.003452 |
| train_size_gated_sample_aware_induction | 0.751 | 0.143 | 248640.8 | 12424576 | 0.001249 |
| compact_train_size_gated_induction | 0.740 | 0.132 | 154800.8 | 7790464 | 0.001860 |
| density_capped_compact_induction | 0.690 | 0.082 | 51520.0 | 3162112 | 0.003452 |
| counterfactual_expansion | 0.826 | 0.218 | 515200.0 | 26341632 | 0.000921 |
