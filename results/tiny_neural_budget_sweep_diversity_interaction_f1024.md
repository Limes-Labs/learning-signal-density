# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T23:16:36Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_diversity_interaction`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 64 | 0.013 | 0.002192 |
| self_ranked_induction | 48 | 64 | 0.132 | 0.004537 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.158 | 0.005544 |
| diverse_self_ranked_induction | 48 | 64 | 0.151 | 0.005159 |
| sample_aware_diverse_self_ranked_induction | 48 | 64 | 0.169 | 0.005908 |
| train_size_gated_sample_aware_induction | 48 | 64 | 0.158 | 0.005544 |
| compact_train_size_gated_induction | 48 | 64 | 0.135 | 0.007594 |
| compact_diverse_train_size_gated_induction | 48 | 64 | 0.117 | 0.006573 |
| counterfactual_expansion | 48 | 64 | 0.231 | 0.003902 |

Comparison target: `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.600 | 0.000 | 6496.0 | 399616 | 0.000000 |
| self_ranked_induction | 0.463 | -0.042 | 31545.2 | 1570304 | -0.022979 |
| sample_aware_self_ranked_induction | 0.505 | 0.000 | 22569.2 | 1126144 | -0.000026 |
| diverse_self_ranked_induction | 0.453 | -0.053 | 31545.2 | 1569792 | -0.028753 |
| sample_aware_diverse_self_ranked_induction | 0.526 | -0.074 | 22569.2 | 1126144 | -0.056398 |
| train_size_gated_sample_aware_induction | 0.600 | 0.000 | 6496.0 | 399616 | 0.000000 |
| compact_train_size_gated_induction | 0.600 | 0.000 | 6496.0 | 399616 | 0.000000 |
| compact_diverse_train_size_gated_induction | 0.600 | 0.000 | 6496.0 | 399616 | 0.000000 |
| counterfactual_expansion | 0.484 | -0.116 | 64960.0 | 3326336 | -0.030732 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.628 | 0.000 | 9632.0 | 592896 | 0.000000 |
| self_ranked_induction | 0.490 | -0.138 | 47165.2 | 2329344 | -0.034026 |
| sample_aware_self_ranked_induction | 0.503 | -0.124 | 42677.2 | 2107264 | -0.033864 |
| diverse_self_ranked_induction | 0.524 | -0.103 | 47165.2 | 2328960 | -0.025532 |
| sample_aware_diverse_self_ranked_induction | 0.490 | -0.138 | 42677.2 | 2106752 | -0.037600 |
| train_size_gated_sample_aware_induction | 0.628 | 0.000 | 9632.0 | 592896 | 0.000000 |
| compact_train_size_gated_induction | 0.628 | 0.000 | 9632.0 | 592896 | 0.000000 |
| compact_diverse_train_size_gated_induction | 0.628 | 0.000 | 9632.0 | 592896 | 0.000000 |
| counterfactual_expansion | 0.621 | -0.007 | 96320.0 | 4934016 | -0.000833 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | -0.042 | 12992.0 | 798976 | -0.027939 |
| self_ranked_induction | 0.537 | -0.084 | 63599.2 | 3139712 | -0.011410 |
| sample_aware_self_ranked_induction | 0.521 | -0.100 | 57683.2 | 2847232 | -0.014963 |
| diverse_self_ranked_induction | 0.563 | -0.058 | 63599.2 | 3139328 | -0.007837 |
| sample_aware_diverse_self_ranked_induction | 0.511 | -0.111 | 57683.2 | 2846464 | -0.016517 |
| train_size_gated_sample_aware_induction | 0.579 | -0.042 | 12992.0 | 798976 | -0.027939 |
| compact_train_size_gated_induction | 0.579 | -0.042 | 12992.0 | 798976 | -0.027939 |
| compact_diverse_train_size_gated_induction | 0.579 | -0.042 | 12992.0 | 798976 | -0.027939 |
| counterfactual_expansion | 0.637 | 0.016 | 129920.0 | 6651136 | 0.001048 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.593 | 0.010 | 19264.0 | 1185408 | 0.003122 |
| self_ranked_induction | 0.672 | 0.090 | 93888.8 | 4656768 | 0.005551 |
| sample_aware_self_ranked_induction | 0.672 | 0.090 | 93888.8 | 4656768 | 0.005551 |
| diverse_self_ranked_induction | 0.662 | 0.079 | 93888.8 | 4657664 | 0.004910 |
| sample_aware_diverse_self_ranked_induction | 0.662 | 0.079 | 93888.8 | 4657664 | 0.004910 |
| train_size_gated_sample_aware_induction | 0.672 | 0.090 | 93888.8 | 4656768 | 0.005551 |
| compact_train_size_gated_induction | 0.672 | 0.090 | 93888.8 | 4656768 | 0.005551 |
| compact_diverse_train_size_gated_induction | 0.672 | 0.090 | 93888.8 | 4656768 | 0.005551 |
| counterfactual_expansion | 0.769 | 0.186 | 192640.0 | 9863808 | 0.005620 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.592 | 0.013 | 25760.0 | 1583616 | 0.002192 |
| self_ranked_induction | 0.712 | 0.132 | 126970.0 | 6223744 | 0.004537 |
| sample_aware_self_ranked_induction | 0.738 | 0.158 | 124255.6 | 6223488 | 0.005544 |
| diverse_self_ranked_induction | 0.730 | 0.151 | 126970.0 | 6224256 | 0.005159 |
| sample_aware_diverse_self_ranked_induction | 0.748 | 0.169 | 124255.6 | 6223488 | 0.005908 |
| train_size_gated_sample_aware_induction | 0.738 | 0.158 | 124255.6 | 6223488 | 0.005544 |
| compact_train_size_gated_induction | 0.714 | 0.135 | 77335.6 | 3903872 | 0.007594 |
| compact_diverse_train_size_gated_induction | 0.696 | 0.117 | 77335.6 | 3903872 | 0.006573 |
| counterfactual_expansion | 0.810 | 0.231 | 257600.0 | 13183616 | 0.003902 |
