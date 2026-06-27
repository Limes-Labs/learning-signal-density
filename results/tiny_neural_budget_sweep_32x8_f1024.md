# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T20:56:22Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_feature_sweep_wide.json`

Backend: `tiny_mlp`
Profile label: `epochs=32_hidden=8_features=1024`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 48 | 64 | 0.070 | 0.005918 |
| qa_expansion | 48 | 64 | 0.101 | 0.003089 |
| self_ranked_induction | 48 | 64 | 0.145 | 0.002623 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.156 | 0.002843 |
| counterfactual_expansion | 24 | 48 | 0.252 | 0.003882 |

Comparison target: `results/tiny_neural_budget_sweep_32x8_f256.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.495 | -0.084 | 12992.0 | 799232 | -0.111754 |
| qa_expansion | 0.537 | -0.042 | 35960.0 | 1969664 | -0.020188 |
| self_ranked_induction | 0.516 | -0.063 | 60169.2 | 3140864 | -0.018075 |
| sample_aware_self_ranked_induction | 0.526 | -0.053 | 42745.2 | 2252288 | -0.021165 |
| counterfactual_expansion | 0.600 | 0.021 | 127136.0 | 6652672 | 0.002855 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.600 | -0.055 | 19264.0 | 1185792 | -0.033302 |
| qa_expansion | 0.593 | -0.062 | 53320.0 | 2921984 | -0.013536 |
| self_ranked_induction | 0.621 | -0.034 | 89662.8 | 4659712 | -0.004490 |
| sample_aware_self_ranked_induction | 0.634 | -0.021 | 80950.8 | 4215296 | -0.002984 |
| counterfactual_expansion | 0.759 | 0.103 | 188512.0 | 9868032 | 0.006381 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.421 | -0.211 | 25984.0 | 1597952 | -0.069846 |
| qa_expansion | 0.516 | -0.116 | 71920.0 | 3938304 | -0.013879 |
| self_ranked_induction | 0.500 | -0.132 | 121005.6 | 6278144 | -0.009376 |
| sample_aware_self_ranked_induction | 0.542 | -0.089 | 109521.6 | 5693440 | -0.007043 |
| counterfactual_expansion | 0.716 | 0.084 | 254272.0 | 13302272 | 0.002855 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.607 | 0.038 | 38528.0 | 2370816 | 0.005724 |
| qa_expansion | 0.634 | 0.066 | 106640.0 | 5842432 | 0.003572 |
| self_ranked_induction | 0.686 | 0.117 | 179205.6 | 9312256 | 0.003803 |
| sample_aware_self_ranked_induction | 0.686 | 0.117 | 179205.6 | 9312256 | 0.003803 |
| counterfactual_expansion | 0.821 | 0.252 | 377024.0 | 19727616 | 0.003882 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.662 | 0.070 | 51520.0 | 3167232 | 0.005918 |
| qa_expansion | 0.694 | 0.101 | 142600.0 | 7806464 | 0.003089 |
| self_ranked_induction | 0.738 | 0.145 | 241050.0 | 12448000 | 0.002623 |
| sample_aware_self_ranked_induction | 0.748 | 0.156 | 238335.6 | 12447744 | 0.002843 |
| counterfactual_expansion | 0.839 | 0.247 | 504160.0 | 26367232 | 0.002128 |
