# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T21:09:50Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_profile_sweep_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 48 | 0.003 | 0.001041 |
| qa_expansion | 48 | 64 | 0.052 | 0.003107 |
| self_ranked_induction | 48 | 64 | 0.153 | 0.005247 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.151 | 0.005270 |
| counterfactual_expansion | 24 | 64 | 0.239 | 0.004033 |

Comparison target: `results/tiny_neural_budget_sweep_32x8_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | 0.000 | 6496.0 | 399616 | 0.000000 |
| qa_expansion | 0.537 | -0.042 | 18328.0 | 984832 | -0.039609 |
| self_ranked_induction | 0.505 | -0.074 | 31401.2 | 1570432 | -0.040421 |
| sample_aware_self_ranked_induction | 0.495 | -0.084 | 22425.2 | 1126144 | -0.064766 |
| counterfactual_expansion | 0.516 | -0.063 | 64960.0 | 3326336 | -0.016763 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.655 | 0.000 | 9632.0 | 592896 | 0.000000 |
| qa_expansion | 0.607 | -0.048 | 27176.0 | 1460992 | -0.020656 |
| self_ranked_induction | 0.552 | -0.103 | 47006.8 | 2329856 | -0.025595 |
| sample_aware_self_ranked_induction | 0.614 | -0.041 | 42518.8 | 2107648 | -0.011346 |
| counterfactual_expansion | 0.731 | 0.076 | 96320.0 | 4934016 | 0.009158 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.568 | -0.063 | 12992.0 | 798976 | -0.041908 |
| qa_expansion | 0.526 | -0.105 | 36656.0 | 1969152 | -0.024756 |
| self_ranked_induction | 0.521 | -0.111 | 63469.6 | 3139072 | -0.015011 |
| sample_aware_self_ranked_induction | 0.532 | -0.100 | 57553.6 | 2846720 | -0.014971 |
| counterfactual_expansion | 0.668 | 0.037 | 129920.0 | 6651136 | 0.002445 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | 0.003 | 19264.0 | 1185408 | 0.001041 |
| qa_expansion | 0.610 | 0.041 | 54352.0 | 2921216 | 0.004427 |
| self_ranked_induction | 0.645 | 0.076 | 93893.6 | 4656128 | 0.004695 |
| sample_aware_self_ranked_induction | 0.645 | 0.076 | 93893.6 | 4656128 | 0.004695 |
| counterfactual_expansion | 0.762 | 0.193 | 192640.0 | 9863808 | 0.005828 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.574 | -0.018 | 25760.0 | 1583616 | -0.003069 |
| qa_expansion | 0.644 | 0.052 | 72680.0 | 3903232 | 0.003107 |
| self_ranked_induction | 0.745 | 0.153 | 126970.0 | 6224000 | 0.005247 |
| sample_aware_self_ranked_induction | 0.743 | 0.151 | 124255.6 | 6223872 | 0.005270 |
| counterfactual_expansion | 0.831 | 0.239 | 257600.0 | 13183616 | 0.004033 |
