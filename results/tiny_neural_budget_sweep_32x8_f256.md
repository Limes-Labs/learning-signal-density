# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T19:31:38Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_feature_sweep.json`

Backend: `tiny_mlp`
Profile label: `epochs=32_hidden=8_features=256`
Hidden units: `8`
Feature dimension: `256`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 64 | 64 | 0.031 | 0.002630 |
| qa_expansion | 48 | 64 | 0.078 | 0.002376 |
| self_ranked_induction | 48 | 64 | 0.132 | 0.002389 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.130 | 0.002369 |
| counterfactual_expansion | 32 | 48 | 0.197 | 0.003031 |

Comparison target: `results/tiny_neural_budget_sweep_32x8.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.484 | -0.095 | 12992.0 | 788224 | -0.125723 |
| qa_expansion | 0.516 | -0.063 | 35960.0 | 1932288 | -0.030282 |
| self_ranked_induction | 0.600 | 0.021 | 60169.2 | 3071488 | 0.006033 |
| sample_aware_self_ranked_induction | 0.526 | -0.053 | 42745.2 | 2209024 | -0.021243 |
| counterfactual_expansion | 0.568 | -0.011 | 127136.0 | 6504704 | -0.001428 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | -0.083 | 19264.0 | 1168640 | -0.049954 |
| qa_expansion | 0.579 | -0.076 | 53320.0 | 2868480 | -0.016544 |
| self_ranked_induction | 0.593 | -0.062 | 89662.8 | 4566016 | -0.008061 |
| sample_aware_self_ranked_induction | 0.607 | -0.048 | 80950.8 | 4132864 | -0.006937 |
| counterfactual_expansion | 0.648 | -0.007 | 188512.0 | 9666560 | -0.000425 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.389 | -0.242 | 25984.0 | 1570560 | -0.080323 |
| qa_expansion | 0.547 | -0.084 | 71920.0 | 3852800 | -0.010094 |
| self_ranked_induction | 0.489 | -0.142 | 121005.6 | 6129664 | -0.010124 |
| sample_aware_self_ranked_induction | 0.532 | -0.100 | 109521.6 | 5558016 | -0.007875 |
| counterfactual_expansion | 0.737 | 0.105 | 254272.0 | 12977920 | 0.003569 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.562 | -0.007 | 38528.0 | 2336512 | -0.001041 |
| qa_expansion | 0.621 | 0.052 | 106640.0 | 5730304 | 0.002820 |
| self_ranked_induction | 0.641 | 0.072 | 179205.6 | 9110272 | 0.002350 |
| sample_aware_self_ranked_induction | 0.641 | 0.072 | 179205.6 | 9110272 | 0.002350 |
| counterfactual_expansion | 0.766 | 0.197 | 377024.0 | 19296512 | 0.003031 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.623 | 0.031 | 51520.0 | 3122432 | 0.002630 |
| qa_expansion | 0.670 | 0.078 | 142600.0 | 7655424 | 0.002376 |
| self_ranked_induction | 0.725 | 0.132 | 241050.0 | 12174336 | 0.002389 |
| sample_aware_self_ranked_induction | 0.722 | 0.130 | 238335.6 | 12176128 | 0.002369 |
| counterfactual_expansion | 0.766 | 0.174 | 504160.0 | 25784832 | 0.001501 |
