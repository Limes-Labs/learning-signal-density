# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T21:22:01Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_profile_sweep_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=8_hidden=8_features=1024`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 16 | 0.000 | 0.000000 |
| qa_expansion | not reached | 16 | 0.000 | 0.000000 |
| self_ranked_induction | not reached | 64 | 0.021 | 0.001291 |
| sample_aware_self_ranked_induction | 64 | 64 | 0.047 | 0.003021 |
| counterfactual_expansion | 48 | 64 | 0.200 | 0.006474 |

Comparison target: `results/tiny_neural_budget_sweep_16x8_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | 0.000 | 3248.0 | 199808 | 0.000000 |
| qa_expansion | 0.579 | 0.000 | 9512.0 | 492416 | 0.000000 |
| self_ranked_induction | 0.579 | 0.000 | 17017.2 | 785216 | 0.000000 |
| sample_aware_self_ranked_induction | 0.589 | 0.011 | 12265.2 | 563072 | 0.014937 |
| counterfactual_expansion | 0.558 | -0.021 | 33872.0 | 1663168 | -0.010716 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.655 | 0.000 | 4816.0 | 296448 | 0.000000 |
| qa_expansion | 0.655 | 0.000 | 14104.0 | 730496 | 0.000000 |
| self_ranked_induction | 0.655 | 0.000 | 25678.8 | 1164928 | 0.000000 |
| sample_aware_self_ranked_induction | 0.662 | 0.007 | 23302.8 | 1053824 | 0.003414 |
| counterfactual_expansion | 0.641 | -0.014 | 50224.0 | 2467008 | -0.003193 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.595 | -0.037 | 6496.0 | 399488 | -0.048892 |
| qa_expansion | 0.584 | -0.047 | 19024.0 | 984576 | -0.021465 |
| self_ranked_induction | 0.600 | -0.032 | 34701.6 | 1569536 | -0.007850 |
| sample_aware_self_ranked_induction | 0.516 | -0.116 | 31569.6 | 1423360 | -0.031608 |
| counterfactual_expansion | 0.658 | 0.026 | 67744.0 | 3325568 | 0.003349 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.569 | 0.000 | 9632.0 | 592704 | 0.000000 |
| qa_expansion | 0.569 | -0.000 | 28208.0 | 1460608 | 0.000000 |
| self_ranked_induction | 0.514 | -0.055 | 51237.6 | 2328064 | -0.006265 |
| sample_aware_self_ranked_induction | 0.514 | -0.055 | 51237.6 | 2328064 | -0.006265 |
| counterfactual_expansion | 0.662 | 0.093 | 100448.0 | 4931904 | 0.005389 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.566 | -0.026 | 12880.0 | 791808 | -0.008768 |
| qa_expansion | 0.532 | -0.060 | 37720.0 | 1951616 | -0.006886 |
| self_ranked_induction | 0.613 | 0.021 | 69930.0 | 3112000 | 0.001291 |
| sample_aware_self_ranked_induction | 0.639 | 0.047 | 67215.6 | 3111936 | 0.003021 |
| counterfactual_expansion | 0.792 | 0.200 | 134320.0 | 6591808 | 0.006474 |
