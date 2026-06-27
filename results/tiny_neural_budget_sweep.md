# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T18:44:21Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_confirmation.json`

Backend: `tiny_mlp`
Hidden units: `32`
Feature dimension: `128`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 48 | -0.031 | -0.004683 |
| qa_expansion | 48 | 64 | 0.036 | 0.001109 |
| self_ranked_induction | 48 | 64 | 0.073 | 0.001312 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.062 | 0.001137 |
| counterfactual_expansion | 32 | 48 | 0.121 | 0.001861 |

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.484 | -0.095 | 12992.0 | 3125248 | -0.125723 |
| qa_expansion | 0.495 | -0.084 | 35960.0 | 7516160 | -0.040376 |
| self_ranked_induction | 0.537 | -0.042 | 60169.2 | 11889664 | -0.012059 |
| sample_aware_self_ranked_induction | 0.495 | -0.084 | 42745.2 | 8566784 | -0.033944 |
| counterfactual_expansion | 0.516 | -0.063 | 127136.0 | 25074688 | -0.008565 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | -0.083 | 19264.0 | 4639744 | -0.049954 |
| qa_expansion | 0.559 | -0.097 | 53320.0 | 11200512 | -0.021056 |
| self_ranked_induction | 0.593 | -0.062 | 89662.8 | 17757184 | -0.008064 |
| sample_aware_self_ranked_induction | 0.600 | -0.055 | 80950.8 | 16072704 | -0.007934 |
| counterfactual_expansion | 0.614 | -0.041 | 188512.0 | 37412864 | -0.002552 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.358 | -0.274 | 25984.0 | 6237184 | -0.090800 |
| qa_expansion | 0.574 | -0.058 | 71920.0 | 15039488 | -0.006940 |
| self_ranked_induction | 0.563 | -0.068 | 121005.6 | 23815168 | -0.004874 |
| sample_aware_self_ranked_induction | 0.526 | -0.105 | 109521.6 | 21604352 | -0.008289 |
| counterfactual_expansion | 0.679 | 0.047 | 254272.0 | 50217984 | 0.001606 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.538 | -0.031 | 38528.0 | 9247744 | -0.004683 |
| qa_expansion | 0.603 | 0.034 | 106640.0 | 22306816 | 0.001880 |
| self_ranked_induction | 0.607 | 0.038 | 179205.6 | 35335168 | 0.001230 |
| sample_aware_self_ranked_induction | 0.607 | 0.038 | 179205.6 | 35335168 | 0.001230 |
| counterfactual_expansion | 0.690 | 0.121 | 377024.0 | 74573824 | 0.001861 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.548 | -0.044 | 51520.0 | 12396544 | -0.003726 |
| qa_expansion | 0.629 | 0.036 | 142600.0 | 29894656 | 0.001109 |
| self_ranked_induction | 0.665 | 0.073 | 241050.0 | 47281152 | 0.001312 |
| sample_aware_self_ranked_induction | 0.655 | 0.062 | 238335.6 | 47263744 | 0.001137 |
| counterfactual_expansion | 0.699 | 0.106 | 504160.0 | 99861504 | 0.000918 |
