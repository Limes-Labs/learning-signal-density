# Learning Signal Density Tiny Neural Profile Sweep

Generated: `2026-06-27T18:56:37Z`

This sweep reruns the deterministic CPU tiny-MLP profile grid at one external sample budget.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep.json`

Material count: `64`
Feature dimension: `128`
Learning rate: `0.03`
Target signed gain over majority: `0.03`

| Condition | Best gain profile | Best signed gain | Best LSD profile | Best signed LSD/1M | Lowest-op target profile | Target-profile ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | epochs=8_hidden=8 | -0.008 | epochs=8_hidden=8 | -0.002630 | None | 0 |
| qa_expansion | epochs=32_hidden=16 | 0.042 | epochs=16_hidden=16 | 0.002175 | epochs=32_hidden=8 | 7473664 |
| self_ranked_induction | epochs=32_hidden=8 | 0.078 | epochs=32_hidden=8 | 0.001406 | epochs=32_hidden=8 | 11820288 |
| sample_aware_self_ranked_induction | epochs=32_hidden=8 | 0.065 | epochs=16_hidden=32 | 0.001361 | epochs=16_hidden=8 | 5907968 |
| counterfactual_expansion | epochs=32_hidden=16 | 0.106 | epochs=8_hidden=32 | 0.003195 | epochs=8_hidden=8 | 6241344 |

## Profile epochs=8_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.584 | -0.008 | 12880.0 | 774784 | -0.002630 |
| qa_expansion | 0.569 | -0.023 | 37720.0 | 1868416 | -0.002695 |
| self_ranked_induction | 0.592 | 0.000 | 69930.0 | 2955072 | -0.000001 |
| sample_aware_self_ranked_induction | 0.590 | -0.003 | 67215.6 | 2953984 | -0.000173 |
| counterfactual_expansion | 0.683 | 0.091 | 134320.0 | 6241344 | 0.002943 |

## Profile epochs=16_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.574 | -0.018 | 25760.0 | 1549568 | -0.003069 |
| qa_expansion | 0.616 | 0.023 | 72680.0 | 3736832 | 0.001399 |
| self_ranked_induction | 0.621 | 0.029 | 126970.0 | 5910144 | 0.000977 |
| sample_aware_self_ranked_induction | 0.626 | 0.034 | 124255.6 | 5907968 | 0.001179 |
| counterfactual_expansion | 0.683 | 0.091 | 257600.0 | 12482688 | 0.001534 |

## Profile epochs=32_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.545 | -0.047 | 51520.0 | 3099136 | -0.003946 |
| qa_expansion | 0.629 | 0.036 | 142600.0 | 7473664 | 0.001109 |
| self_ranked_induction | 0.670 | 0.078 | 241050.0 | 11820288 | 0.001406 |
| sample_aware_self_ranked_induction | 0.657 | 0.065 | 238335.6 | 11815936 | 0.001184 |
| counterfactual_expansion | 0.694 | 0.101 | 504160.0 | 24965376 | 0.000874 |

## Profile epochs=8_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.569 | -0.023 | 12880.0 | 1549568 | -0.007891 |
| qa_expansion | 0.564 | -0.029 | 37720.0 | 3736832 | -0.003293 |
| self_ranked_induction | 0.579 | -0.013 | 69930.0 | 5910144 | -0.000808 |
| sample_aware_self_ranked_induction | 0.582 | -0.010 | 67215.6 | 5907968 | -0.000677 |
| counterfactual_expansion | 0.678 | 0.086 | 134320.0 | 12482688 | 0.002775 |

## Profile epochs=16_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.566 | -0.026 | 25760.0 | 3099136 | -0.004384 |
| qa_expansion | 0.629 | 0.036 | 72680.0 | 7473664 | 0.002175 |
| self_ranked_induction | 0.618 | 0.026 | 126970.0 | 11820288 | 0.000888 |
| sample_aware_self_ranked_induction | 0.623 | 0.031 | 124255.6 | 11815936 | 0.001089 |
| counterfactual_expansion | 0.683 | 0.091 | 257600.0 | 24965376 | 0.001534 |

## Profile epochs=32_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.548 | -0.044 | 51520.0 | 6198272 | -0.003726 |
| qa_expansion | 0.634 | 0.042 | 142600.0 | 14947328 | 0.001267 |
| self_ranked_induction | 0.668 | 0.075 | 241050.0 | 23640576 | 0.001359 |
| sample_aware_self_ranked_induction | 0.657 | 0.065 | 238335.6 | 23631872 | 0.001184 |
| counterfactual_expansion | 0.699 | 0.106 | 504160.0 | 49930752 | 0.000918 |

## Profile epochs=8_hidden=32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.577 | -0.016 | 12880.0 | 3099136 | -0.005261 |
| qa_expansion | 0.556 | -0.036 | 37720.0 | 7473664 | -0.004191 |
| self_ranked_induction | 0.590 | -0.003 | 69930.0 | 11820288 | -0.000163 |
| sample_aware_self_ranked_induction | 0.584 | -0.008 | 67215.6 | 11815936 | -0.000510 |
| counterfactual_expansion | 0.691 | 0.099 | 134320.0 | 24965376 | 0.003195 |

## Profile epochs=16_hidden=32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.574 | -0.018 | 25760.0 | 6198272 | -0.003069 |
| qa_expansion | 0.626 | 0.034 | 72680.0 | 14947328 | 0.002020 |
| self_ranked_induction | 0.623 | 0.031 | 126970.0 | 23640576 | 0.001066 |
| sample_aware_self_ranked_induction | 0.631 | 0.039 | 124255.6 | 23631872 | 0.001361 |
| counterfactual_expansion | 0.683 | 0.091 | 257600.0 | 49930752 | 0.001534 |

## Profile epochs=32_hidden=32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.548 | -0.044 | 51520.0 | 12396544 | -0.003726 |
| qa_expansion | 0.629 | 0.036 | 142600.0 | 29894656 | 0.001109 |
| self_ranked_induction | 0.665 | 0.073 | 241050.0 | 47281152 | 0.001312 |
| sample_aware_self_ranked_induction | 0.655 | 0.062 | 238335.6 | 47263744 | 0.001137 |
| counterfactual_expansion | 0.699 | 0.106 | 504160.0 | 99861504 | 0.000918 |
