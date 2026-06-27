# Learning Signal Density Tiny Neural Profile Sweep

Generated: `2026-06-27T21:07:33Z`

This sweep reruns the deterministic CPU tiny-MLP profile grid at one external sample budget.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_32x8_f1024.json`

Material count: `64`
Feature dimension: `1024`
Learning rate: `0.03`
Target signed gain over majority: `0.03`

| Condition | Best gain profile | Best signed gain | Best LSD profile | Best signed LSD/1M | Lowest-op target profile | Target-profile ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | epochs=64_hidden=8 | 0.088 | epochs=32_hidden=32 | 0.006357 | epochs=32_hidden=8 | 3167232 |
| qa_expansion | epochs=32_hidden=16 | 0.106 | epochs=32_hidden=16 | 0.003247 | epochs=16_hidden=8 | 3903232 |
| self_ranked_induction | epochs=16_hidden=8 | 0.153 | epochs=16_hidden=8 | 0.005247 | epochs=16_hidden=8 | 6224000 |
| sample_aware_self_ranked_induction | epochs=32_hidden=8 | 0.156 | epochs=16_hidden=8 | 0.005270 | epochs=8_hidden=8 | 3111936 |
| counterfactual_expansion | epochs=64_hidden=8 | 0.314 | epochs=8_hidden=8 | 0.006474 | epochs=8_hidden=8 | 6591808 |

## Profile epochs=8_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.566 | -0.026 | 12880.0 | 791808 | -0.008768 |
| qa_expansion | 0.532 | -0.060 | 37720.0 | 1951616 | -0.006886 |
| self_ranked_induction | 0.613 | 0.021 | 69930.0 | 3112000 | 0.001291 |
| sample_aware_self_ranked_induction | 0.639 | 0.047 | 67215.6 | 3111936 | 0.003021 |
| counterfactual_expansion | 0.792 | 0.200 | 134320.0 | 6591808 | 0.006474 |

## Profile epochs=16_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.574 | -0.018 | 25760.0 | 1583616 | -0.003069 |
| qa_expansion | 0.644 | 0.052 | 72680.0 | 3903232 | 0.003107 |
| self_ranked_induction | 0.745 | 0.153 | 126970.0 | 6224000 | 0.005247 |
| sample_aware_self_ranked_induction | 0.743 | 0.151 | 124255.6 | 6223872 | 0.005270 |
| counterfactual_expansion | 0.831 | 0.239 | 257600.0 | 13183616 | 0.004033 |

## Profile epochs=32_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.662 | 0.070 | 51520.0 | 3167232 | 0.005918 |
| qa_expansion | 0.694 | 0.101 | 142600.0 | 7806464 | 0.003089 |
| self_ranked_induction | 0.738 | 0.145 | 241050.0 | 12448000 | 0.002623 |
| sample_aware_self_ranked_induction | 0.748 | 0.156 | 238335.6 | 12447744 | 0.002843 |
| counterfactual_expansion | 0.839 | 0.247 | 504160.0 | 26367232 | 0.002128 |

## Profile epochs=64_hidden=8

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.681 | 0.088 | 103040.0 | 6334464 | 0.003726 |
| qa_expansion | 0.696 | 0.104 | 282440.0 | 15612928 | 0.001599 |
| self_ranked_induction | 0.714 | 0.122 | 469210.0 | 24896000 | 0.001131 |
| sample_aware_self_ranked_induction | 0.709 | 0.117 | 466495.6 | 24895488 | 0.001089 |
| counterfactual_expansion | 0.906 | 0.314 | 997280.0 | 52734464 | 0.001370 |

## Profile epochs=8_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.592 | 0.000 | 12880.0 | 1583616 | 0.000000 |
| qa_expansion | 0.530 | -0.062 | 37720.0 | 3903232 | -0.007185 |
| self_ranked_induction | 0.618 | 0.026 | 69930.0 | 6224000 | 0.001614 |
| sample_aware_self_ranked_induction | 0.634 | 0.042 | 67215.6 | 6223872 | 0.002684 |
| counterfactual_expansion | 0.787 | 0.195 | 134320.0 | 13183616 | 0.006306 |

## Profile epochs=16_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.566 | -0.026 | 25760.0 | 3167232 | -0.004384 |
| qa_expansion | 0.639 | 0.047 | 72680.0 | 7806464 | 0.002797 |
| self_ranked_induction | 0.740 | 0.148 | 126970.0 | 12448000 | 0.005069 |
| sample_aware_self_ranked_induction | 0.740 | 0.148 | 124255.6 | 12447744 | 0.005179 |
| counterfactual_expansion | 0.834 | 0.242 | 257600.0 | 26367232 | 0.004077 |

## Profile epochs=32_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.657 | 0.065 | 51520.0 | 6334464 | 0.005480 |
| qa_expansion | 0.699 | 0.106 | 142600.0 | 15612928 | 0.003247 |
| self_ranked_induction | 0.738 | 0.145 | 241050.0 | 24896000 | 0.002623 |
| sample_aware_self_ranked_induction | 0.745 | 0.153 | 238335.6 | 24895488 | 0.002795 |
| counterfactual_expansion | 0.834 | 0.242 | 504160.0 | 52734464 | 0.002083 |

## Profile epochs=64_hidden=16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.681 | 0.088 | 103040.0 | 12668928 | 0.003726 |
| qa_expansion | 0.694 | 0.101 | 282440.0 | 31225856 | 0.001559 |
| self_ranked_induction | 0.712 | 0.119 | 469210.0 | 49792000 | 0.001107 |
| sample_aware_self_ranked_induction | 0.699 | 0.106 | 466495.6 | 49790976 | 0.000992 |
| counterfactual_expansion | 0.888 | 0.296 | 997280.0 | 105468928 | 0.001291 |

## Profile epochs=32_hidden=32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.668 | 0.075 | 51520.0 | 12668928 | 0.006357 |
| qa_expansion | 0.699 | 0.106 | 142600.0 | 31225856 | 0.003247 |
| self_ranked_induction | 0.740 | 0.148 | 241050.0 | 49792000 | 0.002670 |
| sample_aware_self_ranked_induction | 0.745 | 0.153 | 238335.6 | 49790976 | 0.002795 |
| counterfactual_expansion | 0.834 | 0.242 | 504160.0 | 105468928 | 0.002083 |
