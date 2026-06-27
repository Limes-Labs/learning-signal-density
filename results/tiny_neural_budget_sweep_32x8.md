# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-27T19:12:44Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_profile_sweep.json`

Backend: `tiny_mlp`
Profile label: `epochs=32_hidden=8`
Hidden units: `8`
Feature dimension: `128`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 48 | -0.038 | -0.005724 |
| qa_expansion | 48 | 48 | 0.041 | 0.002256 |
| self_ranked_induction | 48 | 64 | 0.078 | 0.001406 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.065 | 0.001184 |
| counterfactual_expansion | 32 | 48 | 0.124 | 0.001914 |

Comparison target: `results/tiny_neural_budget_sweep.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.526 | -0.053 | 12992.0 | 781312 | -0.069846 |
| qa_expansion | 0.474 | -0.105 | 35960.0 | 1879040 | -0.050470 |
| self_ranked_induction | 0.516 | -0.063 | 60169.2 | 2972416 | -0.018096 |
| sample_aware_self_ranked_induction | 0.474 | -0.105 | 42745.2 | 2141696 | -0.042451 |
| counterfactual_expansion | 0.495 | -0.084 | 127136.0 | 6268672 | -0.011420 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.538 | -0.117 | 19264.0 | 1159936 | -0.070768 |
| qa_expansion | 0.552 | -0.103 | 53320.0 | 2800128 | -0.022560 |
| self_ranked_induction | 0.586 | -0.069 | 89662.8 | 4439296 | -0.008958 |
| sample_aware_self_ranked_induction | 0.593 | -0.062 | 80950.8 | 4018176 | -0.008928 |
| counterfactual_expansion | 0.600 | -0.055 | 188512.0 | 9353216 | -0.003403 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.353 | -0.279 | 25984.0 | 1559296 | -0.092546 |
| qa_expansion | 0.568 | -0.063 | 71920.0 | 3759872 | -0.007570 |
| self_ranked_induction | 0.553 | -0.079 | 121005.6 | 5953792 | -0.005624 |
| sample_aware_self_ranked_induction | 0.526 | -0.105 | 109521.6 | 5401088 | -0.008288 |
| counterfactual_expansion | 0.679 | 0.047 | 254272.0 | 12554496 | 0.001606 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.531 | -0.038 | 38528.0 | 2311936 | -0.005724 |
| qa_expansion | 0.610 | 0.041 | 106640.0 | 5576704 | 0.002256 |
| self_ranked_induction | 0.607 | 0.038 | 179205.6 | 8833792 | 0.001230 |
| sample_aware_self_ranked_induction | 0.607 | 0.038 | 179205.6 | 8833792 | 0.001230 |
| counterfactual_expansion | 0.693 | 0.124 | 377024.0 | 18643456 | 0.001914 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.545 | -0.047 | 51520.0 | 3099136 | -0.003946 |
| qa_expansion | 0.629 | 0.036 | 142600.0 | 7473664 | 0.001109 |
| self_ranked_induction | 0.670 | 0.078 | 241050.0 | 11820288 | 0.001406 |
| sample_aware_self_ranked_induction | 0.657 | 0.065 | 238335.6 | 11815936 | 0.001184 |
| counterfactual_expansion | 0.694 | 0.101 | 504160.0 | 24965376 | 0.000874 |
