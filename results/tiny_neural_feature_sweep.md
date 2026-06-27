# Learning Signal Density Tiny Neural Feature-Dimension Sweep

Generated: `2026-06-27T19:28:15Z`

This sweep reruns the deterministic CPU tiny-MLP profile across hashed feature dimensions.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_32x8.json`

Backend: `tiny_mlp`
Profile label: `epochs=32_hidden=8`
Material count: `64`
Hidden units: `8`
Feature dimensions: `16, 32, 64, 128, 256`
Target signed gain over majority: `0.03`
Comparison target: `results/tiny_neural_budget_sweep_32x8.json`

| Condition | Best gain dimension | Best signed gain | Best LSD dimension | Best signed LSD/1M | Lowest-op target dimension | Target-dimension ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 256 | 0.031 | 256 | 0.002630 | 256 | 3122432 |
| qa_expansion | 256 | 0.078 | 256 | 0.002376 | 128 | 7473664 |
| self_ranked_induction | 256 | 0.132 | 256 | 0.002389 | 128 | 11820288 |
| sample_aware_self_ranked_induction | 256 | 0.130 | 256 | 0.002369 | 128 | 11815936 |
| counterfactual_expansion | 256 | 0.174 | 256 | 0.001501 | 64 | 23553536 |

## Feature Dimension 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.556 | -0.036 | 51520.0 | 2575872 | -0.003069 |
| qa_expansion | 0.540 | -0.052 | 142600.0 | 5794304 | -0.001584 |
| self_ranked_induction | 0.595 | 0.003 | 241050.0 | 8997888 | 0.000047 |
| sample_aware_self_ranked_induction | 0.597 | 0.005 | 238335.6 | 9001472 | 0.000094 |
| counterfactual_expansion | 0.574 | -0.018 | 504160.0 | 18652672 | -0.000157 |

## Feature Dimension 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.564 | -0.029 | 51520.0 | 2921472 | -0.002411 |
| qa_expansion | 0.512 | -0.081 | 142600.0 | 6827008 | -0.002455 |
| self_ranked_induction | 0.566 | -0.026 | 241050.0 | 10691328 | -0.000468 |
| sample_aware_self_ranked_induction | 0.564 | -0.029 | 238335.6 | 10689024 | -0.000521 |
| counterfactual_expansion | 0.582 | -0.010 | 504160.0 | 22424832 | -0.000090 |

## Feature Dimension 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.538 | -0.055 | 51520.0 | 3026176 | -0.004603 |
| qa_expansion | 0.548 | -0.044 | 142600.0 | 7134720 | -0.001346 |
| self_ranked_induction | 0.571 | -0.021 | 241050.0 | 11216640 | -0.000375 |
| sample_aware_self_ranked_induction | 0.566 | -0.026 | 238335.6 | 11211008 | -0.000474 |
| counterfactual_expansion | 0.652 | 0.060 | 504160.0 | 23553536 | 0.000515 |

## Feature Dimension 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.545 | -0.047 | 51520.0 | 3099136 | -0.003946 |
| qa_expansion | 0.629 | 0.036 | 142600.0 | 7473664 | 0.001109 |
| self_ranked_induction | 0.670 | 0.078 | 241050.0 | 11820288 | 0.001406 |
| sample_aware_self_ranked_induction | 0.657 | 0.065 | 238335.6 | 11815936 | 0.001184 |
| counterfactual_expansion | 0.694 | 0.101 | 504160.0 | 24965376 | 0.000874 |

## Feature Dimension 256

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.623 | 0.031 | 51520.0 | 3122432 | 0.002630 |
| qa_expansion | 0.670 | 0.078 | 142600.0 | 7655424 | 0.002376 |
| self_ranked_induction | 0.725 | 0.132 | 241050.0 | 12174336 | 0.002389 |
| sample_aware_self_ranked_induction | 0.722 | 0.130 | 238335.6 | 12176128 | 0.002369 |
| counterfactual_expansion | 0.766 | 0.174 | 504160.0 | 25784832 | 0.001501 |
