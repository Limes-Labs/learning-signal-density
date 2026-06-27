# Learning Signal Density Tiny Neural Feature-Dimension Sweep

Generated: `2026-06-27T20:54:34Z`

This sweep reruns the deterministic CPU tiny-MLP profile across hashed feature dimensions.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_feature_sweep.json`

Backend: `tiny_mlp`
Profile label: `epochs=32_hidden=8`
Material count: `64`
Hidden units: `8`
Feature dimensions: `128, 256, 512, 1024`
Target signed gain over majority: `0.03`
Comparison target: `results/tiny_neural_feature_sweep.json`

| Condition | Best gain dimension | Best signed gain | Best LSD dimension | Best signed LSD/1M | Lowest-op target dimension | Target-dimension ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 1024 | 0.070 | 1024 | 0.005918 | 256 | 3122432 |
| qa_expansion | 1024 | 0.101 | 1024 | 0.003089 | 128 | 7473664 |
| self_ranked_induction | 1024 | 0.145 | 1024 | 0.002623 | 128 | 11820288 |
| sample_aware_self_ranked_induction | 1024 | 0.156 | 1024 | 0.002843 | 128 | 11815936 |
| counterfactual_expansion | 1024 | 0.247 | 1024 | 0.002128 | 128 | 24965376 |

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

## Feature Dimension 512

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.657 | 0.065 | 51520.0 | 3155456 | 0.005480 |
| qa_expansion | 0.678 | 0.086 | 142600.0 | 7773952 | 0.002614 |
| self_ranked_induction | 0.701 | 0.109 | 241050.0 | 12390912 | 0.001968 |
| sample_aware_self_ranked_induction | 0.712 | 0.119 | 238335.6 | 12390912 | 0.002179 |
| counterfactual_expansion | 0.836 | 0.244 | 504160.0 | 26251776 | 0.002106 |

## Feature Dimension 1024

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.662 | 0.070 | 51520.0 | 3167232 | 0.005918 |
| qa_expansion | 0.694 | 0.101 | 142600.0 | 7806464 | 0.003089 |
| self_ranked_induction | 0.738 | 0.145 | 241050.0 | 12448000 | 0.002623 |
| sample_aware_self_ranked_induction | 0.748 | 0.156 | 238335.6 | 12447744 | 0.002843 |
| counterfactual_expansion | 0.839 | 0.247 | 504160.0 | 26367232 | 0.002128 |
