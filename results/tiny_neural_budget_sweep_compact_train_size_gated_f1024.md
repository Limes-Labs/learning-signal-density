# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T14:54:16Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_compact_train_size_gated`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 24 | 0.000 | 0.000000 |
| train_size_gated_sample_aware_induction | 48 | 64 | 0.132 | 0.004634 |
| compact_train_size_gated_induction | 48 | 64 | 0.140 | 0.007883 |

Comparison target: `results/tiny_neural_budget_sweep_train_size_gated_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.484 | -0.084 | 6496.0 | 399744 | -0.223508 |
| train_size_gated_sample_aware_induction | 0.484 | -0.084 | 6496.0 | 399744 | -0.223508 |
| compact_train_size_gated_induction | 0.484 | -0.084 | 6496.0 | 399744 | -0.223508 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.628 | 0.000 | 9632.0 | 593152 | 0.000000 |
| train_size_gated_sample_aware_induction | 0.628 | 0.000 | 9632.0 | 593152 | 0.000000 |
| compact_train_size_gated_induction | 0.628 | 0.000 | 9632.0 | 593152 | 0.000000 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.558 | -0.063 | 12992.0 | 799360 | -0.041908 |
| train_size_gated_sample_aware_induction | 0.558 | -0.063 | 12992.0 | 799360 | -0.041908 |
| compact_train_size_gated_induction | 0.558 | -0.063 | 12992.0 | 799360 | -0.041908 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | -0.007 | 19264.0 | 1185280 | -0.002081 |
| train_size_gated_sample_aware_induction | 0.666 | 0.086 | 93888.8 | 4656512 | 0.005338 |
| compact_train_size_gated_induction | 0.666 | 0.086 | 93888.8 | 4656512 | 0.005338 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.597 | -0.031 | 25760.0 | 1584128 | -0.005261 |
| train_size_gated_sample_aware_induction | 0.761 | 0.132 | 124274.8 | 6223360 | 0.004634 |
| compact_train_size_gated_induction | 0.769 | 0.140 | 77354.8 | 3903232 | 0.007883 |
