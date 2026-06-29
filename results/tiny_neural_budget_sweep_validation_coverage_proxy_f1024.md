# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T14:02:08Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_train_size_gated_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_validation_coverage_proxy`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 64 | 0.018 | 0.003069 |
| sample_aware_self_ranked_induction | 64 | 64 | 0.190 | 0.006637 |
| train_size_gated_sample_aware_induction | 64 | 64 | 0.190 | 0.006637 |
| validation_coverage_proxy_selector | 64 | 64 | 0.171 | 0.003164 |
| validation_abstaining_proxy_selector | 64 | 64 | 0.114 | 0.001704 |
| validation_portfolio_selector | 48 | 64 | 0.195 | 0.001119 |
| counterfactual_expansion | 32 | 64 | 0.213 | 0.003595 |

Comparison target: `results/generated_coverage_audit_selector_transfer_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.642 | 0.000 | 6496.0 | 398720 | 0.000000 |
| sample_aware_self_ranked_induction | 0.516 | -0.032 | 22540.4 | 1124352 | -0.024171 |
| train_size_gated_sample_aware_induction | 0.642 | 0.000 | 6496.0 | 398720 | 0.000000 |
| validation_coverage_proxy_selector | 0.642 | 0.000 | 20695.2 | 398720 | 0.000000 |
| validation_abstaining_proxy_selector | 0.600 | -0.042 | 46662.0 | 834150 | -0.013630 |
| validation_portfolio_selector | 0.537 | -0.011 | 147063.2 | 7323264 | -0.001194 |
| counterfactual_expansion | 0.516 | -0.126 | 64960.0 | 3325440 | -0.033526 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.641 | 0.000 | 9632.0 | 592128 | 0.000000 |
| sample_aware_self_ranked_induction | 0.490 | -0.152 | 42547.6 | 2105984 | -0.041452 |
| train_size_gated_sample_aware_induction | 0.641 | 0.000 | 9632.0 | 592128 | 0.000000 |
| validation_coverage_proxy_selector | 0.559 | -0.083 | 81610.2 | 2712038 | -0.013000 |
| validation_abstaining_proxy_selector | 0.607 | -0.034 | 80228.2 | 1407258 | -0.004117 |
| validation_portfolio_selector | 0.634 | -0.007 | 246955.0 | 12172390 | -0.000322 |
| counterfactual_expansion | 0.579 | -0.062 | 96320.0 | 4933248 | -0.007493 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.611 | 0.000 | 12992.0 | 798592 | 0.000000 |
| sample_aware_self_ranked_induction | 0.574 | -0.037 | 57798.4 | 2847104 | -0.005497 |
| train_size_gated_sample_aware_induction | 0.611 | 0.000 | 12992.0 | 798592 | 0.000000 |
| validation_coverage_proxy_selector | 0.621 | 0.011 | 63546.6 | 1259418 | 0.000833 |
| validation_abstaining_proxy_selector | 0.547 | -0.063 | 129228.2 | 2605670 | -0.003955 |
| validation_portfolio_selector | 0.611 | -0.000 | 333480.2 | 16156826 | -0.000009 |
| counterfactual_expansion | 0.658 | 0.047 | 129920.0 | 6650752 | 0.003143 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.610 | -0.003 | 19264.0 | 1184768 | -0.001041 |
| sample_aware_self_ranked_induction | 0.638 | 0.024 | 93836.0 | 4654976 | 0.001496 |
| train_size_gated_sample_aware_induction | 0.638 | 0.024 | 93836.0 | 4654976 | 0.001496 |
| validation_coverage_proxy_selector | 0.638 | 0.024 | 173559.0 | 4654976 | 0.000795 |
| validation_abstaining_proxy_selector | 0.638 | 0.024 | 254115.6 | 4936986 | 0.000526 |
| validation_portfolio_selector | 0.652 | 0.038 | 535407.0 | 24984550 | 0.000409 |
| counterfactual_expansion | 0.745 | 0.131 | 192640.0 | 9863168 | 0.003955 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | 0.018 | 25760.0 | 1582848 | 0.003069 |
| sample_aware_self_ranked_induction | 0.751 | 0.190 | 124198.0 | 6222208 | 0.006637 |
| train_size_gated_sample_aware_induction | 0.751 | 0.190 | 124198.0 | 6222208 | 0.006637 |
| validation_coverage_proxy_selector | 0.732 | 0.171 | 235368.2 | 6222208 | 0.003164 |
| validation_abstaining_proxy_selector | 0.675 | 0.114 | 305079.2 | 5269043 | 0.001704 |
| validation_portfolio_selector | 0.756 | 0.195 | 746680.8 | 34789197 | 0.001119 |
| counterfactual_expansion | 0.774 | 0.213 | 257600.0 | 13182848 | 0.003595 |
