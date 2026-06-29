# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T14:28:47Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_tempered_sample_aware`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 24 | 0.007 | 0.008326 |
| qa_expansion | 64 | 64 | 0.031 | 0.001865 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.125 | 0.004363 |
| tempered_sample_aware_self_ranked_induction | 48 | 64 | 0.125 | 0.004363 |
| train_size_gated_sample_aware_induction | 48 | 64 | 0.125 | 0.004363 |
| validation_coverage_proxy_selector | 48 | 64 | 0.122 | 0.002215 |
| validation_abstaining_proxy_selector | 64 | 64 | 0.099 | 0.001389 |
| validation_portfolio_selector | 48 | 64 | 0.132 | 0.000780 |
| counterfactual_expansion | 32 | 64 | 0.210 | 0.003551 |

Comparison target: `results/tiny_neural_budget_sweep_train_size_gated_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.568 | -0.063 | 6496.0 | 398720 | -0.167631 |
| qa_expansion | 0.526 | -0.105 | 18328.0 | 983040 | -0.099022 |
| sample_aware_self_ranked_induction | 0.516 | -0.021 | 22511.6 | 1124608 | -0.016115 |
| tempered_sample_aware_self_ranked_induction | 0.516 | -0.021 | 22511.6 | 1124608 | -0.016115 |
| train_size_gated_sample_aware_induction | 0.568 | -0.063 | 6496.0 | 398720 | -0.167631 |
| validation_coverage_proxy_selector | 0.568 | -0.063 | 20229.6 | 398720 | -0.054175 |
| validation_abstaining_proxy_selector | 0.547 | 0.011 | 42750.4 | 688666 | 0.001937 |
| validation_portfolio_selector | 0.495 | -0.042 | 143448.8 | 7158579 | -0.005115 |
| counterfactual_expansion | 0.568 | -0.063 | 64960.0 | 3325440 | -0.016763 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.641 | 0.007 | 9632.0 | 592128 | 0.008326 |
| qa_expansion | 0.586 | -0.048 | 27176.0 | 1459456 | -0.020656 |
| sample_aware_self_ranked_induction | 0.497 | -0.138 | 42562.0 | 2106112 | -0.037728 |
| tempered_sample_aware_self_ranked_induction | 0.538 | -0.097 | 38278.0 | 1894144 | -0.029315 |
| train_size_gated_sample_aware_induction | 0.641 | 0.007 | 9632.0 | 592128 | 0.008326 |
| validation_coverage_proxy_selector | 0.579 | -0.055 | 59925.8 | 1760026 | -0.007536 |
| validation_abstaining_proxy_selector | 0.538 | -0.097 | 85978.2 | 1754342 | -0.010890 |
| validation_portfolio_selector | 0.628 | -0.007 | 238338.2 | 11762253 | -0.000365 |
| counterfactual_expansion | 0.641 | 0.007 | 96320.0 | 4933248 | 0.000832 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.547 | -0.047 | 12992.0 | 798592 | -0.031431 |
| qa_expansion | 0.537 | -0.058 | 36656.0 | 1968384 | -0.013616 |
| sample_aware_self_ranked_induction | 0.442 | -0.153 | 57640.0 | 2847232 | -0.022816 |
| tempered_sample_aware_self_ranked_induction | 0.516 | -0.079 | 51724.0 | 2554880 | -0.013168 |
| train_size_gated_sample_aware_induction | 0.547 | -0.047 | 12992.0 | 798592 | -0.031431 |
| validation_coverage_proxy_selector | 0.547 | -0.047 | 52658.6 | 798592 | -0.007580 |
| validation_abstaining_proxy_selector | 0.521 | -0.074 | 109462.6 | 1676544 | -0.004486 |
| validation_portfolio_selector | 0.526 | -0.068 | 341874.6 | 16573952 | -0.001740 |
| counterfactual_expansion | 0.658 | 0.063 | 129920.0 | 6650752 | 0.004191 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | -0.031 | 19264.0 | 1184768 | -0.009366 |
| qa_expansion | 0.614 | 0.010 | 54352.0 | 2919936 | 0.001107 |
| sample_aware_self_ranked_induction | 0.690 | 0.086 | 93831.2 | 4655104 | 0.005341 |
| tempered_sample_aware_self_ranked_induction | 0.690 | 0.086 | 93831.2 | 4655104 | 0.005341 |
| train_size_gated_sample_aware_induction | 0.690 | 0.086 | 93831.2 | 4655104 | 0.005341 |
| validation_coverage_proxy_selector | 0.679 | 0.076 | 172102.4 | 4654976 | 0.002571 |
| validation_abstaining_proxy_selector | 0.621 | 0.017 | 241474.2 | 4121062 | 0.000439 |
| validation_portfolio_selector | 0.652 | 0.048 | 529769.6 | 24765773 | 0.000531 |
| counterfactual_expansion | 0.786 | 0.183 | 192640.0 | 9863168 | 0.005516 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.587 | -0.026 | 25760.0 | 1582848 | -0.004384 |
| qa_expansion | 0.644 | 0.031 | 72680.0 | 3901696 | 0.001865 |
| sample_aware_self_ranked_induction | 0.738 | 0.125 | 124217.2 | 6221952 | 0.004363 |
| tempered_sample_aware_self_ranked_induction | 0.738 | 0.125 | 124217.2 | 6221952 | 0.004363 |
| train_size_gated_sample_aware_induction | 0.738 | 0.125 | 124217.2 | 6221952 | 0.004363 |
| validation_coverage_proxy_selector | 0.735 | 0.122 | 237755.2 | 6222080 | 0.002215 |
| validation_abstaining_proxy_selector | 0.712 | 0.099 | 294099.8 | 5293696 | 0.001389 |
| validation_portfolio_selector | 0.745 | 0.132 | 725720.8 | 33599206 | 0.000780 |
| counterfactual_expansion | 0.823 | 0.210 | 257600.0 | 13182848 | 0.003551 |
