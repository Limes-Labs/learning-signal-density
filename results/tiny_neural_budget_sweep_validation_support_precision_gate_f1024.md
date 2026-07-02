# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T19:51:48Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_support_precision_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_validation_support_precision_gate`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 96 | 128 | 0.138 | 0.005809 |
| compact_train_size_gated_induction | 64 | 80 | 0.202 | 0.008893 |
| support_ramped_compact_induction | 64 | 80 | 0.202 | 0.008893 |
| density_window_compact_induction | 64 | 80 | 0.202 | 0.008893 |
| support_probe_window_selector | 64 | 80 | 0.202 | 0.008893 |
| validation_support_precision_selector | 64 | 80 | 0.202 | 0.008893 |
| validation_support_precision_gate_selector | 64 | 80 | 0.202 | 0.008893 |
| train_support_density_selector | 64 | 80 | 0.202 | 0.007557 |
| density_capped_compact_induction | 64 | 80 | 0.202 | 0.008893 |
| counterfactual_expansion | 64 | 80 | 0.254 | 0.002736 |

Comparison target: `results/tiny_neural_budget_sweep_validation_support_precision_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.548 | -0.029 | 25760.0 | 1583616 | -0.004822 |
| compact_train_size_gated_induction | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| support_ramped_compact_induction | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| density_window_compact_induction | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| support_probe_window_selector | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| validation_support_precision_selector | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| validation_support_precision_gate_selector | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| train_support_density_selector | 0.712 | 0.135 | 92109.2 | 3904256 | 0.006376 |
| density_capped_compact_induction | 0.712 | 0.135 | 77316.4 | 3904256 | 0.007595 |
| counterfactual_expansion | 0.761 | 0.184 | 257600.0 | 13183616 | 0.003113 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.583 | 0.029 | 32256.0 | 1984768 | 0.003140 |
| compact_train_size_gated_induction | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| support_ramped_compact_induction | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| density_window_compact_induction | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| support_probe_window_selector | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| validation_support_precision_selector | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| validation_support_precision_gate_selector | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| train_support_density_selector | 0.756 | 0.202 | 93004.8 | 4076134 | 0.007557 |
| density_capped_compact_induction | 0.756 | 0.202 | 79027.2 | 4076134 | 0.008893 |
| counterfactual_expansion | 0.808 | 0.254 | 322560.0 | 16515328 | 0.002736 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.649 | 0.054 | 38752.0 | 2381056 | 0.004021 |
| compact_train_size_gated_induction | 0.727 | 0.132 | 100703.6 | 5159091 | 0.003786 |
| support_ramped_compact_induction | 0.727 | 0.132 | 100703.6 | 5159091 | 0.003786 |
| density_window_compact_induction | 0.649 | 0.054 | 38752.0 | 2381056 | 0.004021 |
| support_probe_window_selector | 0.649 | 0.054 | 38752.0 | 2381056 | 0.004021 |
| validation_support_precision_selector | 0.701 | 0.106 | 77489.4 | 4037197 | 0.004849 |
| validation_support_precision_gate_selector | 0.701 | 0.106 | 77489.4 | 4037197 | 0.004849 |
| train_support_density_selector | 0.727 | 0.132 | 118776.4 | 5159091 | 0.003209 |
| density_capped_compact_induction | 0.727 | 0.132 | 100703.6 | 5159091 | 0.003786 |
| counterfactual_expansion | 0.812 | 0.217 | 387520.0 | 19821056 | 0.001621 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.685 | 0.075 | 41888.0 | 2572544 | 0.004800 |
| compact_train_size_gated_induction | 0.747 | 0.138 | 125813.2 | 6342016 | 0.002924 |
| support_ramped_compact_induction | 0.762 | 0.152 | 86755.6 | 4546483 | 0.004687 |
| density_window_compact_induction | 0.685 | 0.075 | 41888.0 | 2572544 | 0.004800 |
| support_probe_window_selector | 0.762 | 0.152 | 86755.6 | 4546483 | 0.004687 |
| validation_support_precision_selector | 0.720 | 0.110 | 61969.8 | 3346330 | 0.005245 |
| validation_support_precision_gate_selector | 0.720 | 0.110 | 61969.8 | 3346330 | 0.005245 |
| train_support_density_selector | 0.762 | 0.152 | 106185.2 | 4546483 | 0.003829 |
| density_capped_compact_induction | 0.685 | 0.075 | 41888.0 | 2572544 | 0.004800 |
| counterfactual_expansion | 0.781 | 0.171 | 418880.0 | 21421184 | 0.001093 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.670 | 0.070 | 45248.0 | 2780160 | 0.003838 |
| compact_train_size_gated_induction | 0.781 | 0.181 | 134476.0 | 6809856 | 0.003324 |
| support_ramped_compact_induction | 0.787 | 0.187 | 78748.0 | 4210944 | 0.005864 |
| density_window_compact_induction | 0.787 | 0.187 | 78748.0 | 4210944 | 0.005864 |
| support_probe_window_selector | 0.787 | 0.187 | 78748.0 | 4210944 | 0.005864 |
| validation_support_precision_selector | 0.787 | 0.187 | 78748.0 | 4210944 | 0.005864 |
| validation_support_precision_gate_selector | 0.743 | 0.143 | 67417.2 | 3638502 | 0.005031 |
| train_support_density_selector | 0.787 | 0.187 | 97412.0 | 4210944 | 0.004741 |
| density_capped_compact_induction | 0.670 | 0.070 | 45248.0 | 2780160 | 0.003838 |
| counterfactual_expansion | 0.839 | 0.239 | 452480.0 | 23145600 | 0.001306 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.678 | 0.118 | 48384.0 | 2976384 | 0.005648 |
| compact_train_size_gated_induction | 0.718 | 0.158 | 145269.6 | 7331200 | 0.002523 |
| support_ramped_compact_induction | 0.750 | 0.190 | 75772.8 | 4111770 | 0.005807 |
| density_window_compact_induction | 0.678 | 0.118 | 48384.0 | 2976384 | 0.005648 |
| support_probe_window_selector | 0.678 | 0.118 | 48384.0 | 2976384 | 0.005648 |
| validation_support_precision_selector | 0.717 | 0.157 | 62265.6 | 3435955 | 0.005751 |
| validation_support_precision_gate_selector | 0.717 | 0.157 | 62265.6 | 3435955 | 0.005751 |
| train_support_density_selector | 0.729 | 0.169 | 79634.4 | 3457664 | 0.004920 |
| density_capped_compact_induction | 0.678 | 0.118 | 48384.0 | 2976384 | 0.005648 |
| counterfactual_expansion | 0.794 | 0.235 | 483840.0 | 24769664 | 0.001123 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.722 | 0.138 | 51520.0 | 3163776 | 0.005809 |
| compact_train_size_gated_induction | 0.726 | 0.142 | 154839.2 | 7796864 | 0.001987 |
| support_ramped_compact_induction | 0.755 | 0.170 | 77981.6 | 4246579 | 0.004744 |
| density_window_compact_induction | 0.722 | 0.138 | 51520.0 | 3163776 | 0.005809 |
| support_probe_window_selector | 0.722 | 0.138 | 51520.0 | 3163776 | 0.005809 |
| validation_support_precision_selector | 0.722 | 0.138 | 55818.0 | 3163776 | 0.005361 |
| validation_support_precision_gate_selector | 0.722 | 0.138 | 55818.0 | 3163776 | 0.005361 |
| train_support_density_selector | 0.722 | 0.138 | 72321.6 | 3163776 | 0.004139 |
| density_capped_compact_induction | 0.722 | 0.138 | 51520.0 | 3163776 | 0.005809 |
| counterfactual_expansion | 0.834 | 0.249 | 515200.0 | 26343296 | 0.001052 |
