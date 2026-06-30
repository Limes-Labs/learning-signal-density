# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T21:24:48Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_support_selector_transfer`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 80 | 128 | 0.148 | 0.006247 |
| compact_train_size_gated_induction | 64 | 80 | 0.190 | 0.008284 |
| support_ramped_compact_induction | 64 | 80 | 0.190 | 0.008284 |
| density_window_compact_induction | 64 | 80 | 0.190 | 0.008284 |
| support_probe_window_selector | 64 | 80 | 0.190 | 0.008284 |
| validation_support_precision_selector | 64 | 80 | 0.190 | 0.008284 |
| validation_support_precision_gate_selector | 64 | 80 | 0.190 | 0.008284 |
| train_support_density_selector | 64 | 80 | 0.190 | 0.007038 |
| density_capped_compact_induction | 64 | 80 | 0.190 | 0.008284 |
| counterfactual_expansion | 64 | 128 | 0.258 | 0.001090 |

Comparison target: `results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.582 | -0.029 | 25760.0 | 1584128 | -0.004822 |
| compact_train_size_gated_induction | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| support_ramped_compact_induction | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| density_window_compact_induction | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| support_probe_window_selector | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| validation_support_precision_selector | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| validation_support_precision_gate_selector | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| train_support_density_selector | 0.719 | 0.109 | 92109.2 | 3903616 | 0.005149 |
| density_capped_compact_induction | 0.719 | 0.109 | 77316.4 | 3903616 | 0.006134 |
| counterfactual_expansion | 0.839 | 0.229 | 257600.0 | 13184128 | 0.003858 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.600 | 0.031 | 32256.0 | 1984640 | 0.003364 |
| compact_train_size_gated_induction | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| support_ramped_compact_induction | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| density_window_compact_induction | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| support_probe_window_selector | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| validation_support_precision_selector | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| validation_support_precision_gate_selector | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| train_support_density_selector | 0.758 | 0.190 | 93638.4 | 4100403 | 0.007038 |
| density_capped_compact_induction | 0.758 | 0.190 | 79545.6 | 4100403 | 0.008284 |
| counterfactual_expansion | 0.790 | 0.221 | 322560.0 | 16515200 | 0.002377 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.685 | 0.092 | 38752.0 | 2380544 | 0.006875 |
| compact_train_size_gated_induction | 0.762 | 0.169 | 101222.0 | 5183488 | 0.004817 |
| support_ramped_compact_induction | 0.762 | 0.169 | 101222.0 | 5183488 | 0.004817 |
| density_window_compact_induction | 0.685 | 0.092 | 38752.0 | 2380544 | 0.006875 |
| support_probe_window_selector | 0.685 | 0.092 | 38752.0 | 2380544 | 0.006875 |
| validation_support_precision_selector | 0.729 | 0.136 | 78007.8 | 4061594 | 0.005816 |
| validation_support_precision_gate_selector | 0.729 | 0.136 | 78007.8 | 4061594 | 0.005816 |
| train_support_density_selector | 0.762 | 0.169 | 119410.0 | 5183488 | 0.004083 |
| density_capped_compact_induction | 0.762 | 0.169 | 101222.0 | 5183488 | 0.004817 |
| counterfactual_expansion | 0.795 | 0.202 | 387520.0 | 19820544 | 0.001505 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.659 | 0.066 | 41888.0 | 2572160 | 0.004187 |
| compact_train_size_gated_induction | 0.717 | 0.123 | 125784.4 | 6343936 | 0.002619 |
| support_ramped_compact_induction | 0.734 | 0.141 | 86237.2 | 4522982 | 0.004345 |
| density_window_compact_induction | 0.659 | 0.066 | 41888.0 | 2572160 | 0.004187 |
| support_probe_window_selector | 0.734 | 0.141 | 86237.2 | 4522982 | 0.004345 |
| validation_support_precision_selector | 0.683 | 0.090 | 61451.4 | 3322829 | 0.004530 |
| validation_support_precision_gate_selector | 0.683 | 0.090 | 61451.4 | 3322829 | 0.004530 |
| train_support_density_selector | 0.734 | 0.141 | 105580.4 | 4522982 | 0.003550 |
| density_capped_compact_induction | 0.659 | 0.066 | 41888.0 | 2572160 | 0.004187 |
| counterfactual_expansion | 0.803 | 0.210 | 418880.0 | 21420800 | 0.001338 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.748 | 0.142 | 45248.0 | 2780288 | 0.007757 |
| compact_train_size_gated_induction | 0.739 | 0.133 | 134476.0 | 6810368 | 0.002445 |
| support_ramped_compact_induction | 0.739 | 0.133 | 78748.0 | 4209920 | 0.004175 |
| density_window_compact_induction | 0.739 | 0.133 | 78748.0 | 4209920 | 0.004175 |
| support_probe_window_selector | 0.739 | 0.133 | 78748.0 | 4209920 | 0.004175 |
| validation_support_precision_selector | 0.739 | 0.133 | 78748.0 | 4209920 | 0.004175 |
| validation_support_precision_gate_selector | 0.752 | 0.146 | 61282.8 | 3351834 | 0.006517 |
| train_support_density_selector | 0.739 | 0.133 | 97412.0 | 4209920 | 0.003375 |
| density_capped_compact_induction | 0.748 | 0.142 | 45248.0 | 2780288 | 0.007757 |
| counterfactual_expansion | 0.860 | 0.254 | 452480.0 | 23145728 | 0.001388 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.682 | 0.113 | 48384.0 | 2975232 | 0.005382 |
| compact_train_size_gated_induction | 0.714 | 0.144 | 145298.4 | 7329408 | 0.002301 |
| support_ramped_compact_induction | 0.711 | 0.142 | 76291.2 | 4134886 | 0.004309 |
| density_window_compact_induction | 0.682 | 0.113 | 48384.0 | 2975232 | 0.005382 |
| support_probe_window_selector | 0.682 | 0.113 | 48384.0 | 2975232 | 0.005382 |
| validation_support_precision_selector | 0.689 | 0.119 | 57600.0 | 3216512 | 0.004504 |
| validation_support_precision_gate_selector | 0.689 | 0.119 | 57600.0 | 3216512 | 0.004504 |
| train_support_density_selector | 0.674 | 0.104 | 85509.6 | 3699072 | 0.002765 |
| density_capped_compact_induction | 0.682 | 0.113 | 48384.0 | 2975232 | 0.005382 |
| counterfactual_expansion | 0.807 | 0.237 | 483840.0 | 24768512 | 0.001136 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.738 | 0.148 | 51520.0 | 3163136 | 0.006247 |
| compact_train_size_gated_induction | 0.743 | 0.153 | 154810.4 | 7794304 | 0.002152 |
| support_ramped_compact_induction | 0.738 | 0.148 | 77463.2 | 4220518 | 0.004145 |
| density_window_compact_induction | 0.738 | 0.148 | 51520.0 | 3163136 | 0.006247 |
| support_probe_window_selector | 0.738 | 0.148 | 51520.0 | 3163136 | 0.006247 |
| validation_support_precision_selector | 0.738 | 0.148 | 55818.0 | 3163136 | 0.005766 |
| validation_support_precision_gate_selector | 0.738 | 0.148 | 55818.0 | 3163136 | 0.005766 |
| train_support_density_selector | 0.738 | 0.148 | 72235.2 | 3163136 | 0.004454 |
| density_capped_compact_induction | 0.738 | 0.148 | 51520.0 | 3163136 | 0.006247 |
| counterfactual_expansion | 0.848 | 0.258 | 515200.0 | 26342656 | 0.001090 |
