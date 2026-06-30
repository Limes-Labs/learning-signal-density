# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T23:03:07Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_support_utility_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_validation_support_gain_gate`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 104 | 112 | 0.116 | 0.006369 |
| compact_train_size_gated_induction | 64 | 80 | 0.158 | 0.006934 |
| support_ramped_compact_induction | 64 | 112 | 0.175 | 0.005489 |
| density_window_compact_induction | 64 | 112 | 0.175 | 0.005489 |
| support_probe_window_selector | 64 | 112 | 0.175 | 0.005489 |
| validation_support_precision_selector | 64 | 112 | 0.175 | 0.005489 |
| validation_support_precision_gate_selector | 64 | 80 | 0.158 | 0.006934 |
| validation_support_utility_selector | 64 | 80 | 0.158 | 0.006934 |
| validation_support_gain_gate_selector | 64 | 80 | 0.158 | 0.006934 |
| train_support_density_selector | 64 | 112 | 0.175 | 0.004437 |
| density_capped_compact_induction | 64 | 80 | 0.158 | 0.006934 |
| counterfactual_expansion | 64 | 128 | 0.268 | 0.001129 |

Comparison target: `results/support_selector_error_audit_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.595 | 0.029 | 25760.0 | 1583104 | 0.004822 |
| compact_train_size_gated_induction | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| support_ramped_compact_induction | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| density_window_compact_induction | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| support_probe_window_selector | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| validation_support_precision_selector | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| validation_support_precision_gate_selector | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| validation_support_utility_selector | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| validation_support_gain_gate_selector | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| train_support_density_selector | 0.722 | 0.156 | 92166.8 | 3903488 | 0.007353 |
| density_capped_compact_induction | 0.722 | 0.156 | 77335.6 | 3903488 | 0.008762 |
| counterfactual_expansion | 0.818 | 0.252 | 257600.0 | 13183104 | 0.004253 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.619 | 0.027 | 32256.0 | 1984768 | 0.002915 |
| compact_train_size_gated_induction | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| support_ramped_compact_induction | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| density_window_compact_induction | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| support_probe_window_selector | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| validation_support_precision_selector | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| validation_support_precision_gate_selector | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| validation_support_utility_selector | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| validation_support_gain_gate_selector | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| train_support_density_selector | 0.750 | 0.158 | 93849.6 | 4107981 | 0.005892 |
| density_capped_compact_induction | 0.750 | 0.158 | 79718.4 | 4107981 | 0.006934 |
| counterfactual_expansion | 0.840 | 0.248 | 322560.0 | 16515328 | 0.002669 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.621 | 0.028 | 38752.0 | 2380800 | 0.002075 |
| compact_train_size_gated_induction | 0.750 | 0.157 | 100703.6 | 5156915 | 0.004481 |
| support_ramped_compact_induction | 0.750 | 0.157 | 100703.6 | 5156915 | 0.004481 |
| density_window_compact_induction | 0.621 | 0.028 | 38752.0 | 2380800 | 0.002075 |
| support_probe_window_selector | 0.621 | 0.028 | 38752.0 | 2380800 | 0.002075 |
| validation_support_precision_selector | 0.723 | 0.130 | 89499.0 | 4596224 | 0.004411 |
| validation_support_precision_gate_selector | 0.723 | 0.130 | 89499.0 | 4596224 | 0.004411 |
| validation_support_utility_selector | 0.677 | 0.083 | 69713.4 | 3500646 | 0.003092 |
| validation_support_gain_gate_selector | 0.645 | 0.052 | 72528.6 | 2940083 | 0.002095 |
| train_support_density_selector | 0.750 | 0.157 | 118776.4 | 5156915 | 0.003799 |
| density_capped_compact_induction | 0.750 | 0.157 | 100703.6 | 5156915 | 0.004481 |
| counterfactual_expansion | 0.814 | 0.221 | 387520.0 | 19820800 | 0.001647 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.704 | 0.069 | 41888.0 | 2572928 | 0.004392 |
| compact_train_size_gated_induction | 0.754 | 0.118 | 125813.2 | 6342272 | 0.002516 |
| support_ramped_compact_induction | 0.765 | 0.130 | 86755.6 | 4546867 | 0.003989 |
| density_window_compact_induction | 0.704 | 0.069 | 41888.0 | 2572928 | 0.004392 |
| support_probe_window_selector | 0.765 | 0.130 | 86755.6 | 4546867 | 0.003989 |
| validation_support_precision_selector | 0.733 | 0.098 | 70696.2 | 3754854 | 0.003744 |
| validation_support_precision_gate_selector | 0.733 | 0.098 | 70696.2 | 3754854 | 0.003744 |
| validation_support_utility_selector | 0.704 | 0.069 | 49093.8 | 2572928 | 0.003755 |
| validation_support_gain_gate_selector | 0.718 | 0.083 | 74078.4 | 3388186 | 0.003057 |
| train_support_density_selector | 0.765 | 0.130 | 106185.2 | 4546867 | 0.003259 |
| density_capped_compact_induction | 0.704 | 0.069 | 41888.0 | 2572928 | 0.004392 |
| counterfactual_expansion | 0.829 | 0.194 | 418880.0 | 21421568 | 0.001236 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.699 | 0.116 | 45248.0 | 2780032 | 0.006369 |
| compact_train_size_gated_induction | 0.736 | 0.154 | 134476.0 | 6807040 | 0.002830 |
| support_ramped_compact_induction | 0.757 | 0.175 | 78748.0 | 4210816 | 0.005489 |
| density_window_compact_induction | 0.757 | 0.175 | 78748.0 | 4210816 | 0.005489 |
| support_probe_window_selector | 0.757 | 0.175 | 78748.0 | 4210816 | 0.005489 |
| validation_support_precision_selector | 0.757 | 0.175 | 78748.0 | 4210816 | 0.005489 |
| validation_support_precision_gate_selector | 0.716 | 0.134 | 55148.4 | 3066829 | 0.006058 |
| validation_support_utility_selector | 0.699 | 0.116 | 50017.2 | 2780032 | 0.005788 |
| validation_support_gain_gate_selector | 0.716 | 0.134 | 58548.0 | 3066829 | 0.005855 |
| train_support_density_selector | 0.757 | 0.175 | 97412.0 | 4210816 | 0.004437 |
| density_capped_compact_induction | 0.699 | 0.116 | 45248.0 | 2780032 | 0.006369 |
| counterfactual_expansion | 0.818 | 0.236 | 452480.0 | 23145472 | 0.001290 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.688 | 0.104 | 48384.0 | 2977024 | 0.004984 |
| compact_train_size_gated_induction | 0.733 | 0.150 | 145269.6 | 7330432 | 0.002390 |
| support_ramped_compact_induction | 0.736 | 0.153 | 75772.8 | 4111898 | 0.004646 |
| density_window_compact_induction | 0.688 | 0.104 | 48384.0 | 2977024 | 0.004984 |
| support_probe_window_selector | 0.688 | 0.104 | 48384.0 | 2977024 | 0.004984 |
| validation_support_precision_selector | 0.735 | 0.151 | 72115.2 | 3894374 | 0.004977 |
| validation_support_precision_gate_selector | 0.735 | 0.151 | 72115.2 | 3894374 | 0.004977 |
| validation_support_utility_selector | 0.688 | 0.104 | 55987.2 | 2977024 | 0.004349 |
| validation_support_gain_gate_selector | 0.731 | 0.147 | 81792.0 | 3676083 | 0.004390 |
| train_support_density_selector | 0.729 | 0.146 | 79634.4 | 3459584 | 0.004115 |
| density_capped_compact_induction | 0.688 | 0.104 | 48384.0 | 2977024 | 0.004984 |
| counterfactual_expansion | 0.815 | 0.232 | 483840.0 | 24770304 | 0.001110 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.647 | 0.048 | 51520.0 | 3163264 | 0.002028 |
| compact_train_size_gated_induction | 0.730 | 0.131 | 154824.8 | 7796352 | 0.001842 |
| support_ramped_compact_induction | 0.742 | 0.143 | 77722.4 | 4234189 | 0.003983 |
| density_window_compact_induction | 0.647 | 0.048 | 51520.0 | 3163264 | 0.002028 |
| support_probe_window_selector | 0.647 | 0.048 | 51520.0 | 3163264 | 0.002028 |
| validation_support_precision_selector | 0.668 | 0.069 | 60742.8 | 3392026 | 0.002232 |
| validation_support_precision_gate_selector | 0.668 | 0.069 | 60742.8 | 3392026 | 0.002232 |
| validation_support_utility_selector | 0.647 | 0.048 | 56734.8 | 3163264 | 0.001821 |
| validation_support_gain_gate_selector | 0.647 | 0.048 | 60012.0 | 3163264 | 0.001692 |
| train_support_density_selector | 0.647 | 0.048 | 72278.4 | 3163264 | 0.001442 |
| density_capped_compact_induction | 0.647 | 0.048 | 51520.0 | 3163264 | 0.002028 |
| counterfactual_expansion | 0.866 | 0.268 | 515200.0 | 26342784 | 0.001129 |
