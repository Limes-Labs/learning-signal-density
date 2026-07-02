# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T19:27:30Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_support_probe_window_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_validation_support_precision`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 80 | 120 | 0.131 | 0.006246 |
| compact_train_size_gated_induction | 64 | 80 | 0.229 | 0.010010 |
| support_ramped_compact_induction | 64 | 80 | 0.229 | 0.010010 |
| density_window_compact_induction | 64 | 80 | 0.229 | 0.010010 |
| support_probe_window_selector | 64 | 80 | 0.229 | 0.010010 |
| validation_support_precision_selector | 64 | 80 | 0.229 | 0.010010 |
| train_support_density_selector | 64 | 80 | 0.229 | 0.008505 |
| density_capped_compact_induction | 64 | 80 | 0.229 | 0.010010 |
| counterfactual_expansion | 64 | 80 | 0.317 | 0.003409 |

Comparison target: `results/tiny_neural_budget_sweep_support_probe_window_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | -0.016 | 25760.0 | 1583104 | -0.002630 |
| compact_train_size_gated_induction | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| support_ramped_compact_induction | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| density_window_compact_induction | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| support_probe_window_selector | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| validation_support_precision_selector | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| train_support_density_selector | 0.743 | 0.148 | 92166.8 | 3904640 | 0.006979 |
| density_capped_compact_induction | 0.743 | 0.148 | 77335.6 | 3904640 | 0.008321 |
| counterfactual_expansion | 0.829 | 0.234 | 257600.0 | 13183104 | 0.003946 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.608 | 0.052 | 32256.0 | 1984896 | 0.005607 |
| compact_train_size_gated_induction | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| support_ramped_compact_induction | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| density_window_compact_induction | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| support_probe_window_selector | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| validation_support_precision_selector | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| train_support_density_selector | 0.785 | 0.229 | 93427.2 | 4092314 | 0.008505 |
| density_capped_compact_induction | 0.785 | 0.229 | 79372.8 | 4092314 | 0.010010 |
| counterfactual_expansion | 0.873 | 0.317 | 322560.0 | 16515456 | 0.003409 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.619 | 0.028 | 38752.0 | 2381568 | 0.002075 |
| compact_train_size_gated_induction | 0.711 | 0.120 | 100703.6 | 5158963 | 0.003436 |
| support_ramped_compact_induction | 0.711 | 0.120 | 100703.6 | 5158963 | 0.003436 |
| density_window_compact_induction | 0.619 | 0.028 | 38752.0 | 2381568 | 0.002075 |
| support_probe_window_selector | 0.619 | 0.028 | 38752.0 | 2381568 | 0.002075 |
| validation_support_precision_selector | 0.704 | 0.113 | 89499.0 | 4598272 | 0.004198 |
| train_support_density_selector | 0.711 | 0.120 | 118776.4 | 5158963 | 0.002912 |
| density_capped_compact_induction | 0.711 | 0.120 | 100703.6 | 5158963 | 0.003436 |
| counterfactual_expansion | 0.805 | 0.214 | 387520.0 | 19821568 | 0.001595 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.691 | 0.059 | 41888.0 | 2572928 | 0.003779 |
| compact_train_size_gated_induction | 0.738 | 0.106 | 125794.0 | 6339968 | 0.002243 |
| support_ramped_compact_induction | 0.773 | 0.141 | 86410.0 | 4529536 | 0.004331 |
| density_window_compact_induction | 0.691 | 0.059 | 41888.0 | 2572928 | 0.003779 |
| support_probe_window_selector | 0.773 | 0.141 | 86410.0 | 4529536 | 0.004331 |
| validation_support_precision_selector | 0.757 | 0.125 | 78558.6 | 4121523 | 0.004455 |
| train_support_density_selector | 0.773 | 0.141 | 105782.0 | 4529536 | 0.003539 |
| density_capped_compact_induction | 0.691 | 0.059 | 41888.0 | 2572928 | 0.003779 |
| counterfactual_expansion | 0.827 | 0.195 | 418880.0 | 21421568 | 0.001246 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.694 | 0.104 | 45248.0 | 2780032 | 0.005715 |
| compact_train_size_gated_induction | 0.752 | 0.163 | 134476.0 | 6809344 | 0.002994 |
| support_ramped_compact_induction | 0.761 | 0.172 | 78748.0 | 4211968 | 0.005395 |
| density_window_compact_induction | 0.761 | 0.172 | 78748.0 | 4211968 | 0.005395 |
| support_probe_window_selector | 0.761 | 0.172 | 78748.0 | 4211968 | 0.005395 |
| validation_support_precision_selector | 0.761 | 0.172 | 78748.0 | 4211968 | 0.005395 |
| train_support_density_selector | 0.761 | 0.172 | 97412.0 | 4211968 | 0.004361 |
| density_capped_compact_induction | 0.694 | 0.104 | 45248.0 | 2780032 | 0.005715 |
| counterfactual_expansion | 0.819 | 0.230 | 452480.0 | 23145472 | 0.001257 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.707 | 0.131 | 48384.0 | 2977536 | 0.006246 |
| compact_train_size_gated_induction | 0.710 | 0.133 | 145269.6 | 7332480 | 0.002125 |
| support_ramped_compact_induction | 0.728 | 0.151 | 75772.8 | 4112154 | 0.004633 |
| density_window_compact_induction | 0.707 | 0.131 | 48384.0 | 2977536 | 0.006246 |
| support_probe_window_selector | 0.707 | 0.131 | 48384.0 | 2977536 | 0.006246 |
| validation_support_precision_selector | 0.726 | 0.150 | 62265.6 | 3437107 | 0.005723 |
| train_support_density_selector | 0.701 | 0.125 | 79634.4 | 3458816 | 0.003704 |
| density_capped_compact_induction | 0.707 | 0.131 | 48384.0 | 2977536 | 0.006246 |
| counterfactual_expansion | 0.810 | 0.233 | 483840.0 | 24770816 | 0.001116 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.714 | 0.123 | 51520.0 | 3164416 | 0.005206 |
| compact_train_size_gated_induction | 0.749 | 0.158 | 154848.8 | 7799808 | 0.002224 |
| support_ramped_compact_induction | 0.729 | 0.138 | 78154.4 | 4256461 | 0.003824 |
| density_window_compact_induction | 0.714 | 0.123 | 51520.0 | 3164416 | 0.005206 |
| support_probe_window_selector | 0.714 | 0.123 | 51520.0 | 3164416 | 0.005206 |
| validation_support_precision_selector | 0.729 | 0.138 | 60742.8 | 3393178 | 0.004866 |
| train_support_density_selector | 0.714 | 0.123 | 72350.4 | 3164416 | 0.003708 |
| density_capped_compact_induction | 0.714 | 0.123 | 51520.0 | 3164416 | 0.005206 |
| counterfactual_expansion | 0.840 | 0.249 | 515200.0 | 26343936 | 0.001052 |
