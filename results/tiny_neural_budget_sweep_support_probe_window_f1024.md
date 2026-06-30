# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T18:55:06Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_train_support_density_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_support_probe_window`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 96 | 112 | 0.130 | 0.007103 |
| compact_train_size_gated_induction | 64 | 80 | 0.181 | 0.007941 |
| support_ramped_compact_induction | 64 | 80 | 0.181 | 0.007941 |
| density_window_compact_induction | 64 | 80 | 0.181 | 0.007941 |
| train_support_density_selector | 64 | 80 | 0.181 | 0.006747 |
| support_probe_window_selector | 64 | 80 | 0.181 | 0.007941 |
| density_capped_compact_induction | 64 | 80 | 0.181 | 0.007941 |
| counterfactual_expansion | 64 | 112 | 0.249 | 0.001363 |

Comparison target: `results/tiny_neural_budget_sweep_train_support_density_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.556 | -0.034 | 25760.0 | 1583360 | -0.005699 |
| compact_train_size_gated_induction | 0.699 | 0.109 | 77297.2 | 3903744 | 0.006136 |
| support_ramped_compact_induction | 0.699 | 0.109 | 77297.2 | 3903744 | 0.006136 |
| density_window_compact_induction | 0.699 | 0.109 | 77297.2 | 3903744 | 0.006136 |
| train_support_density_selector | 0.699 | 0.109 | 92051.6 | 3903744 | 0.005152 |
| support_probe_window_selector | 0.699 | 0.109 | 77297.2 | 3903744 | 0.006136 |
| density_capped_compact_induction | 0.699 | 0.109 | 77297.2 | 3903744 | 0.006136 |
| counterfactual_expansion | 0.779 | 0.190 | 257600.0 | 13183360 | 0.003200 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.610 | 0.015 | 32256.0 | 1984256 | 0.001570 |
| compact_train_size_gated_induction | 0.777 | 0.181 | 79718.4 | 4108877 | 0.007941 |
| support_ramped_compact_induction | 0.777 | 0.181 | 79718.4 | 4108877 | 0.007941 |
| density_window_compact_induction | 0.777 | 0.181 | 79718.4 | 4108877 | 0.007941 |
| train_support_density_selector | 0.777 | 0.181 | 93849.6 | 4108877 | 0.006747 |
| support_probe_window_selector | 0.777 | 0.181 | 79718.4 | 4108877 | 0.007941 |
| density_capped_compact_induction | 0.777 | 0.181 | 79718.4 | 4108877 | 0.007941 |
| counterfactual_expansion | 0.792 | 0.196 | 322560.0 | 16514816 | 0.002108 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.649 | 0.049 | 38752.0 | 2380800 | 0.003632 |
| compact_train_size_gated_induction | 0.772 | 0.172 | 100703.6 | 5158963 | 0.004921 |
| support_ramped_compact_induction | 0.772 | 0.172 | 100703.6 | 5158963 | 0.004921 |
| density_window_compact_induction | 0.649 | 0.049 | 38752.0 | 2380800 | 0.003632 |
| train_support_density_selector | 0.772 | 0.172 | 118776.4 | 5158963 | 0.004172 |
| support_probe_window_selector | 0.649 | 0.049 | 38752.0 | 2380800 | 0.003632 |
| density_capped_compact_induction | 0.772 | 0.172 | 100703.6 | 5158963 | 0.004921 |
| counterfactual_expansion | 0.833 | 0.233 | 387520.0 | 19820800 | 0.001738 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.659 | 0.053 | 41888.0 | 2572544 | 0.003370 |
| compact_train_size_gated_induction | 0.744 | 0.138 | 125832.4 | 6344320 | 0.002924 |
| support_ramped_compact_induction | 0.758 | 0.152 | 87101.2 | 4563686 | 0.004655 |
| density_window_compact_induction | 0.659 | 0.053 | 41888.0 | 2572544 | 0.003370 |
| train_support_density_selector | 0.758 | 0.152 | 106588.4 | 4563686 | 0.003805 |
| support_probe_window_selector | 0.758 | 0.152 | 87101.2 | 4563686 | 0.004655 |
| density_capped_compact_induction | 0.659 | 0.053 | 41888.0 | 2572544 | 0.003370 |
| counterfactual_expansion | 0.818 | 0.211 | 418880.0 | 21421184 | 0.001348 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.730 | 0.130 | 45248.0 | 2780544 | 0.007103 |
| compact_train_size_gated_induction | 0.707 | 0.107 | 134476.0 | 6809216 | 0.001978 |
| support_ramped_compact_induction | 0.764 | 0.164 | 78748.0 | 4210432 | 0.005161 |
| density_window_compact_induction | 0.764 | 0.164 | 78748.0 | 4210432 | 0.005161 |
| train_support_density_selector | 0.764 | 0.164 | 97412.0 | 4210432 | 0.004172 |
| support_probe_window_selector | 0.764 | 0.164 | 78748.0 | 4210432 | 0.005161 |
| density_capped_compact_induction | 0.730 | 0.130 | 45248.0 | 2780544 | 0.007103 |
| counterfactual_expansion | 0.849 | 0.249 | 452480.0 | 23145984 | 0.001363 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.667 | 0.075 | 48384.0 | 2976768 | 0.003588 |
| compact_train_size_gated_induction | 0.717 | 0.125 | 145298.4 | 7329792 | 0.001991 |
| support_ramped_compact_induction | 0.732 | 0.140 | 76291.2 | 4135398 | 0.004240 |
| density_window_compact_induction | 0.667 | 0.075 | 48384.0 | 2976768 | 0.003588 |
| train_support_density_selector | 0.718 | 0.126 | 85509.6 | 3700608 | 0.003266 |
| support_probe_window_selector | 0.667 | 0.075 | 48384.0 | 2976768 | 0.003588 |
| density_capped_compact_induction | 0.667 | 0.075 | 48384.0 | 2976768 | 0.003588 |
| counterfactual_expansion | 0.800 | 0.208 | 483840.0 | 24770048 | 0.000997 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.716 | 0.105 | 51520.0 | 3163264 | 0.004439 |
| compact_train_size_gated_induction | 0.696 | 0.086 | 154829.6 | 7793792 | 0.001203 |
| support_ramped_compact_induction | 0.732 | 0.122 | 77808.8 | 4237722 | 0.003409 |
| density_window_compact_induction | 0.716 | 0.105 | 51520.0 | 3163264 | 0.004439 |
| train_support_density_selector | 0.716 | 0.105 | 72292.8 | 3163264 | 0.003161 |
| support_probe_window_selector | 0.716 | 0.105 | 51520.0 | 3163264 | 0.004439 |
| density_capped_compact_induction | 0.716 | 0.105 | 51520.0 | 3163264 | 0.004439 |
| counterfactual_expansion | 0.817 | 0.206 | 515200.0 | 26342784 | 0.000871 |
