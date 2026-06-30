# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T00:08:21Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_density_window_compact_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_train_support_density`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 96 | 128 | 0.138 | 0.005809 |
| compact_train_size_gated_induction | 64 | 80 | 0.188 | 0.008171 |
| support_ramped_compact_induction | 64 | 80 | 0.188 | 0.008171 |
| train_support_density_selector | 64 | 80 | 0.188 | 0.006941 |
| density_window_compact_induction | 64 | 80 | 0.188 | 0.008171 |
| density_capped_compact_induction | 64 | 80 | 0.188 | 0.008171 |
| counterfactual_expansion | 64 | 80 | 0.269 | 0.002893 |

Comparison target: `results/tiny_neural_budget_sweep_density_window_compact_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.551 | -0.057 | 25760.0 | 1584128 | -0.009645 |
| compact_train_size_gated_induction | 0.722 | 0.114 | 77354.8 | 3904256 | 0.006423 |
| support_ramped_compact_induction | 0.722 | 0.114 | 77354.8 | 3904256 | 0.006423 |
| train_support_density_selector | 0.722 | 0.114 | 92224.4 | 3904256 | 0.005386 |
| density_window_compact_induction | 0.722 | 0.114 | 77354.8 | 3904256 | 0.006423 |
| density_capped_compact_induction | 0.722 | 0.114 | 77354.8 | 3904256 | 0.006423 |
| counterfactual_expansion | 0.792 | 0.184 | 257600.0 | 13184128 | 0.003113 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.596 | 0.019 | 32256.0 | 1984768 | 0.002018 |
| compact_train_size_gated_induction | 0.765 | 0.188 | 79891.2 | 4117094 | 0.008171 |
| support_ramped_compact_induction | 0.765 | 0.188 | 79891.2 | 4117094 | 0.008171 |
| train_support_density_selector | 0.765 | 0.188 | 94060.8 | 4117094 | 0.006941 |
| density_window_compact_induction | 0.765 | 0.188 | 79891.2 | 4117094 | 0.008171 |
| density_capped_compact_induction | 0.765 | 0.188 | 79891.2 | 4117094 | 0.008171 |
| counterfactual_expansion | 0.846 | 0.269 | 322560.0 | 16515328 | 0.002893 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.640 | 0.050 | 38752.0 | 2381312 | 0.003761 |
| compact_train_size_gated_induction | 0.720 | 0.130 | 100703.6 | 5161267 | 0.003727 |
| support_ramped_compact_induction | 0.720 | 0.130 | 100703.6 | 5161267 | 0.003727 |
| train_support_density_selector | 0.720 | 0.130 | 118776.4 | 5161267 | 0.003159 |
| density_window_compact_induction | 0.640 | 0.050 | 38752.0 | 2381312 | 0.003761 |
| density_capped_compact_induction | 0.720 | 0.130 | 100703.6 | 5161267 | 0.003727 |
| counterfactual_expansion | 0.817 | 0.228 | 387520.0 | 19821312 | 0.001699 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.646 | 0.050 | 41888.0 | 2572160 | 0.003166 |
| compact_train_size_gated_induction | 0.706 | 0.109 | 125755.6 | 6342016 | 0.002313 |
| support_ramped_compact_induction | 0.725 | 0.128 | 85718.8 | 4497818 | 0.003994 |
| train_support_density_selector | 0.725 | 0.128 | 104975.6 | 4497818 | 0.003262 |
| density_window_compact_induction | 0.646 | 0.050 | 41888.0 | 2572160 | 0.003166 |
| density_capped_compact_induction | 0.646 | 0.050 | 41888.0 | 2572160 | 0.003166 |
| counterfactual_expansion | 0.818 | 0.221 | 418880.0 | 21420800 | 0.001409 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.691 | 0.072 | 45248.0 | 2780288 | 0.003919 |
| compact_train_size_gated_induction | 0.760 | 0.140 | 134476.0 | 6812672 | 0.002582 |
| support_ramped_compact_induction | 0.745 | 0.125 | 78748.0 | 4211072 | 0.003941 |
| train_support_density_selector | 0.745 | 0.125 | 97412.0 | 4211072 | 0.003186 |
| density_window_compact_induction | 0.745 | 0.125 | 78748.0 | 4211072 | 0.003941 |
| density_capped_compact_induction | 0.691 | 0.072 | 45248.0 | 2780288 | 0.003919 |
| counterfactual_expansion | 0.793 | 0.173 | 452480.0 | 23145728 | 0.000947 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.703 | 0.115 | 48384.0 | 2975744 | 0.005515 |
| compact_train_size_gated_induction | 0.713 | 0.125 | 145240.8 | 7333760 | 0.001992 |
| support_ramped_compact_induction | 0.708 | 0.121 | 75254.4 | 4088909 | 0.003719 |
| train_support_density_selector | 0.707 | 0.119 | 73759.2 | 3218304 | 0.003844 |
| density_window_compact_induction | 0.703 | 0.115 | 48384.0 | 2975744 | 0.005515 |
| density_capped_compact_induction | 0.703 | 0.115 | 48384.0 | 2975744 | 0.005515 |
| counterfactual_expansion | 0.804 | 0.217 | 483840.0 | 24769024 | 0.001036 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.752 | 0.138 | 51520.0 | 3164288 | 0.005809 |
| compact_train_size_gated_induction | 0.734 | 0.119 | 154810.4 | 7798016 | 0.001678 |
| support_ramped_compact_induction | 0.781 | 0.166 | 77463.2 | 4223078 | 0.004661 |
| train_support_density_selector | 0.752 | 0.138 | 72235.2 | 3164288 | 0.004142 |
| density_window_compact_induction | 0.752 | 0.138 | 51520.0 | 3164288 | 0.005809 |
| density_capped_compact_induction | 0.752 | 0.138 | 51520.0 | 3164288 | 0.005809 |
| counterfactual_expansion | 0.835 | 0.221 | 515200.0 | 26343808 | 0.000932 |
