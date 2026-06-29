# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T17:27:15Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_density_capped_compact_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_support_ramped_compact`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 80 | 128 | 0.155 | 0.006521 |
| train_size_gated_sample_aware_induction | 64 | 80 | 0.204 | 0.005162 |
| compact_train_size_gated_induction | 64 | 80 | 0.185 | 0.008230 |
| support_ramped_compact_induction | 64 | 80 | 0.185 | 0.008230 |
| density_capped_compact_induction | 64 | 80 | 0.185 | 0.008230 |
| counterfactual_expansion | 64 | 80 | 0.260 | 0.002803 |

Comparison target: `results/tiny_neural_budget_sweep_density_capped_compact_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.561 | -0.031 | 25760.0 | 1584896 | -0.005261 |
| train_size_gated_sample_aware_induction | 0.706 | 0.114 | 124294.0 | 6225280 | 0.003998 |
| compact_train_size_gated_induction | 0.738 | 0.145 | 77374.0 | 3904384 | 0.008173 |
| support_ramped_compact_induction | 0.738 | 0.145 | 77374.0 | 3904384 | 0.008173 |
| density_capped_compact_induction | 0.738 | 0.145 | 77374.0 | 3904384 | 0.008173 |
| counterfactual_expansion | 0.782 | 0.190 | 257600.0 | 13184896 | 0.003200 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.608 | 0.052 | 32256.0 | 1985152 | 0.005607 |
| train_size_gated_sample_aware_induction | 0.760 | 0.204 | 137433.6 | 6966835 | 0.005162 |
| compact_train_size_gated_induction | 0.742 | 0.185 | 78681.6 | 4060083 | 0.008230 |
| support_ramped_compact_induction | 0.742 | 0.185 | 78681.6 | 4060083 | 0.008230 |
| density_capped_compact_induction | 0.742 | 0.185 | 78681.6 | 4060083 | 0.008230 |
| counterfactual_expansion | 0.817 | 0.260 | 322560.0 | 16515712 | 0.002803 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.673 | 0.082 | 38752.0 | 2381056 | 0.006096 |
| train_size_gated_sample_aware_induction | 0.774 | 0.183 | 171287.6 | 8649779 | 0.003076 |
| compact_train_size_gated_induction | 0.767 | 0.176 | 100703.6 | 5161523 | 0.005027 |
| support_ramped_compact_induction | 0.767 | 0.176 | 100703.6 | 5161523 | 0.005027 |
| density_capped_compact_induction | 0.767 | 0.176 | 100703.6 | 5161523 | 0.005027 |
| counterfactual_expansion | 0.835 | 0.243 | 387520.0 | 19821056 | 0.001816 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.624 | 0.034 | 41888.0 | 2571776 | 0.002145 |
| train_size_gated_sample_aware_induction | 0.730 | 0.139 | 202051.6 | 10110592 | 0.001842 |
| compact_train_size_gated_induction | 0.725 | 0.134 | 125755.6 | 6342016 | 0.002857 |
| support_ramped_compact_induction | 0.710 | 0.120 | 85718.8 | 4497562 | 0.003735 |
| density_capped_compact_induction | 0.624 | 0.034 | 41888.0 | 2571776 | 0.002145 |
| counterfactual_expansion | 0.816 | 0.226 | 418880.0 | 21420416 | 0.001440 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.707 | 0.091 | 45248.0 | 2780032 | 0.004980 |
| train_size_gated_sample_aware_induction | 0.793 | 0.176 | 216892.0 | 10886656 | 0.002010 |
| compact_train_size_gated_induction | 0.740 | 0.124 | 134476.0 | 6813824 | 0.002280 |
| support_ramped_compact_induction | 0.775 | 0.158 | 78748.0 | 4210560 | 0.004973 |
| density_capped_compact_induction | 0.707 | 0.091 | 45248.0 | 2780032 | 0.004980 |
| counterfactual_expansion | 0.833 | 0.216 | 452480.0 | 23145472 | 0.001184 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.668 | 0.100 | 48384.0 | 2974208 | 0.004784 |
| train_size_gated_sample_aware_induction | 0.710 | 0.142 | 233368.8 | 11690752 | 0.001405 |
| compact_train_size_gated_induction | 0.706 | 0.138 | 145240.8 | 7334144 | 0.002192 |
| support_ramped_compact_induction | 0.701 | 0.133 | 75254.4 | 4089165 | 0.004110 |
| density_capped_compact_induction | 0.668 | 0.100 | 48384.0 | 2974208 | 0.004784 |
| counterfactual_expansion | 0.800 | 0.232 | 483840.0 | 24767488 | 0.001110 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.752 | 0.155 | 51520.0 | 3164160 | 0.006521 |
| train_size_gated_sample_aware_induction | 0.765 | 0.168 | 248664.8 | 12434688 | 0.001464 |
| compact_train_size_gated_induction | 0.717 | 0.119 | 154824.8 | 7798528 | 0.001677 |
| support_ramped_compact_induction | 0.782 | 0.184 | 77722.4 | 4234189 | 0.005153 |
| density_capped_compact_induction | 0.752 | 0.155 | 51520.0 | 3164160 | 0.006521 |
| counterfactual_expansion | 0.849 | 0.252 | 515200.0 | 26343680 | 0.001063 |
