# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T17:51:43Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_late_confidence_ramped_compact`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 96 | 160 | 0.177 | 0.004765 |
| compact_train_size_gated_induction | 96 | 160 | 0.197 | 0.001708 |
| support_ramped_compact_induction | 96 | 144 | 0.171 | 0.002705 |
| late_confidence_ramped_compact_induction | 96 | 144 | 0.171 | 0.002705 |
| density_capped_compact_induction | 96 | 160 | 0.177 | 0.004765 |
| counterfactual_expansion | 96 | 144 | 0.268 | 0.000892 |

Comparison target: `results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json`

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.659 | 0.071 | 38752.0 | 2381568 | 0.005318 |
| compact_train_size_gated_induction | 0.741 | 0.153 | 100703.6 | 5161011 | 0.004378 |
| support_ramped_compact_induction | 0.741 | 0.153 | 100703.6 | 5161011 | 0.004378 |
| late_confidence_ramped_compact_induction | 0.741 | 0.153 | 100703.6 | 5161011 | 0.004378 |
| density_capped_compact_induction | 0.741 | 0.153 | 100703.6 | 5161011 | 0.004378 |
| counterfactual_expansion | 0.821 | 0.233 | 387520.0 | 19821568 | 0.001738 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.646 | 0.053 | 41888.0 | 2572544 | 0.003371 |
| compact_train_size_gated_induction | 0.722 | 0.128 | 125813.2 | 6341888 | 0.002720 |
| support_ramped_compact_induction | 0.746 | 0.152 | 86755.6 | 4546355 | 0.004690 |
| late_confidence_ramped_compact_induction | 0.746 | 0.152 | 86755.6 | 4546355 | 0.004690 |
| density_capped_compact_induction | 0.646 | 0.053 | 41888.0 | 2572544 | 0.003371 |
| counterfactual_expansion | 0.830 | 0.237 | 418880.0 | 21421184 | 0.001511 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.709 | 0.094 | 45248.0 | 2780544 | 0.005144 |
| compact_train_size_gated_induction | 0.751 | 0.136 | 134476.0 | 6811520 | 0.002500 |
| support_ramped_compact_induction | 0.739 | 0.124 | 78748.0 | 4211584 | 0.003894 |
| late_confidence_ramped_compact_induction | 0.739 | 0.124 | 78748.0 | 4211584 | 0.003894 |
| density_capped_compact_induction | 0.709 | 0.094 | 45248.0 | 2780544 | 0.005144 |
| counterfactual_expansion | 0.846 | 0.231 | 452480.0 | 23145984 | 0.001266 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.710 | 0.146 | 48384.0 | 2977280 | 0.006977 |
| compact_train_size_gated_induction | 0.671 | 0.107 | 145269.6 | 7331840 | 0.001704 |
| support_ramped_compact_induction | 0.699 | 0.135 | 75772.8 | 4111386 | 0.004119 |
| late_confidence_ramped_compact_induction | 0.703 | 0.139 | 74563.2 | 4054758 | 0.004322 |
| density_capped_compact_induction | 0.710 | 0.146 | 48384.0 | 2977280 | 0.006977 |
| counterfactual_expansion | 0.797 | 0.233 | 483840.0 | 24770560 | 0.001116 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.725 | 0.144 | 51520.0 | 3164416 | 0.006083 |
| compact_train_size_gated_induction | 0.762 | 0.182 | 154868.0 | 7796992 | 0.002552 |
| support_ramped_compact_induction | 0.738 | 0.157 | 78500.0 | 4271872 | 0.004363 |
| late_confidence_ramped_compact_induction | 0.738 | 0.157 | 78500.0 | 4271872 | 0.004363 |
| density_capped_compact_induction | 0.725 | 0.144 | 51520.0 | 3164416 | 0.006083 |
| counterfactual_expansion | 0.810 | 0.230 | 515200.0 | 26343936 | 0.000970 |

## Material Count 144

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.724 | 0.134 | 58016.0 | 3563008 | 0.004462 |
| compact_train_size_gated_induction | 0.786 | 0.197 | 177610.0 | 8789504 | 0.002136 |
| support_ramped_compact_induction | 0.761 | 0.171 | 122122.0 | 6386560 | 0.002705 |
| late_confidence_ramped_compact_induction | 0.761 | 0.171 | 122122.0 | 6386560 | 0.002705 |
| density_capped_compact_induction | 0.724 | 0.134 | 58016.0 | 3563008 | 0.004462 |
| counterfactual_expansion | 0.858 | 0.268 | 580160.0 | 29659648 | 0.000892 |

## Material Count 160

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.768 | 0.177 | 64512.0 | 3964160 | 0.004765 |
| compact_train_size_gated_induction | 0.787 | 0.197 | 200112.0 | 9768960 | 0.001708 |
| support_ramped_compact_induction | 0.750 | 0.159 | 142632.0 | 7418240 | 0.001940 |
| late_confidence_ramped_compact_induction | 0.750 | 0.159 | 142632.0 | 7418240 | 0.001940 |
| density_capped_compact_induction | 0.768 | 0.177 | 64512.0 | 3964160 | 0.004765 |
| counterfactual_expansion | 0.850 | 0.259 | 645120.0 | 33000320 | 0.000698 |
