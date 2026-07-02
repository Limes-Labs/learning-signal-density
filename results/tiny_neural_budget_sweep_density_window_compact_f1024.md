# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T23:45:17Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_density_window_compact`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 64 | 120 | 0.118 | 0.005648 |
| compact_train_size_gated_induction | 64 | 80 | 0.194 | 0.008591 |
| support_ramped_compact_induction | 64 | 80 | 0.194 | 0.008591 |
| density_window_compact_induction | 64 | 80 | 0.194 | 0.008591 |
| density_capped_compact_induction | 64 | 80 | 0.194 | 0.008591 |
| counterfactual_expansion | 64 | 128 | 0.255 | 0.001074 |

Comparison target: `results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.603 | 0.039 | 25760.0 | 1583616 | 0.006576 |
| compact_train_size_gated_induction | 0.719 | 0.156 | 77335.6 | 3903104 | 0.008762 |
| support_ramped_compact_induction | 0.719 | 0.156 | 77335.6 | 3903104 | 0.008762 |
| density_window_compact_induction | 0.719 | 0.156 | 77335.6 | 3903104 | 0.008762 |
| density_capped_compact_induction | 0.719 | 0.156 | 77335.6 | 3903104 | 0.008762 |
| counterfactual_expansion | 0.805 | 0.242 | 257600.0 | 13183616 | 0.004077 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.635 | 0.050 | 32256.0 | 1984640 | 0.005382 |
| compact_train_size_gated_induction | 0.779 | 0.194 | 78508.8 | 4051354 | 0.008591 |
| support_ramped_compact_induction | 0.779 | 0.194 | 78508.8 | 4051354 | 0.008591 |
| density_window_compact_induction | 0.779 | 0.194 | 78508.8 | 4051354 | 0.008591 |
| density_capped_compact_induction | 0.779 | 0.194 | 78508.8 | 4051354 | 0.008591 |
| counterfactual_expansion | 0.838 | 0.252 | 322560.0 | 16515200 | 0.002714 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.654 | 0.064 | 38752.0 | 2380288 | 0.004799 |
| compact_train_size_gated_induction | 0.739 | 0.150 | 100703.6 | 5157043 | 0.004277 |
| support_ramped_compact_induction | 0.739 | 0.150 | 100703.6 | 5157043 | 0.004277 |
| density_window_compact_induction | 0.654 | 0.064 | 38752.0 | 2380288 | 0.004799 |
| density_capped_compact_induction | 0.739 | 0.150 | 100703.6 | 5157043 | 0.004277 |
| counterfactual_expansion | 0.800 | 0.210 | 387520.0 | 19820288 | 0.001569 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.683 | 0.067 | 41888.0 | 2572544 | 0.004289 |
| compact_train_size_gated_induction | 0.766 | 0.150 | 125832.4 | 6344448 | 0.003196 |
| support_ramped_compact_induction | 0.722 | 0.106 | 87101.2 | 4563814 | 0.003237 |
| density_window_compact_induction | 0.683 | 0.067 | 41888.0 | 2572544 | 0.004289 |
| density_capped_compact_induction | 0.683 | 0.067 | 41888.0 | 2572544 | 0.004289 |
| counterfactual_expansion | 0.818 | 0.202 | 418880.0 | 21421184 | 0.001287 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.660 | 0.073 | 45248.0 | 2780160 | 0.004001 |
| compact_train_size_gated_induction | 0.751 | 0.164 | 134476.0 | 6807552 | 0.003022 |
| support_ramped_compact_induction | 0.722 | 0.136 | 78748.0 | 4209792 | 0.004269 |
| density_window_compact_induction | 0.722 | 0.136 | 78748.0 | 4209792 | 0.004269 |
| density_capped_compact_induction | 0.660 | 0.073 | 45248.0 | 2780160 | 0.004001 |
| counterfactual_expansion | 0.818 | 0.231 | 452480.0 | 23145600 | 0.001265 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.693 | 0.118 | 48384.0 | 2975872 | 0.005648 |
| compact_train_size_gated_induction | 0.708 | 0.133 | 145269.6 | 7329408 | 0.002124 |
| support_ramped_compact_induction | 0.736 | 0.161 | 75772.8 | 4111514 | 0.004899 |
| density_window_compact_induction | 0.693 | 0.118 | 48384.0 | 2975872 | 0.005648 |
| density_capped_compact_induction | 0.693 | 0.118 | 48384.0 | 2975872 | 0.005648 |
| counterfactual_expansion | 0.794 | 0.219 | 483840.0 | 24769152 | 0.001050 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.674 | 0.088 | 51520.0 | 3162624 | 0.003726 |
| compact_train_size_gated_induction | 0.722 | 0.136 | 154844.0 | 7793536 | 0.001915 |
| support_ramped_compact_induction | 0.740 | 0.155 | 78068.0 | 4248320 | 0.004290 |
| density_window_compact_induction | 0.674 | 0.088 | 51520.0 | 3162624 | 0.003726 |
| density_capped_compact_induction | 0.674 | 0.088 | 51520.0 | 3162624 | 0.003726 |
| counterfactual_expansion | 0.840 | 0.255 | 515200.0 | 26342144 | 0.001074 |
