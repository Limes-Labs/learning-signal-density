# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-30T22:35:19Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/support_mechanism_audit_f1024.json`

Backend: `tiny_mlp`
Profile label: `f1024_16x8_validation_support_utility`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 80 | 128 | 0.131 | 0.005535 |
| compact_train_size_gated_induction | 64 | 80 | 0.188 | 0.008200 |
| support_ramped_compact_induction | 64 | 80 | 0.188 | 0.008200 |
| density_window_compact_induction | 64 | 80 | 0.188 | 0.008200 |
| support_probe_window_selector | 64 | 80 | 0.188 | 0.008200 |
| validation_support_precision_selector | 64 | 80 | 0.188 | 0.008200 |
| validation_support_precision_gate_selector | 64 | 80 | 0.188 | 0.008200 |
| validation_support_utility_selector | 64 | 80 | 0.188 | 0.008200 |
| train_support_density_selector | 64 | 80 | 0.188 | 0.006966 |
| density_capped_compact_induction | 64 | 80 | 0.188 | 0.008200 |
| counterfactual_expansion | 64 | 80 | 0.269 | 0.002893 |

Comparison target: `results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json`

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.587 | 0.008 | 25760.0 | 1583616 | 0.001315 |
| compact_train_size_gated_induction | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| support_ramped_compact_induction | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| density_window_compact_induction | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| support_probe_window_selector | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| validation_support_precision_selector | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| validation_support_precision_gate_selector | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| validation_support_utility_selector | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| train_support_density_selector | 0.722 | 0.143 | 92166.8 | 3903872 | 0.006739 |
| density_capped_compact_induction | 0.722 | 0.143 | 77335.6 | 3903872 | 0.008032 |
| counterfactual_expansion | 0.771 | 0.192 | 257600.0 | 13183616 | 0.003244 |

## Material Count 80

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.637 | 0.065 | 32256.0 | 1984768 | 0.006952 |
| compact_train_size_gated_induction | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| support_ramped_compact_induction | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| density_window_compact_induction | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| support_probe_window_selector | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| validation_support_precision_selector | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| validation_support_precision_gate_selector | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| validation_support_utility_selector | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| train_support_density_selector | 0.760 | 0.188 | 94060.8 | 4116582 | 0.006966 |
| density_capped_compact_induction | 0.760 | 0.188 | 79891.2 | 4116582 | 0.008200 |
| counterfactual_expansion | 0.842 | 0.269 | 322560.0 | 16515328 | 0.002893 |

## Material Count 96

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.654 | 0.064 | 38752.0 | 2381056 | 0.004799 |
| compact_train_size_gated_induction | 0.722 | 0.132 | 100703.6 | 5159091 | 0.003788 |
| support_ramped_compact_induction | 0.722 | 0.132 | 100703.6 | 5159091 | 0.003788 |
| density_window_compact_induction | 0.654 | 0.064 | 38752.0 | 2381056 | 0.004799 |
| support_probe_window_selector | 0.654 | 0.064 | 38752.0 | 2381056 | 0.004799 |
| validation_support_precision_selector | 0.685 | 0.096 | 77489.4 | 4037197 | 0.004556 |
| validation_support_precision_gate_selector | 0.685 | 0.096 | 77489.4 | 4037197 | 0.004556 |
| validation_support_utility_selector | 0.652 | 0.063 | 57427.8 | 2941619 | 0.004063 |
| train_support_density_selector | 0.722 | 0.132 | 118776.4 | 5159091 | 0.003212 |
| density_capped_compact_induction | 0.722 | 0.132 | 100703.6 | 5159091 | 0.003788 |
| counterfactual_expansion | 0.828 | 0.238 | 387520.0 | 19821056 | 0.001777 |

## Material Count 104

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.664 | 0.054 | 41888.0 | 2572544 | 0.003472 |
| compact_train_size_gated_induction | 0.733 | 0.123 | 125784.4 | 6342016 | 0.002619 |
| support_ramped_compact_induction | 0.707 | 0.098 | 86237.2 | 4522342 | 0.003017 |
| density_window_compact_induction | 0.664 | 0.054 | 41888.0 | 2572544 | 0.003472 |
| support_probe_window_selector | 0.707 | 0.098 | 86237.2 | 4522342 | 0.003017 |
| validation_support_precision_selector | 0.699 | 0.090 | 61969.8 | 3346330 | 0.004015 |
| validation_support_precision_gate_selector | 0.699 | 0.090 | 61969.8 | 3346330 | 0.004015 |
| validation_support_utility_selector | 0.664 | 0.054 | 47824.2 | 2572544 | 0.003137 |
| train_support_density_selector | 0.707 | 0.098 | 105580.4 | 4522342 | 0.002465 |
| density_capped_compact_induction | 0.664 | 0.054 | 41888.0 | 2572544 | 0.003472 |
| counterfactual_expansion | 0.826 | 0.216 | 418880.0 | 21421184 | 0.001379 |

## Material Count 112

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.704 | 0.097 | 45248.0 | 2780160 | 0.005307 |
| compact_train_size_gated_induction | 0.758 | 0.151 | 134476.0 | 6809856 | 0.002775 |
| support_ramped_compact_induction | 0.742 | 0.134 | 78748.0 | 4210944 | 0.004222 |
| density_window_compact_induction | 0.742 | 0.134 | 78748.0 | 4210944 | 0.004222 |
| support_probe_window_selector | 0.742 | 0.134 | 78748.0 | 4210944 | 0.004222 |
| validation_support_precision_selector | 0.742 | 0.134 | 78748.0 | 4210944 | 0.004222 |
| validation_support_precision_gate_selector | 0.730 | 0.122 | 61282.8 | 3352602 | 0.004846 |
| validation_support_utility_selector | 0.704 | 0.097 | 51020.4 | 2780160 | 0.004697 |
| train_support_density_selector | 0.742 | 0.134 | 97412.0 | 4210944 | 0.003413 |
| density_capped_compact_induction | 0.704 | 0.097 | 45248.0 | 2780160 | 0.005307 |
| counterfactual_expansion | 0.827 | 0.219 | 452480.0 | 23145600 | 0.001200 |

## Material Count 120

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.690 | 0.119 | 48384.0 | 2976384 | 0.005715 |
| compact_train_size_gated_induction | 0.713 | 0.142 | 145240.8 | 7331968 | 0.002258 |
| support_ramped_compact_induction | 0.732 | 0.161 | 75254.4 | 4088269 | 0.004935 |
| density_window_compact_induction | 0.690 | 0.119 | 48384.0 | 2976384 | 0.005715 |
| support_probe_window_selector | 0.690 | 0.119 | 48384.0 | 2976384 | 0.005715 |
| validation_support_precision_selector | 0.724 | 0.153 | 66931.2 | 3652454 | 0.005466 |
| validation_support_precision_gate_selector | 0.724 | 0.153 | 66931.2 | 3652454 | 0.005466 |
| validation_support_utility_selector | 0.690 | 0.119 | 55065.6 | 2976384 | 0.005076 |
| train_support_density_selector | 0.704 | 0.133 | 73759.2 | 3217664 | 0.004003 |
| density_capped_compact_induction | 0.690 | 0.119 | 48384.0 | 2976384 | 0.005715 |
| counterfactual_expansion | 0.812 | 0.242 | 483840.0 | 24769664 | 0.001156 |

## Material Count 128

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.730 | 0.131 | 51520.0 | 3163776 | 0.005535 |
| compact_train_size_gated_induction | 0.760 | 0.161 | 154810.4 | 7796992 | 0.002261 |
| support_ramped_compact_induction | 0.730 | 0.131 | 77463.2 | 4222566 | 0.003672 |
| density_window_compact_induction | 0.730 | 0.131 | 51520.0 | 3163776 | 0.005535 |
| support_probe_window_selector | 0.730 | 0.131 | 51520.0 | 3163776 | 0.005535 |
| validation_support_precision_selector | 0.730 | 0.131 | 55818.0 | 3163776 | 0.005109 |
| validation_support_precision_gate_selector | 0.730 | 0.131 | 55818.0 | 3163776 | 0.005109 |
| validation_support_utility_selector | 0.730 | 0.131 | 55818.0 | 3163776 | 0.005109 |
| train_support_density_selector | 0.730 | 0.131 | 72235.2 | 3163776 | 0.003949 |
| density_capped_compact_induction | 0.730 | 0.131 | 51520.0 | 3163776 | 0.005535 |
| counterfactual_expansion | 0.843 | 0.244 | 515200.0 | 26343296 | 0.001030 |
