# Learning Signal Density Sample Budget Sweep

Generated: `2026-06-27T18:05:38Z`

This sweep reruns the pilot across multiple external sample budgets.
It is still synthetic and non-neural; it is meant to test whether a mechanism is stable across data budgets.

This is a fresh-seed confirmation sweep.
Confirmation target: `results/sample_budget_sweep.json`

Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain |
| --- | ---: | ---: | ---: |
| raw_text | 48 | 48 | 0.045 |
| selected_text | 48 | 48 | 0.079 |
| qa_expansion | 16 | 64 | 0.070 |
| induced_rule_expansion | 48 | 48 | 0.097 |
| validation_gated_induction | 48 | 64 | 0.151 |
| direct_validation_gated_induction | 48 | 64 | 0.158 |
| validation_ranked_induction | 48 | 64 | 0.156 |
| train_calibrated_ranked_induction | 48 | 64 | 0.143 |
| self_ranked_induction | 48 | 64 | 0.122 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.135 |
| diverse_self_ranked_induction | 48 | 48 | 0.114 |
| mdl_rule_expansion | 48 | 48 | 0.059 |
| counterfactual_expansion | 24 | 64 | 0.242 |
| prioritized_replay | 48 | 48 | 0.066 |
| selected_counterfactual_replay | 24 | 48 | 0.217 |

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.495 | -0.084 | 2030.0 | -0.715224 |
| selected_text | 0.526 | -0.053 | 2093.0 | -0.424965 |
| qa_expansion | 0.632 | 0.053 | 6206.0 | 0.146220 |
| induced_rule_expansion | 0.526 | -0.053 | 11623.2 | -0.075320 |
| validation_gated_induction | 0.558 | -0.021 | 111588.0 | -0.003105 |
| direct_validation_gated_induction | 0.547 | -0.032 | 24941.2 | -0.027036 |
| validation_ranked_induction | 0.432 | -0.147 | 11756.2 | -0.215515 |
| train_calibrated_ranked_induction | 0.432 | -0.147 | 12029.2 | -0.210637 |
| self_ranked_induction | 0.432 | -0.147 | 11623.2 | -0.217974 |
| sample_aware_self_ranked_induction | 0.411 | -0.168 | 8455.2 | -0.341757 |
| diverse_self_ranked_induction | 0.484 | -0.095 | 11623.2 | -0.140472 |
| mdl_rule_expansion | 0.516 | -0.063 | 11312.8 | -0.070460 |
| counterfactual_expansion | 0.526 | -0.053 | 22214.0 | -0.040850 |
| prioritized_replay | 0.537 | -0.042 | 13146.8 | -0.053207 |
| selected_counterfactual_replay | 0.579 | 0.000 | 18866.6 | 0.002003 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.566 | -0.090 | 3010.0 | -0.346346 |
| selected_text | 0.586 | -0.069 | 3003.0 | -0.266047 |
| qa_expansion | 0.621 | -0.034 | 9202.0 | -0.043573 |
| induced_rule_expansion | 0.607 | -0.048 | 19912.8 | -0.026660 |
| validation_gated_induction | 0.566 | -0.090 | 170853.6 | -0.006130 |
| direct_validation_gated_induction | 0.593 | -0.062 | 37958.0 | -0.019014 |
| validation_ranked_induction | 0.531 | -0.124 | 17883.8 | -0.080164 |
| train_calibrated_ranked_induction | 0.531 | -0.124 | 18282.8 | -0.078426 |
| self_ranked_induction | 0.531 | -0.124 | 17680.8 | -0.081077 |
| sample_aware_self_ranked_induction | 0.641 | -0.014 | 16096.8 | -0.010338 |
| diverse_self_ranked_induction | 0.607 | -0.048 | 17680.8 | -0.031434 |
| mdl_rule_expansion | 0.641 | -0.014 | 17346.2 | -0.009597 |
| counterfactual_expansion | 0.703 | 0.048 | 32938.0 | 0.017043 |
| prioritized_replay | 0.524 | -0.131 | 19080.4 | -0.079648 |
| selected_counterfactual_replay | 0.738 | 0.083 | 26875.8 | 0.034702 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.526 | -0.105 | 4060.0 | -0.223508 |
| selected_text | 0.511 | -0.121 | 4088.0 | -0.257912 |
| qa_expansion | 0.547 | -0.084 | 12412.0 | -0.058488 |
| induced_rule_expansion | 0.521 | -0.111 | 27249.6 | -0.034947 |
| validation_gated_induction | 0.505 | -0.126 | 239390.4 | -0.004542 |
| direct_validation_gated_induction | 0.553 | -0.079 | 51092.0 | -0.013321 |
| validation_ranked_induction | 0.532 | -0.100 | 24179.6 | -0.035654 |
| train_calibrated_ranked_induction | 0.532 | -0.100 | 24725.6 | -0.034867 |
| self_ranked_induction | 0.532 | -0.100 | 23913.6 | -0.036051 |
| sample_aware_self_ranked_induction | 0.542 | -0.089 | 21825.6 | -0.035317 |
| diverse_self_ranked_induction | 0.547 | -0.084 | 23913.6 | -0.030311 |
| mdl_rule_expansion | 0.542 | -0.089 | 33875.4 | -0.023723 |
| counterfactual_expansion | 0.700 | 0.068 | 44428.0 | 0.013276 |
| prioritized_replay | 0.489 | -0.142 | 25890.4 | -0.047550 |
| selected_counterfactual_replay | 0.658 | 0.026 | 36660.8 | 0.005769 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.614 | 0.045 | 6020.0 | 0.043293 |
| selected_text | 0.648 | 0.079 | 6090.0 | 0.075470 |
| qa_expansion | 0.621 | 0.052 | 18404.0 | 0.016340 |
| induced_rule_expansion | 0.666 | 0.097 | 39105.6 | 0.014322 |
| validation_gated_induction | 0.672 | 0.103 | 387657.6 | 0.001530 |
| direct_validation_gated_induction | 0.710 | 0.141 | 75916.0 | 0.010827 |
| validation_ranked_induction | 0.676 | 0.107 | 35647.6 | 0.017412 |
| train_calibrated_ranked_induction | 0.676 | 0.107 | 36445.6 | 0.017031 |
| self_ranked_induction | 0.686 | 0.117 | 35241.6 | 0.019319 |
| sample_aware_self_ranked_induction | 0.686 | 0.117 | 35241.6 | 0.019319 |
| diverse_self_ranked_induction | 0.683 | 0.114 | 35241.6 | 0.018774 |
| mdl_rule_expansion | 0.628 | 0.059 | 68770.8 | 0.005059 |
| counterfactual_expansion | 0.772 | 0.203 | 65876.0 | 0.017956 |
| prioritized_replay | 0.634 | 0.066 | 38506.4 | 0.009876 |
| selected_counterfactual_replay | 0.786 | 0.217 | 54670.8 | 0.023119 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.618 | 0.026 | 8050.0 | 0.014029 |
| selected_text | 0.595 | 0.003 | 8099.0 | 0.000950 |
| qa_expansion | 0.662 | 0.070 | 24610.0 | 0.012390 |
| induced_rule_expansion | 0.688 | 0.096 | 60780.0 | 0.006888 |
| validation_gated_induction | 0.743 | 0.151 | 523106.4 | 0.001255 |
| direct_validation_gated_induction | 0.751 | 0.158 | 100655.6 | 0.006845 |
| validation_ranked_induction | 0.748 | 0.156 | 49079.0 | 0.013804 |
| train_calibrated_ranked_induction | 0.735 | 0.143 | 50150.0 | 0.012386 |
| self_ranked_induction | 0.714 | 0.122 | 48540.0 | 0.010932 |
| sample_aware_self_ranked_induction | 0.727 | 0.135 | 45825.6 | 0.012809 |
| diverse_self_ranked_induction | 0.688 | 0.096 | 48540.0 | 0.008608 |
| mdl_rule_expansion | 0.647 | 0.055 | 104092.4 | 0.002339 |
| counterfactual_expansion | 0.834 | 0.242 | 88090.0 | 0.011923 |
| prioritized_replay | 0.649 | 0.057 | 51307.6 | 0.004786 |
| selected_counterfactual_replay | 0.803 | 0.210 | 72618.2 | 0.012550 |
