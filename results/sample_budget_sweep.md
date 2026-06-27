# Learning Signal Density Sample Budget Sweep

Generated: `2026-06-27T16:37:04Z`

This sweep reruns the pilot across multiple external sample budgets.
It is still synthetic and non-neural; it is meant to test whether a mechanism is stable across data budgets.

Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain |
| --- | ---: | ---: | ---: |
| raw_text | not reached | 64 | 0.016 |
| selected_text | not reached | 64 | 0.010 |
| qa_expansion | not reached | 64 | 0.021 |
| induced_rule_expansion | 48 | 64 | 0.096 |
| validation_gated_induction | 48 | 64 | 0.125 |
| direct_validation_gated_induction | 48 | 64 | 0.119 |
| counterfactual_expansion | 24 | 64 | 0.127 |
| prioritized_replay | 64 | 64 | 0.068 |
| selected_counterfactual_replay | 16 | 64 | 0.182 |

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.463 | -0.158 | 2030.0 | -1.341046 |
| selected_text | 0.568 | -0.053 | 2030.0 | -0.412881 |
| qa_expansion | 0.526 | -0.095 | 6206.0 | -0.263196 |
| induced_rule_expansion | 0.463 | -0.063 | 11968.8 | -0.089166 |
| validation_gated_induction | 0.547 | 0.021 | 112624.8 | 0.003139 |
| direct_validation_gated_induction | 0.537 | 0.011 | 24682.0 | 0.006302 |
| counterfactual_expansion | 0.505 | -0.116 | 22214.0 | -0.089870 |
| prioritized_replay | 0.442 | -0.179 | 12887.6 | -0.232943 |
| selected_counterfactual_replay | 0.653 | 0.032 | 18177.2 | 0.033079 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.497 | -0.131 | 3010.0 | -0.506198 |
| selected_text | 0.469 | -0.159 | 3052.0 | -0.605104 |
| qa_expansion | 0.517 | -0.110 | 9202.0 | -0.139435 |
| induced_rule_expansion | 0.469 | -0.159 | 19394.4 | -0.095186 |
| validation_gated_induction | 0.510 | -0.083 | 173330.4 | -0.005651 |
| direct_validation_gated_induction | 0.531 | -0.062 | 36748.4 | -0.019014 |
| counterfactual_expansion | 0.676 | 0.048 | 32938.0 | 0.017043 |
| prioritized_replay | 0.593 | -0.034 | 19282.0 | -0.020339 |
| selected_counterfactual_replay | 0.703 | 0.076 | 27412.0 | 0.032500 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.458 | -0.179 | 4060.0 | -0.379963 |
| selected_text | 0.532 | -0.105 | 4011.0 | -0.226676 |
| qa_expansion | 0.474 | -0.163 | 12412.0 | -0.113320 |
| induced_rule_expansion | 0.553 | -0.084 | 28200.0 | -0.026563 |
| validation_gated_induction | 0.532 | -0.105 | 235416.0 | -0.003664 |
| direct_validation_gated_induction | 0.563 | -0.074 | 51092.0 | -0.012433 |
| counterfactual_expansion | 0.684 | 0.047 | 44428.0 | 0.009191 |
| prioritized_replay | 0.495 | -0.142 | 25573.6 | -0.047963 |
| selected_counterfactual_replay | 0.721 | 0.084 | 35818.2 | 0.020087 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.607 | -0.003 | 6020.0 | -0.003330 |
| selected_text | 0.603 | -0.007 | 6013.0 | -0.008087 |
| qa_expansion | 0.603 | -0.007 | 18404.0 | -0.002179 |
| induced_rule_expansion | 0.655 | 0.045 | 38731.2 | 0.006787 |
| validation_gated_induction | 0.683 | 0.072 | 389788.8 | 0.001070 |
| direct_validation_gated_induction | 0.672 | 0.062 | 75916.0 | 0.004754 |
| counterfactual_expansion | 0.728 | 0.117 | 65876.0 | 0.010347 |
| prioritized_replay | 0.610 | 0.000 | 38189.6 | -0.000047 |
| selected_counterfactual_replay | 0.666 | 0.055 | 53828.2 | 0.006045 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | 0.631 | 0.016 | 8050.0 | 0.008417 |
| selected_text | 0.626 | 0.010 | 8057.0 | 0.004982 |
| qa_expansion | 0.636 | 0.021 | 24610.0 | 0.003671 |
| induced_rule_expansion | 0.712 | 0.096 | 60664.8 | 0.006901 |
| validation_gated_induction | 0.740 | 0.125 | 534424.8 | 0.001009 |
| direct_validation_gated_induction | 0.735 | 0.119 | 101462.0 | 0.005120 |
| counterfactual_expansion | 0.743 | 0.127 | 88090.0 | 0.006282 |
| prioritized_replay | 0.683 | 0.068 | 51134.8 | 0.005701 |
| selected_counterfactual_replay | 0.797 | 0.182 | 72158.6 | 0.010911 |
