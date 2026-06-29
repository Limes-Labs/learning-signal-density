# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T11:43:27Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_selector_transfer`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 16 | 0.000 | 0.000000 |
| self_ranked_induction | 48 | 64 | 0.140 | 0.004805 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.143 | 0.005000 |
| agreement_gated_self_ranked_induction | 64 | 64 | 0.081 | 0.002788 |
| validation_ranked_induction | 48 | 64 | 0.135 | 0.004607 |
| mdl_rule_expansion | 48 | 48 | 0.066 | 0.003011 |
| validation_abstaining_proxy_selector | 48 | 64 | 0.114 | 0.001532 |
| validation_linear_proxy_selector | 48 | 64 | 0.114 | 0.001532 |
| validation_portfolio_selector | 48 | 64 | 0.127 | 0.000752 |
| counterfactual_expansion | 48 | 64 | 0.236 | 0.003989 |

Comparison target: `results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.621 | 0.000 | 6496.0 | 398848 | 0.000000 |
| self_ranked_induction | 0.516 | -0.011 | 31487.6 | 1569408 | -0.005722 |
| sample_aware_self_ranked_induction | 0.547 | 0.021 | 22511.6 | 1124864 | 0.016062 |
| agreement_gated_self_ranked_induction | 0.558 | -0.063 | 21950.0 | 1124480 | -0.049668 |
| validation_ranked_induction | 0.516 | -0.011 | 31620.6 | 1569408 | -0.005698 |
| mdl_rule_expansion | 0.600 | -0.021 | 30103.4 | 1473715 | -0.013747 |
| validation_abstaining_proxy_selector | 0.621 | 0.000 | 44801.6 | 763878 | 0.000000 |
| validation_linear_proxy_selector | 0.547 | 0.021 | 51103.2 | 1053952 | 0.006994 |
| validation_portfolio_selector | 0.621 | 0.000 | 145537.2 | 7260723 | 0.000000 |
| counterfactual_expansion | 0.558 | -0.063 | 64960.0 | 3325568 | -0.016763 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.628 | 0.000 | 9632.0 | 592384 | 0.000000 |
| self_ranked_induction | 0.538 | -0.090 | 47122.0 | 2328320 | -0.022194 |
| sample_aware_self_ranked_induction | 0.531 | -0.097 | 42634.0 | 2106368 | -0.026400 |
| agreement_gated_self_ranked_induction | 0.559 | -0.069 | 42014.8 | 2106624 | -0.019096 |
| validation_ranked_induction | 0.538 | -0.090 | 47325.0 | 2328320 | -0.022099 |
| mdl_rule_expansion | 0.538 | -0.090 | 50848.0 | 2448512 | -0.021308 |
| validation_abstaining_proxy_selector | 0.593 | -0.034 | 89688.6 | 1892378 | -0.004147 |
| validation_linear_proxy_selector | 0.559 | -0.069 | 96215.6 | 2172851 | -0.008438 |
| validation_portfolio_selector | 0.593 | -0.034 | 241663.8 | 11910528 | -0.001666 |
| counterfactual_expansion | 0.648 | 0.021 | 96320.0 | 4933504 | 0.002498 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.595 | -0.021 | 12992.0 | 798976 | -0.013969 |
| self_ranked_induction | 0.547 | -0.068 | 63584.8 | 3140096 | -0.009275 |
| sample_aware_self_ranked_induction | 0.442 | -0.174 | 57668.8 | 2847488 | -0.025984 |
| agreement_gated_self_ranked_induction | 0.484 | -0.132 | 57352.0 | 2846720 | -0.019837 |
| validation_ranked_induction | 0.547 | -0.068 | 63850.8 | 3140096 | -0.009236 |
| mdl_rule_expansion | 0.532 | -0.084 | 81909.6 | 3740058 | -0.008959 |
| validation_abstaining_proxy_selector | 0.532 | -0.084 | 120158.4 | 2144717 | -0.005236 |
| validation_linear_proxy_selector | 0.532 | -0.084 | 144641.8 | 3232614 | -0.005236 |
| validation_portfolio_selector | 0.547 | -0.068 | 340094.0 | 16513434 | -0.001742 |
| counterfactual_expansion | 0.632 | 0.016 | 129920.0 | 6651136 | 0.001048 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | -0.014 | 19264.0 | 1184640 | -0.004163 |
| self_ranked_induction | 0.652 | 0.059 | 93821.6 | 4655104 | 0.003632 |
| sample_aware_self_ranked_induction | 0.652 | 0.059 | 93821.6 | 4655104 | 0.003632 |
| agreement_gated_self_ranked_induction | 0.610 | 0.017 | 94100.0 | 4653952 | 0.001057 |
| validation_ranked_induction | 0.655 | 0.062 | 94227.6 | 4655104 | 0.003829 |
| mdl_rule_expansion | 0.659 | 0.066 | 128309.6 | 4835840 | 0.003011 |
| validation_abstaining_proxy_selector | 0.652 | 0.059 | 237715.8 | 4776422 | 0.001462 |
| validation_linear_proxy_selector | 0.652 | 0.059 | 237715.8 | 4776422 | 0.001462 |
| validation_portfolio_selector | 0.652 | 0.059 | 527720.4 | 24639744 | 0.000648 |
| counterfactual_expansion | 0.779 | 0.186 | 192640.0 | 9863040 | 0.005620 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.610 | -0.000 | 25760.0 | 1583360 | -0.000000 |
| self_ranked_induction | 0.751 | 0.140 | 126917.2 | 6222464 | 0.004805 |
| sample_aware_self_ranked_induction | 0.753 | 0.143 | 124217.2 | 6222080 | 0.005000 |
| agreement_gated_self_ranked_induction | 0.691 | 0.081 | 125318.8 | 6221184 | 0.002788 |
| validation_ranked_induction | 0.745 | 0.135 | 127456.2 | 6222336 | 0.004607 |
| mdl_rule_expansion | 0.668 | 0.057 | 189894.2 | 7150720 | 0.001249 |
| validation_abstaining_proxy_selector | 0.725 | 0.114 | 331253.8 | 6190618 | 0.001532 |
| validation_linear_proxy_selector | 0.725 | 0.114 | 331253.8 | 6190618 | 0.001532 |
| validation_portfolio_selector | 0.738 | 0.127 | 725107.6 | 33622144 | 0.000752 |
| counterfactual_expansion | 0.847 | 0.236 | 257600.0 | 13183360 | 0.003989 |
