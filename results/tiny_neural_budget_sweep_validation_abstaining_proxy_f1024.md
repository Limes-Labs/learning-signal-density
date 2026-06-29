# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T11:28:26Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_validation_abstaining_proxy`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 48 | 0.003 | 0.001041 |
| self_ranked_induction | 48 | 64 | 0.153 | 0.005247 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.151 | 0.005270 |
| agreement_gated_self_ranked_induction | 48 | 48 | 0.062 | 0.003836 |
| validation_ranked_induction | 48 | 64 | 0.148 | 0.005047 |
| mdl_rule_expansion | 48 | 48 | 0.086 | 0.003776 |
| validation_abstaining_proxy_selector | 48 | 64 | 0.153 | 0.002091 |
| validation_linear_proxy_selector | 48 | 64 | 0.153 | 0.002091 |
| validation_portfolio_selector | 48 | 64 | 0.109 | 0.000627 |
| counterfactual_expansion | 24 | 64 | 0.239 | 0.004033 |

Comparison target: `results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.579 | 0.000 | 6496.0 | 399616 | 0.000000 |
| self_ranked_induction | 0.505 | -0.074 | 31401.2 | 1570432 | -0.040421 |
| sample_aware_self_ranked_induction | 0.495 | -0.084 | 22425.2 | 1126144 | -0.064766 |
| agreement_gated_self_ranked_induction | 0.453 | -0.126 | 21940.4 | 1126272 | -0.099387 |
| validation_ranked_induction | 0.505 | -0.074 | 31534.2 | 1570432 | -0.040251 |
| mdl_rule_expansion | 0.589 | 0.011 | 29295.6 | 1433165 | 0.006546 |
| validation_abstaining_proxy_selector | 0.579 | 0.000 | 36481.4 | 399616 | 0.000000 |
| validation_linear_proxy_selector | 0.474 | -0.105 | 50863.0 | 1069645 | -0.034019 |
| validation_portfolio_selector | 0.589 | 0.011 | 144460.6 | 7226061 | 0.001284 |
| counterfactual_expansion | 0.516 | -0.063 | 64960.0 | 3326336 | -0.016763 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.655 | 0.000 | 9632.0 | 592896 | 0.000000 |
| self_ranked_induction | 0.552 | -0.103 | 47006.8 | 2329856 | -0.025595 |
| sample_aware_self_ranked_induction | 0.614 | -0.041 | 42518.8 | 2107648 | -0.011346 |
| agreement_gated_self_ranked_induction | 0.607 | -0.048 | 41998.0 | 2107392 | -0.013414 |
| validation_ranked_induction | 0.552 | -0.103 | 47209.8 | 2329856 | -0.025485 |
| mdl_rule_expansion | 0.634 | -0.021 | 43979.4 | 2122803 | -0.005054 |
| validation_abstaining_proxy_selector | 0.662 | 0.007 | 73876.8 | 1243469 | 0.000819 |
| validation_linear_proxy_selector | 0.614 | -0.041 | 81317.2 | 1590554 | -0.004938 |
| validation_portfolio_selector | 0.607 | -0.048 | 234432.8 | 11590451 | -0.002284 |
| counterfactual_expansion | 0.731 | 0.076 | 96320.0 | 4934016 | 0.009158 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.568 | -0.063 | 12992.0 | 798976 | -0.041908 |
| self_ranked_induction | 0.521 | -0.111 | 63469.6 | 3139072 | -0.015011 |
| sample_aware_self_ranked_induction | 0.532 | -0.100 | 57553.6 | 2846720 | -0.014971 |
| agreement_gated_self_ranked_induction | 0.532 | -0.100 | 57229.6 | 2846080 | -0.015086 |
| validation_ranked_induction | 0.521 | -0.111 | 63735.6 | 3139072 | -0.014949 |
| mdl_rule_expansion | 0.595 | -0.037 | 79635.4 | 3612544 | -0.004033 |
| validation_abstaining_proxy_selector | 0.568 | -0.063 | 138628.4 | 2907750 | -0.003804 |
| validation_linear_proxy_selector | 0.568 | -0.063 | 148698.0 | 3375923 | -0.003804 |
| validation_portfolio_selector | 0.574 | -0.058 | 337351.8 | 16382464 | -0.001496 |
| counterfactual_expansion | 0.668 | 0.037 | 129920.0 | 6651136 | 0.002445 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | 0.003 | 19264.0 | 1185408 | 0.001041 |
| self_ranked_induction | 0.645 | 0.076 | 93893.6 | 4656128 | 0.004695 |
| sample_aware_self_ranked_induction | 0.645 | 0.076 | 93893.6 | 4656128 | 0.004695 |
| agreement_gated_self_ranked_induction | 0.631 | 0.062 | 94112.0 | 4656128 | 0.003836 |
| validation_ranked_induction | 0.655 | 0.086 | 94299.6 | 4656000 | 0.005312 |
| mdl_rule_expansion | 0.655 | 0.086 | 133468.4 | 5119514 | 0.003776 |
| validation_abstaining_proxy_selector | 0.652 | 0.083 | 233459.2 | 4002150 | 0.001981 |
| validation_linear_proxy_selector | 0.672 | 0.103 | 257006.2 | 4842573 | 0.002368 |
| validation_portfolio_selector | 0.645 | 0.076 | 533107.2 | 24929306 | 0.000828 |
| counterfactual_expansion | 0.762 | 0.193 | 192640.0 | 9863808 | 0.005828 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.574 | -0.018 | 25760.0 | 1583616 | -0.003069 |
| self_ranked_induction | 0.745 | 0.153 | 126970.0 | 6224000 | 0.005247 |
| sample_aware_self_ranked_induction | 0.743 | 0.151 | 124255.6 | 6223872 | 0.005270 |
| agreement_gated_self_ranked_induction | 0.647 | 0.055 | 125568.4 | 6221440 | 0.001885 |
| validation_ranked_induction | 0.740 | 0.148 | 127509.0 | 6224000 | 0.005047 |
| mdl_rule_expansion | 0.636 | 0.044 | 198494.4 | 7444096 | 0.000957 |
| validation_abstaining_proxy_selector | 0.745 | 0.153 | 316914.6 | 6224000 | 0.002091 |
| validation_linear_proxy_selector | 0.745 | 0.153 | 316914.6 | 6224000 | 0.002091 |
| validation_portfolio_selector | 0.701 | 0.109 | 734101.4 | 33921024 | 0.000627 |
| counterfactual_expansion | 0.831 | 0.239 | 257600.0 | 13183616 | 0.004033 |
