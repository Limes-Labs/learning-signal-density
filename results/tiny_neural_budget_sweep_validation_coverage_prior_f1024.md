# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T18:20:32Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_validation_coverage_prior`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 64 | 0.010 | 0.001754 |
| sample_aware_self_ranked_induction | 16 | 64 | 0.174 | 0.006089 |
| train_size_gated_sample_aware_induction | 48 | 64 | 0.174 | 0.006089 |
| validation_coverage_proxy_selector | 48 | 64 | 0.174 | 0.003171 |
| validation_coverage_prior_selector | 48 | 64 | 0.174 | 0.005001 |
| validation_abstaining_proxy_selector | 48 | 64 | 0.135 | 0.001789 |
| validation_portfolio_selector | 48 | 64 | 0.166 | 0.000981 |
| counterfactual_expansion | 24 | 64 | 0.231 | 0.003902 |

Comparison target: `results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.568 | -0.021 | 6496.0 | 399232 | -0.055877 |
| sample_aware_self_ranked_induction | 0.547 | 0.053 | 22554.8 | 1125632 | 0.040335 |
| train_size_gated_sample_aware_induction | 0.568 | -0.021 | 6496.0 | 399232 | -0.055877 |
| validation_coverage_proxy_selector | 0.568 | -0.021 | 20400.0 | 399232 | -0.018071 |
| validation_coverage_prior_selector | 0.568 | -0.021 | 6496.0 | 399232 | -0.055877 |
| validation_abstaining_proxy_selector | 0.537 | -0.053 | 50942.8 | 1056998 | -0.020743 |
| validation_portfolio_selector | 0.516 | 0.021 | 143081.6 | 7135130 | 0.002559 |
| counterfactual_expansion | 0.495 | -0.095 | 64960.0 | 3325952 | -0.025145 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.621 | 0.000 | 9632.0 | 592768 | 0.000000 |
| sample_aware_self_ranked_induction | 0.559 | -0.062 | 42677.2 | 2107008 | -0.016960 |
| train_size_gated_sample_aware_induction | 0.621 | 0.000 | 9632.0 | 592768 | 0.000000 |
| validation_coverage_proxy_selector | 0.559 | -0.062 | 60625.4 | 1768883 | -0.009929 |
| validation_coverage_prior_selector | 0.621 | 0.000 | 9632.0 | 592768 | 0.000000 |
| validation_abstaining_proxy_selector | 0.600 | -0.021 | 68697.2 | 939981 | -0.002375 |
| validation_portfolio_selector | 0.634 | 0.014 | 238949.6 | 11775462 | 0.000683 |
| counterfactual_expansion | 0.662 | 0.041 | 96320.0 | 4933888 | 0.004996 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.500 | -0.095 | 12992.0 | 798848 | -0.062862 |
| sample_aware_self_ranked_induction | 0.468 | -0.126 | 57625.6 | 2847104 | -0.018902 |
| train_size_gated_sample_aware_induction | 0.500 | -0.095 | 12992.0 | 798848 | -0.062862 |
| validation_coverage_proxy_selector | 0.500 | -0.095 | 52592.2 | 798848 | -0.015535 |
| validation_coverage_prior_selector | 0.500 | -0.095 | 26289.2 | 798848 | -0.031320 |
| validation_abstaining_proxy_selector | 0.537 | -0.058 | 127610.6 | 2494822 | -0.003603 |
| validation_portfolio_selector | 0.468 | -0.126 | 345264.2 | 16755712 | -0.003145 |
| counterfactual_expansion | 0.674 | 0.079 | 129920.0 | 6651008 | 0.005238 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.572 | -0.021 | 19264.0 | 1184640 | -0.006244 |
| sample_aware_self_ranked_induction | 0.683 | 0.090 | 93860.0 | 4656128 | 0.005552 |
| train_size_gated_sample_aware_induction | 0.683 | 0.090 | 93860.0 | 4656128 | 0.005552 |
| validation_coverage_proxy_selector | 0.686 | 0.093 | 172306.6 | 4656000 | 0.003147 |
| validation_coverage_prior_selector | 0.686 | 0.093 | 113693.6 | 4656000 | 0.004756 |
| validation_abstaining_proxy_selector | 0.655 | 0.062 | 250572.6 | 4855194 | 0.001427 |
| validation_portfolio_selector | 0.652 | 0.059 | 530708.6 | 24813466 | 0.000645 |
| counterfactual_expansion | 0.797 | 0.203 | 192640.0 | 9863040 | 0.006140 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.610 | 0.010 | 25760.0 | 1583616 | 0.001754 |
| sample_aware_self_ranked_induction | 0.774 | 0.174 | 124255.6 | 6222464 | 0.006089 |
| train_size_gated_sample_aware_induction | 0.774 | 0.174 | 124255.6 | 6222464 | 0.006089 |
| validation_coverage_proxy_selector | 0.774 | 0.174 | 238320.6 | 6222592 | 0.003171 |
| validation_coverage_prior_selector | 0.774 | 0.174 | 151251.2 | 6222592 | 0.005001 |
| validation_abstaining_proxy_selector | 0.735 | 0.135 | 334591.8 | 6190746 | 0.001789 |
| validation_portfolio_selector | 0.766 | 0.166 | 728023.6 | 33666560 | 0.000981 |
| counterfactual_expansion | 0.831 | 0.231 | 257600.0 | 13183616 | 0.003902 |
