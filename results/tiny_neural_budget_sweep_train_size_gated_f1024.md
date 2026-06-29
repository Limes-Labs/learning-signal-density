# Learning Signal Density Tiny Neural Budget Sweep

Generated: `2026-06-29T12:02:55Z`

This sweep reruns the deterministic CPU tiny-MLP profile across external sample budgets.
It is still synthetic and not a language-model result.

Fresh-seed confirmation sweep: `true`
Confirmation target: `results/tiny_neural_budget_sweep_selector_transfer_f1024.json`

Backend: `tiny_mlp`
Profile label: `epochs=16_hidden=8_features=1024_train_size_gated`
Hidden units: `8`
Feature dimension: `1024`
Target signed gain over majority: `0.03`

| Condition | First budget reaching target | Best budget | Best signed gain | Best signed LSD/1M |
| --- | ---: | ---: | ---: | ---: |
| raw_text | not reached | 64 | 0.008 | 0.001315 |
| self_ranked_induction | 48 | 64 | 0.140 | 0.004803 |
| sample_aware_self_ranked_induction | 48 | 64 | 0.145 | 0.005090 |
| train_size_gated_sample_aware_induction | 48 | 64 | 0.145 | 0.005090 |
| validation_ranked_induction | 48 | 64 | 0.140 | 0.004783 |
| mdl_rule_expansion | 48 | 48 | 0.034 | 0.001494 |
| validation_abstaining_proxy_selector | 48 | 64 | 0.104 | 0.001437 |
| validation_linear_proxy_selector | 48 | 64 | 0.104 | 0.001437 |
| validation_portfolio_selector | 48 | 64 | 0.145 | 0.000854 |
| counterfactual_expansion | 24 | 64 | 0.200 | 0.003376 |

Comparison target: `results/tiny_neural_budget_sweep_selector_transfer_f1024.json`

## Material Count 16

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.547 | -0.032 | 6496.0 | 399744 | -0.083815 |
| self_ranked_induction | 0.432 | -0.053 | 31516.4 | 1570688 | -0.028687 |
| sample_aware_self_ranked_induction | 0.463 | -0.021 | 22540.4 | 1126656 | -0.016193 |
| train_size_gated_sample_aware_induction | 0.547 | -0.032 | 6496.0 | 399744 | -0.083815 |
| validation_ranked_induction | 0.432 | -0.053 | 31649.4 | 1570688 | -0.028567 |
| mdl_rule_expansion | 0.547 | -0.032 | 27994.6 | 1366682 | -0.021392 |
| validation_abstaining_proxy_selector | 0.432 | -0.053 | 47944.8 | 925056 | -0.016918 |
| validation_linear_proxy_selector | 0.421 | -0.063 | 51047.6 | 1070541 | -0.020465 |
| validation_portfolio_selector | 0.474 | -0.011 | 143514.8 | 7160858 | -0.001279 |
| counterfactual_expansion | 0.432 | -0.147 | 64960.0 | 3326464 | -0.039114 |

## Material Count 24

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.614 | -0.007 | 9632.0 | 593152 | -0.008326 |
| self_ranked_induction | 0.552 | -0.069 | 47078.8 | 2329728 | -0.017000 |
| sample_aware_self_ranked_induction | 0.497 | -0.124 | 42590.8 | 2107776 | -0.033893 |
| train_size_gated_sample_aware_induction | 0.614 | -0.007 | 9632.0 | 593152 | -0.008326 |
| validation_ranked_induction | 0.552 | -0.069 | 47281.8 | 2329728 | -0.016927 |
| mdl_rule_expansion | 0.545 | -0.076 | 45358.6 | 2179558 | -0.019242 |
| validation_abstaining_proxy_selector | 0.607 | -0.014 | 79088.4 | 1453158 | -0.001858 |
| validation_linear_proxy_selector | 0.579 | -0.041 | 92061.2 | 2059110 | -0.005338 |
| validation_portfolio_selector | 0.566 | -0.055 | 236028.0 | 11648230 | -0.002825 |
| counterfactual_expansion | 0.669 | 0.048 | 96320.0 | 4934272 | 0.005828 |

## Material Count 32

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.568 | -0.074 | 12992.0 | 799360 | -0.048892 |
| self_ranked_induction | 0.537 | -0.105 | 63570.4 | 3139200 | -0.014293 |
| sample_aware_self_ranked_induction | 0.537 | -0.105 | 57654.4 | 2846848 | -0.015772 |
| train_size_gated_sample_aware_induction | 0.568 | -0.074 | 12992.0 | 799360 | -0.048892 |
| validation_ranked_induction | 0.537 | -0.105 | 63836.4 | 3139200 | -0.014234 |
| mdl_rule_expansion | 0.563 | -0.079 | 82555.0 | 3786982 | -0.008508 |
| validation_abstaining_proxy_selector | 0.584 | -0.058 | 107144.2 | 1619174 | -0.004031 |
| validation_linear_proxy_selector | 0.558 | -0.084 | 137554.6 | 3023693 | -0.005282 |
| validation_portfolio_selector | 0.605 | -0.037 | 340593.0 | 16559462 | -0.000925 |
| counterfactual_expansion | 0.737 | 0.095 | 129920.0 | 6651520 | 0.006286 |

## Material Count 48

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.569 | -0.021 | 19264.0 | 1185280 | -0.006244 |
| self_ranked_induction | 0.693 | 0.103 | 93869.6 | 4656512 | 0.006407 |
| sample_aware_self_ranked_induction | 0.693 | 0.103 | 93869.6 | 4656512 | 0.006407 |
| train_size_gated_sample_aware_induction | 0.693 | 0.103 | 93869.6 | 4656512 | 0.006407 |
| validation_ranked_induction | 0.676 | 0.086 | 94275.6 | 4656384 | 0.005316 |
| mdl_rule_expansion | 0.624 | 0.034 | 131143.8 | 5004493 | 0.001494 |
| validation_abstaining_proxy_selector | 0.648 | 0.059 | 250406.6 | 4853709 | 0.001350 |
| validation_linear_proxy_selector | 0.648 | 0.059 | 250406.6 | 4853709 | 0.001350 |
| validation_portfolio_selector | 0.628 | 0.038 | 530638.6 | 24815821 | 0.000416 |
| counterfactual_expansion | 0.745 | 0.155 | 192640.0 | 9863680 | 0.004683 |

## Material Count 64

| Condition | Heldout acc. | Signed gain | Compute units | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.621 | 0.008 | 25760.0 | 1584128 | 0.001315 |
| self_ranked_induction | 0.753 | 0.140 | 126970.0 | 6223616 | 0.004803 |
| sample_aware_self_ranked_induction | 0.758 | 0.145 | 124255.6 | 6223360 | 0.005090 |
| train_size_gated_sample_aware_induction | 0.758 | 0.145 | 124255.6 | 6223360 | 0.005090 |
| validation_ranked_induction | 0.753 | 0.140 | 127509.0 | 6223488 | 0.004783 |
| mdl_rule_expansion | 0.636 | 0.023 | 193817.0 | 7178803 | 0.000407 |
| validation_abstaining_proxy_selector | 0.717 | 0.104 | 326014.0 | 6198963 | 0.001437 |
| validation_linear_proxy_selector | 0.717 | 0.104 | 326014.0 | 6198963 | 0.001437 |
| validation_portfolio_selector | 0.758 | 0.145 | 729287.2 | 33658291 | 0.000854 |
| counterfactual_expansion | 0.813 | 0.200 | 257600.0 | 13184128 | 0.003376 |
