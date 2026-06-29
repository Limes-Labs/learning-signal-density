# Learning Signal Density Generated-label Audit

Source artifact: `results/tiny_neural_budget_sweep_selector_transfer_f1024.json`
Source generated: `2026-06-29T11:43:27Z`
Profile label: `epochs=16_hidden=8_features=1024_selector_transfer`

This hidden-rulebook audit is non-deployable. It compares generated synthetic labels with the synthetic world's true rulebook after the source sweep has already been run. The purpose is diagnostic: separate generated-label correctness from the tiny learner's downstream gain.

| Materials | Condition | Synthetic labels | Label precision | Linked gain | Linked LSD/1M |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | raw_text | 0 | n/a | 0.000000 | 0.000000 |
| 16 | self_ranked_induction | 290 | 0.741379 | -0.010526 | -0.005722 |
| 16 | sample_aware_self_ranked_induction | 70 | 0.742857 | 0.021053 | 0.016062 |
| 16 | agreement_gated_self_ranked_induction | 70 | 0.957143 | -0.063158 | -0.049668 |
| 16 | validation_ranked_induction | 290 | 0.741379 | -0.010526 | -0.005698 |
| 16 | mdl_rule_expansion | 243 | 0.617284 | -0.021053 | -0.013747 |
| 16 | counterfactual_expansion | 1160 | 1.000000 | -0.063158 | -0.016763 |
| 24 | raw_text | 0 | n/a | 0.000000 | 0.000000 |
| 24 | self_ranked_induction | 430 | 0.718605 | -0.089655 | -0.022194 |
| 24 | sample_aware_self_ranked_induction | 320 | 0.700000 | -0.096552 | -0.026400 |
| 24 | agreement_gated_self_ranked_induction | 320 | 0.903125 | -0.068966 | -0.019096 |
| 24 | validation_ranked_induction | 430 | 0.718605 | -0.089655 | -0.022099 |
| 24 | mdl_rule_expansion | 490 | 0.608163 | -0.089655 | -0.021308 |
| 24 | counterfactual_expansion | 1720 | 1.000000 | 0.020690 | 0.002498 |
| 32 | raw_text | 0 | n/a | -0.021053 | -0.013969 |
| 32 | self_ranked_induction | 580 | 0.691379 | -0.068421 | -0.009275 |
| 32 | sample_aware_self_ranked_induction | 435 | 0.675862 | -0.173684 | -0.025984 |
| 32 | agreement_gated_self_ranked_induction | 435 | 0.917241 | -0.131579 | -0.019837 |
| 32 | validation_ranked_induction | 580 | 0.691379 | -0.068421 | -0.009236 |
| 32 | mdl_rule_expansion | 879 | 0.680319 | -0.084210 | -0.008959 |
| 32 | counterfactual_expansion | 2320 | 1.000000 | 0.015790 | 0.001048 |
| 48 | raw_text | 0 | n/a | -0.013793 | -0.004163 |
| 48 | self_ranked_induction | 860 | 0.783721 | 0.058621 | 0.003632 |
| 48 | sample_aware_self_ranked_induction | 860 | 0.783721 | 0.058621 | 0.003632 |
| 48 | agreement_gated_self_ranked_induction | 860 | 0.946512 | 0.017241 | 0.001057 |
| 48 | validation_ranked_induction | 860 | 0.779070 | 0.062069 | 0.003829 |
| 48 | mdl_rule_expansion | 950 | 0.832632 | 0.065517 | 0.003011 |
| 48 | counterfactual_expansion | 3440 | 1.000000 | 0.186207 | 0.005620 |
| 64 | raw_text | 0 | n/a | -0.000000 | -0.000000 |
| 64 | self_ranked_induction | 1150 | 0.834783 | 0.140260 | 0.004805 |
| 64 | sample_aware_self_ranked_induction | 1150 | 0.829565 | 0.142857 | 0.005000 |
| 64 | agreement_gated_self_ranked_induction | 1150 | 0.954783 | 0.080519 | 0.002788 |
| 64 | validation_ranked_induction | 1150 | 0.822609 | 0.135065 | 0.004607 |
| 64 | mdl_rule_expansion | 1610 | 0.767081 | 0.057143 | 0.001249 |
| 64 | counterfactual_expansion | 4600 | 1.000000 | 0.236363 | 0.003989 |

## Scope

- The audit uses the hidden rulebook, so it cannot be used as a deployable policy selector.
- The source sweep remains the neural experiment; this artifact explains a mechanism candidate.
- High generated-label precision is not sufficient evidence of neural gain at scarce budgets.

```json
{
  "audit_only": true,
  "deployable_policy": false,
  "fresh_seed_confirmation": true,
  "heldout_used_for_selection": false,
  "hidden_rulebook_available_to_policies": false,
  "neural_model": true,
  "paper_ready_claim": false,
  "synthetic_domain": true,
  "uses_hidden_rulebook_for_audit": true
}
```
