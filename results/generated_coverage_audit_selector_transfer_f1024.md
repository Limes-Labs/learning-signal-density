# Learning Signal Density Generated-coverage Audit

Source artifact: `results/tiny_neural_budget_sweep_selector_transfer_f1024.json`
Source generated: `2026-06-29T11:43:27Z`
Profile label: `epochs=16_hidden=8_features=1024_selector_transfer`

This heldout-distribution audit is non-deployable. It compares where generated synthetic labels land in motif space against the heldout motif distribution after the source sweep has already been run.

| Materials | Condition | Label precision | Heldout triple coverage | Triple L1 distance | Linked gain |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | raw_text | n/a | 0.000000 | n/a | 0.000000 |
| 16 | self_ranked_induction | 0.741379 | 0.000000 | 1.000000 | -0.010526 |
| 16 | sample_aware_self_ranked_induction | 0.742857 | 0.000000 | 1.000000 | 0.021053 |
| 16 | agreement_gated_self_ranked_induction | 0.957143 | 0.000000 | 1.000000 | -0.063158 |
| 16 | validation_ranked_induction | 0.741379 | 0.000000 | 1.000000 | -0.010526 |
| 16 | mdl_rule_expansion | 0.617284 | 0.357895 | 0.901235 | -0.021053 |
| 16 | counterfactual_expansion | 1.000000 | 0.473684 | 0.908621 | -0.063158 |
| 24 | raw_text | n/a | 0.000000 | n/a | 0.000000 |
| 24 | self_ranked_induction | 0.718605 | 0.365517 | 0.791339 | -0.089655 |
| 24 | sample_aware_self_ranked_induction | 0.700000 | 0.317241 | 0.811099 | -0.096552 |
| 24 | agreement_gated_self_ranked_induction | 0.903125 | 0.372414 | 0.815517 | -0.068966 |
| 24 | validation_ranked_induction | 0.718605 | 0.365517 | 0.791339 | -0.089655 |
| 24 | mdl_rule_expansion | 0.608163 | 0.682759 | 0.763969 | -0.089655 |
| 24 | counterfactual_expansion | 1.000000 | 0.896552 | 0.742442 | 0.020690 |
| 32 | raw_text | n/a | 0.000000 | n/a | -0.021053 |
| 32 | self_ranked_induction | 0.691379 | 0.673684 | 0.683666 | -0.068421 |
| 32 | sample_aware_self_ranked_induction | 0.675862 | 0.463158 | 0.792015 | -0.173684 |
| 32 | agreement_gated_self_ranked_induction | 0.917241 | 0.536842 | 0.764428 | -0.131579 |
| 32 | validation_ranked_induction | 0.691379 | 0.673684 | 0.683666 | -0.068421 |
| 32 | mdl_rule_expansion | 0.680319 | 0.615789 | 0.776582 | -0.084210 |
| 32 | counterfactual_expansion | 1.000000 | 0.947368 | 0.662137 | 0.015790 |
| 48 | raw_text | n/a | 0.000000 | n/a | -0.013793 |
| 48 | self_ranked_induction | 0.783721 | 0.641379 | 0.551002 | 0.058621 |
| 48 | sample_aware_self_ranked_induction | 0.783721 | 0.641379 | 0.551002 | 0.058621 |
| 48 | agreement_gated_self_ranked_induction | 0.946512 | 0.717241 | 0.618204 | 0.017241 |
| 48 | validation_ranked_induction | 0.779070 | 0.675862 | 0.542863 | 0.062069 |
| 48 | mdl_rule_expansion | 0.832632 | 0.568966 | 0.692015 | 0.065517 |
| 48 | counterfactual_expansion | 1.000000 | 1.000000 | 0.529200 | 0.186207 |
| 64 | raw_text | n/a | 0.000000 | n/a | -0.000000 |
| 64 | self_ranked_induction | 0.834783 | 0.670130 | 0.496138 | 0.140260 |
| 64 | sample_aware_self_ranked_induction | 0.829565 | 0.690909 | 0.477843 | 0.142857 |
| 64 | agreement_gated_self_ranked_induction | 0.954783 | 0.727273 | 0.576443 | 0.080519 |
| 64 | validation_ranked_induction | 0.822609 | 0.758442 | 0.471790 | 0.135065 |
| 64 | mdl_rule_expansion | 0.767081 | 0.841558 | 0.537211 | 0.057143 |
| 64 | counterfactual_expansion | 1.000000 | 1.000000 | 0.468642 | 0.236363 |

## Scope

- The audit uses the hidden rulebook and heldout motif distribution, so it cannot be used as a deployable selector.
- Lower triple L1 distance means generated labels are distributed more like heldout motifs.
- The source sweep remains the neural experiment; this artifact tests a mechanism hypothesis.

```json
{
  "audit_only": true,
  "deployable_policy": false,
  "fresh_seed_confirmation": true,
  "heldout_distribution_available_to_policies": false,
  "heldout_used_for_selection": false,
  "hidden_rulebook_available_to_policies": false,
  "neural_model": true,
  "paper_ready_claim": false,
  "synthetic_domain": true,
  "uses_heldout_distribution_for_audit": true,
  "uses_hidden_rulebook_for_label_audit": true
}
```
