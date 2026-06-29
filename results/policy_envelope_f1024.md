# Learning Signal Density Post-hoc Policy Envelope

Source artifact: `results/tiny_neural_budget_sweep_agreement_gated_f1024.json`
Source generated: `2026-06-28T23:55:06Z`
Profile label: `epochs=16_hidden=8_features=1024_agreement_gated`

This is a post-hoc policy envelope. It uses heldout sweep results to choose the best non-oracle condition at each material budget, so it is not deployable. Its purpose is to measure the selection problem left for a future adaptive policy.

Target signed gain: `0.03`
First envelope budget reaching target: `48`
Best envelope condition: `self_ranked_induction` at `64` materials

| Materials | Envelope condition | Signed gain | Charged compute | Oracle gain | Oracle gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | mdl_rule_expansion | 0.010526 | 29295.600000 | -0.063158 | -0.073684 |
| 24 | raw_text | 0.000000 | 9632.000000 | 0.075862 | 0.075862 |
| 32 | mdl_rule_expansion | -0.036842 | 79635.400000 | 0.036842 | 0.073684 |
| 48 | validation_ranked_induction | 0.086207 | 94299.600000 | 0.193104 | 0.106897 |
| 64 | self_ranked_induction | 0.153247 | 126970.000000 | 0.238961 | 0.085714 |

## Scope

- Non-deployable reason: The envelope selects the best non-oracle condition separately at each material budget using heldout accuracy from the completed sweep. It is a diagnostic upper bound for policy selection, not a deployable policy.
- Oracle counterfactual expansion is excluded from the envelope and reported only as headroom.
- Validation-ranked and MDL rows use validation labels and charge that selection work in the source artifact.
