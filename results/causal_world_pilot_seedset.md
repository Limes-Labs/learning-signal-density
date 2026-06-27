# Learning Signal Density Pilot

Generated: `2026-06-27T16:50:01Z`

This is a controlled pilot on a synthetic causal-text domain. It is not a neural-language-model result.
The heldout split is not used for selection or transformation. Counterfactual expansion is oracle-generated inside the synthetic world.

| Condition | Heldout acc. | Majority acc. | Signed gain | External events | Internal tokens | Compute units | Signed LSD/1M | Clipped LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.607 | 0.610 | -0.003 | 172.0 | 1204.0 | 6020.0 | -0.003330 | 0.013321 |
| selected_text | 0.603 | 0.610 | -0.007 | 172.0 | 961.8 | 6013.0 | -0.008087 | 0.025977 |
| qa_expansion | 0.603 | 0.610 | -0.007 | 172.0 | 3268.0 | 18404.0 | -0.002179 | 0.010893 |
| induced_rule_expansion | 0.655 | 0.610 | 0.045 | 172.0 | 6455.2 | 38731.2 | 0.006787 | 0.006787 |
| validation_gated_induction | 0.683 | 0.610 | 0.072 | 172.0 | 9750.4 | 389788.8 | 0.001070 | 0.001070 |
| direct_validation_gated_induction | 0.672 | 0.610 | 0.062 | 172.0 | 11524.0 | 75916.0 | 0.004754 | 0.005018 |
| mdl_rule_expansion | 0.631 | 0.610 | 0.021 | 172.0 | 5910.4 | 69019.2 | 0.001899 | 0.002473 |
| counterfactual_expansion | 0.728 | 0.610 | 0.117 | 172.0 | 11524.0 | 65876.0 | 0.010347 | 0.010347 |
| prioritized_replay | 0.610 | 0.610 | 0.000 | 172.0 | 6565.6 | 38189.6 | -0.000047 | 0.003612 |
| selected_counterfactual_replay | 0.666 | 0.610 | 0.055 | 172.0 | 9205.8 | 53828.2 | 0.006045 | 0.007878 |

## Pareto Frontier

`counterfactual_expansion`, `induced_rule_expansion`, `prioritized_replay`, `raw_text`, `selected_counterfactual_replay`, `selected_text`

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "neural_model": false,
  "oracle_transform": true,
  "paper_ready_claim": false,
  "synthetic_domain": true
}
```

## Condition Scope

| Condition | Oracle labels | Train-only selection | Train-only induction | Validation-gated threshold |
| --- | ---: | ---: | ---: | ---: |
| raw_text | false | false | false | false |
| selected_text | false | true | false | false |
| qa_expansion | false | false | false | false |
| induced_rule_expansion | false | false | true | false |
| validation_gated_induction | false | false | true | true |
| direct_validation_gated_induction | false | false | true | true |
| mdl_rule_expansion | false | false | true | true |
| counterfactual_expansion | true | false | false | false |
| prioritized_replay | false | false | false | false |
| selected_counterfactual_replay | true | true | false | false |

## Interpretation

- External sample efficiency charges the original observations only.
- Compute efficiency charges training tokens, train-only selection cost, and synthetic transform tokens.
- Validation-gated conditions also charge threshold-search overhead.
- MDL conditions charge rule-search, validation scoring, and selected-rule description length.
- Signed metrics preserve negative results; clipped metrics count only per-seed positive improvements.
- Learning-signal density is reported as heldout improvement per external event per charged internal unit, scaled by 1M.
- The first useful scientific question is the Pareto frontier, not a single winning condition.
