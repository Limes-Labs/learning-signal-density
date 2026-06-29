# Learning Signal Density Pilot

Generated: `2026-06-29T12:07:50Z`

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
| validation_ranked_induction | 0.659 | 0.610 | 0.048 | 172.0 | 5332.0 | 35585.2 | 0.007878 | 0.008442 |
| train_calibrated_ranked_induction | 0.659 | 0.610 | 0.048 | 172.0 | 5332.0 | 36383.2 | 0.007705 | 0.008257 |
| self_ranked_induction | 0.666 | 0.610 | 0.055 | 172.0 | 5332.0 | 35179.2 | 0.009108 | 0.009108 |
| sample_aware_self_ranked_induction | 0.666 | 0.610 | 0.055 | 172.0 | 5332.0 | 35179.2 | 0.009108 | 0.009108 |
| diverse_self_ranked_induction | 0.652 | 0.610 | 0.041 | 172.0 | 5332.0 | 35179.2 | 0.006832 | 0.006832 |
| mdl_rule_expansion | 0.631 | 0.610 | 0.021 | 172.0 | 5910.4 | 69019.2 | 0.001899 | 0.002473 |
| counterfactual_expansion | 0.728 | 0.610 | 0.117 | 172.0 | 11524.0 | 65876.0 | 0.010347 | 0.010347 |
| prioritized_replay | 0.610 | 0.610 | 0.000 | 172.0 | 6565.6 | 38189.6 | -0.000047 | 0.003612 |
| selected_counterfactual_replay | 0.666 | 0.610 | 0.055 | 172.0 | 9205.8 | 53828.2 | 0.006045 | 0.007878 |

## Pareto Frontier

`counterfactual_expansion`, `raw_text`, `sample_aware_self_ranked_induction`, `selected_text`, `self_ranked_induction`

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

| Condition | Oracle labels | Train-only selection | Train-only induction | Validation threshold | Validation transform selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_text | false | false | false | false | false |
| selected_text | false | true | false | false | false |
| qa_expansion | false | false | false | false | false |
| induced_rule_expansion | false | false | true | false | false |
| validation_gated_induction | false | false | true | true | true |
| direct_validation_gated_induction | false | false | true | true | true |
| validation_ranked_induction | false | false | true | false | true |
| train_calibrated_ranked_induction | false | true | true | false | false |
| self_ranked_induction | false | true | true | false | false |
| sample_aware_self_ranked_induction | false | true | true | false | false |
| diverse_self_ranked_induction | false | true | true | false | false |
| mdl_rule_expansion | false | false | true | false | true |
| counterfactual_expansion | true | false | false | false | false |
| prioritized_replay | false | false | false | false | false |
| selected_counterfactual_replay | true | true | false | false | false |

## Interpretation

- External sample efficiency charges the original observations only.
- Compute efficiency charges training tokens, train-only selection cost, and synthetic transform tokens.
- Validation-gated conditions also charge threshold-search overhead.
- Ranked induction conditions charge candidate-ranking overhead and any calibration scoring they use.
- MDL conditions charge rule-search, validation scoring, and selected-rule description length.
- Signed metrics preserve negative results; clipped metrics count only per-seed positive improvements.
- Learning-signal density is reported as heldout improvement per external event per charged internal unit, scaled by 1M.
- The first useful scientific question is the Pareto frontier, not a single winning condition.
