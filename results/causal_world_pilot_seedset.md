# Learning Signal Density Pilot

Generated: `2026-06-27T16:09:38Z`

This is a controlled pilot on a synthetic causal-text domain. It is not a neural-language-model result.
The heldout split is not used for selection or transformation. Counterfactual expansion is oracle-generated inside the synthetic world.

| Condition | Heldout acc. | Majority acc. | External events | Internal tokens | Compute units | External eff. | Compute eff./10k | LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.607 | 0.610 | 172.0 | 1204.0 | 6020.0 | 0.000080 | 0.022912 | 0.013321 |
| selected_text | 0.603 | 0.610 | 172.0 | 961.8 | 6013.0 | 0.000160 | 0.044681 | 0.025977 |
| qa_expansion | 0.603 | 0.610 | 172.0 | 3268.0 | 18404.0 | 0.000201 | 0.018737 | 0.010893 |
| counterfactual_expansion | 0.728 | 0.610 | 172.0 | 11524.0 | 65876.0 | 0.000681 | 0.017797 | 0.010347 |
| prioritized_replay | 0.610 | 0.610 | 172.0 | 6565.6 | 38189.6 | 0.000140 | 0.006213 | 0.003612 |
| selected_counterfactual_replay | 0.666 | 0.610 | 172.0 | 9205.8 | 53828.2 | 0.000421 | 0.013551 | 0.007878 |

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

## Interpretation

- External sample efficiency charges the original observations only.
- Compute efficiency charges training tokens, train-only selection cost, and synthetic transform tokens.
- Learning-signal density is reported as heldout improvement per external event per charged internal unit, scaled by 1M.
- The first useful scientific question is the Pareto frontier, not a single winning condition.
