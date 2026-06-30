# Learning Signal Density Support-selector Error Audit

This is a post-hoc support-selector error audit. It reads committed fresh-seed
neural sweeps and asks whether extra selector information has positive
expected value after charged inspection or validation cost.

## Source Summary

| Source | Best simple | Best simple LSD | Least-regret selector | Selector LSD | Avg. regret | Wins | Avg. selector cost |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Train support-density selector | density_window_compact_induction | 0.005255 | train_support_density_selector | 0.004274 | 0.001099 | 0/7 | 17904.9 |
| Support-probe window selector | density_capped_compact_induction | 0.005357 | support_probe_window_selector | 0.005079 | 0.000555 | 0/7 | 1941.0 |
| Validation support-precision selector | density_capped_compact_induction | 0.006102 | validation_support_precision_selector | 0.006138 | 0.000042 | 2/7 | 1309.2 |
| Validation support-precision gate | density_window_compact_induction | 0.006090 | validation_support_precision_selector | 0.006223 | -0.000110 | 2/7 | 1620.0 |
| Support-selector transfer stress | density_capped_compact_induction | 0.006115 | validation_support_precision_gate_selector | 0.005936 | 0.000496 | 1/7 | 2082.8 |

## Selector Details

| Source | Selector | Avg. LSD | Avg. regret | Worst budget | Worst regret | Avg. selector cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train support-density selector | train_support_density_selector | 0.004274 | 0.001099 | 120 | 0.001671 | 17904.9 |
| Support-probe window selector | train_support_density_selector | 0.004354 | 0.001280 | 112 | 0.002931 | 17948.8 |
| Support-probe window selector | support_probe_window_selector | 0.005079 | 0.000555 | 112 | 0.001942 | 1941.0 |
| Validation support-precision selector | train_support_density_selector | 0.004815 | 0.001365 | 120 | 0.002542 | 17928.2 |
| Validation support-precision selector | support_probe_window_selector | 0.005941 | 0.000240 | 96 | 0.001361 | 1930.0 |
| Validation support-precision selector | validation_support_precision_selector | 0.006138 | 0.000042 | 120 | 0.000523 | 1309.2 |
| Validation support-precision gate | train_support_density_selector | 0.004967 | 0.001145 | 128 | 0.001670 | 17915.9 |
| Validation support-precision gate | support_probe_window_selector | 0.006074 | 0.000039 | 120 | 0.000159 | 1935.5 |
| Validation support-precision gate | validation_support_precision_selector | 0.006223 | -0.000110 | 128 | 0.000448 | 1620.0 |
| Validation support-precision gate | validation_support_precision_gate_selector | 0.006104 | 0.000009 | 112 | 0.000833 | 1915.6 |
| Support-selector transfer stress | train_support_density_selector | 0.004345 | 0.002087 | 112 | 0.004382 | 17936.5 |
| Support-selector transfer stress | support_probe_window_selector | 0.005920 | 0.000512 | 112 | 0.003582 | 1927.3 |
| Support-selector transfer stress | validation_support_precision_selector | 0.005601 | 0.000831 | 112 | 0.003582 | 1706.4 |
| Support-selector transfer stress | validation_support_precision_gate_selector | 0.005936 | 0.000496 | 112 | 0.001240 | 2082.8 |

## Recommendation

- Promote support selector: `false`.
- Strongest transfer selector: `validation_support_precision_gate_selector`.
- Transfer best simple comparator: `density_capped_compact_induction`.
- Reason: Do not promote a support selector yet: the least-regret transfer selector still loses to the best simple comparator after charged selection cost.

## Scope

- This audit uses completed heldout outcomes after the source sweeps have run.
- The heldout outcomes are not available to any deployable policy.
- Treat this as mechanism evidence and a promotion gate, not as a new selector.

```json
{
  "deployable_policy": false,
  "heldout_available_to_policies": false,
  "heldout_used_for_error_analysis": true,
  "neural_model": true,
  "paper_ready_claim": false,
  "post_hoc_diagnostic": true,
  "synthetic_domain": true,
  "uses_committed_fresh_seed_artifacts": true
}
```
