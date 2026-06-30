# Learning Signal Density Support-ramp Mechanism Audit

Source artifact: `results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json`
Source generated: `2026-06-30T21:24:48Z`
Profile label: `f1024_16x8_support_selector_transfer`

This support-ramp mechanism audit is non-deployable. It reconstructs candidate pipelines on the support-selector transfer seeds and compares generated labels with the hidden rulebook and heldout motif distribution after the neural sweep has already run.

## Transition Diagnostics

| Materials | Support precision | Compact precision | Support coverage | Compact coverage | Support LSD | Density-cap LSD | Support minus density-cap LSD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 104 | 0.820911 | 0.904278 | 0.569600 | 0.760000 | 0.004345 | 0.004187 | 0.000158 |
| 112 | 0.784507 | 0.923500 | 0.408955 | 0.671642 | 0.004175 | 0.007757 | -0.003582 |
| 120 | 0.809028 | 0.946759 | 0.312500 | 0.770833 | 0.004309 | 0.005382 | -0.001073 |
| 128 | 0.823194 | 0.959565 | 0.251948 | 0.751948 | 0.004145 | 0.006247 | -0.002102 |

## Mechanism Summary

- Support precision improvements over compact: `0`.
- Support coverage losses versus compact: `4`.
- Support density wins over density cap: `1`.
- Promote support-ramp mechanism: `false`.
- Interpretation: The support ramp is a cost-reduction mechanism, not a reliability improvement: on transition budgets it lowers synthetic volume and usually loses heldout motif coverage and signed density versus the density-capped raw fallback.

## Scope

- The hidden rulebook and heldout motif distribution are used only after the source sweep.
- The source neural artifact remains the downstream experiment; this artifact explains mechanism failures.
- Treat the result as a design constraint for future expected-value selectors.

```json
{
  "audit_only": true,
  "deployable_policy": false,
  "fresh_seed_confirmation": true,
  "heldout_available_to_policies": false,
  "heldout_used_for_selection": false,
  "hidden_rulebook_available_to_policies": false,
  "neural_model": true,
  "paper_ready_claim": false,
  "synthetic_domain": true,
  "uses_heldout_distribution_for_audit": true,
  "uses_hidden_rulebook_for_label_audit": true
}
```
