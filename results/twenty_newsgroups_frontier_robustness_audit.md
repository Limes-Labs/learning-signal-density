# Twenty Newsgroups Frontier Robustness Audit

Generated: `2026-07-02T01:01:02Z`

This is a paired-seed robustness audit over committed Twenty Newsgroups artifacts.
It introduces no new policy and tests whether mean density wins survive a conservative seed bootstrap.

| Comparison | Budget | Mean ratio | Paired wins | Delta CI95 | Robust win? | Robust loss? |
| --- | ---: | ---: | ---: | --- | --- | --- |
| class_balanced_80_vs_random | 80 | 1.258 | 2/3 | [-0.001440, 0.008371] | False | False |
| prototype_40_vs_random | 40 | 0.030 | 0/3 | [-0.073751, -0.042834] | False | True |
| prototype_160_vs_random | 160 | 0.112 | 0/3 | [-0.005790, -0.004871] | False | True |
| budgeted_margin_1x_160_vs_class_balanced | 160 | 1.034 | 1/3 | [-0.001785, 0.002073] | False | False |
| budgeted_margin_2x_160_vs_class_balanced | 160 | 1.045 | 2/3 | [-0.000269, 0.000652] | False | False |
| budgeted_margin_2x_160_vs_random | 160 | 0.744 | 0/3 | [-0.002109, -0.000967] | False | True |

## Interpretation

- Mean density wins: 3.
- Robust density wins under the paired-bootstrap rule: 0.
- Robust density losses under the paired-bootstrap rule: 3.
- The current positive density frontiers should be read as hypotheses until confirmed on more seeds.

## Scope Flags

```json
{
  "exact_seed_bootstrap": true,
  "heldout_used_for_selection": false,
  "introduces_new_policy": false,
  "metadata_stripped": true,
  "paired_seed_audit": true,
  "paper_ready_claim": false,
  "post_hoc_diagnostic": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
