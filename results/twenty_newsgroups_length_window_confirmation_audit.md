# Twenty Newsgroups Length-Window Confirmation Audit

Generated: `2026-07-02T00:45:37Z`

This is a real NLP stress audit over UCI Twenty Newsgroups mini.
It tests whether sampled length-window selectors produce a stable random-density win.

| Budget | Dev best | Dev LSD | Dev random | Dev win? | Confirm same LSD | Confirm random | Same win? | Confirm best |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 40 | length_window_shortest_2x | 0.034693 | 0.060065 | False | 0.026764 | 0.030603 | True | length_window_top_0.25_2x |
| 80 | length_window_shortest_2x | 0.010818 | 0.013418 | False | 0.010625 | 0.010652 | False | length_window_band_0.20_0.70_1x |
| 160 | length_window_short_diverse_2x | 0.004636 | 0.006005 | False | 0.003433 | 0.003930 | False | length_window_shortest_2x |

## Interpretation

- The development grid finds no sampled length-window row that beats random density.
- Some confirmation-phase best rows beat same-phase random at other budgets, but those are not development-selected wins.
- The result is a mixed stress test for cheap length-window selection, not a deployable policy claim.

## Scope Flags

```json
{
  "confirmation_seeds_disjoint": true,
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "oracle_train_labels_used_for_selection": false,
  "paper_ready_claim": false,
  "post_hoc_development_selection": true,
  "real_dataset": true,
  "scan_window_sampled_before_length_selection": true,
  "synthetic_domain": false,
  "teacher_used_for_selection": false,
  "validation_used_for_selection": false
}
```
