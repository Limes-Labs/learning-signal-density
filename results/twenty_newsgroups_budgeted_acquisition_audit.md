# Twenty Newsgroups Budgeted Active Label-Acquisition Audit

Generated: `2026-07-02T00:19:01Z`

This is a real NLP budgeted active label-acquisition audit over UCI Twenty Newsgroups mini.
The teacher scores only a sampled window of the train-only pool before true labels are acquired.

| Budget | Best density condition | Acc | Scan size | LSD | Random LSD | Class LSD | Density win? |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 40 | short_margin_uncertainty_1x | 0.084 | 20.0 | 0.027690 | 0.060065 | 0.036293 | False |
| 80 | short_margin_uncertainty_2x | 0.099 | 120.0 | 0.011165 | 0.013418 | 0.016884 | False |
| 160 | margin_uncertainty_2x | 0.133 | 240.0 | 0.004467 | 0.006005 | 0.004276 | True |

## Interpretation

- This audit tests whether reducing active-acquisition scan cost can move the density frontier.
- The scan window is sampled before teacher scoring; true labels are used only after acquisition.
- A density win requires the acquired examples to pay for seed selection, teacher training, window scoring, and final training.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "oracle_train_labels_used_for_acquisition": false,
  "paper_ready_claim": false,
  "post_hoc_optimization_attempt": true,
  "real_dataset": true,
  "scan_window_sampled_without_text_scoring": true,
  "synthetic_domain": false,
  "true_labels_acquired_after_selection": true,
  "validation_used_for_selection": false
}
```
