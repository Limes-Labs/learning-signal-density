# Twenty Newsgroups Active Label-Acquisition Audit

Generated: `2026-07-02T00:04:59Z`

This is a real NLP active label-acquisition audit over UCI Twenty Newsgroups mini.
The teacher selects from the train-only pool before true labels for acquired records are used in the final student.

| Budget | Best density condition | Acc | LSD | Random LSD | Class LSD | Density win? |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 40 | balanced_margin_uncertainty | 0.079 | 0.002935 | 0.060065 | 0.036293 | False |
| 80 | short_margin_uncertainty | 0.090 | 0.001835 | 0.013418 | 0.016884 | False |
| 160 | short_margin_uncertainty | 0.127 | 0.001604 | 0.006005 | 0.004276 | False |

## Interpretation

- This audit tests active acquisition with true labels acquired after teacher selection.
- It is label-aware only for the charged class-balanced seed; acquisition itself uses teacher predictions and margins.
- A density win requires the acquired examples to pay for seed selection, teacher training, scoring, and final training.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "oracle_train_labels_used_for_acquisition": false,
  "paper_ready_claim": false,
  "post_hoc_optimization_attempt": true,
  "real_dataset": true,
  "synthetic_domain": false,
  "true_labels_acquired_after_selection": true,
  "validation_used_for_selection": false
}
```
