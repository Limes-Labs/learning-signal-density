# Twenty Newsgroups Self-Training Pseudo-Label Audit

Generated: `2026-07-01T23:41:50Z`

This is a real NLP pseudo-label/self-training audit over UCI Twenty Newsgroups mini.
Pseudo-labels are teacher predictions only; oracle train labels are recorded for diagnostics but not used for selection or student labels.

| Budget | Best condition | Acc | Pseudo agree | LSD | Class-balanced LSD | Random LSD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 40 | class_balanced_self_training_balanced_margin_2x | 0.072 | 0.154 | 0.001270 | 0.036293 | 0.060065 |
| 80 | class_balanced_self_training_balanced_margin_1x | 0.083 | 0.175 | 0.000875 | 0.016884 | 0.013418 |
| 160 | class_balanced_self_training_balanced_margin_1x | 0.107 | 0.226 | 0.000583 | 0.004276 | 0.006005 |

## Interpretation

- The teacher pseudo-labels are too noisy in this scarce-label setting.
- Confidence filtering by teacher margin does not rescue learning-signal density once scoring and student training costs are charged.
- This is a negative control for distillation and synthetic-data filtering claims in the current repo.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "oracle_train_labels_used_for_pseudo_label_selection": false,
  "paper_ready_claim": false,
  "post_hoc_optimization_attempt": true,
  "pseudo_labels_use_teacher_predictions": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
