# Real Text Selection-Cost Pilot

Generated: `2026-07-01T21:59:10Z`

Dataset: UCI SMS Spam Collection (10.24432/C5CC84, CC BY 4.0, 5574 records).

This artifact uses real SMS text, not the synthetic causal-text world.
The heldout split is never used for sampling-policy selection. Validation is used only by the declared selector.
Primary quality metric: spam-class F1 improvement over the train-sample majority baseline.

## Train Budget 32

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.408 | 0.000 | 0.408 | 2595.0 | 4.991051 |
| class_balanced_sample | class_balanced_sample: 5 | 0.472 | 0.000 | 0.472 | 60398.6 | 0.244137 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.550 | 0.000 | 0.550 | 6655.0 | 2.585588 |
| validation_sample_selector | class_balanced_sample: 4, random_sample: 1 | 0.582 | 0.000 | 0.582 | 88226.6 | 0.206003 |
| validation_label_index_selector | label_index_balanced_sample: 4, random_sample: 1 | 0.582 | 0.000 | 0.582 | 34505.0 | 0.526972 |

Pareto frontier: `label_index_balanced_sample`, `random_sample`, `validation_label_index_selector`

## Train Budget 64

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.489 | 0.000 | 0.489 | 4940.0 | 1.559989 |
| class_balanced_sample | class_balanced_sample: 5 | 0.694 | 0.000 | 0.694 | 63567.6 | 0.170597 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.595 | 0.000 | 0.595 | 9701.0 | 0.957780 |
| validation_sample_selector | class_balanced_sample: 3, random_sample: 2 | 0.540 | 0.000 | 0.540 | 93033.2 | 0.090784 |
| validation_label_index_selector | label_index_balanced_sample: 3, random_sample: 2 | 0.540 | 0.000 | 0.540 | 39311.6 | 0.215414 |

Pareto frontier: `class_balanced_sample`, `label_index_balanced_sample`, `random_sample`

## Train Budget 128

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.719 | 0.000 | 0.719 | 9920.0 | 0.568491 |
| class_balanced_sample | class_balanced_sample: 5 | 0.725 | 0.000 | 0.725 | 70065.6 | 0.080854 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.720 | 0.000 | 0.720 | 16042.0 | 0.350824 |
| validation_sample_selector | class_balanced_sample: 1, random_sample: 4 | 0.742 | 0.000 | 0.742 | 102381.2 | 0.056594 |
| validation_label_index_selector | label_index_balanced_sample: 1, random_sample: 4 | 0.742 | 0.000 | 0.742 | 48659.6 | 0.119136 |

Pareto frontier: `label_index_balanced_sample`, `random_sample`, `validation_label_index_selector`

## Train Budget 256

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.808 | 0.000 | 0.808 | 20097.0 | 0.157204 |
| class_balanced_sample | class_balanced_sample: 5 | 0.771 | 0.000 | 0.771 | 82401.6 | 0.036545 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.728 | 0.000 | 0.728 | 28548.0 | 0.099703 |
| validation_sample_selector | class_balanced_sample: 1, random_sample: 4 | 0.781 | 0.000 | 0.781 | 122984.8 | 0.024847 |
| validation_label_index_selector | label_index_balanced_sample: 1, random_sample: 4 | 0.781 | 0.000 | 0.781 | 69263.2 | 0.044178 |

Pareto frontier: `random_sample`

## Train Budget 512

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.836 | 0.000 | 0.836 | 40515.0 | 0.040287 |
| class_balanced_sample | class_balanced_sample: 5 | 0.842 | 0.000 | 0.842 | 107686.6 | 0.015278 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.795 | 0.000 | 0.795 | 54468.0 | 0.028530 |
| validation_sample_selector | random_sample: 5 | 0.858 | 0.000 | 0.858 | 160327.8 | 0.010460 |
| validation_label_index_selector | random_sample: 5 | 0.858 | 0.000 | 0.858 | 106606.2 | 0.015733 |

Pareto frontier: `random_sample`, `validation_label_index_selector`

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "neural_model": false,
  "paper_ready_claim": false,
  "real_dataset": true,
  "synthetic_domain": false,
  "validation_used_by_selector": true
}
```
