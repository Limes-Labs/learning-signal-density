# Real Text Selection-Cost Pilot

Generated: `2026-07-01T21:59:07Z`

Dataset: UCI SMS Spam Collection (10.24432/C5CC84, CC BY 4.0, 5574 records).

This artifact uses real SMS text, not the synthetic causal-text world.
The heldout split is never used for sampling-policy selection. Validation is used only by the declared selector.
Primary quality metric: spam-class F1 improvement over the train-sample majority baseline.

## Train Budget 32

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.425 | 0.000 | 0.425 | 2396.0 | 5.879975 |
| class_balanced_sample | class_balanced_sample: 5 | 0.486 | 0.000 | 0.486 | 70221.6 | 0.216139 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.499 | 0.000 | 0.499 | 7197.0 | 2.168642 |
| validation_sample_selector | class_balanced_sample: 4, random_sample: 1 | 0.524 | 0.000 | 0.524 | 78800.2 | 0.207702 |
| validation_label_index_selector | label_index_balanced_sample: 4, random_sample: 1 | 0.524 | 0.000 | 0.524 | 16034.6 | 1.019898 |

Pareto frontier: `label_index_balanced_sample`, `random_sample`, `validation_label_index_selector`

## Train Budget 64

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.600 | 0.000 | 0.600 | 4749.0 | 1.988980 |
| class_balanced_sample | class_balanced_sample: 5 | 0.645 | 0.000 | 0.645 | 73443.6 | 0.137339 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.635 | 0.000 | 0.635 | 10371.0 | 0.956476 |
| validation_sample_selector | class_balanced_sample: 2, random_sample: 3 | 0.589 | 0.000 | 0.589 | 83372.0 | 0.110425 |
| validation_label_index_selector | label_index_balanced_sample: 2, random_sample: 3 | 0.589 | 0.000 | 0.589 | 20606.4 | 0.446796 |

Pareto frontier: `class_balanced_sample`, `label_index_balanced_sample`, `random_sample`

## Train Budget 128

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.709 | 0.000 | 0.709 | 9798.0 | 0.565809 |
| class_balanced_sample | class_balanced_sample: 5 | 0.619 | 0.000 | 0.619 | 79700.6 | 0.060637 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.613 | 0.000 | 0.613 | 16558.0 | 0.289603 |
| validation_sample_selector | random_sample: 5 | 0.727 | 0.000 | 0.727 | 92608.4 | 0.061352 |
| validation_label_index_selector | random_sample: 5 | 0.727 | 0.000 | 0.727 | 29842.8 | 0.190498 |

Pareto frontier: `random_sample`, `validation_label_index_selector`

## Train Budget 256

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.776 | 0.000 | 0.776 | 19820.0 | 0.153031 |
| class_balanced_sample | class_balanced_sample: 5 | 0.698 | 0.000 | 0.698 | 92760.6 | 0.029394 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.794 | 0.000 | 0.794 | 29316.0 | 0.105836 |
| validation_sample_selector | random_sample: 5 | 0.805 | 0.000 | 0.805 | 111962.0 | 0.028100 |
| validation_label_index_selector | random_sample: 5 | 0.805 | 0.000 | 0.805 | 49196.4 | 0.063968 |

Pareto frontier: `label_index_balanced_sample`, `random_sample`, `validation_label_index_selector`

## Train Budget 512

| Condition | Selected policy counts | Heldout spam F1 | Majority F1 | Signed F1 gain | Compute units | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_sample | random_sample: 5 | 0.847 | 0.000 | 0.847 | 40820.0 | 0.040546 |
| class_balanced_sample | class_balanced_sample: 5 | 0.780 | 0.000 | 0.780 | 117756.6 | 0.012937 |
| label_index_balanced_sample | label_index_balanced_sample: 5 | 0.790 | 0.000 | 0.790 | 54903.0 | 0.028125 |
| validation_sample_selector | class_balanced_sample: 2, random_sample: 3 | 0.801 | 0.000 | 0.801 | 154658.4 | 0.010140 |
| validation_label_index_selector | label_index_balanced_sample: 2, random_sample: 3 | 0.801 | 0.000 | 0.801 | 91892.8 | 0.017108 |

Pareto frontier: `random_sample`

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
