# Twenty Newsgroups Active Selection-Cost Pilot

Generated: `2026-07-01T22:52:31Z`

This is a real NLP classification pilot over UCI Twenty Newsgroups mini.
Headers, quote lines, and reply boilerplate are stripped before splitting.

| Budget | Condition | Acc. | Gain | LSD | Cost | Selected policy counts |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 40 | random_sample | 0.093 | 0.043 | 0.060065 | 18414.0 | {"random_sample": 3} |
| 40 | class_balanced_sample | 0.083 | 0.033 | 0.036293 | 23861.0 | {"class_balanced_sample": 3} |
| 40 | length_curriculum_sample | 0.062 | 0.012 | 0.001143 | 255473.7 | {"length_curriculum_sample": 3} |
| 40 | prototype_retrieval_sample | 0.102 | 0.052 | 0.001772 | 726875.3 | {"prototype_retrieval_sample": 3} |
| 40 | validation_selector | 0.102 | 0.052 | 0.000911 | 1417903.7 | {"prototype_retrieval_sample": 3} |
| 80 | random_sample | 0.102 | 0.052 | 0.013418 | 48276.0 | {"random_sample": 3} |
| 80 | class_balanced_sample | 0.102 | 0.052 | 0.016884 | 40911.0 | {"class_balanced_sample": 3} |
| 80 | length_curriculum_sample | 0.071 | 0.021 | 0.001038 | 258251.7 | {"length_curriculum_sample": 3} |
| 80 | prototype_retrieval_sample | 0.078 | 0.028 | 0.000472 | 792048.3 | {"prototype_retrieval_sample": 3} |
| 80 | validation_selector | 0.075 | 0.025 | 0.000248 | 1282578.7 | {"class_balanced_sample": 2, "random_sample": 1} |
| 160 | random_sample | 0.141 | 0.091 | 0.006005 | 94309.0 | {"random_sample": 3} |
| 160 | class_balanced_sample | 0.124 | 0.074 | 0.004276 | 111163.0 | {"class_balanced_sample": 3} |
| 160 | length_curriculum_sample | 0.104 | 0.054 | 0.001270 | 266164.7 | {"length_curriculum_sample": 3} |
| 160 | prototype_retrieval_sample | 0.145 | 0.095 | 0.000675 | 885630.3 | {"prototype_retrieval_sample": 3} |
| 160 | validation_selector | 0.111 | 0.061 | 0.000272 | 1392944.3 | {"class_balanced_sample": 1, "length_curriculum_sample": 1, "random_sample": 1} |

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "neural_model": false,
  "paper_ready_claim": false,
  "real_dataset": true,
  "synthetic_domain": false
}
```
