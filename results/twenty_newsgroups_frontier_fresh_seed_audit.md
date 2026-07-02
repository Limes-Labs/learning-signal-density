# Twenty Newsgroups Fresh-Seed Frontier Audit

Generated: `2026-07-02T10:05:41Z`

Fresh seeds: `[439, 443, 449, 457, 461]`

| Comparison | Mean ratio | Paired wins | Delta CI95 | Sign p(win) | Sign p(loss) | Reading |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| class_balanced_80_vs_random | 1.326 | 4/5 | [-0.000911, 0.007691] | 0.187500 | 0.968750 | fragile mean win |
| prototype_40_vs_random | 0.027 | 1/5 | [-0.063294, -0.014366] | 0.968750 | 0.187500 | mixed/no win |
| prototype_160_vs_random | 0.119 | 0/5 | [-0.004679, -0.003074] | 1.000000 | 0.031250 | robust density loss |
| budgeted_margin_1x_160_vs_class_balanced | 0.576 | 1/5 | [-0.003633, 0.000027] | 0.968750 | 0.187500 | mixed/no win |
| budgeted_margin_2x_160_vs_class_balanced | 0.713 | 1/5 | [-0.002039, -0.000366] | 0.968750 | 0.187500 | mixed/no win |
| budgeted_margin_2x_160_vs_random | 0.682 | 0/5 | [-0.002071, -0.000747] | 1.000000 | 0.031250 | robust density loss |

## Interpretation

- The fresh-seed audit is a replication check, not a policy-search stage.
- Robust density wins under the fresh-seed rule: 0.
- Robust density losses under the fresh-seed rule: 2.
- The positive class-balanced 80-label mean win remains fragile on untouched seeds.
- Budgeted active-acquisition rows do not replicate as density wins on untouched seeds.
