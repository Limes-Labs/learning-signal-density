# Twenty Newsgroups Length-Penalized Retrieval-Cost Audit

Generated: `2026-07-01T23:26:19Z`

This is a real NLP retrieval-cost optimization audit over UCI Twenty Newsgroups mini.
It tests whether penalizing selected-document length can make prototype retrieval cheaper enough to beat random density.

| Budget | Best acc alpha | Acc | LSD | Best density alpha | Acc | LSD | Random LSD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0 | 0.102 | 0.001772 | 0 | 0.102 | 0.001772 | 0.060065 |
| 80 | 0.25 | 0.113 | 0.001505 | 0.25 | 0.113 | 0.001505 | 0.013418 |
| 160 | 0 | 0.145 | 0.000675 | 0.5 | 0.120 | 0.000811 | 0.006005 |

## Interpretation

- Alpha 0.0 is the original prototype-retrieval score.
- Larger alpha penalizes long selected documents and reduces nonreusable training-token cost.
- In this artifact, length penalties improve some prototype-retrieval rows but do not beat random sampling on density.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "paper_ready_claim": false,
  "post_hoc_optimization_attempt": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
