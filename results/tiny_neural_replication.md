# Learning Signal Density Tiny Neural Replication

Generated: `2026-06-27T18:19:39Z`

This is a deterministic CPU tiny-MLP replication of the causal-domain pilot.
It is not a language-model result and not a paper-ready frontier claim.

Backend: `tiny_mlp`
Hidden units: `32`
Feature dimension: `128`

| Condition | Heldout acc. | Signed gain | Compute units | Neural params | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.500 | -0.110 | 38528.0 | 4161 | 9234432 | -0.016651 |
| qa_expansion | 0.586 | -0.024 | 106640.0 | 4161 | 22316032 | -0.001316 |
| self_ranked_induction | 0.652 | 0.041 | 179143.2 | 4161 | 35346432 | 0.001343 |
| sample_aware_self_ranked_induction | 0.652 | 0.041 | 179143.2 | 4161 | 35346432 | 0.001343 |
| counterfactual_expansion | 0.666 | 0.055 | 377024.0 | 4161 | 74560512 | 0.000851 |
