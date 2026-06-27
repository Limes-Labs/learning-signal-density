# Learning Signal Density Tiny Neural Confirmation

Generated: `2026-06-27T18:34:28Z`

This is a deterministic CPU tiny-MLP replication of the causal-domain pilot.
It is not a language-model result and not a paper-ready frontier claim.

Fresh-seed confirmation: `true`
Confirmation target: `results/tiny_neural_replication.json`
Target signed gain over majority: `0.03`

Backend: `tiny_mlp`
Hidden units: `32`
Feature dimension: `128`

| Condition | Heldout acc. | Signed gain | Reaches target | Compute units | Neural params | Neural train ops | Signed LSD/1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_text | 0.538 | -0.031 | no | 38528.0 | 4161 | 9247744 | -0.004683 |
| qa_expansion | 0.603 | 0.034 | yes | 106640.0 | 4161 | 22306816 | 0.001880 |
| self_ranked_induction | 0.607 | 0.038 | yes | 179205.6 | 4161 | 35335168 | 0.001230 |
| sample_aware_self_ranked_induction | 0.607 | 0.038 | yes | 179205.6 | 4161 | 35335168 | 0.001230 |
| counterfactual_expansion | 0.690 | 0.121 | yes | 377024.0 | 4161 | 74573824 | 0.001861 |
