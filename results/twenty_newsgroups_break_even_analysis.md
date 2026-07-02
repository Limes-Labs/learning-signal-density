# Twenty Newsgroups Break-Even Selection-Cost Analysis

Generated: `2026-07-01T23:06:41Z`

This artifact is a mathematical audit over the committed Twenty Newsgroups active-selection JSON.
It does not introduce a new policy. It asks when each active/curriculum/retrieval policy pays for its charged event-compute multiplier.

Break-even condition:

```text
G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)
```

| Budget | Candidate | Quality mult. | Event-compute mult. | Density ratio | Break-even gain | Cost over BE | Reuses to win | Fully amortized ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | class_balanced_sample | 0.750 | 1.296 | 0.579 | 0.056 | 1.728 | never | 0.609 |
| 40 | length_curriculum_sample | 0.269 | 13.874 | 0.019 | 0.601 | 51.530 | 73 | 3.382 |
| 40 | prototype_retrieval_sample | 1.192 | 39.474 | 0.030 | 1.711 | 33.107 | never | 0.100 |
| 40 | validation_selector | 1.192 | 77.001 | 0.015 | 3.337 | 64.581 | never | 0.100 |
| 80 | class_balanced_sample | 1.000 | 0.847 | 1.180 | 0.044 | 0.847 | 1 | 1.216 |
| 80 | length_curriculum_sample | 0.403 | 5.349 | 0.075 | 0.276 | 13.267 | 17 | 4.587 |
| 80 | prototype_retrieval_sample | 0.548 | 16.407 | 0.033 | 0.848 | 29.919 | never | 0.093 |
| 80 | validation_selector | 0.484 | 26.568 | 0.018 | 1.373 | 54.907 | never | 0.505 |
| 160 | class_balanced_sample | 0.817 | 1.179 | 0.693 | 0.107 | 1.444 | never | 0.700 |
| 160 | length_curriculum_sample | 0.596 | 2.822 | 0.211 | 0.256 | 4.733 | 6 | 4.626 |
| 160 | prototype_retrieval_sample | 1.046 | 9.391 | 0.111 | 0.853 | 8.979 | never | 0.261 |
| 160 | validation_selector | 0.670 | 14.770 | 0.045 | 1.342 | 22.054 | never | 0.774 |

## Interpretation

- `density_ratio > 1` means the candidate beats random under learning-signal density.
- `quality_multiplier > 1` means the candidate improves heldout accuracy gain over random before cost is charged.
- `event_compute_multiplier` measures the full external-events times charged-compute bill.
- `compute_over_break_even` is the factor by which charged compute exceeds what the observed quality gain can afford.
- `amortized_reuses_to_density_win` is the minimum independent downstream reuse count needed if selector and validation costs are reusable.
- `fully_amortized_density_ratio` is the limiting ratio after the reusable cost is spread over infinitely many uses.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "metadata_stripped": true,
  "paper_ready_claim": false,
  "post_hoc_diagnostic": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
