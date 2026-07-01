# SMS Spam Break-Even Selection-Cost Analysis

Generated: `2026-07-01T22:37:45Z`

This artifact is a mathematical audit over committed SMS Spam result JSON.
It does not introduce a new policy. It asks whether each non-random policy's quality gain is large enough to pay for its charged event-compute multiplier.

Break-even condition:

```text
G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)
```

## SMS Spam v800

| Budget | Candidate | Quality mult. | Event-compute mult. | Density ratio | Max possible density | Break-even quality | Compute over break-even | Reuses to win | Fully amortized ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | label_index_balanced_sample | 1.348 | 2.565 | 0.526 | 0.956 | 1.046 | 1.902 | 9 | 1.136 |
| 32 | class_balanced_sample | 1.157 | 23.275 | 0.050 | 0.105 | 9.492 | 20.120 | never | 0.967 |
| 32 | validation_label_index_selector | 1.426 | 13.297 | 0.107 | 0.184 | 5.423 | 9.323 | 47 | 1.226 |
| 32 | validation_sample_selector | 1.426 | 33.999 | 0.042 | 0.072 | 13.866 | 23.838 | 125 | 1.226 |
| 64 | label_index_balanced_sample | 1.215 | 1.964 | 0.619 | 1.041 | 0.961 | 1.616 | never | 0.980 |
| 64 | class_balanced_sample | 1.418 | 12.868 | 0.110 | 0.159 | 6.297 | 9.074 | 79 | 1.117 |
| 64 | validation_label_index_selector | 1.104 | 7.958 | 0.139 | 0.257 | 3.894 | 7.207 | never | 0.981 |
| 64 | validation_sample_selector | 1.104 | 18.833 | 0.059 | 0.109 | 9.216 | 17.056 | never | 0.981 |
| 128 | label_index_balanced_sample | 1.002 | 1.617 | 0.620 | 0.860 | 1.162 | 1.614 | never | 0.797 |
| 128 | class_balanced_sample | 1.009 | 7.063 | 0.143 | 0.197 | 5.076 | 6.999 | never | 0.784 |
| 128 | validation_label_index_selector | 1.032 | 4.905 | 0.210 | 0.284 | 3.525 | 4.753 | never | 0.995 |
| 128 | validation_sample_selector | 1.032 | 10.321 | 0.100 | 0.135 | 7.417 | 10.001 | never | 0.995 |
| 256 | label_index_balanced_sample | 0.902 | 1.421 | 0.635 | 0.872 | 1.147 | 1.575 | never | 0.726 |
| 256 | class_balanced_sample | 0.954 | 4.100 | 0.233 | 0.302 | 3.311 | 4.297 | never | 0.764 |
| 256 | validation_label_index_selector | 0.968 | 3.446 | 0.281 | 0.359 | 2.783 | 3.562 | never | 0.900 |
| 256 | validation_sample_selector | 0.968 | 6.120 | 0.158 | 0.202 | 4.942 | 6.325 | never | 0.900 |
| 512 | label_index_balanced_sample | 0.951 | 1.344 | 0.708 | 0.890 | 1.123 | 1.413 | never | 0.757 |
| 512 | class_balanced_sample | 1.008 | 2.658 | 0.379 | 0.450 | 2.221 | 2.637 | never | 0.810 |
| 512 | validation_label_index_selector | 1.027 | 2.631 | 0.390 | 0.455 | 2.199 | 2.561 | 82 | 1.020 |
| 512 | validation_sample_selector | 1.027 | 3.957 | 0.260 | 0.302 | 3.307 | 3.852 | 148 | 1.020 |

## SMS Spam v200

| Budget | Candidate | Quality mult. | Event-compute mult. | Density ratio | Max possible density | Break-even quality | Compute over break-even | Reuses to win | Fully amortized ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | label_index_balanced_sample | 1.175 | 3.004 | 0.391 | 0.784 | 1.276 | 2.556 | never | 0.932 |
| 32 | class_balanced_sample | 1.143 | 29.308 | 0.039 | 0.080 | 12.453 | 25.643 | never | 0.834 |
| 32 | validation_label_index_selector | 1.234 | 6.692 | 0.184 | 0.352 | 2.844 | 5.425 | never | 0.934 |
| 32 | validation_sample_selector | 1.234 | 32.888 | 0.038 | 0.072 | 13.974 | 26.662 | never | 0.934 |
| 64 | label_index_balanced_sample | 1.057 | 2.184 | 0.484 | 0.763 | 1.311 | 2.066 | never | 0.810 |
| 64 | class_balanced_sample | 1.075 | 15.465 | 0.069 | 0.108 | 9.286 | 14.391 | never | 0.785 |
| 64 | validation_label_index_selector | 0.982 | 4.339 | 0.226 | 0.384 | 2.605 | 4.420 | never | 0.840 |
| 64 | validation_sample_selector | 0.982 | 17.556 | 0.056 | 0.095 | 10.541 | 17.882 | never | 0.840 |
| 128 | label_index_balanced_sample | 0.865 | 1.690 | 0.512 | 0.835 | 1.198 | 1.953 | never | 0.684 |
| 128 | class_balanced_sample | 0.873 | 8.134 | 0.107 | 0.173 | 5.767 | 9.321 | never | 0.670 |
| 128 | validation_label_index_selector | 1.025 | 3.046 | 0.337 | 0.463 | 2.159 | 2.971 | never | 0.988 |
| 128 | validation_sample_selector | 1.025 | 9.452 | 0.108 | 0.149 | 6.701 | 9.219 | never | 0.988 |
| 256 | label_index_balanced_sample | 1.023 | 1.479 | 0.691 | 0.871 | 1.148 | 1.446 | never | 0.806 |
| 256 | class_balanced_sample | 0.899 | 4.680 | 0.192 | 0.275 | 3.633 | 5.205 | never | 0.690 |
| 256 | validation_label_index_selector | 1.038 | 2.482 | 0.418 | 0.519 | 1.927 | 2.392 | 161 | 1.009 |
| 256 | validation_sample_selector | 1.038 | 5.649 | 0.184 | 0.228 | 4.385 | 5.445 | 509 | 1.009 |
| 512 | label_index_balanced_sample | 0.933 | 1.345 | 0.693 | 0.878 | 1.140 | 1.442 | never | 0.750 |
| 512 | class_balanced_sample | 0.921 | 2.885 | 0.319 | 0.409 | 2.444 | 3.134 | never | 0.739 |
| 512 | validation_label_index_selector | 0.946 | 2.251 | 0.420 | 0.524 | 1.907 | 2.380 | never | 0.863 |
| 512 | validation_sample_selector | 0.946 | 3.789 | 0.250 | 0.312 | 3.210 | 4.006 | never | 0.863 |

## Interpretation

- `density_ratio > 1` means the candidate beats random under learning-signal density.
- `break_even_quality >= quality_upper_bound` means the candidate could not strictly beat random even with perfect spam F1 under the charged cost model.
- `max_possible_density_ratio` is the best density ratio reachable at the configured quality upper bound.
- `compute_over_break_even` is the factor by which charged compute exceeds what the observed quality gain can afford.
- `amortized_reuses_to_density_win` is the minimum number of independent downstream uses needed if selector and validation-tuning costs are reusable.
- `fully_amortized_density_ratio` is the limiting ratio after the reusable cost is spread over infinitely many uses.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "paper_ready_claim": false,
  "post_hoc_diagnostic": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
