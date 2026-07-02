# Twenty Newsgroups Class-Balanced Confirmation Audit

Generated: `2026-07-02T10:12:07Z`

Confirmation seeds: `[2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141]`

| Candidate | Reference | Budget | Mean ratio | Paired wins | Delta CI95 | Sign p(win) | Robust win? |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| class_balanced_sample | random_sample | 80 | 1.067 | 10/20 | [-0.002197, 0.003466] | 0.588099 | False |

## Interpretation

- This audit is a targeted confirmation, not a new selector search.
- It tests whether the most plausible positive Newsgroups frontier survives a larger untouched seed block.
- Robust density win under the confirmation rule: False.
- The candidate wins 10 of 20 paired seeds with one-sided sign-test p=0.588099.
