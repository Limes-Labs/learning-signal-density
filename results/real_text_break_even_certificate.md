# Real-Text Break-Even Frontier Certificate

Generated: `2026-07-01T23:53:42Z`

This is a mathematical certificate over committed real-text result artifacts.
It introduces no new policy; it classifies whether observed quality gains clear the charged event-compute break-even condition.

Break-even condition:

```text
G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)
```

| Family | Rows | Quality wins | Density wins | Q-win density losses | Finite reuse | Nonreusable too high | K=1 bound impossible | Mean density ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sms_spam_selection | 40 | 26 | 0 | 26 | 8 | 1 | 39 | 0.287 |
| twenty_newsgroups_active_selection | 12 | 3 | 1 | 3 | 0 | 1 | 4 | 0.251 |
| twenty_newsgroups_retrieval_alpha | 18 | 3 | 0 | 3 | 1 | 1 | 6 | 0.067 |
| twenty_newsgroups_self_training | 24 | 0 | 0 | 0 | 0 | 0 | 3 | 0.048 |

## Frontier Witnesses

- Strongest observed density win: `Twenty Newsgroups active b=80 class_balanced_sample vs random_sample`.
- Largest quality win without density win: `SMS Spam v800 b=32 validation_label_index_selector vs random_sample`.
- Cheapest finite reuse frontier: `SMS Spam v800 b=32 label_index_balanced_sample vs random_sample`.
- Worst cost over break-even among quality wins: `Twenty Newsgroups active b=40 validation_selector vs random_sample`.

## Scope Flags

```json
{
  "heldout_used_for_selection": false,
  "introduces_new_policy": false,
  "mathematical_certificate": true,
  "paper_ready_claim": false,
  "post_hoc_diagnostic": true,
  "real_dataset": true,
  "synthetic_domain": false
}
```
