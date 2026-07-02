# Paper Build And Release Readiness

The canonical manuscript source is `paper/main.tex`.

The current paper is a working draft, not a versioned public paper release. The
tables in `paper/generated/results_tables.tex` are generated from committed
JSON artifacts and should not be edited by hand.

## Regenerate Paper Tables

```bash
python3 scripts/build_paper_tables.py
```

The generator currently reads:

- `results/sms_spam_real_text_selection_cost.json`
- `results/sms_spam_real_text_selection_cost_v200.json`
- `results/sms_spam_break_even_analysis.json`
- `results/twenty_newsgroups_active_selection.json`
- `results/twenty_newsgroups_break_even_analysis.json`
- `results/twenty_newsgroups_retrieval_cost_audit.json`
- `results/twenty_newsgroups_self_training_audit.json`
- `results/twenty_newsgroups_active_acquisition_audit.json`
- `results/twenty_newsgroups_budgeted_acquisition_audit.json`
- `results/twenty_newsgroups_length_window_confirmation_audit.json`
- `results/twenty_newsgroups_frontier_robustness_audit.json`
- `results/twenty_newsgroups_frontier_fresh_seed_audit.json`
- `results/real_text_break_even_certificate.json`
- `results/tiny_neural_feature_sweep_wide.json`
- `results/tiny_neural_profile_sweep_f1024.json`
- `results/tiny_neural_budget_sweep_32x8_f1024.json`
- `results/tiny_neural_budget_sweep_16x8_f1024.json`
- `results/tiny_neural_budget_sweep_8x8_f1024.json`
- `results/tiny_neural_budget_sweep_validation_selected_f1024.json`
- `results/tiny_neural_budget_sweep_agreement_gated_f1024.json`
- `results/policy_envelope_f1024.json`
- `results/tiny_neural_budget_sweep_validation_portfolio_f1024.json`
- `results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json`
- `results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json`
- `results/tiny_neural_budget_sweep_selector_transfer_f1024.json`
- `results/generated_label_audit_selector_transfer_f1024.json`
- `results/generated_coverage_audit_selector_transfer_f1024.json`
- `results/tiny_neural_budget_sweep_train_size_gated_f1024.json`
- `results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json`
- `results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.json`
- `results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json`
- `results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json`
- `results/tiny_neural_budget_sweep_diversity_interaction_f1024.json`
- `results/tiny_neural_budget_sweep_density_capped_compact_f1024.json`
- `results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json`
- `results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json`
- `results/tiny_neural_budget_sweep_density_window_compact_f1024.json`
- `results/tiny_neural_budget_sweep_train_support_density_f1024.json`
- `results/tiny_neural_budget_sweep_support_probe_window_f1024.json`
- `results/tiny_neural_budget_sweep_validation_support_precision_f1024.json`
- `results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json`
- `results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json`

It refuses artifacts whose claim scope says heldout data was used for selection,
the artifact is not a fresh-seed confirmation, or the artifact is already marked
as paper-ready. That keeps the manuscript tied to the current evidence level.
The policy-envelope artifact is the one deliberate exception: it must disclose
heldout policy selection, mark itself as post-hoc and non-deployable, and
exclude the oracle condition from its non-oracle envelope.
The validation-portfolio artifact must disclose validation policy selection,
avoid heldout selection, and charge the candidate portfolio search. The
linear-proxy selector artifact must additionally mark the low-fidelity proxy
selector scope and charge the proxy fits before final heldout evaluation.
The abstaining-proxy selector artifact must also mark raw-text abstention and
record the validation-example margin used before leaving raw text.
The selector-transfer artifact must use fresh seeds and keep the same
heldout-isolation flags, so selector claims are checked beyond the development
selector artifacts.
The generated-label audit must disclose hidden-rulebook audit labels, mark the
hidden rulebook unavailable to policies, keep heldout out of selection, and
remain non-deployable.
The generated-coverage audit must disclose heldout-distribution audit use, mark
that distribution unavailable to policies, keep heldout out of selection, and
remain non-deployable.
The train-size gated artifact must use another unseen seed set, mark the
train-only schedule scope, and keep validation and heldout out of policy
selection.
The validation-coverage proxy artifact must use fresh seeds, disclose
validation motif distribution as the policy-selection signal, keep heldout
distribution out of selection, and mark that validation labels are not used for
the selector score.
The coverage-prior selector artifact must use fresh seeds, disclose the
validation motif distribution, train-size prior, lean candidate set, and
compute penalty used for policy selection, keep heldout distribution out of
selection, and mark that validation labels are not used for the selector score.
The tempered sample-aware artifact must use fresh seeds, mark train-only
selection and induction, and keep validation, heldout, and oracle labels out of
the synthetic-budget policy.
The compact train-size gated artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
and disclose that original QA duplicates are dropped only at the large-sample
tier.
The diversity-interaction artifact must use fresh seeds, mark train-only
selection and induction for both sample-aware diversity and compact diversity,
keep validation and heldout out of policy selection, and disclose that compact
diversity applies its diversity penalty only after the compact large-sample
tier.
The density-capped compact artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
and disclose both the compact large-sample tier and abundant-data raw fallback.
The support-ramped compact artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
and disclose both the compact large-sample tier and abundant-data support
ramp.
The late-confidence compact artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
and disclose both the abundant-data support ramp and the later confidence
floor.
The density-window compact artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
and disclose the fixed compact, raw, support-ramped, and abundant raw fallback
windows.
The train-support-density artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
disclose the support-density threshold, and charge candidate inspection before
the selected final fit.
The support-probe window artifact must use fresh seeds, mark train-only
selection and induction, keep validation and heldout out of policy selection,
disclose the support-probe train-event window, and mark selected-candidate
construction reuse.
The validation support-precision artifact must use fresh seeds, disclose
validation-based policy and transform selection, keep heldout out of selection,
disclose the fixed precision threshold and transition window, and mark
selected-candidate construction reuse.
The no-window validation support-precision gate artifact must use fresh seeds,
disclose validation-based policy and transform selection, keep heldout out of
selection, mark that the fixed transition prior was removed, and mark
selected-candidate construction reuse.
The support-selector transfer artifact must use the fresh transfer seed block,
disclose the same selector scopes, keep heldout out of selection, and identify
the no-window validation gate artifact it stress-tests.
The real-text SMS Spam artifacts must identify the UCI dataset, record the
checked SHA-256 and CC BY 4.0 license, mark `synthetic_domain=false`, keep
heldout out of selection, disclose label-index selection cost, and keep
`paper_ready_claim=false`.
The SMS break-even artifact must analyze those committed real-text artifacts,
use random sampling as the reference condition, state the selector break-even
inequality, mark itself as a post-hoc diagnostic, and preserve the current
finding that no tested non-random SMS selector beats random on density. It must
also record the configured quality upper bound used for the perfect-quality
feasibility check and the reusable-cost keys used for selector-cost
amortization.
The Twenty Newsgroups artifact must use the UCI mini corpus, mark itself as
real and non-synthetic, strip metadata/quotes before splitting, keep heldout
closed for selection, and include random, class-balanced, curriculum,
prototype-retrieval, and validation-selector conditions.
The Twenty Newsgroups break-even artifact must analyze that committed active
selection artifact, use random sampling as the reference, state the same
break-even inequality, preserve the class-balanced 80-document density win, and
preserve the finding that prototype retrieval can improve quality without
beating random density.
The Twenty Newsgroups retrieval-cost audit must disclose that it is a post-hoc
optimization attempt, sweep the committed length-penalty grid, keep heldout
closed, and preserve the finding that length penalties improve some
retrieval-family rows without producing a random-density win.
The Twenty Newsgroups self-training audit must use teacher predictions as
pseudo-labels, keep oracle train labels diagnostic-only, keep heldout closed,
and preserve the finding that pseudo-label agreement is too low for the tested
margin filters to beat random or class-balanced density.
The Twenty Newsgroups active-acquisition audit must use a charged
class-balanced seed, select unlabeled train-pool records without validation or
heldout access, acquire true labels only after selection, and preserve the
finding that no tested acquisition mode beats random or class-balanced density
without explicit reuse.
The Twenty Newsgroups budgeted-window active-acquisition audit must sample the
scan window before teacher scoring, use the full label budget, acquire true
labels only after selection, preserve the no-random-density-win finding, and
preserve the 160-label class-balanced density wins for margin uncertainty.
The Twenty Newsgroups sampled length-window audit must use disjoint development
and confirmation seeds, keep selection label-free and teacher-free, preserve
the no-development-density-win result, and preserve the 80-label confirmation
failure for the development-selected policy.
The Twenty Newsgroups frontier robustness audit must introduce no new policy,
use paired seed rows from the committed Newsgroups artifacts, preserve three
fragile mean density wins, zero robust density wins, and three robust density
losses under the exact seed-bootstrap rule.
The Twenty Newsgroups fresh-seed frontier audit must rerun only the frontier
comparisons on untouched seeds 439, 443, 449, 457, and 461, preserve zero robust
density wins, two robust density losses, and one fragile mean density win under
the exact seed-bootstrap rule.
The real-text break-even certificate must introduce no new policy, certify the
committed SMS Spam and Twenty Newsgroups break-even artifacts, preserve the
172-row comparison set, and keep the current separation between 38 observed
quality wins, three observed density wins, and 13 finite-reuse frontiers.

## Release Checklist

Mirror the stronger Limes Labs release pattern only after the evidence supports
it:

- Rendered author PDF from `paper/main.tex`.
- LaTeX/source bundle containing `paper/`, generated TeX inputs, bibliography,
  scripts, and the exact JSON artifacts used by the paper.
- Full source archive at the release tag.
- Result manifest with SHA-256 checksums and PDF page-count metadata.
- Artifact README with rebuild and verification commands.
- Submission metadata for Zenodo, OSF, Figshare, OpenReview, or other target
  venue.
- Fresh verification log for tests, smoke experiment, paper table generation,
  LaTeX build, and `git diff --check`.

## Non-Release State

Do not tag a paper release until the repo has at least one stronger replication
than the current tiny-MLP evidence. The current draft is useful for shaping the
argument and keeping claims artifact-backed, but it is not yet a final paper
claim about language models or frontier systems.
