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
- `results/tiny_neural_budget_sweep_train_size_gated_f1024.json`

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
The train-size gated artifact must use another unseen seed set, mark the
train-only schedule scope, and keep validation and heldout out of policy
selection.

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
