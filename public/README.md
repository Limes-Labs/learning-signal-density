# Public Paper Artifacts

The canonical paper source is `paper/main.tex`.

No rendered PDF is committed yet. Add one only after the result set is strong
enough to justify a paper-formatted public artifact. The current manuscript
tables are generated from committed JSON by `scripts/build_paper_tables.py` and
stored in `paper/generated/results_tables.tex`. The current generated tables
also include a post-hoc policy envelope from `results/policy_envelope_f1024.json`;
that envelope is explicitly non-deployable and is included only as a selector
diagnostic. The deployable validation-portfolio selector artifact is included
as a negative-control selector experiment: it charges the full candidate search
and does not use heldout labels for selection. The linear-proxy selector
artifact is included as a partial selector-cost improvement: it replaces full
candidate MLP training with charged two-epoch linear proxy fits before one
final heldout evaluation, while still failing at the lowest budgets. The
abstaining-proxy selector artifact is included as a downside-control probe: it
falls back to raw text unless a non-raw candidate clears a three-validation
example margin, improving 16/24-material behavior but still failing at 32
materials. The selector-transfer artifact is included as a fresh-seed stress
test showing that current deployable selectors do not yet transfer strongly
enough for paper-ready claims.
