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
and does not use heldout labels for selection.
