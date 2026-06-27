# Research Roadmap

## Phase 1: Controlled Causal-Domain Audit

Status: implemented in this first slice.

- Generate a synthetic causal-text domain with hidden rules.
- Freeze train, validation, and heldout splits by `(material, stimulus)` pair.
- Compare raw, selected, transformed, counterfactual, and replay pipelines.
- Charge internal tokens and transform cost.
- Publish JSON and Markdown result artifacts.
- Add signed metrics and sample-budget sweeps so negative or budget-sensitive
  results stay visible.
- Add validation-gated induction to measure whether spending validation compute
  improves non-oracle transformation quality enough to justify its overhead.
- Add direct validation gating to test whether a cheaper threshold-selection
  proxy preserves most of the gain without paying per-candidate retraining cost.
- Add validation-ranked induction to test whether a fixed-budget subset of
  validation-scored induced counterfactuals improves non-oracle density.
- Add train-calibrated and self-ranked induction ablations to isolate whether
  validation reliability estimates, train-only calibration, or confidence/support
  ranking drive the gain.
- Add sample-aware self-ranked induction to test whether train-size-aware
  synthetic budgets reduce low-sample generated-label noise without validation
  or oracle labels.
- Add a fresh-seed confirmation sweep so post-hoc ranked-policy improvements
  are checked against seed overfitting before being used in the paper.
- Add diverse self-ranked induction to test whether balancing feature coverage
  improves or harms the fixed ranked budget.
- Add MDL-style rule compression to test whether compact transform policies can
  reduce internal examples and make non-oracle transformation more efficient.

## Phase 2: Tiny Neural Replication

Status: started with a deterministic CPU MLP artifact, fresh-seed
confirmation, and a neural sample-budget sweep.

- Start with a dependency-light MLP or small transformer.
- Add a nanoGPT-compatible backend only after the CPU smoke path is stable.
- Keep the exact same split boundary and artifact schema.
- Report parameter count, training steps, estimated training operations,
  wall-clock time, and framework/device metadata.
- Compare oracle counterfactual expansion against train-only induced
  counterfactual expansion, validation-gated induction, and direct validation
  gating, ranked induction variants, and MDL-compressed rules to measure the
  cost of imperfect transformation and transform selection.
- Keep fresh-seed confirmation before promoting any neural gain discovered by
  changing MLP width, epoch count, learning rate, or condition set. The first
  confirmation keeps ranked train-only conditions above target, but does not
  replace a small-transformer or nanoGPT-scale replication.
- Treat the current tiny-MLP signal as budget-dependent: ranked train-only
  conditions first reach the signed-gain target at 48 materials, while oracle
  counterfactual expansion reaches it at 32 materials.

## Phase 3: Continual-Learning Replay

Add three synthetic domains trained sequentially:

1. Fictional materials.
2. Fictional biology.
3. Fictional legal rules.

Compare no replay, uniform replay, prioritized replay, generated replay, and
retrieval memory. Report stability and plasticity separately.

## Phase 4: Dense Feedback

Add decomposable tasks with step labels. Compare outcome-only labels with
process labels and noisy process labels. The result should include verifier
cost and reward-hacking or false-positive rates where measurable.

## Phase 5: Paper

The paper should not claim that the first pilot proves anything about frontier
LLMs. It should build toward a stronger claim through staged evidence:

- Controlled pilot.
- Tiny neural replication.
- Continual replay study.
- Dense feedback study.
- Optional nanoGPT-scale run if compute is available.

Each stage should preserve negative results and make the external-data versus
internal-compute tradeoff visible.
