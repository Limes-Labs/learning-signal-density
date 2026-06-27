# Research Roadmap

## Phase 1: Controlled Causal-Domain Audit

Status: implemented in this first slice.

- Generate a synthetic causal-text domain with hidden rules.
- Freeze train, validation, and heldout splits by `(material, stimulus)` pair.
- Compare raw, selected, transformed, counterfactual, and replay pipelines.
- Charge internal tokens and transform cost.
- Publish JSON and Markdown result artifacts.

## Phase 2: Tiny Neural Replication

Replace the online linear learner with a tiny neural model:

- Start with a dependency-light MLP or small transformer.
- Add a nanoGPT-compatible backend only after the CPU smoke path is stable.
- Keep the exact same split boundary and artifact schema.
- Report wall-clock time and framework/device metadata.

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
