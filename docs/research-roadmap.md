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
confirmation, a neural sample-budget sweep, a profile-efficiency sweep, and an
efficient-profile budget confirmation. It now also includes a feature-dimension
frontier plus 256-feature and 1024-feature budget reruns for the efficient
`32x8` profile, followed by a 1024-feature epoch/width profile sweep and a
`16x8` budget confirmation. A further `8x8` ablation tests whether lower
training budget can reduce scarce-sample overfitting, and a charged
validation-selected rerun tests whether validation-ranked or MDL rule-selection
policies repair the low-budget generated-label failure. A train-only
agreement-gated rerun tests whether independent induced-rule projections are a
cheap enough reliability signal. A post-hoc non-oracle policy envelope now
quantifies the remaining selector problem while explicitly marking itself
non-deployable because it uses completed heldout results. A validation-portfolio
selector probe tests the deployable version of that idea by training six
non-oracle candidate policies, choosing on validation, charging the full search,
and then evaluating heldout once. A linear-proxy selector probe tests a cheaper
model hierarchy by scoring the same candidates with two-epoch linear fits,
charging that search, training only one final tiny MLP, and then evaluating
heldout once. An abstaining-proxy selector adds a raw-text fallback that
requires three extra validation-correct examples before selecting a non-raw
policy. A fresh-seed selector-transfer stress test reruns the selector family
on seeds `37 41 43 47 53` to check whether the development selector result
survives new worlds. A train-size gated schedule baseline then uses a second
unseen seed set, `59 61 67 71 73`, to test whether a cheap train-only switch
from raw text to sample-aware induction is a harder baseline than validation
portfolio selection. A hidden-rulebook generated-label audit then checks the
selector-transfer synthetic labels after the fact and shows that label precision
alone does not explain the remaining neural-gain bottleneck. A
heldout-distribution generated-coverage audit then tests whether the motif
distribution of generated labels explains part of the remaining gap. A
validation-coverage proxy now turns that mechanism diagnostic into a deployable
selector probe on fresh seeds `103 107 109 113 127`, using validation motif
distribution rather than heldout distribution or validation labels for the
selector score. A tempered sample-aware ablation then tests whether lowering
the mid-budget train-only synthetic ratio from `0.75` to `0.50` repairs the
same 24/32-material failure without validation selection.
A compact train-size gated efficiency probe then keeps the same schedule but
drops original QA duplicates at the large-sample tier, testing whether density
can improve by removing redundant transformed originals instead of changing
generated labels.
A density-capped compact probe then extends the budget range through 128
materials and tests whether the policy should return to raw text once external
evidence is abundant enough that generated-label transforms no longer pay back
their charged compute.
A support-ramped compact probe then tests the middle of that high-budget
frontier, keeping compact induction but raising induced-label minimum support
from `3` to `4` after the abundant-data tier.

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
- Treat profile selection as part of the compute frontier. The current
  64-material profile sweep shows that `32x8` beats `32x32` for self-ranked
  induction with substantially lower neural training operations.
- Confirm profile choices across the full sample-budget frontier. The current
  `32x8` rerun preserves the target-reaching material counts from the original
  `32x32` sweep, slightly improves best signed gains for the positive
  conditions, and uses one quarter of the estimated neural training
  multiply-adds at matched budgets.
- Treat feature hashing dimension as a separate capacity and collision frontier.
  The current `32x8` feature sweep rejects 16, 32, and 64 features for ranked
  train-only conditions at 64 materials, while 256 features improves the
  128-feature frontier with only a small neural-op increase.
- Continue widening the feature map until the sparse hash frontier visibly
  saturates. The current 1024-feature sweep improves the 256-feature frontier
  with only a small additional operation cost, but still leaves train-only
  ranked methods negative at 24 and 32 materials.
- Re-check profile efficiency whenever the representation changes. At 1024
  features, `16x8` becomes the strongest self-ranked profile at the
  64-material point and halves neural training operations versus `32x8`, but
  weakens raw text, QA, and oracle best gain in the full budget rerun.
- Treat the 24/32-material ranked failure as a transform-policy problem, not
  only an optimizer-size problem. The `8x8` ablation improves the smallest
  sample-aware budgets, but loses the self-ranked target entirely and does not
  repair the 32-material failure.
- Treat charged validation selection as informative but not solved. The
  validation-ranked rerun reaches target only at 48 materials, like the
  train-only ranked policies. MDL rule expansion is less negative at 32
  materials but still below target and lower at peak gain, so the next policy
  needs better scarce-sample reliability without excessive selection cost.
- Treat source agreement alone as insufficient. The agreement-gated train-only
  probe reaches target only at 48 materials, is worse than sample-aware ranking
  at 16 materials, and has much lower peak signed gain. Independent projection
  agreement is therefore not enough without better uncertainty calibration or
  coverage control.
- Treat policy selection as the next bottleneck. The post-hoc non-oracle
  envelope switches from MDL at 16 materials to raw text at 24, MDL at 32,
  validation-ranked induction at 48, and self-ranked induction at 64. Because
  this selection uses heldout results after the fact, it is not deployable, but
  it defines the target for a preregistered adaptive selector using only
  train/validation signals.
- Treat naive validation portfolios as too expensive. The first deployable
  selector matches the envelope at 16 materials but stays negative at 24 and
  32, peaks below fixed self-ranked induction, and has much lower signed LSD
  because every candidate training is charged. The next selector should learn a
  cheaper abstain/compress/rank rule instead of brute-force trying every policy.
- Treat low-fidelity selector hierarchy as partial progress, not the answer.
  The linear-proxy selector improves over the full validation portfolio at 48
  and 64 materials and raises 64-material signed LSD from `0.000627` to
  `0.002091`, but it is still negative at 16, 24, and 32 materials and remains
  less density-efficient than fixed self-ranked or sample-aware self-ranked
  induction. The next selector needs a cheaper pre-filter or abstention signal
  before paying for even proxy fits.
- Treat raw-text abstention as downside control, not a frontier solution. The
  abstaining-proxy selector moves 16 materials from `-0.105` to `0.000` and 24
  materials from `-0.041` to `0.006897`, but it still fails at 32 materials and
  lowers the 48-material gain from `0.103` to `0.082759`. The next experiment
  should search for a reliability feature that distinguishes the 32-material
  MDL/raw/self-ranked cases before heldout, not merely raise the abstention
  threshold.
- Treat selector transfer as the new promotion gate. On fresh seeds `37 41 43
  47 53`, fixed sample-aware self-ranked induction beats the deployable
  selector family at 64 materials (`0.142857` versus `0.114286` for the proxy
  selectors), and raw text is less negative than the selector family at 32. A
  future adaptive selector should not be promoted unless it improves both the
  development artifact and this fresh-seed transfer stress test.
- Treat generated-label precision as necessary but not sufficient. The
  hidden-rulebook audit finds `0.917241` precision for agreement-gated labels
  at 32 materials, yet the linked neural gain is `-0.131579`; at 64 materials,
  agreement-gated precision is higher than sample-aware precision
  (`0.954783` versus `0.829565`) while gain is lower (`0.080519` versus
  `0.142857`). Future policies must measure coverage, distribution, and learner
  interaction rather than optimizing label precision alone.
- Treat generated-label coverage as the next mechanism target. On the
  selector-transfer seeds, sample-aware induction has lower 64-material
  generated-versus-heldout motif L1 distance than agreement-gated induction
  (`0.477843` versus `0.576443`) and higher gain, while validation-ranked
  induction has the lowest 32-material distance among non-oracle generated
  policies (`0.683666`) and is least negative. The next experiment should
  search for a deployable train/validation proxy for heldout motif coverage.
- Treat validation motif coverage as useful but not sufficient. The fresh-seed
  validation-coverage proxy turns 32 materials positive (`0.010526`) and
  reaches `0.171428` at 64 materials, but it selects MDL rule expansion at
  every 24-material seed and drops to `-0.082759`. The next selector should
  combine motif coverage with a low-budget abstention floor or train-size
  prior before paying for generated-label policies.
- Treat synthetic-budget tempering as partial damage control. On fresh seeds
  `157 163 167 173 179`, lowering the mid-budget sample-aware ratio from
  `0.75` to `0.50` improves 24-material gain from `-0.137931` to `-0.096552`
  and 32-material gain from `-0.152632` to `-0.078947`, while preserving the
  high-budget gains. It still loses to the train-size raw fallback at 24 and
  32, so the next policy needs a principled abstention signal, not only a
  smaller generated-label budget.
- Treat the train-size gated schedule as the new cheap-selector baseline. On
  unseen seeds `59 61 67 71 73`, raw text below 144 train events plus
  sample-aware induction above that threshold reaches the same `0.145454` best
  gain and `0.005090` best signed LSD as fixed sample-aware induction, without
  validation selector search. It remains negative at 16 and 32 materials, so it
  is not a solution; it is the baseline a future adaptive selector must beat
  after charging its selection cost.
- Treat representation cost as part of the policy. On fresh seeds
  `181 191 193 197 199`, compact train-size gating matches the train-size gate
  through 48 materials, then improves the 64-material row from `0.132467` gain
  and `0.004634` signed LSD to `0.140260` gain and `0.007883` signed LSD by
  keeping raw originals but dropping original QA duplicates at the large-sample
  tier.
- Treat abundant-data raw fallback as a density frontier candidate. On fresh
  seeds `293 307 311 313 317`, density-capped compact induction matches
  compact through 96 materials, then returns to raw text from 104 onward. At
  128 materials it gives up gain (`0.132468` to `0.081818`) but improves signed
  LSD (`0.001860` to `0.003452`), so future policies should optimize a
  gain-density Pareto frontier rather than only absolute heldout gain.
- Treat abundant-data support floors as another Pareto axis, not a full
  replacement for raw fallback. On fresh seeds `401 409 419 421 431`,
  support-ramped compact induction matches compact through 96 materials, then
  raises induced-label support to `4`. At 104 materials it improves signed LSD
  over both compact and raw fallback (`0.003735` versus `0.002857` and
  `0.002145`). At 128 materials it recovers more gain than raw fallback
  (`0.184416` versus `0.154545`), but raw fallback remains denser (`0.006521`
  versus `0.005153`). Future selectors should choose among raw, compact, and
  higher-support compact views according to the target point on the frontier.

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
