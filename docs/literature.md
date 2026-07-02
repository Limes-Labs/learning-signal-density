# Literature Map

This page records the first source-backed map for the project. It is not a
complete survey.

## Data Selection And Curation

Rho-1 proposes Selective Language Modeling: a reference model scores pretraining
tokens, and training focuses the loss on tokens aligned with a desired
distribution. The arXiv abstract reports large math-pretraining efficiency gains
in the authors' setup and is a direct prior for the selection axis.

- Lin et al., "Rho-1: Not All Tokens Are What You Need", 2024.
  <https://arxiv.org/abs/2404.07965>

DataComp-LM provides a controlled testbed for language-model dataset curation:
standardized corpora, recipes, model scales, and many downstream evaluations.
It is relevant because this repo should eventually move from a synthetic pilot
to standardized data-selection experiments.

- Li et al., "DataComp-LM: In search of the next generation of training sets
  for language models", 2024. <https://arxiv.org/abs/2406.11794>

LESS selects influential instruction-tuning data using low-rank gradient
similarity. It is relevant because it treats data value as target-dependent
rather than universal.

- Xia et al., "LESS: Selecting Influential Data for Targeted Instruction
  Tuning", 2024. <https://arxiv.org/abs/2402.04333>

MacKay's information-based active data selection is an early mathematical prior
for choosing data by expected informativeness. It is useful here because it
frames selection as a value problem rather than a generic preprocessing step.

- MacKay, "Information-Based Objective Functions for Active Data Selection",
  1992. <https://authors.library.caltech.edu/records/efefp-2j353>

Settles' active-learning survey and later practice-oriented paper provide the
right caution for this repo: active learning can reduce labeled examples, but
real deployments may still fail to reduce total cost once practical overhead is
included.

- Settles, "Active Learning Literature Survey", 2009.
  <https://research.cs.wisc.edu/techreports/2009/TR1648.pdf>
- Settles, "From Theories to Queries: Active Learning in Practice", 2011.
  <https://proceedings.mlr.press/v16/settles11a.html>

Baldridge and Osborne make the reuse caveat concrete for language technology:
training material selected actively for one model can be brittle when reused
with other models, so total annotation cost and reusability have to be measured
instead of assumed.

- Baldridge and Osborne, "Active Learning and the Total Cost of Annotation",
  2004. <https://aclanthology.org/W04-3202/>

Howard's information value theory is the broader decision-analysis ancestor:
information should be valued jointly with the decision it improves and the cost
of acquiring it.

- Howard, "Information Value Theory", 1966.
  <https://ui.adsabs.harvard.edu/abs/1966ITSSC...2...22H/abstract>

## Real-Text Dataset Probe

UCI Twenty Newsgroups is the first broader NLP active-selection pilot in this
repo. It is multi-class topic classification rather than binary filtering, and
it supports active-learning, curriculum, retrieval/prototype, and selector-cost
questions. The experiment strips headers, quote lines, and reply boilerplate
before splitting because metadata leakage is a known practical issue for this
dataset family. The follow-up break-even artifact treats the result as a
mathematical cost audit, not a leaderboard: a retrieval or selector policy must
beat random sampling on heldout-quality multiplier by more than it increases
event-compute multiplier. A second retrieval-cost audit sweeps a length penalty
inside prototype retrieval; it is useful because it shows that cheaper selected
documents help some retrieval rows, but do not remove the dominant full-scan
cost. A self-training audit then tests pseudo-label filtering as a small
distillation analogue; the current teacher's pseudo-label agreement is too low
for confidence-margin filtering to pay for itself.

- "Twenty Newsgroups", UCI Machine Learning Repository.
  <https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups>
- Rennie's 20 Newsgroups page records the common description of the corpus as
  approximately 20,000 documents partitioned across 20 newsgroups.
  <https://qwone.com/~jason/20Newsgroups/>

UCI SMS Spam Collection is the first non-synthetic dataset used in this repo.
It is small enough for stdlib-only CI-style experiments, public, licensed CC BY
4.0, and directly aligned with class-imbalanced binary text classification. The
current use is not the central NLP benchmark and not an SMS-filtering
leaderboard; it is a selection-cost sanity check where spam-class F1 is
measured against charged sampling and validation overhead.

- Almeida and Hidalgo, "SMS Spam Collection", UCI Machine Learning Repository,
  2011. <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>

## Transformation And Synthetic Views

The phi-1 work is a useful reference for the transformation axis because it
combines selected textbook-quality data with synthetically generated textbooks
and exercises.

- Gunasekar et al., "Textbooks Are All You Need", 2023.
  <https://arxiv.org/abs/2306.11644>

TinyStories is relevant because it shows a deliberately constrained synthetic
language domain can be useful for small-model analysis. This project should keep
that same humility: controlled synthetic domains are instruments, not evidence
that broad language competence has been solved.

- Eldan and Li, "TinyStories: How Small Can Language Models Be and Still Speak
  Coherent English?", 2023. <https://arxiv.org/abs/2305.07759>

## Dense Feedback

"Let's Verify Step by Step" compares outcome supervision with process
supervision and reports that process supervision performed better for the MATH
setup studied by the authors. It also motivates active selection of feedback
examples.

- Lightman et al., "Let's Verify Step by Step", 2023.
  <https://arxiv.org/abs/2305.20050>

## Replay And Internal Processing

Prioritized Experience Replay is the direct engineering reference for replaying
important experiences more frequently than uniform replay.

- Schaul et al., "Prioritized Experience Replay", 2015.
  <https://arxiv.org/abs/1511.05952>

DreamerV3 and MuZero motivate the broader claim that agents can trade external
interaction for internal modeling, replay, or imagination. They are not direct
baselines for this first repo slice.

- Hafner et al., "Mastering diverse control tasks through world models", 2025.
  <https://www.nature.com/articles/s41586-025-08744-2>
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with
  a Learned Model", 2019. <https://arxiv.org/abs/1911.08265>

## Current Gap

The open gap this repository targets is not simply "which trick improves a
benchmark." The gap is measurement: compare external observations, internal
tokens, transform cost, replay cost, and heldout improvement in one artifact.
