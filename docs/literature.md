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
