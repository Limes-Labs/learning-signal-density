#!/usr/bin/env bash
set -euo pipefail

python3 -m unittest discover -s tests
python3 -m py_compile learning_signal_density/*.py
python3 -m learning_signal_density \
  --output-json results/causal_world_pilot_seedset.json \
  --output-md results/causal_world_pilot_seedset.md \
  --seeds 3 5 7 11 13 \
  --material-count 48 \
  --epochs 5
git diff --check
