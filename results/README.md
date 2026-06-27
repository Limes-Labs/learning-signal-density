# Results

This directory stores checked-in result artifacts that can be regenerated from
the repository.

Current canonical pilot:

```bash
python3 -m learning_signal_density \
  --output-json results/causal_world_pilot_seedset.json \
  --output-md results/causal_world_pilot_seedset.md \
  --seeds 3 5 7 11 13 \
  --material-count 48 \
  --epochs 5
```

Current sample-budget sweep:

```bash
python3 -m learning_signal_density.sweep \
  --output-json results/sample_budget_sweep.json \
  --output-md results/sample_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 3 5 7 11 13 \
  --epochs 5 \
  --target-signed-gain 0.03
```

Fresh-seed confirmation sweep:

```bash
python3 -m learning_signal_density.sweep \
  --output-json results/confirmation_budget_sweep.json \
  --output-md results/confirmation_budget_sweep.md \
  --material-counts 16 24 32 48 64 \
  --seeds 17 19 23 29 31 \
  --epochs 5 \
  --target-signed-gain 0.03 \
  --fresh-seed-confirmation \
  --confirmation-of results/sample_budget_sweep.json
```

Do not edit generated result JSON by hand. If the code changes, regenerate the
artifact and rerun tests.
