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

Do not edit generated result JSON by hand. If the code changes, regenerate the
artifact and rerun tests.
