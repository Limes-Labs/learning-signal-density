import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.neural_profile_sweep import run_neural_profile_sweep


class NeuralProfileSweepTests(unittest.TestCase):
    def test_neural_profile_sweep_writes_gain_and_efficiency_frontiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "neural_profile_sweep.json"
            out_md = Path(temp_dir) / "neural_profile_sweep.md"

            run_neural_profile_sweep(
                profiles=[
                    {"epochs": 2, "hidden_units": 4},
                    {"epochs": 3, "hidden_units": 4},
                ],
                seeds=[3],
                conditions=["raw_text", "sample_aware_self_ranked_induction"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                feature_dimension=32,
                learning_rate=0.03,
                target_signed_gain=0.03,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(saved["title"], "Learning Signal Density Tiny Neural Profile Sweep")
            self.assertEqual(saved["claim_scope"]["neural_model"], True)
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(len(saved["profiles"]), 2)
            self.assertIn("epochs=2_hidden=4", saved["profile_results"])
            self.assertIn("sample_aware_self_ranked_induction", saved["frontier"])
            summary = saved["frontier"]["raw_text"]
            self.assertIn("best_signed_gain_profile", summary)
            self.assertIn("lowest_neural_ops_reaching_target_profile", summary)
            raw_profile = saved["profile_results"]["epochs=2_hidden=4"]["conditions"]["raw_text"]
            self.assertGreater(raw_profile["estimated_neural_training_multiply_adds_mean"], 0)
            self.assertIn("Tiny Neural Profile Sweep", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
