import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.neural_feature_sweep import run_neural_feature_sweep


class NeuralFeatureSweepTests(unittest.TestCase):
    def test_neural_feature_sweep_writes_feature_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "neural_feature_sweep.json"
            out_md = Path(temp_dir) / "neural_feature_sweep.md"

            run_neural_feature_sweep(
                feature_dimensions=[16, 32],
                seeds=[3],
                conditions=["raw_text", "sample_aware_self_ranked_induction"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=2,
                hidden_units=4,
                learning_rate=0.03,
                target_signed_gain=0.03,
                profile_label="epochs=2_hidden=4",
                comparison_of="results/baseline_feature_profile.json",
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(saved["title"], "Learning Signal Density Tiny Neural Feature-Dimension Sweep")
            self.assertEqual(saved["profile_label"], "epochs=2_hidden=4")
            self.assertEqual(saved["comparison_of"], "results/baseline_feature_profile.json")
            self.assertEqual(saved["feature_dimensions"], [16, 32])
            self.assertIn("features=16", saved["dimension_results"])
            self.assertIn("sample_aware_self_ranked_induction", saved["frontier"])
            summary = saved["frontier"]["raw_text"]
            self.assertIn("best_signed_gain_feature_dimension", summary)
            self.assertIn("lowest_neural_ops_reaching_target_feature_dimension", summary)
            raw_dimension = saved["dimension_results"]["features=16"]["conditions"]["raw_text"]
            self.assertGreater(raw_dimension["estimated_neural_training_multiply_adds_mean"], 0)
            markdown = out_md.read_text()
            self.assertIn("Tiny Neural Feature-Dimension Sweep", markdown)
            self.assertIn("Comparison target: `results/baseline_feature_profile.json`", markdown)


if __name__ == "__main__":
    unittest.main()
