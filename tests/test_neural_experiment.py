import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.neural_experiment import run_neural_condition, run_neural_seedset


class NeuralExperimentArtifactTests(unittest.TestCase):
    def test_neural_seedset_writes_scoped_artifact_with_neural_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "neural.json"
            out_md = Path(temp_dir) / "neural.md"
            result = run_neural_seedset(
                seeds=[3],
                conditions=["raw_text", "sample_aware_self_ranked_induction"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=4,
                hidden_units=8,
                feature_dimension=64,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(saved["claim_scope"]["neural_model"], True)
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(saved["learner_backend"], "tiny_mlp")
            self.assertIn("sample_aware_self_ranked_induction", saved["conditions"])
            self.assertGreater(saved["conditions"]["raw_text"]["neural_parameter_count_mean"], 0)
            self.assertGreater(saved["conditions"]["raw_text"]["neural_training_step_count_mean"], 0)
            self.assertGreater(saved["conditions"]["raw_text"]["estimated_neural_training_multiply_adds_mean"], 0)
            self.assertEqual(result["feature_dimension"], 64)
            self.assertIn("Tiny Neural Replication", out_md.read_text())

    def test_neural_seedset_marks_fresh_seed_confirmation_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "neural_confirmation.json"
            out_md = Path(temp_dir) / "neural_confirmation.md"
            run_neural_seedset(
                seeds=[17],
                conditions=["raw_text", "sample_aware_self_ranked_induction"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=4,
                hidden_units=8,
                feature_dimension=64,
                fresh_seed_confirmation=True,
                confirmation_of="results/tiny_neural_replication.json",
                target_signed_gain=0.03,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(saved["title"], "Learning Signal Density Tiny Neural Confirmation")
            self.assertEqual(saved["claim_scope"]["fresh_seed_confirmation"], True)
            self.assertEqual(saved["confirmation_of"], "results/tiny_neural_replication.json")
            self.assertEqual(saved["target_signed_gain"], 0.03)
            self.assertEqual(saved["confirmation"]["sample_aware_self_ranked_induction"]["reaches_target"], False)
            markdown = out_md.read_text()
            self.assertIn("Fresh-seed confirmation", markdown)
            self.assertIn("results/tiny_neural_replication.json", markdown)

    def test_validation_gated_conditions_fail_until_neural_gate_is_implemented(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a neural validation gate"):
            run_neural_condition(
                seed=3,
                condition="validation_gated_induction",
                material_count=16,
                epochs=2,
                hidden_units=8,
                feature_dimension=64,
                learning_rate=0.03,
            )


if __name__ == "__main__":
    unittest.main()
