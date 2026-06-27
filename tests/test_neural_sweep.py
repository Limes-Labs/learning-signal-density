import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.neural_sweep import run_neural_budget_sweep


class NeuralBudgetSweepTests(unittest.TestCase):
    def test_neural_budget_sweep_writes_thresholds_and_budgeted_neural_costs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "neural_budget_sweep.json"
            out_md = Path(temp_dir) / "neural_budget_sweep.md"

            run_neural_budget_sweep(
                material_counts=[16, 24],
                seeds=[3],
                conditions=["raw_text", "sample_aware_self_ranked_induction"],
                output_json=out_json,
                output_markdown=out_md,
                epochs=2,
                hidden_units=4,
                feature_dimension=32,
                learning_rate=0.03,
                target_signed_gain=0.03,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(saved["title"], "Learning Signal Density Tiny Neural Budget Sweep")
            self.assertEqual(saved["claim_scope"]["neural_model"], True)
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(saved["material_counts"], [16, 24])
            self.assertIn("sample_aware_self_ranked_induction", saved["thresholds"])
            self.assertIn("first_material_count_reaching_target", saved["thresholds"]["raw_text"])
            raw_budget = saved["budgets"]["16"]["raw_text"]
            self.assertGreater(raw_budget["neural_parameter_count_mean"], 0)
            self.assertGreater(raw_budget["estimated_neural_training_multiply_adds_mean"], 0)
            self.assertIn("Tiny Neural Budget Sweep", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
