import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.sweep import run_sample_budget_sweep


class SampleBudgetSweepTests(unittest.TestCase):
    def test_sweep_writes_budget_artifact_and_threshold_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "sweep.json"
            out_md = Path(temp_dir) / "sweep.md"
            result = run_sample_budget_sweep(
                material_counts=[16, 24],
                seeds=[3],
                conditions=["raw_text", "induced_rule_expansion"],
                output_json=out_json,
                output_markdown=out_md,
                epochs=2,
                target_signed_gain=0.01,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(result["claim_scope"]["paper_ready_claim"], False)
            self.assertEqual(saved["material_counts"], [16, 24])
            self.assertIn("induced_rule_expansion", saved["thresholds"])
            self.assertIn("first_material_count_reaching_target", saved["thresholds"]["induced_rule_expansion"])
            self.assertIn("Sample Budget Sweep", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
