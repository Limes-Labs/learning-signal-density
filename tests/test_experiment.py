import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.experiment import run_seedset


class ExperimentArtifactTests(unittest.TestCase):
    def test_seedset_writes_costed_artifact_with_honest_scope_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "result.json"
            out_md = Path(temp_dir) / "result.md"
            result = run_seedset(
                seeds=[3, 5],
                conditions=["raw_text", "counterfactual_expansion"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=24,
                epochs=3,
            )

            saved = json.loads(out_json.read_text())
            self.assertEqual(result["claim_scope"]["neural_model"], False)
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertIn("raw_text", saved["conditions"])
            self.assertIn("counterfactual_expansion", saved["conditions"])
            self.assertGreater(saved["conditions"]["raw_text"]["external_events_mean"], 0)
            self.assertGreater(saved["conditions"]["counterfactual_expansion"]["internal_tokens_mean"], 0)
            self.assertIn("Learning Signal Density Pilot", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
