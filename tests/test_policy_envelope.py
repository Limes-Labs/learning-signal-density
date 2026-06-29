import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_policy_envelope.py"


def load_envelope_builder():
    spec = importlib.util.spec_from_file_location("build_policy_envelope", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PolicyEnvelopeArtifactTests(unittest.TestCase):
    def test_builds_non_deployable_post_hoc_policy_envelope(self) -> None:
        envelope_builder = load_envelope_builder()

        artifact = envelope_builder.build_policy_envelope(REPO_ROOT)

        self.assertEqual(artifact["profile_label"], "epochs=16_hidden=8_features=1024_agreement_gated")
        self.assertEqual(
            artifact["source_artifacts"],
            ["results/tiny_neural_budget_sweep_agreement_gated_f1024.json"],
        )
        self.assertEqual(artifact["oracle_condition"], "counterfactual_expansion")
        self.assertNotIn("counterfactual_expansion", artifact["conditions_considered"])
        self.assertEqual(artifact["claim_scope"]["synthetic_domain"], True)
        self.assertEqual(artifact["claim_scope"]["neural_model"], True)
        self.assertEqual(artifact["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertEqual(artifact["claim_scope"]["post_hoc_envelope"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_policy_selection"], True)
        self.assertEqual(artifact["claim_scope"]["deployable_policy"], False)
        self.assertEqual(artifact["claim_scope"]["paper_ready_claim"], False)

        expected_best = {
            "16": ("mdl_rule_expansion", 0.010526),
            "24": ("raw_text", 0.0),
            "32": ("mdl_rule_expansion", -0.036842),
            "48": ("validation_ranked_induction", 0.086207),
            "64": ("self_ranked_induction", 0.153247),
        }
        for material, (condition, gain) in expected_best.items():
            with self.subTest(material=material):
                row = artifact["best_by_material"][material]
                self.assertEqual(row["condition"], condition)
                self.assertAlmostEqual(row["signed_gain"], gain)

        self.assertEqual(artifact["first_material_count_reaching_target"], 48)
        self.assertEqual(artifact["best_material_count"], 64)
        self.assertAlmostEqual(artifact["best_signed_gain"], 0.153247)
        self.assertGreater(
            artifact["best_by_material"]["48"]["oracle_gap_signed_gain"],
            artifact["best_by_material"]["32"]["oracle_gap_signed_gain"],
        )

    def test_writes_json_and_markdown_artifacts(self) -> None:
        envelope_builder = load_envelope_builder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "policy_envelope.json"
            output_md = Path(tmpdir) / "policy_envelope.md"
            envelope_builder.write_policy_envelope(REPO_ROOT, output_json, output_md)
            written = json.loads(output_json.read_text())
            markdown = output_md.read_text()

        self.assertEqual(written["first_material_count_reaching_target"], 48)
        self.assertIn("post-hoc", markdown)
        self.assertIn("not deployable", markdown)
        self.assertIn("raw_text", markdown)
        self.assertIn("mdl_rule_expansion", markdown)


if __name__ == "__main__":
    unittest.main()
