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
            self.assertIn("signed_external_sample_efficiency_mean", saved["conditions"]["raw_text"])
            self.assertIn("clipped_external_sample_efficiency_mean", saved["conditions"]["raw_text"])
            self.assertIn("pareto_frontier_conditions", saved)
            self.assertEqual(saved["condition_scope"]["induced_rule_expansion"]["oracle_generated_labels"], False)
            self.assertEqual(saved["condition_scope"]["counterfactual_expansion"]["oracle_generated_labels"], True)
            self.assertTrue(saved["pareto_frontier_conditions"])
            self.assertIn("Learning Signal Density Pilot", out_md.read_text())

    def test_validation_gated_induction_records_thresholds_and_tuning_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "result.json"
            result = run_seedset(
                seeds=[3],
                conditions=["validation_gated_induction"],
                output_json=out_json,
                material_count=24,
                epochs=2,
            )

            saved = json.loads(out_json.read_text())
            row = saved["per_seed"][0]
            self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(saved["condition_scope"]["validation_gated_induction"]["oracle_generated_labels"], False)
            self.assertEqual(saved["condition_scope"]["validation_gated_induction"]["validation_used_for_threshold"], True)
            self.assertGreater(row["validation_tuning_cost_tokens"], 0)
            self.assertGreater(row["validation_gate_candidates"], 1)
            self.assertIn(row["induction_min_support"], [1, 2, 3, 4])
            self.assertGreaterEqual(row["induction_min_confidence"], 0.5)
            self.assertIn("validation_tuning_cost_tokens_mean", saved["conditions"]["validation_gated_induction"])

    def test_direct_validation_gate_is_charged_less_than_retraining_gate(self) -> None:
        result = run_seedset(
            seeds=[3],
            conditions=["validation_gated_induction", "direct_validation_gated_induction"],
            material_count=24,
            epochs=2,
        )

        rows = {row["condition"]: row for row in result["per_seed"]}
        direct = rows["direct_validation_gated_induction"]
        retrained = rows["validation_gated_induction"]
        self.assertEqual(result["condition_scope"]["direct_validation_gated_induction"]["validation_used_for_threshold"], True)
        self.assertLess(direct["validation_tuning_cost_tokens"], retrained["validation_tuning_cost_tokens"])
        self.assertGreater(direct["validation_gate_candidates"], 1)
        self.assertIsNotNone(direct["validation_gate_score"])

    def test_mdl_rule_expansion_records_description_length_and_rule_count(self) -> None:
        result = run_seedset(
            seeds=[3],
            conditions=["mdl_rule_expansion"],
            material_count=32,
            epochs=2,
        )

        row = result["per_seed"][0]
        self.assertEqual(result["condition_scope"]["mdl_rule_expansion"]["oracle_generated_labels"], False)
        self.assertEqual(result["condition_scope"]["mdl_rule_expansion"]["validation_used_for_threshold"], True)
        self.assertGreater(row["mdl_selected_rule_count"], 0)
        self.assertGreater(row["mdl_description_length_tokens"], 0)
        self.assertGreater(row["rule_search_cost_tokens"], 0)
        self.assertGreaterEqual(row["charged_compute_units"], row["rule_search_cost_tokens"])
        self.assertIn("mdl_selected_rule_count_mean", result["conditions"]["mdl_rule_expansion"])


if __name__ == "__main__":
    unittest.main()
