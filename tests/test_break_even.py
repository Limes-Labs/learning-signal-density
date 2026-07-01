import json
import unittest
from pathlib import Path

from learning_signal_density.break_even import break_even_comparison
from scripts.build_break_even_analysis import build_sms_break_even_analysis


class BreakEvenAnalysisTests(unittest.TestCase):
    def test_same_budget_density_break_even_requires_quality_multiplier_above_compute_multiplier(self) -> None:
        reference = {
            "condition": "random_sample",
            "external_events_mean": 32,
            "charged_compute_units_mean": 100.0,
            "spam_f1_improvement_over_majority_mean": 0.25,
        }
        candidate = {
            "condition": "validation_selector",
            "external_events_mean": 32,
            "charged_compute_units_mean": 400.0,
            "spam_f1_improvement_over_majority_mean": 0.50,
        }

        row = break_even_comparison(
            reference=reference,
            candidate=candidate,
            quality_key="spam_f1_improvement_over_majority_mean",
            quality_upper_bound=1.0,
        )

        self.assertEqual(row["event_compute_multiplier"], 4.0)
        self.assertEqual(row["quality_multiplier"], 2.0)
        self.assertEqual(row["break_even_quality"], 1.0)
        self.assertEqual(row["required_quality_delta_to_break_even"], 0.5)
        self.assertEqual(row["density_ratio"], 0.5)
        self.assertFalse(row["candidate_density_wins"])
        self.assertFalse(row["perfect_quality_can_beat"])
        self.assertEqual(row["max_possible_density_ratio"], 1.0)
        self.assertEqual(row["required_fraction_of_quality_bound"], 1.0)

    def test_sms_break_even_analysis_records_real_selector_tax(self) -> None:
        result = build_sms_break_even_analysis(Path("."))

        self.assertEqual(result["title"], "SMS Spam Break-Even Selection-Cost Analysis")
        self.assertEqual(result["claim_scope"]["real_dataset"], True)
        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(result["reference_condition"], "random_sample")

        v800_budget_32 = result["comparisons"]["SMS Spam v800"]["32"]
        label_index = v800_budget_32["label_index_balanced_sample"]
        validation_label_index = v800_budget_32["validation_label_index_selector"]

        self.assertGreater(label_index["candidate_quality"], label_index["reference_quality"])
        self.assertGreater(label_index["event_compute_multiplier"], label_index["quality_multiplier"])
        self.assertLess(label_index["density_ratio"], 1.0)
        self.assertGreater(label_index["break_even_quality"], 1.0)
        self.assertFalse(label_index["perfect_quality_can_beat"])
        self.assertLess(label_index["max_possible_density_ratio"], 1.0)

        self.assertGreater(validation_label_index["candidate_quality"], label_index["candidate_quality"])
        self.assertGreater(validation_label_index["compute_over_break_even"], 1.0)
        self.assertLess(validation_label_index["density_ratio"], label_index["density_ratio"])
        self.assertFalse(validation_label_index["perfect_quality_can_beat"])

        v200_budget_32 = result["comparisons"]["SMS Spam v200"]["32"]["validation_label_index_selector"]
        self.assertLess(v200_budget_32["compute_over_break_even"], validation_label_index["compute_over_break_even"])
        self.assertLess(v200_budget_32["density_ratio"], 1.0)
        self.assertFalse(v200_budget_32["perfect_quality_can_beat"])

    def test_generated_break_even_artifact_matches_builder(self) -> None:
        expected = build_sms_break_even_analysis(Path("."))
        committed = json.loads(Path("results/sms_spam_break_even_analysis.json").read_text())

        self.assertEqual(committed["source_artifacts"], expected["source_artifacts"])
        self.assertEqual(committed["reference_condition"], expected["reference_condition"])
        self.assertEqual(committed["candidate_conditions"], expected["candidate_conditions"])
        self.assertEqual(committed["comparisons"], expected["comparisons"])


if __name__ == "__main__":
    unittest.main()
