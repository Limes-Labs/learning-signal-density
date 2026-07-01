import json
import unittest
from pathlib import Path

from learning_signal_density.break_even import break_even_comparison
from scripts.build_break_even_analysis import build_sms_break_even_analysis, render_markdown
from scripts.build_newsgroups_break_even_analysis import (
    build_newsgroups_break_even_analysis,
    render_markdown as render_newsgroups_markdown,
)


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
            "charged_compute_units_mean": 250.0,
            "selection_cost_tokens_mean": 0.0,
            "validation_tuning_cost_tokens_mean": 150.0,
            "spam_f1_improvement_over_majority_mean": 0.50,
        }

        row = break_even_comparison(
            reference=reference,
            candidate=candidate,
            quality_key="spam_f1_improvement_over_majority_mean",
            quality_upper_bound=1.0,
            reusable_compute_keys=("selection_cost_tokens_mean", "validation_tuning_cost_tokens_mean"),
        )

        self.assertEqual(row["event_compute_multiplier"], 2.5)
        self.assertEqual(row["quality_multiplier"], 2.0)
        self.assertEqual(row["break_even_quality"], 0.625)
        self.assertEqual(row["required_quality_delta_to_break_even"], 0.125)
        self.assertEqual(row["density_ratio"], 0.8)
        self.assertFalse(row["candidate_density_wins"])
        self.assertTrue(row["perfect_quality_can_beat"])
        self.assertEqual(row["max_possible_density_ratio"], 1.6)
        self.assertEqual(row["required_fraction_of_quality_bound"], 0.625)
        self.assertEqual(row["candidate_reusable_compute_units"], 150.0)
        self.assertEqual(row["candidate_nonreusable_compute_units"], 100.0)
        self.assertEqual(row["amortized_reuses_to_density_win"], 2)
        self.assertGreater(row["density_ratio_at_min_reuses"], 1.0)
        self.assertEqual(row["fully_amortized_density_ratio"], 2.0)

    def test_amortized_reuse_reports_impossible_when_nonreusable_cost_is_too_high(self) -> None:
        reference = {
            "condition": "random_sample",
            "external_events_mean": 10,
            "charged_compute_units_mean": 100.0,
            "spam_f1_improvement_over_majority_mean": 0.25,
        }
        candidate = {
            "condition": "costly_fixed_selector",
            "external_events_mean": 10,
            "charged_compute_units_mean": 250.0,
            "selection_cost_tokens_mean": 40.0,
            "validation_tuning_cost_tokens_mean": 0.0,
            "spam_f1_improvement_over_majority_mean": 0.50,
        }

        row = break_even_comparison(
            reference=reference,
            candidate=candidate,
            quality_key="spam_f1_improvement_over_majority_mean",
            reusable_compute_keys=("selection_cost_tokens_mean", "validation_tuning_cost_tokens_mean"),
        )

        self.assertEqual(row["max_affordable_compute_units"], 200.0)
        self.assertEqual(row["candidate_nonreusable_compute_units"], 210.0)
        self.assertIsNone(row["amortized_reuses_to_density_win"])
        self.assertLess(row["fully_amortized_density_ratio"], 1.0)

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
        self.assertGreater(label_index["amortized_reuses_to_density_win"], 1)
        self.assertGreater(label_index["fully_amortized_density_ratio"], 1.0)

        self.assertGreater(validation_label_index["candidate_quality"], label_index["candidate_quality"])
        self.assertGreater(validation_label_index["compute_over_break_even"], 1.0)
        self.assertLess(validation_label_index["density_ratio"], label_index["density_ratio"])
        self.assertFalse(validation_label_index["perfect_quality_can_beat"])
        self.assertGreater(validation_label_index["amortized_reuses_to_density_win"], label_index["amortized_reuses_to_density_win"])
        self.assertGreater(validation_label_index["fully_amortized_density_ratio"], 1.0)

        v200_budget_32 = result["comparisons"]["SMS Spam v200"]["32"]["validation_label_index_selector"]
        self.assertLess(v200_budget_32["compute_over_break_even"], validation_label_index["compute_over_break_even"])
        self.assertLess(v200_budget_32["density_ratio"], 1.0)
        self.assertFalse(v200_budget_32["perfect_quality_can_beat"])
        self.assertIsNone(v200_budget_32["amortized_reuses_to_density_win"])
        self.assertLess(v200_budget_32["fully_amortized_density_ratio"], 1.0)

    def test_generated_break_even_artifact_matches_builder(self) -> None:
        expected = build_sms_break_even_analysis(Path("."))
        committed = json.loads(Path("results/sms_spam_break_even_analysis.json").read_text())

        self.assertEqual(committed["source_artifacts"], expected["source_artifacts"])
        self.assertEqual(committed["reference_condition"], expected["reference_condition"])
        self.assertEqual(committed["candidate_conditions"], expected["candidate_conditions"])
        self.assertEqual(committed["comparisons"], expected["comparisons"])
        self.assertEqual(committed["amortization_model"], expected["amortization_model"])

    def test_newsgroups_break_even_analysis_separates_quality_from_density(self) -> None:
        result = build_newsgroups_break_even_analysis(Path("."))

        self.assertEqual(result["title"], "Twenty Newsgroups Break-Even Selection-Cost Analysis")
        self.assertEqual(result["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(result["reference_condition"], "random_sample")
        self.assertEqual(result["quality_metric"], "accuracy_improvement_over_majority_mean")
        self.assertEqual(result["quality_upper_bound"], 0.95)
        self.assertEqual(result["claim_scope"]["real_dataset"], True)
        self.assertEqual(result["claim_scope"]["metadata_stripped"], True)
        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)

        prototype_budget_40 = result["comparisons"]["40"]["prototype_retrieval_sample"]
        self.assertGreater(prototype_budget_40["quality_multiplier"], 1.0)
        self.assertLess(prototype_budget_40["density_ratio"], 1.0)
        self.assertFalse(prototype_budget_40["candidate_density_wins"])
        self.assertFalse(prototype_budget_40["perfect_quality_can_beat"])

        class_balanced_budget_80 = result["comparisons"]["80"]["class_balanced_sample"]
        self.assertEqual(class_balanced_budget_80["quality_multiplier"], 1.0)
        self.assertLess(class_balanced_budget_80["event_compute_multiplier"], 1.0)
        self.assertGreater(class_balanced_budget_80["density_ratio"], 1.0)
        self.assertTrue(class_balanced_budget_80["candidate_density_wins"])

        prototype_budget_160 = result["comparisons"]["160"]["prototype_retrieval_sample"]
        self.assertGreater(prototype_budget_160["quality_multiplier"], 1.0)
        self.assertLess(prototype_budget_160["density_ratio"], 1.0)
        self.assertIsNone(prototype_budget_160["amortized_reuses_to_density_win"])
        self.assertGreater(prototype_budget_160["compute_over_break_even"], 1.0)

        self.assertEqual(result["summary"]["prototype_retrieval_sample"]["observed_quality_wins"], 2)
        self.assertEqual(result["summary"]["prototype_retrieval_sample"]["density_wins"], 0)
        self.assertEqual(result["summary"]["class_balanced_sample"]["density_wins"], 1)

    def test_generated_newsgroups_break_even_artifact_matches_builder(self) -> None:
        expected = build_newsgroups_break_even_analysis(Path("."))
        committed = json.loads(Path("results/twenty_newsgroups_break_even_analysis.json").read_text())

        self.assertEqual(committed["source_artifacts"], expected["source_artifacts"])
        self.assertEqual(committed["reference_condition"], expected["reference_condition"])
        self.assertEqual(committed["candidate_conditions"], expected["candidate_conditions"])
        self.assertEqual(committed["comparisons"], expected["comparisons"])
        self.assertEqual(committed["amortization_model"], expected["amortization_model"])

    def test_markdown_break_even_table_has_single_header(self) -> None:
        rendered = render_markdown(build_sms_break_even_analysis(Path(".")))

        self.assertEqual(rendered.count("Reuses to win"), 2)
        self.assertEqual(rendered.count("Compute over break-even | Reuses to win"), 2)

    def test_newsgroups_markdown_break_even_table_has_single_header(self) -> None:
        rendered = render_newsgroups_markdown(build_newsgroups_break_even_analysis(Path(".")))

        self.assertEqual(rendered.count("Reuses to win"), 1)
        self.assertEqual(rendered.count("Quality mult. | Event-compute mult."), 1)


if __name__ == "__main__":
    unittest.main()
