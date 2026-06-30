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

    def test_train_size_gated_policy_is_declared_as_train_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "train_size_gate.json"
            out_md = Path(temp_dir) / "train_size_gate.md"
            run_neural_seedset(
                seeds=[17],
                conditions=["train_size_gated_sample_aware_induction"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=48,
                epochs=2,
                hidden_units=4,
                feature_dimension=32,
                fresh_seed_confirmation=True,
            )

            saved = json.loads(out_json.read_text())
            scope = saved["condition_scope"]["train_size_gated_sample_aware_induction"]
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(scope["train_only_selection"], True)
            self.assertEqual(scope["train_only_induction"], True)
            self.assertEqual(scope["validation_used_for_transform_selection"], False)
            self.assertEqual(scope["oracle_generated_labels"], False)

    def test_compact_train_size_gated_policy_reduces_large_sample_neural_compute(self) -> None:
        compact = run_neural_condition(
            seed=181,
            condition="compact_train_size_gated_induction",
            material_count=64,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        train_gated = run_neural_condition(
            seed=181,
            condition="train_size_gated_sample_aware_induction",
            material_count=64,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )

        self.assertEqual(compact["condition"], "compact_train_size_gated_induction")
        self.assertLess(compact["internal_examples"], train_gated["internal_examples"])
        self.assertLess(compact["internal_tokens"], train_gated["internal_tokens"])
        self.assertLess(compact["charged_compute_units"], train_gated["charged_compute_units"])
        self.assertEqual(compact["portfolio_candidate_count"], 0)

    def test_density_capped_compact_policy_returns_to_raw_for_abundant_data(self) -> None:
        density_capped = run_neural_condition(
            seed=293,
            condition="density_capped_compact_induction",
            material_count=104,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        raw = run_neural_condition(
            seed=293,
            condition="raw_text",
            material_count=104,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        compact = run_neural_condition(
            seed=293,
            condition="compact_train_size_gated_induction",
            material_count=104,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )

        self.assertEqual(density_capped["condition"], "density_capped_compact_induction")
        self.assertEqual(density_capped["internal_examples"], raw["internal_examples"])
        self.assertEqual(density_capped["internal_tokens"], raw["internal_tokens"])
        self.assertEqual(density_capped["charged_compute_units"], raw["charged_compute_units"])
        self.assertLess(density_capped["charged_compute_units"], compact["charged_compute_units"])
        self.assertEqual(density_capped["portfolio_candidate_count"], 0)

    def test_support_ramped_compact_policy_declares_train_only_support_ramp(self) -> None:
        result = run_neural_seedset(
            seeds=[401],
            conditions=["support_ramped_compact_induction"],
            material_count=104,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            fresh_seed_confirmation=True,
        )

        scope = result["condition_scope"]["support_ramped_compact_induction"]
        stats = result["conditions"]["support_ramped_compact_induction"]

        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["abundant_data_min_support"], 4)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(stats["portfolio_candidate_count_mean"], 0)

    def test_late_confidence_ramped_compact_policy_declares_train_only_late_confidence_ramp(self) -> None:
        result = run_neural_seedset(
            seeds=[463],
            conditions=["late_confidence_ramped_compact_induction"],
            material_count=120,
            epochs=4,
            hidden_units=4,
            feature_dimension=64,
            fresh_seed_confirmation=True,
        )

        scope = result["condition_scope"]["late_confidence_ramped_compact_induction"]
        stats = result["conditions"]["late_confidence_ramped_compact_induction"]

        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["abundant_data_min_support"], 4)
        self.assertEqual(scope["late_confidence_ramp_min_events"], 432)
        self.assertEqual(scope["late_confidence_min_confidence"], 0.60)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(stats["portfolio_candidate_count_mean"], 0)

    def test_train_support_density_selector_uses_train_only_support_density(self) -> None:
        raw_case = run_neural_condition(
            seed=1009,
            condition="train_support_density_selector",
            material_count=120,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        raw_baseline = run_neural_condition(
            seed=1009,
            condition="raw_text",
            material_count=120,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        support_case = run_neural_condition(
            seed=1031,
            condition="train_support_density_selector",
            material_count=120,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        support_baseline = run_neural_condition(
            seed=1031,
            condition="support_ramped_compact_induction",
            material_count=120,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )

        self.assertEqual(raw_case["portfolio_selected_condition"], "raw_text")
        self.assertEqual(support_case["portfolio_selected_condition"], "support_ramped_compact_induction")
        self.assertEqual(raw_case["portfolio_candidate_count"], 3)
        self.assertEqual(
            raw_case["portfolio_candidate_conditions"],
            [
                "raw_text",
                "compact_train_size_gated_induction",
                "support_ramped_compact_induction",
            ],
        )
        self.assertGreater(raw_case["portfolio_selection_cost_units"], 0)
        self.assertGreater(raw_case["charged_compute_units"], raw_baseline["charged_compute_units"])
        self.assertGreater(support_case["charged_compute_units"], support_baseline["charged_compute_units"])
        self.assertIn("train_support_density", raw_case["portfolio_selection_metric"])
        self.assertNotIn("validation", raw_case["portfolio_selection_metric"])
        self.assertNotIn("heldout", raw_case["portfolio_selection_metric"])

    def test_train_support_density_selector_scope_is_declared_in_neural_artifact(self) -> None:
        result = run_neural_seedset(
            seeds=[1009],
            conditions=["train_support_density_selector"],
            material_count=120,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            fresh_seed_confirmation=True,
        )

        scope = result["condition_scope"]["train_support_density_selector"]
        stats = result["conditions"]["train_support_density_selector"]

        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["support_density_selector"], True)
        self.assertEqual(scope["support_density_min_kept_per_compute"], 0.00145)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(stats["portfolio_candidate_count_mean"], 3)
        self.assertIn("portfolio_selected_condition_counts", stats)

    def test_support_probe_window_selector_reuses_selected_support_construction(self) -> None:
        support_case = run_neural_condition(
            seed=1031,
            condition="support_probe_window_selector",
            material_count=104,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        support_baseline = run_neural_condition(
            seed=1031,
            condition="support_ramped_compact_induction",
            material_count=104,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        raw_case = run_neural_condition(
            seed=1031,
            condition="support_probe_window_selector",
            material_count=128,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )
        raw_baseline = run_neural_condition(
            seed=1031,
            condition="raw_text",
            material_count=128,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            learning_rate=0.03,
        )

        self.assertEqual(support_case["portfolio_selected_condition"], "support_ramped_compact_induction")
        self.assertEqual(support_case["portfolio_candidate_count"], 1)
        self.assertEqual(support_case["portfolio_candidate_conditions"], ["support_ramped_compact_induction"])
        self.assertGreater(support_case["portfolio_selection_cost_units"], 0)
        self.assertEqual(support_case["charged_compute_units"], support_baseline["charged_compute_units"])
        self.assertIn("train_support_probe_window", support_case["portfolio_selection_metric"])
        self.assertNotIn("validation", support_case["portfolio_selection_metric"])
        self.assertNotIn("heldout", support_case["portfolio_selection_metric"])

        self.assertEqual(raw_case["portfolio_selected_condition"], "raw_text")
        self.assertEqual(raw_case["portfolio_candidate_count"], 0)
        self.assertEqual(raw_case["portfolio_selection_cost_units"], 0)
        self.assertEqual(raw_case["charged_compute_units"], raw_baseline["charged_compute_units"])

    def test_support_probe_window_selector_scope_is_declared_in_neural_artifact(self) -> None:
        result = run_neural_seedset(
            seeds=[1031],
            conditions=["support_probe_window_selector"],
            material_count=104,
            epochs=16,
            hidden_units=4,
            feature_dimension=64,
            fresh_seed_confirmation=True,
        )

        scope = result["condition_scope"]["support_probe_window_selector"]
        stats = result["conditions"]["support_probe_window_selector"]

        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["support_probe_window_selector"], True)
        self.assertEqual(scope["support_probe_min_train_events"], 360)
        self.assertEqual(scope["support_probe_max_train_events"], 432)
        self.assertEqual(scope["reuse_selected_candidate_construction"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(stats["portfolio_candidate_count_mean"], 1)
        self.assertIn("portfolio_selected_condition_counts", stats)

    def test_validation_portfolio_selector_charges_candidate_training_without_heldout_selection(self) -> None:
        selector = run_neural_condition(
            seed=17,
            condition="validation_portfolio_selector",
            material_count=16,
            epochs=2,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )
        raw = run_neural_condition(
            seed=17,
            condition="raw_text",
            material_count=16,
            epochs=2,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )

        self.assertEqual(selector["condition"], "validation_portfolio_selector")
        self.assertGreater(selector["portfolio_candidate_count"], 1)
        self.assertGreater(selector["portfolio_selection_cost_units"], raw["charged_compute_units"])
        self.assertGreater(selector["charged_compute_units"], raw["charged_compute_units"])
        self.assertGreater(selector["estimated_neural_training_multiply_adds"], raw["estimated_neural_training_multiply_adds"])
        self.assertIn(
            selector["portfolio_selected_condition"],
            selector["portfolio_candidate_conditions"],
        )
        self.assertNotEqual(selector["portfolio_selected_condition"], "counterfactual_expansion")
        self.assertNotIn("heldout", selector["portfolio_selection_metric"])

    def test_validation_portfolio_selector_scope_is_declared_in_neural_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "selector.json"
            out_md = Path(temp_dir) / "selector.md"
            run_neural_seedset(
                seeds=[17],
                conditions=["validation_portfolio_selector"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=2,
                hidden_units=4,
                feature_dimension=32,
                fresh_seed_confirmation=True,
            )

            saved = json.loads(out_json.read_text())
            scope = saved["condition_scope"]["validation_portfolio_selector"]
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(scope["validation_used_for_policy_selection"], True)
            self.assertEqual(scope["oracle_generated_labels"], False)
            self.assertEqual(
                saved["conditions"]["validation_portfolio_selector"]["portfolio_candidate_count_mean"],
                6,
            )
            self.assertIn("portfolio_selected_condition_counts", saved["conditions"]["validation_portfolio_selector"])

    def test_validation_linear_proxy_selector_charges_proxy_search_but_avoids_portfolio_neural_training(self) -> None:
        proxy = run_neural_condition(
            seed=17,
            condition="validation_linear_proxy_selector",
            material_count=16,
            epochs=4,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )
        portfolio = run_neural_condition(
            seed=17,
            condition="validation_portfolio_selector",
            material_count=16,
            epochs=4,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )
        raw = run_neural_condition(
            seed=17,
            condition="raw_text",
            material_count=16,
            epochs=4,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )

        self.assertEqual(proxy["condition"], "validation_linear_proxy_selector")
        self.assertEqual(proxy["portfolio_candidate_count"], 6)
        self.assertEqual(proxy["portfolio_proxy_epochs"], 2)
        self.assertGreater(proxy["portfolio_selection_cost_units"], raw["charged_compute_units"])
        self.assertLess(proxy["charged_compute_units"], portfolio["charged_compute_units"])
        self.assertLess(
            proxy["estimated_neural_training_multiply_adds"],
            portfolio["estimated_neural_training_multiply_adds"],
        )
        self.assertIn(proxy["portfolio_selected_condition"], proxy["portfolio_candidate_conditions"])
        self.assertNotIn("heldout", proxy["portfolio_selection_metric"])

    def test_validation_linear_proxy_selector_scope_is_declared_in_neural_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "proxy_selector.json"
            out_md = Path(temp_dir) / "proxy_selector.md"
            run_neural_seedset(
                seeds=[17],
                conditions=["validation_linear_proxy_selector"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=2,
                hidden_units=4,
                feature_dimension=32,
                fresh_seed_confirmation=True,
            )

            saved = json.loads(out_json.read_text())
            scope = saved["condition_scope"]["validation_linear_proxy_selector"]
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(scope["validation_used_for_policy_selection"], True)
            self.assertEqual(scope["low_fidelity_proxy_selector"], True)
            self.assertEqual(scope["oracle_generated_labels"], False)
            self.assertEqual(
                saved["conditions"]["validation_linear_proxy_selector"]["portfolio_candidate_count_mean"],
                6,
            )
            self.assertEqual(
                saved["conditions"]["validation_linear_proxy_selector"]["portfolio_proxy_epochs_mean"],
                2,
            )
            self.assertIn(
                "portfolio_selected_condition_counts",
                saved["conditions"]["validation_linear_proxy_selector"],
            )

    def test_validation_abstaining_proxy_selector_requires_extra_validation_examples_before_leaving_raw_text(self) -> None:
        selector = run_neural_condition(
            seed=17,
            condition="validation_abstaining_proxy_selector",
            material_count=16,
            epochs=4,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )
        proxy = run_neural_condition(
            seed=17,
            condition="validation_linear_proxy_selector",
            material_count=16,
            epochs=4,
            hidden_units=4,
            feature_dimension=32,
            learning_rate=0.03,
        )

        self.assertEqual(selector["condition"], "validation_abstaining_proxy_selector")
        self.assertEqual(selector["portfolio_candidate_count"], 6)
        self.assertEqual(selector["portfolio_proxy_epochs"], 2)
        self.assertEqual(selector["portfolio_abstention_extra_correct"], 3)
        self.assertGreater(selector["portfolio_abstention_margin"], 0)
        self.assertEqual(selector["portfolio_raw_text_abstention"], 1)
        self.assertEqual(selector["portfolio_selected_condition"], "raw_text")
        self.assertLessEqual(selector["charged_compute_units"], proxy["charged_compute_units"])
        self.assertIn("raw_text_abstention", selector["portfolio_selection_metric"])
        self.assertNotIn("heldout", selector["portfolio_selection_metric"])

    def test_validation_abstaining_proxy_selector_scope_is_declared_in_neural_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "abstaining_proxy_selector.json"
            out_md = Path(temp_dir) / "abstaining_proxy_selector.md"
            run_neural_seedset(
                seeds=[17],
                conditions=["validation_abstaining_proxy_selector"],
                output_json=out_json,
                output_markdown=out_md,
                material_count=16,
                epochs=2,
                hidden_units=4,
                feature_dimension=32,
                fresh_seed_confirmation=True,
            )

            saved = json.loads(out_json.read_text())
            scope = saved["condition_scope"]["validation_abstaining_proxy_selector"]
            self.assertEqual(saved["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(scope["validation_used_for_policy_selection"], True)
            self.assertEqual(scope["low_fidelity_proxy_selector"], True)
            self.assertEqual(scope["raw_text_abstention"], True)
            self.assertEqual(scope["oracle_generated_labels"], False)
            self.assertEqual(
                saved["conditions"]["validation_abstaining_proxy_selector"][
                    "portfolio_abstention_extra_correct_mean"
                ],
                3,
            )
            self.assertIn(
                "portfolio_selected_condition_counts",
                saved["conditions"]["validation_abstaining_proxy_selector"],
            )

    def test_validation_coverage_proxy_selector_records_motif_proxy_scope(self) -> None:
        result = run_neural_seedset(
            seeds=[109],
            conditions=["validation_coverage_proxy_selector"],
            material_count=32,
            epochs=16,
            hidden_units=8,
            feature_dimension=1024,
            learning_rate=0.03,
        )

        selector = result["conditions"]["validation_coverage_proxy_selector"]
        scope = result["condition_scope"]["validation_coverage_proxy_selector"]

        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_motif_distribution_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], False)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(selector["portfolio_candidate_count_mean"], 6)
        self.assertEqual(selector["portfolio_proxy_epochs_mean"], 0)
        self.assertGreater(selector["portfolio_selection_cost_units_mean"], 0)
        self.assertEqual(selector["portfolio_selected_condition_counts"], {"mdl_rule_expansion": 1})
        self.assertEqual(selector["accuracy_improvement_over_majority_mean"], 0.052632)
        self.assertIn("portfolio_candidate_summaries", result["per_seed"][0])
        self.assertEqual(
            result["per_seed"][0]["portfolio_selection_metric"],
            "validation_motif_coverage_l1_with_pair_coverage_bonus",
        )

    def test_validation_coverage_prior_selector_uses_train_size_floor_before_motif_scan(self) -> None:
        result = run_neural_seedset(
            seeds=[109],
            conditions=["validation_coverage_prior_selector"],
            material_count=24,
            epochs=16,
            hidden_units=8,
            feature_dimension=1024,
            learning_rate=0.03,
        )

        selector = result["conditions"]["validation_coverage_prior_selector"]
        scope = result["condition_scope"]["validation_coverage_prior_selector"]
        row = result["per_seed"][0]

        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_motif_distribution_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], False)
        self.assertEqual(scope["train_size_prior_min_events"], 96)
        self.assertEqual(scope["lean_coverage_candidate_set"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)
        self.assertEqual(selector["portfolio_candidate_count_mean"], 0)
        self.assertEqual(selector["portfolio_proxy_epochs_mean"], 0)
        self.assertEqual(selector["portfolio_selection_cost_units_mean"], 0)
        self.assertEqual(selector["portfolio_selected_condition_counts"], {"raw_text": 1})
        self.assertEqual(selector["accuracy_improvement_over_majority_mean"], 0.0)
        self.assertEqual(row["portfolio_selected_condition"], "raw_text")
        self.assertEqual(
            row["portfolio_selection_metric"],
            "validation_motif_coverage_l1_with_train_size_prior_96_and_lean_cost_penalty",
        )

    def test_validation_coverage_prior_selector_scans_lean_candidates_after_floor(self) -> None:
        result = run_neural_seedset(
            seeds=[109],
            conditions=["validation_coverage_prior_selector"],
            material_count=48,
            epochs=16,
            hidden_units=8,
            feature_dimension=1024,
            learning_rate=0.03,
        )

        selector = result["conditions"]["validation_coverage_prior_selector"]
        row = result["per_seed"][0]

        self.assertEqual(selector["portfolio_candidate_count_mean"], 3)
        self.assertEqual(
            selector["portfolio_candidate_conditions"],
            [
                "raw_text",
                "sample_aware_self_ranked_induction",
                "validation_ranked_induction",
            ],
        )
        self.assertEqual(
            row["portfolio_candidate_conditions"],
            [
                "raw_text",
                "sample_aware_self_ranked_induction",
                "validation_ranked_induction",
            ],
        )
        self.assertGreater(selector["portfolio_selection_cost_units_mean"], 0)
        self.assertIn(row["portfolio_selected_condition"], row["portfolio_candidate_conditions"])
        self.assertNotIn("heldout", row["portfolio_selection_metric"])


if __name__ == "__main__":
    unittest.main()
