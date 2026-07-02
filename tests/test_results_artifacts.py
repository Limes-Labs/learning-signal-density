import json
import unittest
from pathlib import Path


class CommittedResultArtifactTests(unittest.TestCase):
    def test_twenty_newsgroups_artifact_records_real_nlp_selection_tradeoffs(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_active_selection.json").read_text())

        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(artifact["dataset"]["label_count"], 20)
        self.assertEqual(artifact["dataset"]["license"], "CC BY 4.0")
        self.assertEqual(
            artifact["dataset"]["sha256"],
            "cfbb360d6c1e55c06d33a4c5da0789a93b78db74833a70be8ff2e133cc4e6a6e",
        )
        self.assertEqual(artifact["claim_scope"]["real_dataset"], True)
        self.assertEqual(artifact["claim_scope"]["synthetic_domain"], False)
        self.assertEqual(artifact["claim_scope"]["metadata_stripped"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertIn("prototype_retrieval_sample", artifact["condition_scope"])
        self.assertIn("length_curriculum_sample", artifact["condition_scope"])
        self.assertIn("validation_selector", artifact["condition_scope"])

        budget_40 = artifact["budgets"]["40"]["conditions"]
        self.assertGreater(
            budget_40["prototype_retrieval_sample"]["heldout_accuracy_mean"],
            budget_40["random_sample"]["heldout_accuracy_mean"],
        )
        self.assertGreater(
            budget_40["random_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_40["prototype_retrieval_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        self.assertGreater(
            budget_40["validation_selector"]["charged_compute_units_mean"],
            budget_40["prototype_retrieval_sample"]["charged_compute_units_mean"],
        )

        budget_80 = artifact["budgets"]["80"]["conditions"]
        self.assertGreater(
            budget_80["class_balanced_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_80["random_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )

    def test_twenty_newsgroups_break_even_artifact_records_active_selection_math(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_break_even_analysis.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["reference_condition"], "random_sample")
        self.assertEqual(artifact["quality_metric"], "accuracy_improvement_over_majority_mean")
        self.assertEqual(artifact["quality_upper_bound"], 0.95)
        self.assertEqual(artifact["claim_scope"]["real_dataset"], True)
        self.assertEqual(artifact["claim_scope"]["metadata_stripped"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertIn("G_candidate / G_reference", artifact["theorem"]["general_inequality"])

        budget_40 = artifact["comparisons"]["40"]
        self.assertGreater(budget_40["prototype_retrieval_sample"]["quality_multiplier"], 1.0)
        self.assertLess(budget_40["prototype_retrieval_sample"]["density_ratio"], 1.0)
        self.assertFalse(budget_40["prototype_retrieval_sample"]["perfect_quality_can_beat"])

        budget_80 = artifact["comparisons"]["80"]
        self.assertTrue(budget_80["class_balanced_sample"]["candidate_density_wins"])
        self.assertGreater(budget_80["class_balanced_sample"]["density_ratio"], 1.0)
        self.assertLess(budget_80["class_balanced_sample"]["event_compute_multiplier"], 1.0)

        budget_160 = artifact["comparisons"]["160"]
        self.assertGreater(budget_160["prototype_retrieval_sample"]["quality_multiplier"], 1.0)
        self.assertLess(budget_160["prototype_retrieval_sample"]["density_ratio"], 1.0)
        self.assertIsNone(budget_160["prototype_retrieval_sample"]["amortized_reuses_to_density_win"])

    def test_twenty_newsgroups_retrieval_cost_audit_records_negative_optimization_attempt(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_retrieval_cost_audit.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(artifact["alphas"], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
        self.assertEqual(artifact["claim_scope"]["real_dataset"], True)
        self.assertEqual(artifact["claim_scope"]["metadata_stripped"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["post_hoc_optimization_attempt"], True)

        budget_80 = artifact["budgets"]["80"]
        self.assertEqual(budget_80["best_accuracy_alpha"], "0.25")
        self.assertEqual(budget_80["best_density_alpha"], "0.25")
        self.assertGreater(
            budget_80["alpha_results"]["0.25"]["heldout_accuracy_mean"],
            budget_80["alpha_results"]["0"]["heldout_accuracy_mean"],
        )
        self.assertLess(
            budget_80["alpha_results"]["0.25"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_80["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )

        budget_160 = artifact["budgets"]["160"]
        self.assertEqual(budget_160["best_density_alpha"], "0.5")
        self.assertGreater(
            budget_160["alpha_results"]["0.5"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_160["alpha_results"]["0"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        for budget_row in artifact["budgets"].values():
            for row in budget_row["alpha_results"].values():
                self.assertFalse(row["break_even_vs_random"]["candidate_density_wins"])

    def test_twenty_newsgroups_self_training_audit_records_noisy_pseudo_labels(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_self_training_audit.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(artifact["pseudo_multipliers"], [1, 2])
        self.assertEqual(artifact["filter_modes"], ["global_margin", "balanced_margin"])
        self.assertEqual(artifact["claim_scope"]["pseudo_labels_use_teacher_predictions"], True)
        self.assertEqual(artifact["claim_scope"]["oracle_train_labels_used_for_pseudo_label_selection"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)

        budget_40 = artifact["budgets"]["40"]
        self.assertEqual(budget_40["best_self_training_condition"], "class_balanced_self_training_balanced_margin_2x")
        self.assertLess(
            budget_40["condition_results"][budget_40["best_self_training_condition"]]["pseudo_label_agreement_mean"],
            0.2,
        )
        budget_80 = artifact["budgets"]["80"]
        self.assertEqual(budget_80["best_self_training_condition"], "class_balanced_self_training_balanced_margin_1x")
        self.assertLess(
            budget_80["condition_results"][budget_80["best_self_training_condition"]][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            budget_80["class_balanced_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        budget_160 = artifact["budgets"]["160"]
        self.assertLess(
            budget_160["condition_results"][budget_160["best_self_training_condition"]]["pseudo_label_agreement_mean"],
            0.25,
        )
        for budget_row in artifact["budgets"].values():
            for row in budget_row["condition_results"].values():
                self.assertLess(
                    row["signed_learning_signal_density_per_1m_event_compute_mean"],
                    budget_row["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
                )

    def test_twenty_newsgroups_active_acquisition_audit_records_active_label_costs(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_active_acquisition_audit.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(
            artifact["acquisition_modes"],
            [
                "margin_uncertainty",
                "balanced_margin_uncertainty",
                "short_margin_uncertainty",
                "confidence_curriculum",
            ],
        )
        self.assertEqual(artifact["claim_scope"]["true_labels_acquired_after_selection"], True)
        self.assertEqual(artifact["claim_scope"]["oracle_train_labels_used_for_acquisition"], False)
        self.assertEqual(artifact["claim_scope"]["validation_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)

        budget_40 = artifact["budgets"]["40"]
        self.assertEqual(
            budget_40["best_density_condition"],
            "class_balanced_seed_active_balanced_margin_uncertainty",
        )
        self.assertLess(
            budget_40["condition_results"][budget_40["best_density_condition"]][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            budget_40["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )

        budget_160 = artifact["budgets"]["160"]
        self.assertEqual(
            budget_160["best_density_condition"],
            "class_balanced_seed_active_short_margin_uncertainty",
        )
        best_160 = budget_160["condition_results"][budget_160["best_density_condition"]]
        self.assertEqual(best_160["break_even_vs_class_balanced"]["amortized_reuses_to_density_win"], 4)
        self.assertLess(
            best_160["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_160["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        for budget_row in artifact["budgets"].values():
            for row in budget_row["condition_results"].values():
                self.assertFalse(row["break_even_vs_random"]["candidate_density_wins"])
                self.assertFalse(row["break_even_vs_class_balanced"]["candidate_density_wins"])

    def test_twenty_newsgroups_budgeted_acquisition_audit_records_budgeted_scan_frontier(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_budgeted_acquisition_audit.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(
            artifact["acquisition_modes"],
            ["margin_uncertainty", "balanced_margin_uncertainty", "short_margin_uncertainty"],
        )
        self.assertEqual(artifact["scan_multipliers"], [1, 2, 4])
        self.assertEqual(artifact["claim_scope"]["scan_window_sampled_without_text_scoring"], True)
        self.assertEqual(artifact["claim_scope"]["true_labels_acquired_after_selection"], True)
        self.assertEqual(artifact["claim_scope"]["oracle_train_labels_used_for_acquisition"], False)
        self.assertEqual(artifact["claim_scope"]["validation_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)

        budget_160 = artifact["budgets"]["160"]
        self.assertEqual(budget_160["best_density_condition"], "budgeted_active_margin_uncertainty_2x")
        best_160 = budget_160["condition_results"][budget_160["best_density_condition"]]
        self.assertEqual(best_160["external_events_mean"], 160.0)
        self.assertEqual(best_160["scan_window_size_mean"], 240.0)
        self.assertTrue(best_160["break_even_vs_class_balanced"]["candidate_density_wins"])
        self.assertFalse(best_160["break_even_vs_random"]["candidate_density_wins"])

        for budget, budget_row in artifact["budgets"].items():
            for row in budget_row["condition_results"].values():
                self.assertEqual(row["external_events_mean"], float(budget))
                self.assertFalse(row["break_even_vs_random"]["candidate_density_wins"])

    def test_twenty_newsgroups_length_window_audit_records_mixed_confirmation(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_length_window_confirmation_audit.json").read_text())

        self.assertEqual(artifact["source_artifacts"], ["results/twenty_newsgroups_active_selection.json"])
        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["dataset"]["record_count"], 1998)
        self.assertEqual(artifact["development_seeds"], [311, 313, 317])
        self.assertEqual(artifact["confirmation_seeds"], [331, 337, 347, 349, 353])
        self.assertEqual(artifact["claim_scope"]["scan_window_sampled_before_length_selection"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["validation_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["teacher_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["oracle_train_labels_used_for_selection"], False)
        self.assertEqual(artifact["development_density_win_count"], 0)
        self.assertEqual(artifact["confirmation_same_condition_density_win_count"], 1)

        dev_80 = artifact["phases"]["development"]["budgets"]["80"]
        confirm_80 = artifact["phases"]["confirmation"]["budgets"]["80"]
        self.assertEqual(dev_80["best_density_condition"], "length_window_shortest_2x")
        self.assertFalse(
            dev_80["condition_results"]["length_window_shortest_2x"]["break_even_vs_random"][
                "candidate_density_wins"
            ]
        )
        self.assertEqual(confirm_80["density_win_conditions"], [])
        self.assertLess(
            confirm_80["condition_results"]["length_window_shortest_2x"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            confirm_80["random_reference"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )

    def test_twenty_newsgroups_frontier_robustness_audit_demotes_fragile_mean_wins(self) -> None:
        artifact = json.loads(Path("results/twenty_newsgroups_frontier_robustness_audit.json").read_text())

        self.assertEqual(
            artifact["source_artifacts"],
            [
                "results/twenty_newsgroups_active_selection.json",
                "results/twenty_newsgroups_budgeted_acquisition_audit.json",
            ],
        )
        self.assertEqual(artifact["claim_scope"]["paired_seed_audit"], True)
        self.assertEqual(artifact["claim_scope"]["exact_seed_bootstrap"], True)
        self.assertEqual(artifact["claim_scope"]["introduces_new_policy"], False)

        summary = artifact["summary"]
        self.assertEqual(summary["comparisons"], 6)
        self.assertEqual(summary["mean_density_wins"], 3)
        self.assertEqual(summary["robust_density_wins"], 0)
        self.assertEqual(summary["robust_density_losses"], 3)

        by_name = {row["comparison"]: row for row in artifact["comparisons"]}
        class_80 = by_name["class_balanced_80_vs_random"]
        self.assertEqual(class_80["paired_win_count"], 2)
        self.assertGreater(class_80["mean_density_ratio"], 1.0)
        self.assertLess(class_80["bootstrap"]["density_delta_ci95"][0], 0.0)
        self.assertFalse(class_80["robust_density_win"])

        budgeted_2x = by_name["budgeted_margin_2x_160_vs_class_balanced"]
        self.assertGreater(budgeted_2x["mean_density_ratio"], 1.0)
        self.assertFalse(budgeted_2x["robust_density_win"])

        prototype_160 = by_name["prototype_160_vs_random"]
        self.assertTrue(prototype_160["robust_density_loss"])
        self.assertEqual(prototype_160["paired_win_count"], 0)

    def test_real_text_break_even_certificate_records_global_frontier(self) -> None:
        artifact = json.loads(Path("results/real_text_break_even_certificate.json").read_text())

        self.assertEqual(artifact["title"], "Real-Text Break-Even Frontier Certificate")
        self.assertEqual(artifact["claim_scope"]["mathematical_certificate"], True)
        self.assertEqual(artifact["claim_scope"]["introduces_new_policy"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertIn("G_candidate / G_reference", artifact["theorem"]["density_condition"])

        summary = artifact["summary"]
        self.assertEqual(summary["rows"], 172)
        self.assertEqual(summary["observed_quality_wins"], 38)
        self.assertEqual(summary["density_wins"], 3)
        self.assertEqual(summary["quality_win_density_losses"], 36)
        self.assertEqual(summary["finite_reuse_needed"], 13)
        self.assertEqual(summary["bounded_quality_impossible_at_k1"], 53)

        strongest = summary["strongest_observed_density_win"]
        self.assertEqual(strongest["artifact_label"], "Twenty Newsgroups active")
        self.assertEqual(strongest["budget"], "80")
        self.assertEqual(strongest["candidate_condition"], "class_balanced_sample")
        self.assertEqual(strongest["density_ratio"], 1.180025)

        self.assertEqual(
            summary["cheapest_finite_reuse_frontier"]["amortized_reuses_to_density_win"],
            2,
        )
        self.assertEqual(summary["families"]["twenty_newsgroups_self_training"]["density_wins"], 0)
        self.assertEqual(summary["families"]["twenty_newsgroups_active_acquisition"]["density_wins"], 0)
        self.assertEqual(summary["families"]["twenty_newsgroups_budgeted_active_acquisition"]["density_wins"], 2)

    def test_sms_spam_real_text_artifacts_record_dataset_scope_and_cost_tradeoff(self) -> None:
        default = json.loads(Path("results/sms_spam_real_text_selection_cost.json").read_text())
        v200 = json.loads(Path("results/sms_spam_real_text_selection_cost_v200.json").read_text())

        for artifact in (default, v200):
            self.assertEqual(artifact["dataset"]["name"], "UCI SMS Spam Collection")
            self.assertEqual(artifact["dataset"]["record_count"], 5574)
            self.assertEqual(artifact["dataset"]["license"], "CC BY 4.0")
            self.assertEqual(
                artifact["dataset"]["sha256"],
                "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
            )
            self.assertEqual(artifact["claim_scope"]["synthetic_domain"], False)
            self.assertEqual(artifact["claim_scope"]["real_dataset"], True)
            self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
            self.assertEqual(artifact["claim_scope"]["paper_ready_claim"], False)
            self.assertIn("label_index_balanced_sample", artifact["condition_scope"])
            self.assertIn("validation_label_index_selector", artifact["condition_scope"])

        budget_32 = default["budgets"]["32"]["conditions"]
        self.assertGreater(
            budget_32["label_index_balanced_sample"]["heldout_spam_f1_mean"],
            budget_32["random_sample"]["heldout_spam_f1_mean"],
        )
        self.assertGreater(
            budget_32["random_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_32["label_index_balanced_sample"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        self.assertGreater(
            budget_32["validation_label_index_selector"]["signed_learning_signal_density_per_1m_event_compute_mean"],
            budget_32["validation_sample_selector"]["signed_learning_signal_density_per_1m_event_compute_mean"],
        )
        self.assertLess(
            v200["budgets"]["32"]["conditions"]["validation_label_index_selector"]["charged_compute_units_mean"],
            default["budgets"]["32"]["conditions"]["validation_label_index_selector"]["charged_compute_units_mean"],
        )

    def test_sms_spam_break_even_artifact_records_selector_cost_inequality(self) -> None:
        artifact = json.loads(Path("results/sms_spam_break_even_analysis.json").read_text())

        self.assertEqual(artifact["reference_condition"], "random_sample")
        self.assertEqual(
            artifact["theorem"]["general_inequality"],
            "G_candidate / G_reference > (N_candidate C_candidate) / (N_reference C_reference)",
        )
        self.assertEqual(artifact["claim_scope"]["post_hoc_diagnostic"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(artifact["quality_upper_bound"], 1.0)
        self.assertEqual(
            artifact["amortization_model"]["reusable_compute_keys"],
            ["selection_cost_tokens_mean", "validation_tuning_cost_tokens_mean"],
        )

        v800_budget_32 = artifact["comparisons"]["SMS Spam v800"]["32"]
        self.assertGreater(v800_budget_32["label_index_balanced_sample"]["break_even_quality"], 1.0)
        self.assertFalse(v800_budget_32["label_index_balanced_sample"]["perfect_quality_can_beat"])
        self.assertLess(v800_budget_32["label_index_balanced_sample"]["max_possible_density_ratio"], 1.0)
        self.assertEqual(v800_budget_32["label_index_balanced_sample"]["amortized_reuses_to_density_win"], 9)
        self.assertGreater(v800_budget_32["label_index_balanced_sample"]["fully_amortized_density_ratio"], 1.0)
        self.assertGreater(v800_budget_32["validation_label_index_selector"]["break_even_quality"], 1.0)
        self.assertFalse(v800_budget_32["validation_label_index_selector"]["perfect_quality_can_beat"])
        self.assertLess(v800_budget_32["validation_label_index_selector"]["max_possible_density_ratio"], 0.2)
        self.assertEqual(v800_budget_32["validation_label_index_selector"]["amortized_reuses_to_density_win"], 47)
        self.assertEqual(v800_budget_32["validation_label_index_selector"]["candidate_density_wins"], False)

        v200_budget_32 = artifact["comparisons"]["SMS Spam v200"]["32"]["validation_label_index_selector"]
        self.assertIsNone(v200_budget_32["amortized_reuses_to_density_win"])
        self.assertLess(v200_budget_32["fully_amortized_density_ratio"], 1.0)

        for by_budget in artifact["comparisons"].values():
            for by_condition in by_budget.values():
                for row in by_condition.values():
                    self.assertLess(row["density_ratio"], 1.0)

    def test_confirmation_sweep_uses_disjoint_seeds_and_records_sample_aware_result(self) -> None:
        discovery = json.loads(Path("results/sample_budget_sweep.json").read_text())
        confirmation = json.loads(Path("results/confirmation_budget_sweep.json").read_text())

        self.assertTrue(set(discovery["seeds"]).isdisjoint(confirmation["seeds"]))
        self.assertEqual(confirmation["confirmation_of"], "results/sample_budget_sweep.json")
        self.assertEqual(confirmation["claim_scope"]["paper_ready_claim"], False)
        self.assertEqual(confirmation["claim_scope"]["fresh_seed_confirmation"], True)

        thresholds = confirmation["thresholds"]
        sample_aware = thresholds["sample_aware_self_ranked_induction"]
        self_ranked = thresholds["self_ranked_induction"]
        self.assertGreater(sample_aware["best_signed_gain"], self_ranked["best_signed_gain"])
        self.assertEqual(sample_aware["first_material_count_reaching_target"], 48)

        budget_24 = confirmation["budgets"]["24"]
        self.assertGreater(
            budget_24["sample_aware_self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
            budget_24["self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
        )

    def test_feature_dimension_artifacts_record_256_frontier_and_low_budget_limits(self) -> None:
        feature = json.loads(Path("results/tiny_neural_feature_sweep.json").read_text())
        budget_128 = json.loads(Path("results/tiny_neural_budget_sweep_32x8.json").read_text())
        budget_256 = json.loads(Path("results/tiny_neural_budget_sweep_32x8_f256.json").read_text())

        self.assertEqual(feature["profile_label"], "epochs=32_hidden=8")
        self.assertEqual(feature["comparison_of"], "results/tiny_neural_budget_sweep_32x8.json")
        self.assertEqual(feature["feature_dimensions"], [16, 32, 64, 128, 256])
        self.assertEqual(budget_256["profile_label"], "epochs=32_hidden=8_features=256")
        self.assertEqual(budget_256["comparison_of"], "results/tiny_neural_budget_sweep_32x8.json")

        self.assertEqual(feature["frontier"]["self_ranked_induction"]["best_signed_gain_feature_dimension"], 256)
        self.assertLess(
            feature["dimension_results"]["features=64"]["conditions"]["self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0,
        )
        self.assertGreater(
            feature["dimension_results"]["features=256"]["conditions"]["self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            feature["dimension_results"]["features=128"]["conditions"]["self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )

        self.assertEqual(budget_256["thresholds"]["raw_text"]["first_material_count_reaching_target"], 64)
        self.assertEqual(budget_256["thresholds"]["self_ranked_induction"]["first_material_count_reaching_target"], 48)
        self.assertEqual(budget_256["thresholds"]["counterfactual_expansion"]["first_material_count_reaching_target"], 32)
        self.assertGreater(
            budget_256["thresholds"]["self_ranked_induction"]["best_signed_gain"],
            budget_128["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertLess(
            budget_256["budgets"]["32"]["self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
            0,
        )
        ops_ratio = (
            budget_256["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
            / budget_128["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
        )
        self.assertLess(ops_ratio, 1.04)

    def test_wide_feature_dimension_artifact_records_collision_frontier(self) -> None:
        wide = json.loads(Path("results/tiny_neural_feature_sweep_wide.json").read_text())

        self.assertEqual(wide["profile_label"], "epochs=32_hidden=8")
        self.assertEqual(wide["comparison_of"], "results/tiny_neural_feature_sweep.json")
        self.assertEqual(wide["feature_dimensions"], [128, 256, 512, 1024])
        self.assertEqual(wide["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(wide["claim_scope"]["fresh_seed_confirmation"], True)

        self.assertEqual(wide["frontier"]["self_ranked_induction"]["best_signed_gain_feature_dimension"], 1024)
        self.assertEqual(
            wide["frontier"]["sample_aware_self_ranked_induction"]["best_signed_gain_feature_dimension"],
            1024,
        )
        self.assertEqual(wide["frontier"]["counterfactual_expansion"]["best_signed_gain_feature_dimension"], 1024)
        self.assertGreater(
            wide["dimension_results"]["features=512"]["conditions"]["self_ranked_induction"][
                "estimated_neural_training_multiply_adds_mean"
            ],
            wide["dimension_results"]["features=256"]["conditions"]["self_ranked_induction"][
                "estimated_neural_training_multiply_adds_mean"
            ],
        )

    def test_wide_feature_budget_artifact_confirms_1024_frontier(self) -> None:
        budget_256 = json.loads(Path("results/tiny_neural_budget_sweep_32x8_f256.json").read_text())
        budget_1024 = json.loads(Path("results/tiny_neural_budget_sweep_32x8_f1024.json").read_text())

        self.assertEqual(budget_1024["feature_dimension"], 1024)
        self.assertEqual(budget_1024["profile_label"], "epochs=32_hidden=8_features=1024")
        self.assertEqual(budget_1024["comparison_of"], "results/tiny_neural_budget_sweep_32x8_f256.json")
        self.assertEqual(budget_1024["thresholds"]["raw_text"]["first_material_count_reaching_target"], 48)
        self.assertEqual(budget_1024["thresholds"]["self_ranked_induction"]["first_material_count_reaching_target"], 48)
        self.assertEqual(budget_1024["thresholds"]["counterfactual_expansion"]["first_material_count_reaching_target"], 24)
        self.assertGreater(
            budget_1024["thresholds"]["self_ranked_induction"]["best_signed_gain"],
            budget_256["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertGreater(
            budget_1024["thresholds"]["counterfactual_expansion"]["best_signed_gain"],
            budget_256["thresholds"]["counterfactual_expansion"]["best_signed_gain"],
        )
        self.assertLess(
            budget_1024["budgets"]["32"]["self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
            0,
        )
        ops_ratio = (
            budget_1024["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
            / budget_256["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
        )
        self.assertLess(ops_ratio, 1.03)

    def test_f1024_profile_artifact_records_capacity_frontier(self) -> None:
        profile = json.loads(Path("results/tiny_neural_profile_sweep_f1024.json").read_text())

        self.assertEqual(profile["feature_dimension"], 1024)
        self.assertEqual(profile["material_count"], 64)
        self.assertEqual(profile["confirmation_of"], "results/tiny_neural_budget_sweep_32x8_f1024.json")
        self.assertEqual(profile["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(profile["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn({"epochs": 32, "hidden_units": 8}, profile["profiles"])
        self.assertIn("self_ranked_induction", profile["frontier"])
        self.assertGreater(
            profile["profile_results"]["epochs=32_hidden=8"]["conditions"]["self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0,
        )

    def test_f1024_16x8_budget_artifact_confirms_lower_ops_self_ranked_profile(self) -> None:
        budget_32x8 = json.loads(Path("results/tiny_neural_budget_sweep_32x8_f1024.json").read_text())
        budget_16x8 = json.loads(Path("results/tiny_neural_budget_sweep_16x8_f1024.json").read_text())

        self.assertEqual(budget_16x8["feature_dimension"], 1024)
        self.assertEqual(budget_16x8["profile_label"], "epochs=16_hidden=8_features=1024")
        self.assertEqual(budget_16x8["comparison_of"], "results/tiny_neural_budget_sweep_32x8_f1024.json")
        self.assertIsNone(budget_16x8["thresholds"]["raw_text"]["first_material_count_reaching_target"])
        self.assertEqual(budget_16x8["thresholds"]["self_ranked_induction"]["first_material_count_reaching_target"], 48)
        self.assertEqual(
            budget_16x8["thresholds"]["counterfactual_expansion"]["first_material_count_reaching_target"],
            24,
        )
        self.assertGreater(
            budget_16x8["thresholds"]["self_ranked_induction"]["best_signed_gain"],
            budget_32x8["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertLess(
            budget_16x8["thresholds"]["counterfactual_expansion"]["best_signed_gain"],
            budget_32x8["thresholds"]["counterfactual_expansion"]["best_signed_gain"],
        )
        self.assertLess(
            budget_16x8["budgets"]["32"]["self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
            0,
        )
        ops_ratio = (
            budget_16x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
            / budget_32x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
        )
        self.assertEqual(ops_ratio, 0.5)

    def test_f1024_8x8_budget_artifact_records_low_epoch_ablation(self) -> None:
        budget_16x8 = json.loads(Path("results/tiny_neural_budget_sweep_16x8_f1024.json").read_text())
        budget_8x8 = json.loads(Path("results/tiny_neural_budget_sweep_8x8_f1024.json").read_text())

        self.assertEqual(budget_8x8["feature_dimension"], 1024)
        self.assertEqual(budget_8x8["profile_label"], "epochs=8_hidden=8_features=1024")
        self.assertEqual(budget_8x8["comparison_of"], "results/tiny_neural_budget_sweep_16x8_f1024.json")
        self.assertEqual(budget_8x8["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(budget_8x8["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIsNone(budget_8x8["thresholds"]["self_ranked_induction"]["first_material_count_reaching_target"])
        self.assertEqual(
            budget_8x8["thresholds"]["sample_aware_self_ranked_induction"]["first_material_count_reaching_target"],
            64,
        )
        self.assertGreater(
            budget_8x8["budgets"]["16"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0,
        )
        self.assertGreater(
            budget_8x8["budgets"]["24"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0,
        )
        self.assertLess(
            budget_8x8["thresholds"]["self_ranked_induction"]["best_signed_gain"],
            budget_16x8["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertLess(
            budget_8x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"],
            budget_16x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"],
        )
        ops_ratio = (
            budget_8x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
            / budget_16x8["budgets"]["64"]["self_ranked_induction"]["estimated_neural_training_multiply_adds_mean"]
        )
        self.assertEqual(ops_ratio, 0.5)

    def test_f1024_validation_selected_budget_artifact_records_charged_reliability_probe(self) -> None:
        validation_selected = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_selected_f1024.json").read_text()
        )

        self.assertEqual(validation_selected["feature_dimension"], 1024)
        self.assertEqual(validation_selected["profile_label"], "epochs=16_hidden=8_features=1024_validation_selected")
        self.assertEqual(validation_selected["comparison_of"], "results/tiny_neural_budget_sweep_16x8_f1024.json")
        self.assertEqual(validation_selected["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(validation_selected["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_ranked_induction", validation_selected["conditions"])
        self.assertIn("mdl_rule_expansion", validation_selected["conditions"])
        self.assertEqual(
            validation_selected["thresholds"]["validation_ranked_induction"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertEqual(validation_selected["thresholds"]["mdl_rule_expansion"]["first_material_count_reaching_target"], 48)
        self.assertLess(
            validation_selected["budgets"]["32"]["validation_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0,
        )
        self.assertGreater(
            validation_selected["budgets"]["32"]["mdl_rule_expansion"][
                "accuracy_improvement_over_majority_mean"
            ],
            validation_selected["budgets"]["32"]["self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            validation_selected["thresholds"]["mdl_rule_expansion"]["best_signed_gain"],
            validation_selected["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertGreater(
            validation_selected["budgets"]["64"]["mdl_rule_expansion"]["charged_compute_units_mean"],
            validation_selected["budgets"]["64"]["self_ranked_induction"]["charged_compute_units_mean"],
        )

    def test_f1024_agreement_gated_budget_artifact_records_train_only_reliability_probe(self) -> None:
        agreement_gated = json.loads(
            Path("results/tiny_neural_budget_sweep_agreement_gated_f1024.json").read_text()
        )

        self.assertEqual(agreement_gated["feature_dimension"], 1024)
        self.assertEqual(agreement_gated["profile_label"], "epochs=16_hidden=8_features=1024_agreement_gated")
        self.assertEqual(agreement_gated["comparison_of"], "results/tiny_neural_budget_sweep_validation_selected_f1024.json")
        self.assertEqual(agreement_gated["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(agreement_gated["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("agreement_gated_self_ranked_induction", agreement_gated["conditions"])
        self.assertEqual(
            agreement_gated["thresholds"]["agreement_gated_self_ranked_induction"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertLess(
            agreement_gated["budgets"]["16"]["agreement_gated_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            agreement_gated["budgets"]["16"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertEqual(
            agreement_gated["budgets"]["32"]["agreement_gated_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            agreement_gated["budgets"]["32"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            agreement_gated["thresholds"]["agreement_gated_self_ranked_induction"]["best_signed_gain"],
            agreement_gated["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )
        self.assertLess(
            agreement_gated["budgets"]["32"]["agreement_gated_self_ranked_induction"][
                "charged_compute_units_mean"
            ],
            agreement_gated["budgets"]["32"]["sample_aware_self_ranked_induction"][
                "charged_compute_units_mean"
            ],
        )

    def test_f1024_validation_portfolio_artifact_records_deployable_selector_probe(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_portfolio_f1024.json").read_text()
        )

        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "epochs=16_hidden=8_features=1024_validation_portfolio")
        self.assertEqual(selector["comparison_of"], "results/policy_envelope_f1024.json")
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_portfolio_selector", selector["conditions"])

        scope = selector["condition_scope"]["validation_portfolio_selector"]
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material in ("16", "24", "32", "48", "64"):
            row = selector["budgets"][material]["validation_portfolio_selector"]
            self.assertEqual(row["portfolio_candidate_count_mean"], 6)
            self.assertGreater(row["portfolio_selection_cost_units_mean"], 0)
            self.assertIn("portfolio_selected_condition_counts", row)

        self.assertEqual(
            selector["thresholds"]["validation_portfolio_selector"]["first_material_count_reaching_target"],
            48,
        )
        self.assertLess(
            selector["thresholds"]["validation_portfolio_selector"]["best_signed_gain"],
            selector["thresholds"]["self_ranked_induction"]["best_signed_gain"],
        )

    def test_f1024_validation_linear_proxy_artifact_records_low_fidelity_selector_probe(self) -> None:
        proxy = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json").read_text()
        )
        portfolio = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_portfolio_f1024.json").read_text()
        )

        self.assertEqual(proxy["feature_dimension"], 1024)
        self.assertEqual(proxy["profile_label"], "epochs=16_hidden=8_features=1024_validation_linear_proxy")
        self.assertEqual(proxy["comparison_of"], "results/tiny_neural_budget_sweep_validation_portfolio_f1024.json")
        self.assertEqual(proxy["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(proxy["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_linear_proxy_selector", proxy["conditions"])

        scope = proxy["condition_scope"]["validation_linear_proxy_selector"]
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["low_fidelity_proxy_selector"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material in ("16", "24", "32", "48", "64"):
            row = proxy["budgets"][material]["validation_linear_proxy_selector"]
            self.assertEqual(row["portfolio_candidate_count_mean"], 6)
            self.assertEqual(row["portfolio_proxy_epochs_mean"], 2)
            self.assertGreater(row["portfolio_selection_cost_units_mean"], 0)
            self.assertIn("portfolio_selected_condition_counts", row)
            self.assertLess(
                row["portfolio_selection_cost_units_mean"],
                portfolio["budgets"][material]["validation_portfolio_selector"][
                    "portfolio_selection_cost_units_mean"
                ],
            )

        self.assertEqual(
            proxy["thresholds"]["validation_linear_proxy_selector"]["first_material_count_reaching_target"],
            48,
        )
        self.assertGreater(
            proxy["thresholds"]["validation_linear_proxy_selector"]["best_signed_gain"],
            portfolio["thresholds"]["validation_portfolio_selector"]["best_signed_gain"],
        )
        self.assertGreater(
            proxy["budgets"]["64"]["validation_linear_proxy_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            portfolio["budgets"]["64"]["validation_portfolio_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_validation_abstaining_proxy_artifact_records_raw_text_abstention_probe(self) -> None:
        abstaining = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json").read_text()
        )
        proxy = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json").read_text()
        )

        self.assertEqual(abstaining["feature_dimension"], 1024)
        self.assertEqual(
            abstaining["profile_label"],
            "epochs=16_hidden=8_features=1024_validation_abstaining_proxy",
        )
        self.assertEqual(
            abstaining["comparison_of"],
            "results/tiny_neural_budget_sweep_validation_linear_proxy_f1024.json",
        )
        self.assertEqual(abstaining["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(abstaining["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_abstaining_proxy_selector", abstaining["conditions"])

        scope = abstaining["condition_scope"]["validation_abstaining_proxy_selector"]
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["low_fidelity_proxy_selector"], True)
        self.assertEqual(scope["raw_text_abstention"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material in ("16", "24", "32", "48", "64"):
            row = abstaining["budgets"][material]["validation_abstaining_proxy_selector"]
            self.assertEqual(row["portfolio_candidate_count_mean"], 6)
            self.assertEqual(row["portfolio_proxy_epochs_mean"], 2)
            self.assertEqual(row["portfolio_abstention_extra_correct_mean"], 3)
            self.assertGreaterEqual(row["portfolio_raw_text_abstention_mean"], 0)
            self.assertIn("portfolio_selected_condition_counts", row)
            self.assertLessEqual(
                row["charged_compute_units_mean"],
                proxy["budgets"][material]["validation_linear_proxy_selector"]["charged_compute_units_mean"],
            )

        self.assertEqual(
            abstaining["thresholds"]["validation_abstaining_proxy_selector"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertEqual(
            abstaining["budgets"]["16"]["validation_abstaining_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.0,
        )
        self.assertGreater(
            abstaining["budgets"]["24"]["validation_abstaining_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            proxy["budgets"]["24"]["validation_linear_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
        )

    def test_f1024_selector_transfer_artifact_records_fresh_seed_stress_test(self) -> None:
        transfer = json.loads(
            Path("results/tiny_neural_budget_sweep_selector_transfer_f1024.json").read_text()
        )

        self.assertEqual(transfer["seeds"], [37, 41, 43, 47, 53])
        self.assertEqual(transfer["feature_dimension"], 1024)
        self.assertEqual(transfer["profile_label"], "epochs=16_hidden=8_features=1024_selector_transfer")
        self.assertEqual(
            transfer["comparison_of"],
            "results/tiny_neural_budget_sweep_validation_abstaining_proxy_f1024.json",
        )
        self.assertEqual(transfer["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(transfer["claim_scope"]["fresh_seed_confirmation"], True)

        for condition in (
            "validation_abstaining_proxy_selector",
            "validation_linear_proxy_selector",
            "validation_portfolio_selector",
            "sample_aware_self_ranked_induction",
            "counterfactual_expansion",
        ):
            self.assertIn(condition, transfer["conditions"])

        abstaining = transfer["budgets"]["32"]["validation_abstaining_proxy_selector"]
        linear = transfer["budgets"]["32"]["validation_linear_proxy_selector"]
        portfolio = transfer["budgets"]["64"]["validation_portfolio_selector"]
        sample = transfer["budgets"]["64"]["sample_aware_self_ranked_induction"]

        self.assertEqual(abstaining["accuracy_improvement_over_majority_mean"], -0.084211)
        self.assertEqual(linear["accuracy_improvement_over_majority_mean"], -0.084211)
        self.assertEqual(portfolio["accuracy_improvement_over_majority_mean"], 0.127273)
        self.assertGreater(
            sample["accuracy_improvement_over_majority_mean"],
            transfer["budgets"]["64"]["validation_abstaining_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertEqual(
            transfer["thresholds"]["validation_abstaining_proxy_selector"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertEqual(
            transfer["thresholds"]["validation_abstaining_proxy_selector"]["best_signed_gain"],
            0.114286,
        )

    def test_f1024_train_size_gated_artifact_records_unseen_seed_schedule_probe(self) -> None:
        gated = json.loads(
            Path("results/tiny_neural_budget_sweep_train_size_gated_f1024.json").read_text()
        )

        self.assertEqual(gated["seeds"], [59, 61, 67, 71, 73])
        self.assertEqual(gated["feature_dimension"], 1024)
        self.assertEqual(gated["profile_label"], "epochs=16_hidden=8_features=1024_train_size_gated")
        self.assertEqual(
            gated["comparison_of"],
            "results/tiny_neural_budget_sweep_selector_transfer_f1024.json",
        )
        self.assertEqual(gated["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(gated["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("train_size_gated_sample_aware_induction", gated["conditions"])

        scope = gated["condition_scope"]["train_size_gated_sample_aware_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            gated["budgets"]["16"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.031579,
        )
        self.assertEqual(
            gated["budgets"]["32"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.073684,
        )
        self.assertEqual(
            gated["budgets"]["48"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.103448,
        )
        self.assertEqual(
            gated["thresholds"]["train_size_gated_sample_aware_induction"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertEqual(
            gated["thresholds"]["train_size_gated_sample_aware_induction"]["best_signed_gain"],
            0.145454,
        )
        self.assertGreater(
            gated["budgets"]["64"]["train_size_gated_sample_aware_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            gated["budgets"]["64"]["validation_abstaining_proxy_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_generated_label_audit_artifact_records_label_precision_not_gain_mechanism(self) -> None:
        audit = json.loads(Path("results/generated_label_audit_selector_transfer_f1024.json").read_text())

        self.assertEqual(audit["seeds"], [37, 41, 43, 47, 53])
        self.assertEqual(audit["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(
            audit["source_artifacts"],
            ["results/tiny_neural_budget_sweep_selector_transfer_f1024.json"],
        )
        self.assertEqual(audit["claim_scope"]["uses_hidden_rulebook_for_audit"], True)
        self.assertEqual(audit["claim_scope"]["deployable_policy"], False)
        self.assertEqual(audit["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(audit["claim_scope"]["paper_ready_claim"], False)

        material32 = audit["audits"]["32"]
        self.assertIsNone(material32["raw_text"]["label_precision"])
        self.assertEqual(
            material32["agreement_gated_self_ranked_induction"]["label_precision"],
            0.917241,
        )
        self.assertEqual(
            material32["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            -0.131579,
        )
        self.assertEqual(
            material32["sample_aware_self_ranked_induction"]["label_precision"],
            0.675862,
        )
        self.assertEqual(
            material32["sample_aware_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            -0.173684,
        )
        self.assertEqual(material32["counterfactual_expansion"]["label_precision"], 1.0)
        self.assertEqual(
            material32["counterfactual_expansion"]["linked_accuracy_improvement_over_majority_mean"],
            0.01579,
        )

        material64 = audit["audits"]["64"]
        self.assertGreater(
            material64["agreement_gated_self_ranked_induction"]["label_precision"],
            material64["sample_aware_self_ranked_induction"]["label_precision"],
        )
        self.assertLess(
            material64["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            material64["sample_aware_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
        )

    def test_generated_coverage_audit_artifact_records_distribution_not_precision_mechanism(self) -> None:
        audit = json.loads(Path("results/generated_coverage_audit_selector_transfer_f1024.json").read_text())

        self.assertEqual(audit["seeds"], [37, 41, 43, 47, 53])
        self.assertEqual(audit["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(
            audit["source_artifacts"],
            ["results/tiny_neural_budget_sweep_selector_transfer_f1024.json"],
        )
        self.assertEqual(audit["claim_scope"]["uses_hidden_rulebook_for_label_audit"], True)
        self.assertEqual(audit["claim_scope"]["uses_heldout_distribution_for_audit"], True)
        self.assertEqual(audit["claim_scope"]["heldout_distribution_available_to_policies"], False)
        self.assertEqual(audit["claim_scope"]["deployable_policy"], False)
        self.assertEqual(audit["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(audit["claim_scope"]["paper_ready_claim"], False)

        material32 = audit["audits"]["32"]
        self.assertLess(
            material32["validation_ranked_induction"]["generated_vs_heldout_triple_l1_distance"],
            material32["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
        )
        self.assertGreater(
            material32["validation_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            material32["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertEqual(
            material32["validation_ranked_induction"]["generated_vs_heldout_triple_l1_distance"],
            0.683666,
        )

        material64 = audit["audits"]["64"]
        self.assertLess(
            material64["sample_aware_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            material64["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
        )
        self.assertGreater(
            material64["sample_aware_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            material64["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertEqual(
            material64["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            0.576443,
        )

    def test_f1024_validation_coverage_proxy_artifact_records_deployable_coverage_probe(self) -> None:
        coverage = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json").read_text()
        )

        self.assertEqual(coverage["seeds"], [103, 107, 109, 113, 127])
        self.assertEqual(coverage["feature_dimension"], 1024)
        self.assertEqual(
            coverage["profile_label"],
            "epochs=16_hidden=8_features=1024_validation_coverage_proxy",
        )
        self.assertEqual(
            coverage["comparison_of"],
            "results/generated_coverage_audit_selector_transfer_f1024.json",
        )
        self.assertEqual(coverage["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(coverage["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_coverage_proxy_selector", coverage["conditions"])

        scope = coverage["condition_scope"]["validation_coverage_proxy_selector"]
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_motif_distribution_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], False)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            coverage["budgets"]["16"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.0,
        )
        self.assertEqual(
            coverage["budgets"]["24"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.082759,
        )
        self.assertEqual(
            coverage["budgets"]["32"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.010526,
        )
        self.assertEqual(
            coverage["budgets"]["64"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.171428,
        )
        self.assertEqual(
            coverage["thresholds"]["validation_coverage_proxy_selector"][
                "first_material_count_reaching_target"
            ],
            64,
        )
        self.assertLess(
            coverage["budgets"]["64"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            coverage["budgets"]["64"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertGreater(
            coverage["budgets"]["32"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            coverage["budgets"]["32"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )

    def test_f1024_validation_coverage_prior_artifact_records_floor_and_lean_candidate_tradeoff(self) -> None:
        prior = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_coverage_prior_f1024.json").read_text()
        )

        self.assertEqual(prior["seeds"], [601, 607, 613, 617, 619])
        self.assertEqual(prior["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(prior["feature_dimension"], 1024)
        self.assertEqual(
            prior["profile_label"],
            "epochs=16_hidden=8_features=1024_validation_coverage_prior",
        )
        self.assertEqual(
            prior["confirmation_of"],
            "results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json",
        )
        self.assertEqual(
            prior["comparison_of"],
            "results/tiny_neural_budget_sweep_validation_coverage_proxy_f1024.json",
        )
        self.assertEqual(prior["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(prior["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_coverage_prior_selector", prior["conditions"])

        scope = prior["condition_scope"]["validation_coverage_prior_selector"]
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_motif_distribution_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], False)
        self.assertEqual(scope["train_size_prior_min_events"], 96)
        self.assertEqual(scope["lean_coverage_candidate_set"], True)
        self.assertEqual(scope["coverage_utility_compute_penalty"], 0.00001)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            prior["budgets"]["24"]["validation_coverage_prior_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.0,
        )
        self.assertGreater(
            prior["budgets"]["24"]["validation_coverage_prior_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            prior["budgets"]["24"]["validation_coverage_proxy_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertEqual(
            prior["budgets"]["24"]["validation_coverage_prior_selector"][
                "portfolio_candidate_count_mean"
            ],
            0,
        )
        self.assertEqual(
            prior["budgets"]["48"]["validation_coverage_prior_selector"][
                "portfolio_candidate_count_mean"
            ],
            3,
        )
        self.assertGreater(
            prior["budgets"]["48"]["validation_coverage_prior_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            prior["budgets"]["48"]["validation_coverage_proxy_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            prior["budgets"]["64"]["validation_coverage_prior_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            prior["budgets"]["64"]["train_size_gated_sample_aware_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertEqual(
            prior["thresholds"]["validation_coverage_prior_selector"]["first_material_count_reaching_target"],
            48,
        )

    def test_f1024_tempered_sample_aware_artifact_records_train_only_budget_tempering(self) -> None:
        tempered = json.loads(
            Path("results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json").read_text()
        )

        self.assertEqual(tempered["seeds"], [157, 163, 167, 173, 179])
        self.assertEqual(tempered["feature_dimension"], 1024)
        self.assertEqual(
            tempered["profile_label"],
            "epochs=16_hidden=8_features=1024_tempered_sample_aware",
        )
        self.assertEqual(
            tempered["comparison_of"],
            "results/tiny_neural_budget_sweep_train_size_gated_f1024.json",
        )
        self.assertEqual(tempered["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(tempered["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("tempered_sample_aware_self_ranked_induction", tempered["conditions"])

        scope = tempered["condition_scope"]["tempered_sample_aware_self_ranked_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            tempered["budgets"]["16"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.021053,
        )
        self.assertEqual(
            tempered["budgets"]["24"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.096552,
        )
        self.assertEqual(
            tempered["budgets"]["32"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            -0.078947,
        )
        self.assertEqual(
            tempered["budgets"]["64"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.124675,
        )
        self.assertEqual(
            tempered["thresholds"]["tempered_sample_aware_self_ranked_induction"][
                "first_material_count_reaching_target"
            ],
            48,
        )
        self.assertGreater(
            tempered["budgets"]["24"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            tempered["budgets"]["24"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertGreater(
            tempered["budgets"]["32"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            tempered["budgets"]["32"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            tempered["budgets"]["24"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            tempered["budgets"]["24"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            tempered["budgets"]["32"]["tempered_sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            tempered["budgets"]["32"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )

    def test_f1024_compact_train_size_gated_artifact_records_large_sample_density_gain(self) -> None:
        compact = json.loads(
            Path("results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json").read_text()
        )

        self.assertEqual(compact["seeds"], [181, 191, 193, 197, 199])
        self.assertEqual(compact["feature_dimension"], 1024)
        self.assertEqual(compact["profile_label"], "f1024_16x8_compact_train_size_gated")
        self.assertEqual(
            compact["comparison_of"],
            "results/tiny_neural_budget_sweep_train_size_gated_f1024.json",
        )
        self.assertEqual(
            compact["confirmation_of"],
            "results/tiny_neural_budget_sweep_tempered_sample_aware_f1024.json",
        )
        self.assertEqual(compact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(compact["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("compact_train_size_gated_induction", compact["conditions"])

        scope = compact["condition_scope"]["compact_train_size_gated_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material_count in ("16", "24", "32", "48"):
            self.assertEqual(
                compact["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                compact["budgets"][material_count]["train_size_gated_sample_aware_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertEqual(
                compact["budgets"][material_count]["compact_train_size_gated_induction"][
                    "charged_compute_units_mean"
                ],
                compact["budgets"][material_count]["train_size_gated_sample_aware_induction"][
                    "charged_compute_units_mean"
                ],
            )

        self.assertEqual(
            compact["budgets"]["64"]["compact_train_size_gated_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.14026,
        )
        self.assertEqual(
            compact["budgets"]["64"]["compact_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            0.007883,
        )
        self.assertGreater(
            compact["budgets"]["64"]["compact_train_size_gated_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            compact["budgets"]["64"]["train_size_gated_sample_aware_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            compact["budgets"]["64"]["compact_train_size_gated_induction"][
                "charged_compute_units_mean"
            ],
            compact["budgets"]["64"]["train_size_gated_sample_aware_induction"][
                "charged_compute_units_mean"
            ],
        )
        self.assertGreater(
            compact["budgets"]["64"]["compact_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            compact["budgets"]["64"]["train_size_gated_sample_aware_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_diversity_interaction_artifact_records_train_only_tradeoff(self) -> None:
        diversity = json.loads(
            Path("results/tiny_neural_budget_sweep_diversity_interaction_f1024.json").read_text()
        )

        self.assertEqual(diversity["seeds"], [701, 709, 719, 727, 733])
        self.assertEqual(diversity["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(diversity["feature_dimension"], 1024)
        self.assertEqual(
            diversity["profile_label"],
            "epochs=16_hidden=8_features=1024_diversity_interaction",
        )
        self.assertEqual(
            diversity["comparison_of"],
            "results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json",
        )
        self.assertEqual(diversity["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(diversity["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("sample_aware_diverse_self_ranked_induction", diversity["conditions"])
        self.assertIn("compact_diverse_train_size_gated_induction", diversity["conditions"])

        sample_diverse_scope = diversity["condition_scope"]["sample_aware_diverse_self_ranked_induction"]
        self.assertEqual(sample_diverse_scope["train_only_selection"], True)
        self.assertEqual(sample_diverse_scope["train_only_induction"], True)
        self.assertEqual(sample_diverse_scope["validation_used_for_transform_selection"], False)
        self.assertEqual(sample_diverse_scope["oracle_generated_labels"], False)

        compact_diverse_scope = diversity["condition_scope"]["compact_diverse_train_size_gated_induction"]
        self.assertEqual(compact_diverse_scope["train_only_selection"], True)
        self.assertEqual(compact_diverse_scope["train_only_induction"], True)
        self.assertEqual(compact_diverse_scope["validation_used_for_policy_selection"], False)
        self.assertEqual(compact_diverse_scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(compact_diverse_scope["diversity_penalty_after_compaction"], True)
        self.assertEqual(compact_diverse_scope["oracle_generated_labels"], False)

        self.assertEqual(
            diversity["budgets"]["64"]["sample_aware_diverse_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.168831,
        )
        self.assertGreater(
            diversity["budgets"]["64"]["sample_aware_diverse_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            diversity["budgets"]["64"]["sample_aware_self_ranked_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertGreater(
            diversity["budgets"]["64"]["sample_aware_diverse_self_ranked_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            diversity["budgets"]["64"]["sample_aware_self_ranked_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            diversity["budgets"]["64"]["compact_diverse_train_size_gated_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            diversity["budgets"]["64"]["compact_train_size_gated_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            diversity["budgets"]["64"]["compact_diverse_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            diversity["budgets"]["64"]["compact_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertGreater(
            diversity["budgets"]["64"]["compact_diverse_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            diversity["budgets"]["64"]["sample_aware_diverse_self_ranked_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

        for material_count in ("16", "24", "32", "48"):
            self.assertEqual(
                diversity["budgets"][material_count]["compact_diverse_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                diversity["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )

        for material_count in ("24", "32", "64"):
            self.assertGreater(
                diversity["budgets"][material_count]["diverse_self_ranked_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                diversity["budgets"][material_count]["self_ranked_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )

    def test_f1024_density_capped_compact_artifact_records_abundant_raw_fallback(self) -> None:
        density = json.loads(
            Path("results/tiny_neural_budget_sweep_density_capped_compact_f1024.json").read_text()
        )

        self.assertEqual(density["seeds"], [293, 307, 311, 313, 317])
        self.assertEqual(density["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(density["feature_dimension"], 1024)
        self.assertEqual(density["profile_label"], "f1024_16x8_density_capped_compact")
        self.assertEqual(
            density["confirmation_of"],
            "results/tiny_neural_budget_sweep_compact_train_size_gated_f1024.json",
        )
        self.assertEqual(density["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(density["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("density_capped_compact_induction", density["conditions"])

        scope = density["condition_scope"]["density_capped_compact_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["abundant_data_raw_fallback"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material_count in ("64", "80", "96"):
            self.assertEqual(
                density["budgets"][material_count]["density_capped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                density["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertEqual(
                density["budgets"][material_count]["density_capped_compact_induction"][
                    "charged_compute_units_mean"
                ],
                density["budgets"][material_count]["compact_train_size_gated_induction"][
                    "charged_compute_units_mean"
                ],
            )

        for material_count in ("104", "112", "120", "128"):
            self.assertEqual(
                density["budgets"][material_count]["density_capped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                density["budgets"][material_count]["raw_text"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertEqual(
                density["budgets"][material_count]["density_capped_compact_induction"][
                    "charged_compute_units_mean"
                ],
                density["budgets"][material_count]["raw_text"]["charged_compute_units_mean"],
            )
            self.assertGreater(
                density["budgets"][material_count]["density_capped_compact_induction"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
                density["budgets"][material_count]["compact_train_size_gated_induction"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
            )

        self.assertEqual(
            density["budgets"]["128"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            0.003452,
        )
        self.assertLess(
            density["budgets"]["128"]["density_capped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            density["budgets"]["128"]["compact_train_size_gated_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )

    def test_f1024_support_ramped_compact_artifact_records_abundant_support_tradeoff(self) -> None:
        support_ramped = json.loads(
            Path("results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json").read_text()
        )

        self.assertEqual(support_ramped["seeds"], [401, 409, 419, 421, 431])
        self.assertEqual(support_ramped["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(support_ramped["feature_dimension"], 1024)
        self.assertEqual(support_ramped["profile_label"], "f1024_16x8_support_ramped_compact")
        self.assertEqual(
            support_ramped["confirmation_of"],
            "results/tiny_neural_budget_sweep_density_capped_compact_f1024.json",
        )
        self.assertEqual(
            support_ramped["comparison_of"],
            "results/tiny_neural_budget_sweep_density_capped_compact_f1024.json",
        )
        self.assertEqual(support_ramped["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(support_ramped["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("support_ramped_compact_induction", support_ramped["conditions"])

        scope = support_ramped["condition_scope"]["support_ramped_compact_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["abundant_data_support_ramp"], True)
        self.assertEqual(scope["abundant_data_support_ramp_min_events"], 360)
        self.assertEqual(scope["abundant_data_min_support"], 4)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material_count in ("64", "80", "96"):
            self.assertEqual(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                support_ramped["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertEqual(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "charged_compute_units_mean"
                ],
                support_ramped["budgets"][material_count]["compact_train_size_gated_induction"][
                    "charged_compute_units_mean"
                ],
            )

        for material_count in ("104", "112", "120", "128"):
            self.assertLess(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "charged_compute_units_mean"
                ],
                support_ramped["budgets"][material_count]["compact_train_size_gated_induction"][
                    "charged_compute_units_mean"
                ],
            )

        self.assertEqual(
            support_ramped["budgets"]["104"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            0.003735,
        )
        self.assertGreater(
            support_ramped["budgets"]["104"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            support_ramped["budgets"]["104"]["compact_train_size_gated_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertGreater(
            support_ramped["budgets"]["104"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            support_ramped["budgets"]["104"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        for material_count in ("112", "128"):
            self.assertGreater(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                support_ramped["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertGreater(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                support_ramped["budgets"][material_count]["density_capped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertGreater(
                support_ramped["budgets"][material_count]["support_ramped_compact_induction"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
                support_ramped["budgets"][material_count]["compact_train_size_gated_induction"][
                    "signed_learning_signal_density_per_1m_event_compute_mean"
                ],
            )
        self.assertEqual(
            support_ramped["budgets"]["128"]["support_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.184416,
        )
        self.assertLess(
            support_ramped["budgets"]["128"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            support_ramped["budgets"]["128"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_late_confidence_ramped_compact_artifact_records_late_confidence_tradeoff(self) -> None:
        late_confidence = json.loads(
            Path("results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json").read_text()
        )

        self.assertEqual(late_confidence["seeds"], [499, 503, 509, 521, 523])
        self.assertEqual(late_confidence["material_counts"], [96, 104, 112, 120, 128, 144, 160])
        self.assertEqual(late_confidence["feature_dimension"], 1024)
        self.assertEqual(late_confidence["profile_label"], "f1024_16x8_late_confidence_ramped_compact")
        self.assertEqual(
            late_confidence["confirmation_of"],
            "results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json",
        )
        self.assertEqual(
            late_confidence["comparison_of"],
            "results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json",
        )
        self.assertEqual(late_confidence["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(late_confidence["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("late_confidence_ramped_compact_induction", late_confidence["conditions"])

        scope = late_confidence["condition_scope"]["late_confidence_ramped_compact_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["abundant_data_support_ramp"], True)
        self.assertEqual(scope["abundant_data_support_ramp_min_events"], 360)
        self.assertEqual(scope["abundant_data_min_support"], 4)
        self.assertEqual(scope["late_confidence_ramp_min_events"], 432)
        self.assertEqual(scope["late_confidence_min_confidence"], 0.60)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            late_confidence["budgets"]["104"]["late_confidence_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.152,
        )
        self.assertGreater(
            late_confidence["budgets"]["104"]["late_confidence_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            late_confidence["budgets"]["104"]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        for material_count in ("96", "104", "112", "128", "144", "160"):
            self.assertEqual(
                late_confidence["budgets"][material_count]["late_confidence_ramped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                late_confidence["budgets"][material_count]["support_ramped_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )
            self.assertEqual(
                late_confidence["budgets"][material_count]["late_confidence_ramped_compact_induction"][
                    "charged_compute_units_mean"
                ],
                late_confidence["budgets"][material_count]["support_ramped_compact_induction"][
                    "charged_compute_units_mean"
                ],
            )

        self.assertEqual(
            late_confidence["budgets"]["120"]["late_confidence_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            0.004322,
        )
        self.assertGreater(
            late_confidence["budgets"]["120"]["late_confidence_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            late_confidence["budgets"]["120"]["support_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertGreater(
            late_confidence["budgets"]["120"]["late_confidence_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            late_confidence["budgets"]["120"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            late_confidence["budgets"]["120"]["late_confidence_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            late_confidence["budgets"]["120"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            late_confidence["budgets"]["160"]["late_confidence_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            late_confidence["budgets"]["160"]["density_capped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            late_confidence["thresholds"]["late_confidence_ramped_compact_induction"]["best_signed_gain"],
            late_confidence["thresholds"]["compact_train_size_gated_induction"]["best_signed_gain"],
        )

    def test_f1024_density_window_compact_artifact_records_local_window_tradeoff(self) -> None:
        density_window = json.loads(
            Path("results/tiny_neural_budget_sweep_density_window_compact_f1024.json").read_text()
        )

        self.assertEqual(density_window["seeds"], [929, 937, 941, 947, 953])
        self.assertEqual(density_window["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(density_window["feature_dimension"], 1024)
        self.assertEqual(density_window["profile_label"], "f1024_16x8_density_window_compact")
        self.assertEqual(
            density_window["confirmation_of"],
            "results/tiny_neural_budget_sweep_late_confidence_ramped_compact_f1024.json",
        )
        self.assertEqual(
            density_window["comparison_of"],
            "results/tiny_neural_budget_sweep_support_ramped_compact_f1024.json",
        )
        self.assertEqual(density_window["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(density_window["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("density_window_compact_induction", density_window["conditions"])

        scope = density_window["condition_scope"]["density_window_compact_induction"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["compact_original_encoding_at_large_samples"], True)
        self.assertEqual(scope["compact_density_window_max_events"], 320)
        self.assertEqual(scope["transition_support_window_min_events"], 400)
        self.assertEqual(scope["transition_support_window_max_events"], 432)
        self.assertEqual(scope["abundant_data_raw_fallback"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        for material_count in ("64", "80"):
            self.assertEqual(
                density_window["budgets"][material_count]["density_window_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                density_window["budgets"][material_count]["compact_train_size_gated_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )

        for material_count in ("96", "104", "120"):
            self.assertEqual(
                density_window["budgets"][material_count]["density_window_compact_induction"][
                    "accuracy_improvement_over_majority_mean"
                ],
                density_window["budgets"][material_count]["raw_text"][
                    "accuracy_improvement_over_majority_mean"
                ],
            )

        self.assertEqual(
            density_window["budgets"]["112"]["density_window_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
            0.135821,
        )
        self.assertGreater(
            density_window["budgets"]["112"]["density_window_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            density_window["budgets"]["112"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertGreater(
            density_window["budgets"]["120"]["density_window_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            density_window["budgets"]["120"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            density_window["budgets"]["128"]["density_window_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            density_window["budgets"]["128"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_train_support_density_artifact_records_charged_selector_overhead(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_train_support_density_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1033, 1039, 1049, 1051, 1061])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_train_support_density")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_density_window_compact_f1024.json",
        )
        self.assertEqual(
            selector["comparison_of"],
            "results/tiny_neural_budget_sweep_density_window_compact_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("train_support_density_selector", selector["conditions"])

        scope = selector["condition_scope"]["train_support_density_selector"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["support_density_selector"], True)
        self.assertEqual(scope["support_density_min_kept_per_compute"], 0.00145)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["64"]["train_support_density_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"compact_train_size_gated_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["104"]["train_support_density_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"support_ramped_compact_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["120"]["train_support_density_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 4, "support_ramped_compact_induction": 1},
        )
        self.assertEqual(
            selector["budgets"]["128"]["train_support_density_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )
        self.assertEqual(
            selector["budgets"]["104"]["train_support_density_selector"][
                "accuracy_improvement_over_majority_mean"
            ],
            selector["budgets"]["104"]["support_ramped_compact_induction"][
                "accuracy_improvement_over_majority_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["104"]["train_support_density_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["104"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["120"]["train_support_density_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["120"]["density_window_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["128"]["train_support_density_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["128"]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_support_probe_window_artifact_records_reuse_accounting_and_misses(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_support_probe_window_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1063, 1069, 1087, 1091, 1093])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_support_probe_window")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_train_support_density_f1024.json",
        )
        self.assertEqual(
            selector["comparison_of"],
            "results/tiny_neural_budget_sweep_train_support_density_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("support_probe_window_selector", selector["conditions"])

        scope = selector["condition_scope"]["support_probe_window_selector"]
        self.assertEqual(scope["train_only_selection"], True)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], False)
        self.assertEqual(scope["validation_used_for_transform_selection"], False)
        self.assertEqual(scope["support_probe_window_selector"], True)
        self.assertEqual(scope["support_probe_min_train_events"], 360)
        self.assertEqual(scope["support_probe_max_train_events"], 432)
        self.assertEqual(scope["reuse_selected_candidate_construction"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["64"]["support_probe_window_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"compact_train_size_gated_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["96"]["support_probe_window_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )
        self.assertEqual(
            selector["budgets"]["104"]["support_probe_window_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"support_ramped_compact_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["120"]["support_probe_window_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )
        self.assertEqual(
            selector["budgets"]["104"]["support_probe_window_selector"][
                "charged_compute_units_mean"
            ],
            selector["budgets"]["104"]["support_ramped_compact_induction"][
                "charged_compute_units_mean"
            ],
        )
        self.assertGreater(
            selector["budgets"]["104"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["104"]["train_support_density_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["112"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["112"]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["120"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["120"]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_validation_support_precision_artifact_records_boundary_gain_and_misses(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_support_precision_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1259, 1277, 1279, 1283, 1289])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_validation_support_precision")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_support_probe_window_f1024.json",
        )
        self.assertEqual(
            selector["comparison_of"],
            "results/tiny_neural_budget_sweep_support_probe_window_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_support_precision_selector", selector["conditions"])

        scope = selector["condition_scope"]["validation_support_precision_selector"]
        self.assertEqual(scope["train_only_selection"], False)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_used_for_transform_selection"], True)
        self.assertEqual(scope["validation_support_precision_selector"], True)
        self.assertEqual(scope["validation_support_precision_threshold"], 0.825758)
        self.assertEqual(scope["validation_support_compact_max_train_events"], 320)
        self.assertEqual(scope["validation_support_transition_min_train_events"], 400)
        self.assertEqual(scope["validation_support_transition_max_train_events"], 432)
        self.assertEqual(scope["reuse_selected_candidate_construction"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["64"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"compact_train_size_gated_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["96"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 1, "support_ramped_compact_induction": 4},
        )
        self.assertEqual(
            selector["budgets"]["104"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 1, "support_ramped_compact_induction": 4},
        )
        self.assertEqual(
            selector["budgets"]["112"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"support_ramped_compact_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["120"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 3, "support_ramped_compact_induction": 2},
        )
        self.assertEqual(
            selector["budgets"]["128"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 4, "support_ramped_compact_induction": 1},
        )

        self.assertGreater(
            selector["budgets"]["96"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["96"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertGreater(
            selector["budgets"]["104"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["104"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["120"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["120"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["128"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["128"]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

        selector_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        support_probe_average = sum(
            selector["budgets"][str(material)]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        self.assertGreater(selector_average, support_probe_average)

    def test_f1024_validation_support_precision_gate_artifact_records_no_window_control(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1381, 1399, 1409, 1423, 1427])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_validation_support_precision_gate")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_validation_support_precision_f1024.json",
        )
        self.assertEqual(
            selector["comparison_of"],
            "results/tiny_neural_budget_sweep_validation_support_precision_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_support_precision_gate_selector", selector["conditions"])

        scope = selector["condition_scope"]["validation_support_precision_gate_selector"]
        self.assertEqual(scope["train_only_selection"], False)
        self.assertEqual(scope["train_only_induction"], True)
        self.assertEqual(scope["validation_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_used_for_transform_selection"], True)
        self.assertEqual(scope["validation_support_precision_gate_selector"], True)
        self.assertEqual(scope["validation_support_precision_selector"], False)
        self.assertEqual(scope["validation_support_precision_threshold"], 0.825758)
        self.assertEqual(scope["validation_support_compact_max_train_events"], 320)
        self.assertEqual(scope["validation_support_uses_fixed_transition_prior"], False)
        self.assertEqual(scope["reuse_selected_candidate_construction"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["64"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"compact_train_size_gated_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["96"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 2, "support_ramped_compact_induction": 3},
        )
        self.assertEqual(
            selector["budgets"]["104"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 3, "support_ramped_compact_induction": 2},
        )
        self.assertEqual(
            selector["budgets"]["112"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 2, "support_ramped_compact_induction": 3},
        )
        self.assertEqual(
            selector["budgets"]["120"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 3, "support_ramped_compact_induction": 2},
        )
        self.assertEqual(
            selector["budgets"]["128"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )

        self.assertLess(
            selector["budgets"]["112"]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["112"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["128"]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["128"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

        gate_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        current_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        support_probe_average = sum(
            selector["budgets"][str(material)]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        self.assertLess(gate_average, current_average)
        self.assertGreater(gate_average, support_probe_average)

    def test_f1024_support_selector_transfer_artifact_records_unseen_seed_stress(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1459, 1471, 1481, 1483, 1487])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_support_selector_transfer")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_support_precision_selector", selector["conditions"])
        self.assertIn("validation_support_precision_gate_selector", selector["conditions"])
        self.assertEqual(
            selector["budgets"]["96"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 2, "support_ramped_compact_induction": 3},
        )
        self.assertEqual(
            selector["budgets"]["112"]["validation_support_precision_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"support_ramped_compact_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["112"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 3, "support_ramped_compact_induction": 2},
        )
        self.assertEqual(
            selector["budgets"]["128"]["validation_support_precision_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )

        density_capped_average = sum(
            selector["budgets"][str(material)]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        gate_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        fixed_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])

        self.assertGreater(density_capped_average, gate_average)
        self.assertGreater(gate_average, fixed_average)
        self.assertGreater(
            selector["budgets"]["112"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["112"]["validation_support_precision_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["120"]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["120"]["support_probe_window_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_validation_support_utility_artifact_records_fresh_seed_negative_result(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_support_utility_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1601, 1607, 1609, 1613, 1619])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_validation_support_utility")
        self.assertEqual(selector["confirmation_of"], "results/support_mechanism_audit_f1024.json")
        self.assertEqual(
            selector["comparison_of"],
            "results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json",
        )
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_support_utility_selector", selector["conditions"])

        scope = selector["condition_scope"]["validation_support_utility_selector"]
        self.assertEqual(scope["validation_support_utility_selector"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_motif_distribution_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_support_utility_min_score"], 0.0)
        self.assertEqual(scope["validation_support_utility_pair_coverage_weight"], 0.25)
        self.assertEqual(scope["validation_support_utility_triple_l1_weight"], 0.20)
        self.assertEqual(scope["validation_support_utility_compute_penalty"], 0.000001)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["64"]["validation_support_utility_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"compact_train_size_gated_induction": 5},
        )
        self.assertEqual(
            selector["budgets"]["96"]["validation_support_utility_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 4, "support_ramped_compact_induction": 1},
        )
        self.assertEqual(
            selector["budgets"]["104"]["validation_support_utility_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )
        self.assertEqual(
            selector["budgets"]["128"]["validation_support_utility_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )

        utility_average = sum(
            selector["budgets"][str(material)]["validation_support_utility_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        gate_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        density_average = sum(
            selector["budgets"][str(material)]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        raw_average = sum(
            selector["budgets"][str(material)]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])

        self.assertEqual(round(utility_average, 6), 0.005473)
        self.assertGreater(utility_average, raw_average)
        self.assertLess(utility_average, gate_average)
        self.assertLess(utility_average, density_average)
        self.assertGreater(
            selector["budgets"]["96"]["validation_support_utility_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["96"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )
        self.assertLess(
            selector["budgets"]["112"]["validation_support_utility_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
            selector["budgets"]["112"]["density_capped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ],
        )

    def test_f1024_validation_support_gain_gate_artifact_records_prefiltered_negative_result(self) -> None:
        selector = json.loads(
            Path("results/tiny_neural_budget_sweep_validation_support_gain_gate_f1024.json").read_text()
        )

        self.assertEqual(selector["seeds"], [1667, 1669, 1693, 1697, 1699])
        self.assertEqual(selector["material_counts"], [64, 80, 96, 104, 112, 120, 128])
        self.assertEqual(selector["feature_dimension"], 1024)
        self.assertEqual(selector["profile_label"], "f1024_16x8_validation_support_gain_gate")
        self.assertEqual(
            selector["confirmation_of"],
            "results/tiny_neural_budget_sweep_validation_support_utility_f1024.json",
        )
        self.assertEqual(selector["comparison_of"], "results/support_selector_error_audit_f1024.json")
        self.assertEqual(selector["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(selector["claim_scope"]["fresh_seed_confirmation"], True)
        self.assertIn("validation_support_gain_gate_selector", selector["conditions"])

        scope = selector["condition_scope"]["validation_support_gain_gate_selector"]
        self.assertEqual(scope["validation_support_gain_gate_selector"], True)
        self.assertEqual(scope["validation_support_gain_precision_prefilter"], True)
        self.assertEqual(scope["validation_labels_used_for_policy_selection"], True)
        self.assertEqual(scope["validation_support_gain_proxy_epochs"], 2)
        self.assertEqual(scope["validation_support_gain_min_score"], 0.0)
        self.assertEqual(scope["validation_support_gain_compute_penalty"], 0.0000005)
        self.assertEqual(scope["validation_support_precision_threshold"], 0.825758)
        self.assertEqual(scope["reuse_selected_candidate_construction"], True)
        self.assertEqual(scope["oracle_generated_labels"], False)

        self.assertEqual(
            selector["budgets"]["96"]["validation_support_gain_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 4, "support_ramped_compact_induction": 1},
        )
        self.assertEqual(
            selector["budgets"]["104"]["validation_support_gain_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 3, "support_ramped_compact_induction": 2},
        )
        self.assertEqual(
            selector["budgets"]["128"]["validation_support_gain_gate_selector"][
                "portfolio_selected_condition_counts"
            ],
            {"raw_text": 5},
        )

        gain_gate_average = sum(
            selector["budgets"][str(material)]["validation_support_gain_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        utility_average = sum(
            selector["budgets"][str(material)]["validation_support_utility_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        precision_gate_average = sum(
            selector["budgets"][str(material)]["validation_support_precision_gate_selector"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        support_average = sum(
            selector["budgets"][str(material)]["support_ramped_compact_induction"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])
        raw_average = sum(
            selector["budgets"][str(material)]["raw_text"][
                "signed_learning_signal_density_per_1m_event_compute_mean"
            ]
            for material in selector["material_counts"]
        ) / len(selector["material_counts"])

        self.assertEqual(round(gain_gate_average, 6), 0.004684)
        self.assertGreater(gain_gate_average, raw_average)
        self.assertLess(gain_gate_average, utility_average)
        self.assertLess(gain_gate_average, precision_gate_average)
        self.assertLess(gain_gate_average, support_average)


if __name__ == "__main__":
    unittest.main()
