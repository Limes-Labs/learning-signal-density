import json
import unittest
from pathlib import Path


class CommittedResultArtifactTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
