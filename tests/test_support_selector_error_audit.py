import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_support_selector_error_audit.py"
AUDIT_JSON_PATH = REPO_ROOT / "results" / "support_selector_error_audit_f1024.json"
AUDIT_MD_PATH = REPO_ROOT / "results" / "support_selector_error_audit_f1024.md"


def load_audit_builder():
    spec = importlib.util.spec_from_file_location("build_support_selector_error_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SupportSelectorErrorAuditTests(unittest.TestCase):
    def test_committed_audit_records_post_hoc_regret_ledger(self) -> None:
        audit = json.loads(AUDIT_JSON_PATH.read_text())

        self.assertEqual(
            audit["source_artifacts"],
            [
                "results/tiny_neural_budget_sweep_train_support_density_f1024.json",
                "results/tiny_neural_budget_sweep_support_probe_window_f1024.json",
                "results/tiny_neural_budget_sweep_validation_support_precision_f1024.json",
                "results/tiny_neural_budget_sweep_validation_support_precision_gate_f1024.json",
                "results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json",
            ],
        )
        self.assertEqual(audit["metric"], "signed_learning_signal_density_per_1m_event_compute_mean")
        self.assertEqual(audit["claim_scope"]["post_hoc_diagnostic"], True)
        self.assertEqual(audit["claim_scope"]["heldout_available_to_policies"], False)
        self.assertEqual(audit["claim_scope"]["deployable_policy"], False)
        self.assertEqual(audit["claim_scope"]["paper_ready_claim"], False)
        self.assertFalse(audit["recommendation"]["promote_support_selector"])

        transfer = audit["artifact_summaries"]["support_selector_transfer"]
        self.assertEqual(transfer["best_fixed_simple_condition"], "density_capped_compact_induction")
        self.assertEqual(transfer["best_fixed_simple_lsd"], 0.006115)

        fixed = transfer["selector_diagnostics"]["validation_support_precision_selector"]
        gate = transfer["selector_diagnostics"]["validation_support_precision_gate_selector"]
        self.assertEqual(fixed["average_lsd"], 0.005601)
        self.assertEqual(fixed["average_regret_vs_best_simple_lsd"], 0.000831)
        self.assertEqual(gate["average_lsd"], 0.005936)
        self.assertEqual(gate["average_regret_vs_best_simple_lsd"], 0.000496)
        self.assertEqual(gate["budgets_beating_best_simple_count"], 1)
        self.assertEqual(gate["worst_regret_material_count"], 112)
        self.assertEqual(gate["average_selection_cost_units"], 2082.8)
        self.assertLess(
            gate["average_regret_vs_best_simple_lsd"],
            fixed["average_regret_vs_best_simple_lsd"],
        )

        gate_source = audit["artifact_summaries"]["validation_support_precision_gate"]
        local_fixed = gate_source["selector_diagnostics"]["validation_support_precision_selector"]
        self.assertEqual(local_fixed["average_regret_vs_best_simple_lsd"], -0.00011)
        self.assertEqual(local_fixed["local_expected_value_positive"], True)

        precision_source = audit["artifact_summaries"]["validation_support_precision"]
        precision_selector = precision_source["selector_diagnostics"][
            "validation_support_precision_selector"
        ]
        self.assertEqual(precision_selector["average_regret_vs_best_simple_lsd"], 0.000042)
        self.assertEqual(precision_selector["budgets_beating_best_simple_count"], 2)

    def test_generator_rebuilds_committed_audit_and_markdown(self) -> None:
        audit_builder = load_audit_builder()

        rebuilt = audit_builder.build_support_selector_error_audit(REPO_ROOT)
        committed = json.loads(AUDIT_JSON_PATH.read_text())
        for key in (
            "source_artifacts",
            "simple_comparator_conditions",
            "selector_conditions",
            "artifact_summaries",
            "recommendation",
            "claim_scope",
        ):
            self.assertEqual(rebuilt[key], committed[key])

        markdown = AUDIT_MD_PATH.read_text()
        self.assertEqual(markdown, audit_builder.render_markdown(committed))
        self.assertIn("post-hoc support-selector error audit", markdown)
        self.assertIn("validation_support_precision_gate_selector", markdown)
        self.assertIn("0.000496", markdown)


if __name__ == "__main__":
    unittest.main()
