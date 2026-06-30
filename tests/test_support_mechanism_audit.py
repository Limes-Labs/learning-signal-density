import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_support_mechanism_audit.py"
AUDIT_JSON_PATH = REPO_ROOT / "results" / "support_mechanism_audit_f1024.json"
AUDIT_MD_PATH = REPO_ROOT / "results" / "support_mechanism_audit_f1024.md"


def load_audit_builder():
    spec = importlib.util.spec_from_file_location("build_support_mechanism_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SupportMechanismAuditTests(unittest.TestCase):
    def test_committed_audit_records_precision_coverage_and_density_mechanism(self) -> None:
        audit = json.loads(AUDIT_JSON_PATH.read_text())

        self.assertEqual(
            audit["source_artifacts"],
            ["results/tiny_neural_budget_sweep_support_selector_transfer_f1024.json"],
        )
        self.assertEqual(audit["seeds"], [1459, 1471, 1481, 1483, 1487])
        self.assertEqual(
            audit["conditions_audited"],
            [
                "raw_text",
                "compact_train_size_gated_induction",
                "support_ramped_compact_induction",
                "density_window_compact_induction",
                "density_capped_compact_induction",
            ],
        )
        self.assertEqual(audit["claim_scope"]["uses_hidden_rulebook_for_label_audit"], True)
        self.assertEqual(audit["claim_scope"]["uses_heldout_distribution_for_audit"], True)
        self.assertEqual(audit["claim_scope"]["heldout_available_to_policies"], False)
        self.assertEqual(audit["claim_scope"]["deployable_policy"], False)
        self.assertEqual(audit["claim_scope"]["paper_ready_claim"], False)

        material112 = audit["audits"]["112"]
        compact112 = material112["compact_train_size_gated_induction"]
        support112 = material112["support_ramped_compact_induction"]
        density112 = material112["density_capped_compact_induction"]
        self.assertEqual(compact112["label_precision"], 0.9235)
        self.assertEqual(compact112["heldout_triple_coverage"], 0.671642)
        self.assertEqual(support112["label_precision"], 0.784507)
        self.assertEqual(support112["heldout_triple_coverage"], 0.408955)
        self.assertEqual(support112["synthetic_example_count"], 710)
        self.assertEqual(density112["linked_signed_learning_signal_density_per_1m_event_compute_mean"], 0.007757)

        transition112 = audit["transition_diagnostics"]["112"]
        self.assertEqual(transition112["support_minus_compact_label_precision"], -0.138993)
        self.assertEqual(transition112["support_minus_compact_heldout_triple_coverage"], -0.262687)
        self.assertEqual(transition112["support_minus_density_capped_lsd"], -0.003582)

        summary = audit["mechanism_summary"]
        self.assertEqual(summary["support_ramp_precision_improvement_count"], 0)
        self.assertEqual(summary["support_ramp_coverage_loss_count"], 4)
        self.assertEqual(summary["support_ramp_density_win_over_density_cap_count"], 1)
        self.assertEqual(summary["transition_material_counts"], [104, 112, 120, 128])
        self.assertFalse(summary["promote_support_ramp_mechanism"])

    def test_generator_rebuilds_committed_audit_and_markdown(self) -> None:
        audit_builder = load_audit_builder()

        rebuilt = audit_builder.build_support_mechanism_audit(REPO_ROOT)
        committed = json.loads(AUDIT_JSON_PATH.read_text())
        for key in (
            "source_artifacts",
            "conditions_audited",
            "material_counts",
            "audits",
            "transition_diagnostics",
            "mechanism_summary",
            "claim_scope",
        ):
            self.assertEqual(rebuilt[key], committed[key])

        markdown = AUDIT_MD_PATH.read_text()
        self.assertEqual(markdown, audit_builder.render_markdown(committed))
        self.assertIn("support-ramp mechanism audit", markdown)
        self.assertIn("0.784507", markdown)
        self.assertIn("-0.003582", markdown)


if __name__ == "__main__":
    unittest.main()
