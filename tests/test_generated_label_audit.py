import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.pipelines import TrainingExample
from learning_signal_density.domain import build_world, split_observations
from learning_signal_density.neural_experiment import _generated_motif_counts
from learning_signal_density.pipelines import build_pipeline_examples


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_generated_label_audit.py"


def load_audit_builder():
    spec = importlib.util.spec_from_file_location("build_generated_label_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratedLabelAuditTests(unittest.TestCase):
    def test_parses_generated_example_fields_from_pipeline_text(self) -> None:
        audit_builder = load_audit_builder()
        example = TrainingExample(
            text=(
                "Question material=auren_alloy_000 family=auren "
                "has modifier=q9_salt and stimulus=amber_heat. "
                "Will the material become brittle?"
            ),
            label=True,
            pair_key=("auren_alloy_000", "amber_heat"),
            source_observation_id="obs-test-q9_salt",
            source_kind="sample_aware_self_ranked_induced",
        )

        self.assertEqual(
            audit_builder.parse_generated_example_fields(example),
            {
                "family": "auren",
                "stimulus": "amber_heat",
                "modifier": "q9_salt",
            },
        )

    def test_diverse_self_ranked_induction_is_counted_as_generated(self) -> None:
        audit_builder = load_audit_builder()
        world = build_world(seed=37, material_count=48)
        split = split_observations(world.observations)
        pipeline = build_pipeline_examples("diverse_self_ranked_induction", split.train, world.rules)

        diverse_generated = [
            example for example in pipeline.examples if example.source_kind == "diverse_self_ranked_induced"
        ]

        self.assertGreater(len(diverse_generated), 0)
        self.assertTrue(all(audit_builder.is_generated_label_example(example) for example in diverse_generated))
        self.assertEqual(sum(_generated_motif_counts(pipeline).values()), len(diverse_generated))

        sample_aware_diverse = build_pipeline_examples(
            "sample_aware_diverse_self_ranked_induction",
            split.train,
            world.rules,
        )
        sample_aware_diverse_generated = [
            example
            for example in sample_aware_diverse.examples
            if example.source_kind == "sample_aware_diverse_self_ranked_induced"
        ]

        self.assertGreater(len(sample_aware_diverse_generated), 0)
        self.assertTrue(
            all(audit_builder.is_generated_label_example(example) for example in sample_aware_diverse_generated)
        )
        self.assertEqual(
            sum(_generated_motif_counts(sample_aware_diverse).values()),
            len(sample_aware_diverse_generated),
        )

        compact_world = build_world(seed=37, material_count=64)
        compact_split = split_observations(compact_world.observations)
        compact_diverse = build_pipeline_examples(
            "compact_diverse_train_size_gated_induction",
            compact_split.train,
            compact_world.rules,
        )
        compact_diverse_generated = [
            example
            for example in compact_diverse.examples
            if example.source_kind == "compact_diverse_sample_aware_self_ranked_induced"
        ]

        self.assertGreater(len(compact_diverse_generated), 0)
        self.assertTrue(all(audit_builder.is_generated_label_example(example) for example in compact_diverse_generated))
        self.assertEqual(sum(_generated_motif_counts(compact_diverse).values()), len(compact_diverse_generated))

    def test_builds_hidden_rulebook_diagnostic_without_deployable_claims(self) -> None:
        audit_builder = load_audit_builder()

        artifact = audit_builder.build_generated_label_audit(REPO_ROOT)

        self.assertEqual(
            artifact["source_artifacts"],
            ["results/tiny_neural_budget_sweep_selector_transfer_f1024.json"],
        )
        self.assertEqual(artifact["seeds"], [37, 41, 43, 47, 53])
        self.assertEqual(artifact["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(artifact["claim_scope"]["uses_hidden_rulebook_for_audit"], True)
        self.assertEqual(artifact["claim_scope"]["deployable_policy"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["paper_ready_claim"], False)

        material32 = artifact["audits"]["32"]
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

    def test_writes_json_and_markdown_artifacts(self) -> None:
        audit_builder = load_audit_builder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "generated_label_audit.json"
            output_md = Path(tmpdir) / "generated_label_audit.md"
            audit_builder.write_generated_label_audit(REPO_ROOT, output_json, output_md)
            written = json.loads(output_json.read_text())
            markdown = output_md.read_text()

        self.assertEqual(written["audits"]["32"]["agreement_gated_self_ranked_induction"]["label_precision"], 0.917241)
        self.assertIn("hidden-rulebook", markdown)
        self.assertIn("non-deployable", markdown)
        self.assertIn("agreement_gated_self_ranked_induction", markdown)


if __name__ == "__main__":
    unittest.main()
