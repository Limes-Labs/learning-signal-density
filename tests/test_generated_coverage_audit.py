import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from learning_signal_density.domain import build_world, split_observations
from learning_signal_density.pipelines import build_pipeline_examples

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_generated_coverage_audit.py"


def load_coverage_builder():
    spec = importlib.util.spec_from_file_location("build_generated_coverage_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratedCoverageAuditTests(unittest.TestCase):
    def test_l1_distribution_distance_is_zero_for_matching_counts(self) -> None:
        coverage_builder = load_coverage_builder()

        self.assertEqual(
            coverage_builder.distribution_l1_distance(
                {"a": 2, "b": 1},
                {"a": 4, "b": 2},
            ),
            0.0,
        )

    def test_diverse_self_ranked_induction_is_counted_as_generated(self) -> None:
        coverage_builder = load_coverage_builder()
        world = build_world(seed=37, material_count=48)
        split = split_observations(world.observations)
        pipeline = build_pipeline_examples("diverse_self_ranked_induction", split.train, world.rules)

        diverse_generated = [
            example for example in pipeline.examples if example.source_kind == "diverse_self_ranked_induced"
        ]

        self.assertGreater(len(diverse_generated), 0)
        self.assertEqual(coverage_builder.generated_examples(pipeline), tuple(diverse_generated))

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
        self.assertEqual(
            coverage_builder.generated_examples(sample_aware_diverse),
            tuple(sample_aware_diverse_generated),
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
        self.assertEqual(coverage_builder.generated_examples(compact_diverse), tuple(compact_diverse_generated))

    def test_builds_non_deployable_heldout_coverage_diagnostic(self) -> None:
        coverage_builder = load_coverage_builder()

        artifact = coverage_builder.build_generated_coverage_audit(REPO_ROOT)

        self.assertEqual(
            artifact["source_artifacts"],
            ["results/tiny_neural_budget_sweep_selector_transfer_f1024.json"],
        )
        self.assertEqual(artifact["seeds"], [37, 41, 43, 47, 53])
        self.assertEqual(artifact["material_counts"], [16, 24, 32, 48, 64])
        self.assertEqual(artifact["claim_scope"]["uses_hidden_rulebook_for_label_audit"], True)
        self.assertEqual(artifact["claim_scope"]["uses_heldout_distribution_for_audit"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_distribution_available_to_policies"], False)
        self.assertEqual(artifact["claim_scope"]["deployable_policy"], False)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(artifact["claim_scope"]["paper_ready_claim"], False)

        material32 = artifact["audits"]["32"]
        self.assertEqual(
            material32["validation_ranked_induction"]["generated_vs_heldout_triple_l1_distance"],
            0.683666,
        )
        self.assertEqual(
            material32["validation_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            -0.068421,
        )
        self.assertEqual(
            material32["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            0.764428,
        )
        self.assertEqual(
            material32["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            -0.131579,
        )

        material64 = artifact["audits"]["64"]
        self.assertEqual(
            material64["sample_aware_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            0.477843,
        )
        self.assertEqual(
            material64["sample_aware_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            0.142857,
        )
        self.assertEqual(
            material64["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            0.576443,
        )
        self.assertEqual(
            material64["agreement_gated_self_ranked_induction"][
                "linked_accuracy_improvement_over_majority_mean"
            ],
            0.080519,
        )

    def test_writes_json_and_markdown_artifacts(self) -> None:
        coverage_builder = load_coverage_builder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "generated_coverage_audit.json"
            output_md = Path(tmpdir) / "generated_coverage_audit.md"
            coverage_builder.write_generated_coverage_audit(REPO_ROOT, output_json, output_md)
            written = json.loads(output_json.read_text())
            markdown = output_md.read_text()

        self.assertEqual(
            written["audits"]["64"]["agreement_gated_self_ranked_induction"][
                "generated_vs_heldout_triple_l1_distance"
            ],
            0.576443,
        )
        self.assertIn("heldout-distribution", markdown)
        self.assertIn("non-deployable", markdown)
        self.assertIn("agreement_gated_self_ranked_induction", markdown)


if __name__ == "__main__":
    unittest.main()
