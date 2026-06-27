import unittest

from learning_signal_density.domain import build_world, split_observations
from learning_signal_density.pipelines import build_pipeline_examples


class PipelineAccountingTests(unittest.TestCase):
    def test_transforms_keep_external_event_count_fixed(self) -> None:
        world = build_world(seed=31, material_count=24)
        split = split_observations(world.observations)
        baseline = build_pipeline_examples("raw_text", split.train, world.rules)
        expanded = build_pipeline_examples("counterfactual_expansion", split.train, world.rules)
        replayed = build_pipeline_examples("prioritized_replay", split.train, world.rules)

        self.assertEqual(baseline.external_event_count, len(split.train))
        self.assertEqual(expanded.external_event_count, len(split.train))
        self.assertEqual(replayed.external_event_count, len(split.train))
        self.assertGreater(expanded.internal_example_count, baseline.internal_example_count)
        self.assertGreater(replayed.internal_example_count, baseline.internal_example_count)

    def test_pipeline_examples_do_not_include_heldout_pairs(self) -> None:
        world = build_world(seed=37, material_count=30)
        split = split_observations(world.observations)
        heldout_pairs = {item.pair_key for item in split.heldout}

        for condition in (
            "raw_text",
            "qa_expansion",
            "counterfactual_expansion",
            "induced_rule_expansion",
            "prioritized_replay",
        ):
            examples = build_pipeline_examples(condition, split.train, world.rules)
            produced_pairs = {example.pair_key for example in examples.examples}
            self.assertFalse(produced_pairs & heldout_pairs, condition)

    def test_induced_rule_expansion_charges_train_only_modeling_cost(self) -> None:
        world = build_world(seed=41, material_count=40)
        split = split_observations(world.observations)
        examples = build_pipeline_examples("induced_rule_expansion", split.train, world.rules)

        self.assertEqual(examples.external_event_count, len(split.train))
        self.assertGreater(examples.modeling_cost_tokens, 0)
        self.assertGreater(examples.transform_cost_tokens, 0)
        self.assertIn("induced_counterfactual", {example.source_kind for example in examples.examples})

    def test_stricter_induction_threshold_produces_no_more_examples(self) -> None:
        world = build_world(seed=42, material_count=40)
        split = split_observations(world.observations)
        permissive = build_pipeline_examples(
            "validation_gated_induction",
            split.train,
            world.rules,
            induction_min_support=1,
            induction_min_confidence=0.5,
        )
        strict = build_pipeline_examples(
            "validation_gated_induction",
            split.train,
            world.rules,
            induction_min_support=4,
            induction_min_confidence=0.85,
        )

        self.assertLessEqual(strict.internal_example_count, permissive.internal_example_count)
        self.assertEqual(strict.external_event_count, permissive.external_event_count)

    def test_selection_and_replay_do_not_peek_at_hidden_rulebook(self) -> None:
        world = build_world(seed=43, material_count=40)
        alternate_rules = build_world(seed=101, material_count=40).rules
        split = split_observations(world.observations)

        for condition in (
            "selected_text",
            "prioritized_replay",
            "validation_gated_induction",
            "direct_validation_gated_induction",
        ):
            normal = build_pipeline_examples(condition, split.train, world.rules)
            alternate = build_pipeline_examples(condition, split.train, alternate_rules)
            normal_signature = [(example.source_observation_id, example.source_kind) for example in normal.examples]
            alternate_signature = [(example.source_observation_id, example.source_kind) for example in alternate.examples]
            self.assertEqual(normal_signature, alternate_signature, condition)


if __name__ == "__main__":
    unittest.main()
