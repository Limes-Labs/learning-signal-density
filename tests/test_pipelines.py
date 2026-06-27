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

        normal = build_pipeline_examples(
            "mdl_rule_expansion",
            split.train,
            world.rules,
            validation_observations=split.validation,
        )
        alternate = build_pipeline_examples(
            "mdl_rule_expansion",
            split.train,
            alternate_rules,
            validation_observations=split.validation,
        )
        normal_signature = [(example.source_observation_id, example.source_kind) for example in normal.examples]
        alternate_signature = [(example.source_observation_id, example.source_kind) for example in alternate.examples]
        self.assertEqual(normal_signature, alternate_signature, "mdl_rule_expansion")

        normal = build_pipeline_examples(
            "validation_ranked_induction",
            split.train,
            world.rules,
            validation_observations=split.validation,
        )
        alternate = build_pipeline_examples(
            "validation_ranked_induction",
            split.train,
            alternate_rules,
            validation_observations=split.validation,
        )
        normal_signature = [(example.source_observation_id, example.source_kind) for example in normal.examples]
        alternate_signature = [(example.source_observation_id, example.source_kind) for example in alternate.examples]
        self.assertEqual(normal_signature, alternate_signature, "validation_ranked_induction")

        normal = build_pipeline_examples("train_calibrated_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("train_calibrated_ranked_induction", split.train, alternate_rules)
        normal_signature = [(example.source_observation_id, example.source_kind) for example in normal.examples]
        alternate_signature = [(example.source_observation_id, example.source_kind) for example in alternate.examples]
        self.assertEqual(normal_signature, alternate_signature, "train_calibrated_ranked_induction")

        normal = build_pipeline_examples("self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("self_ranked_induction", split.train, alternate_rules)
        normal_signature = [(example.source_observation_id, example.source_kind) for example in normal.examples]
        alternate_signature = [(example.source_observation_id, example.source_kind) for example in alternate.examples]
        self.assertEqual(normal_signature, alternate_signature, "self_ranked_induction")

    def test_mdl_rule_expansion_requires_explicit_validation_split(self) -> None:
        world = build_world(seed=44, material_count=40)
        split = split_observations(world.observations)

        with self.assertRaises(ValueError):
            build_pipeline_examples("mdl_rule_expansion", split.train, world.rules)

    def test_validation_ranked_induction_requires_explicit_validation_split(self) -> None:
        world = build_world(seed=45, material_count=40)
        split = split_observations(world.observations)

        with self.assertRaises(ValueError):
            build_pipeline_examples("validation_ranked_induction", split.train, world.rules)

    def test_mdl_rule_expansion_uses_fewer_synthetic_examples_than_induced_rules(self) -> None:
        world = build_world(seed=47, material_count=48)
        split = split_observations(world.observations)
        induced = build_pipeline_examples("induced_rule_expansion", split.train, world.rules)
        mdl = build_pipeline_examples("mdl_rule_expansion", split.train, world.rules, validation_observations=split.validation)

        self.assertEqual(mdl.external_event_count, induced.external_event_count)
        self.assertLess(mdl.internal_example_count, induced.internal_example_count)
        self.assertGreater(mdl.mdl_selected_rule_count, 0)
        self.assertGreater(mdl.mdl_description_length_tokens, 0)
        self.assertGreater(mdl.rule_search_cost_tokens, len(split.train) * 7)
        self.assertIn("mdl_counterfactual", {example.source_kind for example in mdl.examples})

    def test_validation_ranked_induction_budgets_candidates_and_charges_ranking(self) -> None:
        world = build_world(seed=49, material_count=48)
        split = split_observations(world.observations)
        induced = build_pipeline_examples("induced_rule_expansion", split.train, world.rules)
        ranked = build_pipeline_examples(
            "validation_ranked_induction",
            split.train,
            world.rules,
            validation_observations=split.validation,
        )

        self.assertEqual(ranked.external_event_count, induced.external_event_count)
        self.assertLess(ranked.internal_example_count, induced.internal_example_count)
        self.assertGreater(ranked.ranked_candidate_count, ranked.ranked_kept_candidate_count)
        self.assertGreater(ranked.ranked_kept_candidate_count, 0)
        self.assertGreater(ranked.candidate_ranking_cost_tokens, 0)
        self.assertIn("validation_ranked_induced", {example.source_kind for example in ranked.examples})

    def test_train_calibrated_ranked_induction_uses_train_calibration_not_validation(self) -> None:
        world = build_world(seed=53, material_count=48)
        split = split_observations(world.observations)
        induced = build_pipeline_examples("induced_rule_expansion", split.train, world.rules)
        ranked = build_pipeline_examples("train_calibrated_ranked_induction", split.train, world.rules)

        self.assertEqual(ranked.external_event_count, induced.external_event_count)
        self.assertLess(ranked.internal_example_count, induced.internal_example_count)
        self.assertGreater(ranked.ranked_candidate_count, ranked.ranked_kept_candidate_count)
        self.assertGreater(ranked.ranked_kept_candidate_count, 0)
        self.assertGreater(ranked.candidate_ranking_cost_tokens, 0)
        self.assertGreater(ranked.train_calibration_event_count, 0)
        self.assertEqual(ranked.validation_calibration_event_count, 0)
        self.assertIn("train_calibrated_ranked_induced", {example.source_kind for example in ranked.examples})

    def test_self_ranked_induction_removes_calibration_cost(self) -> None:
        world = build_world(seed=59, material_count=48)
        split = split_observations(world.observations)
        train_calibrated = build_pipeline_examples("train_calibrated_ranked_induction", split.train, world.rules)
        self_ranked = build_pipeline_examples("self_ranked_induction", split.train, world.rules)

        self.assertEqual(self_ranked.external_event_count, train_calibrated.external_event_count)
        self.assertEqual(self_ranked.ranked_kept_candidate_count, train_calibrated.ranked_kept_candidate_count)
        self.assertGreater(self_ranked.ranked_candidate_count, self_ranked.ranked_kept_candidate_count)
        self.assertEqual(self_ranked.train_calibration_event_count, 0)
        self.assertEqual(self_ranked.validation_calibration_event_count, 0)
        self.assertLess(self_ranked.candidate_ranking_cost_tokens, train_calibrated.candidate_ranking_cost_tokens)
        self.assertIn("self_ranked_induced", {example.source_kind for example in self_ranked.examples})


if __name__ == "__main__":
    unittest.main()
