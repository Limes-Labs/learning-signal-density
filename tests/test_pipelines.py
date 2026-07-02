import unittest

from learning_signal_density.domain import build_world, split_observations
from learning_signal_density.pipelines import build_pipeline_examples


class PipelineAccountingTests(unittest.TestCase):
    def _example_signature(self, examples):
        return [
            (
                example.source_observation_id,
                example.source_kind,
                example.pair_key,
                example.text,
                example.label,
            )
            for example in examples.examples
        ]

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
            normal_signature = self._example_signature(normal)
            alternate_signature = self._example_signature(alternate)
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
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
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
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "validation_ranked_induction")

        normal = build_pipeline_examples("train_calibrated_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("train_calibrated_ranked_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "train_calibrated_ranked_induction")

        normal = build_pipeline_examples("self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("self_ranked_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "self_ranked_induction")

        normal = build_pipeline_examples("diverse_self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("diverse_self_ranked_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "diverse_self_ranked_induction")

        normal = build_pipeline_examples("sample_aware_self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("sample_aware_self_ranked_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "sample_aware_self_ranked_induction")

        normal = build_pipeline_examples("sample_aware_diverse_self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples(
            "sample_aware_diverse_self_ranked_induction",
            split.train,
            alternate_rules,
        )
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "sample_aware_diverse_self_ranked_induction")

        normal = build_pipeline_examples("compact_train_size_gated_induction", split.train, world.rules)
        alternate = build_pipeline_examples("compact_train_size_gated_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "compact_train_size_gated_induction")

        normal = build_pipeline_examples("compact_diverse_train_size_gated_induction", split.train, world.rules)
        alternate = build_pipeline_examples("compact_diverse_train_size_gated_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "compact_diverse_train_size_gated_induction")

        normal = build_pipeline_examples("density_capped_compact_induction", split.train, world.rules)
        alternate = build_pipeline_examples("density_capped_compact_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "density_capped_compact_induction")

        normal = build_pipeline_examples("support_ramped_compact_induction", split.train, world.rules)
        alternate = build_pipeline_examples("support_ramped_compact_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "support_ramped_compact_induction")

        normal = build_pipeline_examples("density_window_compact_induction", split.train, world.rules)
        alternate = build_pipeline_examples("density_window_compact_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "density_window_compact_induction")

        normal = build_pipeline_examples("late_confidence_ramped_compact_induction", split.train, world.rules)
        alternate = build_pipeline_examples("late_confidence_ramped_compact_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "late_confidence_ramped_compact_induction")

        normal = build_pipeline_examples("agreement_gated_self_ranked_induction", split.train, world.rules)
        alternate = build_pipeline_examples("agreement_gated_self_ranked_induction", split.train, alternate_rules)
        normal_signature = self._example_signature(normal)
        alternate_signature = self._example_signature(alternate)
        self.assertEqual(normal_signature, alternate_signature, "agreement_gated_self_ranked_induction")

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

    def test_diverse_self_ranked_induction_reduces_modifier_concentration(self) -> None:
        world = build_world(seed=59, material_count=48)
        split = split_observations(world.observations)
        self_ranked = build_pipeline_examples("self_ranked_induction", split.train, world.rules)
        diverse = build_pipeline_examples("diverse_self_ranked_induction", split.train, world.rules)

        self.assertEqual(diverse.external_event_count, self_ranked.external_event_count)
        self.assertEqual(diverse.ranked_kept_candidate_count, self_ranked.ranked_kept_candidate_count)
        self.assertEqual(diverse.train_calibration_event_count, 0)
        self.assertEqual(diverse.validation_calibration_event_count, 0)
        self.assertGreater(diverse.ranked_diversity_penalty, 0.0)
        self.assertLessEqual(diverse.ranked_max_modifier_count, self_ranked.ranked_max_modifier_count)
        self.assertIn("diverse_self_ranked_induced", {example.source_kind for example in diverse.examples})

    def test_sample_aware_self_ranked_induction_adapts_budget_without_calibration(self) -> None:
        small_world = build_world(seed=59, material_count=16)
        small_split = split_observations(small_world.observations)
        small_self_ranked = build_pipeline_examples("self_ranked_induction", small_split.train, small_world.rules)
        small_sample_aware = build_pipeline_examples("sample_aware_self_ranked_induction", small_split.train, small_world.rules)

        self.assertEqual(small_sample_aware.external_event_count, small_self_ranked.external_event_count)
        self.assertEqual(small_sample_aware.train_calibration_event_count, 0)
        self.assertEqual(small_sample_aware.validation_calibration_event_count, 0)
        self.assertLess(small_sample_aware.ranked_synthetic_budget_ratio, small_self_ranked.ranked_synthetic_budget_ratio)
        self.assertLess(small_sample_aware.ranked_kept_candidate_count, small_self_ranked.ranked_kept_candidate_count)
        self.assertEqual(small_sample_aware.ranked_induction_min_support, 2)
        self.assertIn("sample_aware_self_ranked_induced", {example.source_kind for example in small_sample_aware.examples})

    def test_sample_aware_diverse_induction_combines_budget_policy_with_diversity(self) -> None:
        world = build_world(seed=59, material_count=48)
        split = split_observations(world.observations)
        sample_aware = build_pipeline_examples("sample_aware_self_ranked_induction", split.train, world.rules)
        sample_aware_diverse = build_pipeline_examples(
            "sample_aware_diverse_self_ranked_induction",
            split.train,
            world.rules,
        )

        self.assertEqual(sample_aware_diverse.external_event_count, sample_aware.external_event_count)
        self.assertEqual(sample_aware_diverse.ranked_synthetic_budget_ratio, sample_aware.ranked_synthetic_budget_ratio)
        self.assertEqual(sample_aware_diverse.ranked_kept_candidate_count, sample_aware.ranked_kept_candidate_count)
        self.assertEqual(sample_aware_diverse.ranked_induction_min_support, sample_aware.ranked_induction_min_support)
        self.assertEqual(
            sample_aware_diverse.ranked_induction_min_confidence,
            sample_aware.ranked_induction_min_confidence,
        )
        self.assertEqual(sample_aware_diverse.train_calibration_event_count, 0)
        self.assertEqual(sample_aware_diverse.validation_calibration_event_count, 0)
        self.assertGreater(sample_aware_diverse.ranked_diversity_penalty, 0.0)
        self.assertLessEqual(sample_aware_diverse.ranked_max_modifier_count, sample_aware.ranked_max_modifier_count)
        self.assertIn(
            "sample_aware_diverse_self_ranked_induced",
            {example.source_kind for example in sample_aware_diverse.examples},
        )

    def test_tempered_sample_aware_induction_reduces_mid_budget_synthetic_ratio(self) -> None:
        mid_world = build_world(seed=157, material_count=32)
        mid_split = split_observations(mid_world.observations)
        sample_aware = build_pipeline_examples("sample_aware_self_ranked_induction", mid_split.train, mid_world.rules)
        tempered = build_pipeline_examples(
            "tempered_sample_aware_self_ranked_induction",
            mid_split.train,
            mid_world.rules,
        )

        self.assertEqual(tempered.external_event_count, sample_aware.external_event_count)
        self.assertEqual(tempered.train_calibration_event_count, 0)
        self.assertEqual(tempered.validation_calibration_event_count, 0)
        self.assertEqual(tempered.ranked_synthetic_budget_ratio, 0.5)
        self.assertEqual(tempered.ranked_induction_min_support, sample_aware.ranked_induction_min_support)
        self.assertEqual(tempered.ranked_induction_min_confidence, sample_aware.ranked_induction_min_confidence)
        self.assertLess(tempered.ranked_kept_candidate_count, sample_aware.ranked_kept_candidate_count)
        self.assertIn(
            "tempered_sample_aware_self_ranked_induced",
            {example.source_kind for example in tempered.examples},
        )

        large_world = build_world(seed=157, material_count=48)
        large_split = split_observations(large_world.observations)
        large_tempered = build_pipeline_examples(
            "tempered_sample_aware_self_ranked_induction",
            large_split.train,
            large_world.rules,
        )
        self.assertEqual(large_tempered.ranked_synthetic_budget_ratio, 1.0)

    def test_train_size_gated_sample_aware_induction_uses_raw_until_large_train_split(self) -> None:
        small_world = build_world(seed=61, material_count=32)
        small_split = split_observations(small_world.observations)
        small_raw = build_pipeline_examples("raw_text", small_split.train, small_world.rules)
        small_gated = build_pipeline_examples(
            "train_size_gated_sample_aware_induction",
            small_split.train,
            small_world.rules,
        )

        self.assertEqual(self._example_signature(small_gated), self._example_signature(small_raw))
        self.assertEqual(small_gated.ranked_synthetic_budget_ratio, 0.0)
        self.assertEqual(small_gated.ranked_kept_candidate_count, 0)

        large_world = build_world(seed=61, material_count=48)
        large_split = split_observations(large_world.observations)
        large_sample_aware = build_pipeline_examples(
            "sample_aware_self_ranked_induction",
            large_split.train,
            large_world.rules,
        )
        large_gated = build_pipeline_examples(
            "train_size_gated_sample_aware_induction",
            large_split.train,
            large_world.rules,
        )

        self.assertEqual(self._example_signature(large_gated), self._example_signature(large_sample_aware))
        self.assertEqual(
            large_gated.ranked_synthetic_budget_ratio,
            large_sample_aware.ranked_synthetic_budget_ratio,
        )
        self.assertGreater(large_gated.ranked_kept_candidate_count, 0)

        large_world = build_world(seed=59, material_count=64)
        large_split = split_observations(large_world.observations)
        large_sample_aware = build_pipeline_examples("sample_aware_self_ranked_induction", large_split.train, large_world.rules)
        self.assertEqual(large_sample_aware.ranked_synthetic_budget_ratio, 1.0)
        self.assertEqual(large_sample_aware.ranked_induction_min_support, 3)

    def test_compact_train_size_gated_induction_drops_original_qa_only_at_large_samples(self) -> None:
        small_world = build_world(seed=61, material_count=32)
        small_split = split_observations(small_world.observations)
        small_raw = build_pipeline_examples("raw_text", small_split.train, small_world.rules)
        small_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            small_split.train,
            small_world.rules,
        )

        self.assertEqual(self._example_signature(small_compact), self._example_signature(small_raw))
        self.assertEqual(small_compact.ranked_synthetic_budget_ratio, 0.0)
        self.assertEqual(small_compact.ranked_kept_candidate_count, 0)

        mid_world = build_world(seed=61, material_count=48)
        mid_split = split_observations(mid_world.observations)
        mid_sample_aware = build_pipeline_examples(
            "sample_aware_self_ranked_induction",
            mid_split.train,
            mid_world.rules,
        )
        mid_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            mid_split.train,
            mid_world.rules,
        )

        self.assertEqual(self._example_signature(mid_compact), self._example_signature(mid_sample_aware))

        large_world = build_world(seed=59, material_count=64)
        large_split = split_observations(large_world.observations)
        large_sample_aware = build_pipeline_examples(
            "sample_aware_self_ranked_induction",
            large_split.train,
            large_world.rules,
        )
        large_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            large_split.train,
            large_world.rules,
        )
        large_source_kinds = {example.source_kind for example in large_compact.examples}

        self.assertEqual(large_compact.external_event_count, large_sample_aware.external_event_count)
        self.assertLess(large_compact.internal_example_count, large_sample_aware.internal_example_count)
        self.assertLess(large_compact.internal_token_count, large_sample_aware.internal_token_count)
        self.assertEqual(large_compact.ranked_synthetic_budget_ratio, 1.0)
        self.assertEqual(large_compact.ranked_induction_min_support, 3)
        self.assertEqual(large_compact.ranked_kept_candidate_count, large_sample_aware.ranked_kept_candidate_count)
        self.assertGreater(large_compact.ranked_candidate_count, large_compact.ranked_kept_candidate_count)
        self.assertGreater(large_compact.candidate_ranking_cost_tokens, 0)
        self.assertIn("raw", large_source_kinds)
        self.assertNotIn("qa", large_source_kinds)
        self.assertIn("compact_sample_aware_self_ranked_induced", large_source_kinds)

    def test_compact_diverse_train_size_gated_induction_applies_diversity_only_after_compaction(self) -> None:
        small_world = build_world(seed=61, material_count=32)
        small_split = split_observations(small_world.observations)
        small_raw = build_pipeline_examples("raw_text", small_split.train, small_world.rules)
        small_compact_diverse = build_pipeline_examples(
            "compact_diverse_train_size_gated_induction",
            small_split.train,
            small_world.rules,
        )

        self.assertEqual(self._example_signature(small_compact_diverse), self._example_signature(small_raw))
        self.assertEqual(small_compact_diverse.ranked_synthetic_budget_ratio, 0.0)

        mid_world = build_world(seed=61, material_count=48)
        mid_split = split_observations(mid_world.observations)
        mid_sample_aware = build_pipeline_examples(
            "sample_aware_self_ranked_induction",
            mid_split.train,
            mid_world.rules,
        )
        mid_compact_diverse = build_pipeline_examples(
            "compact_diverse_train_size_gated_induction",
            mid_split.train,
            mid_world.rules,
        )

        self.assertEqual(self._example_signature(mid_compact_diverse), self._example_signature(mid_sample_aware))
        self.assertEqual(mid_compact_diverse.ranked_diversity_penalty, 0.0)

        large_world = build_world(seed=59, material_count=64)
        large_split = split_observations(large_world.observations)
        large_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            large_split.train,
            large_world.rules,
        )
        large_compact_diverse = build_pipeline_examples(
            "compact_diverse_train_size_gated_induction",
            large_split.train,
            large_world.rules,
        )
        large_source_kinds = {example.source_kind for example in large_compact_diverse.examples}

        self.assertEqual(large_compact_diverse.external_event_count, large_compact.external_event_count)
        self.assertEqual(large_compact_diverse.internal_example_count, large_compact.internal_example_count)
        self.assertEqual(large_compact_diverse.ranked_synthetic_budget_ratio, large_compact.ranked_synthetic_budget_ratio)
        self.assertEqual(large_compact_diverse.ranked_kept_candidate_count, large_compact.ranked_kept_candidate_count)
        self.assertGreater(large_compact_diverse.ranked_diversity_penalty, 0.0)
        self.assertLessEqual(large_compact_diverse.ranked_max_modifier_count, large_compact.ranked_max_modifier_count)
        self.assertIn("raw", large_source_kinds)
        self.assertNotIn("qa", large_source_kinds)
        self.assertIn("compact_diverse_sample_aware_self_ranked_induced", large_source_kinds)

    def test_density_capped_compact_induction_returns_to_raw_when_data_is_abundant(self) -> None:
        small_world = build_world(seed=71, material_count=32)
        small_split = split_observations(small_world.observations)
        small_raw = build_pipeline_examples("raw_text", small_split.train, small_world.rules)
        small_density_capped = build_pipeline_examples(
            "density_capped_compact_induction",
            small_split.train,
            small_world.rules,
        )

        self.assertEqual(self._example_signature(small_density_capped), self._example_signature(small_raw))
        self.assertEqual(small_density_capped.ranked_synthetic_budget_ratio, 0.0)

        mid_world = build_world(seed=71, material_count=48)
        mid_split = split_observations(mid_world.observations)
        mid_sample_aware = build_pipeline_examples(
            "sample_aware_self_ranked_induction",
            mid_split.train,
            mid_world.rules,
        )
        mid_density_capped = build_pipeline_examples(
            "density_capped_compact_induction",
            mid_split.train,
            mid_world.rules,
        )

        self.assertEqual(self._example_signature(mid_density_capped), self._example_signature(mid_sample_aware))

        compact_world = build_world(seed=71, material_count=80)
        compact_split = split_observations(compact_world.observations)
        compact_baseline = build_pipeline_examples(
            "compact_train_size_gated_induction",
            compact_split.train,
            compact_world.rules,
        )
        compact_density_capped = build_pipeline_examples(
            "density_capped_compact_induction",
            compact_split.train,
            compact_world.rules,
        )

        self.assertEqual(self._example_signature(compact_density_capped), self._example_signature(compact_baseline))
        self.assertIn(
            "compact_sample_aware_self_ranked_induced",
            {example.source_kind for example in compact_density_capped.examples},
        )

        abundant_world = build_world(seed=71, material_count=104)
        abundant_split = split_observations(abundant_world.observations)
        abundant_raw = build_pipeline_examples("raw_text", abundant_split.train, abundant_world.rules)
        abundant_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            abundant_split.train,
            abundant_world.rules,
        )
        abundant_density_capped = build_pipeline_examples(
            "density_capped_compact_induction",
            abundant_split.train,
            abundant_world.rules,
        )

        self.assertEqual(self._example_signature(abundant_density_capped), self._example_signature(abundant_raw))
        self.assertLess(abundant_density_capped.internal_token_count, abundant_compact.internal_token_count)
        self.assertEqual(abundant_density_capped.ranked_synthetic_budget_ratio, 0.0)

    def test_density_window_compact_induction_uses_compact_and_support_only_in_density_windows(self) -> None:
        small_world = build_world(seed=79, material_count=32)
        small_split = split_observations(small_world.observations)
        small_raw = build_pipeline_examples("raw_text", small_split.train, small_world.rules)
        small_density_window = build_pipeline_examples(
            "density_window_compact_induction",
            small_split.train,
            small_world.rules,
        )

        self.assertEqual(self._example_signature(small_density_window), self._example_signature(small_raw))
        self.assertEqual(small_density_window.ranked_synthetic_budget_ratio, 0.0)

        compact_world = build_world(seed=79, material_count=80)
        compact_split = split_observations(compact_world.observations)
        compact_baseline = build_pipeline_examples(
            "compact_train_size_gated_induction",
            compact_split.train,
            compact_world.rules,
        )
        compact_density_window = build_pipeline_examples(
            "density_window_compact_induction",
            compact_split.train,
            compact_world.rules,
        )

        self.assertEqual(self._example_signature(compact_density_window), self._example_signature(compact_baseline))
        self.assertIn(
            "compact_sample_aware_self_ranked_induced",
            {example.source_kind for example in compact_density_window.examples},
        )

        raw_window_world = build_world(seed=79, material_count=104)
        raw_window_split = split_observations(raw_window_world.observations)
        raw_window = build_pipeline_examples("raw_text", raw_window_split.train, raw_window_world.rules)
        density_window_raw = build_pipeline_examples(
            "density_window_compact_induction",
            raw_window_split.train,
            raw_window_world.rules,
        )

        self.assertEqual(self._example_signature(density_window_raw), self._example_signature(raw_window))
        self.assertEqual(density_window_raw.ranked_synthetic_budget_ratio, 0.0)

        transition_world = build_world(seed=79, material_count=112)
        transition_split = split_observations(transition_world.observations)
        transition_support = build_pipeline_examples(
            "support_ramped_compact_induction",
            transition_split.train,
            transition_world.rules,
        )
        transition_density_window = build_pipeline_examples(
            "density_window_compact_induction",
            transition_split.train,
            transition_world.rules,
        )

        self.assertEqual(self._example_signature(transition_density_window), self._example_signature(transition_support))
        self.assertEqual(transition_density_window.ranked_induction_min_support, 4)
        self.assertIn(
            "support_ramped_compact_sample_aware_self_ranked_induced",
            {example.source_kind for example in transition_density_window.examples},
        )

        abundant_world = build_world(seed=79, material_count=120)
        abundant_split = split_observations(abundant_world.observations)
        abundant_raw = build_pipeline_examples("raw_text", abundant_split.train, abundant_world.rules)
        abundant_density_window = build_pipeline_examples(
            "density_window_compact_induction",
            abundant_split.train,
            abundant_world.rules,
        )

        self.assertEqual(self._example_signature(abundant_density_window), self._example_signature(abundant_raw))
        self.assertEqual(abundant_density_window.ranked_synthetic_budget_ratio, 0.0)

    def test_support_ramped_compact_induction_raises_support_at_abundant_samples(self) -> None:
        pre_ramp_world = build_world(seed=73, material_count=96)
        pre_ramp_split = split_observations(pre_ramp_world.observations)
        pre_ramp_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            pre_ramp_split.train,
            pre_ramp_world.rules,
        )
        pre_ramp_support_ramped = build_pipeline_examples(
            "support_ramped_compact_induction",
            pre_ramp_split.train,
            pre_ramp_world.rules,
        )

        self.assertEqual(
            self._example_signature(pre_ramp_support_ramped),
            self._example_signature(pre_ramp_compact),
        )
        self.assertEqual(pre_ramp_support_ramped.ranked_induction_min_support, 3)

        abundant_world = build_world(seed=73, material_count=104)
        abundant_split = split_observations(abundant_world.observations)
        abundant_compact = build_pipeline_examples(
            "compact_train_size_gated_induction",
            abundant_split.train,
            abundant_world.rules,
        )
        abundant_support_ramped = build_pipeline_examples(
            "support_ramped_compact_induction",
            abundant_split.train,
            abundant_world.rules,
        )
        source_kinds = {example.source_kind for example in abundant_support_ramped.examples}

        self.assertEqual(abundant_support_ramped.external_event_count, abundant_compact.external_event_count)
        self.assertEqual(abundant_support_ramped.ranked_synthetic_budget_ratio, 1.0)
        self.assertEqual(abundant_support_ramped.ranked_induction_min_support, 4)
        self.assertEqual(abundant_support_ramped.ranked_induction_min_confidence, 0.55)
        self.assertLessEqual(
            abundant_support_ramped.ranked_kept_candidate_count,
            abundant_compact.ranked_kept_candidate_count,
        )
        self.assertLess(abundant_support_ramped.internal_example_count, abundant_compact.internal_example_count)
        self.assertLess(abundant_support_ramped.internal_token_count, abundant_compact.internal_token_count)
        self.assertIn("raw", source_kinds)
        self.assertNotIn("qa", source_kinds)
        self.assertIn("support_ramped_compact_sample_aware_self_ranked_induced", source_kinds)

    def test_late_confidence_ramped_compact_induction_raises_confidence_only_late(self) -> None:
        pre_late_world = build_world(seed=79, material_count=112)
        pre_late_split = split_observations(pre_late_world.observations)
        pre_late_support = build_pipeline_examples(
            "support_ramped_compact_induction",
            pre_late_split.train,
            pre_late_world.rules,
        )
        pre_late_late_confidence = build_pipeline_examples(
            "late_confidence_ramped_compact_induction",
            pre_late_split.train,
            pre_late_world.rules,
        )

        self.assertEqual(
            self._example_signature(pre_late_late_confidence),
            self._example_signature(pre_late_support),
        )
        self.assertEqual(pre_late_late_confidence.ranked_induction_min_support, 4)
        self.assertEqual(pre_late_late_confidence.ranked_induction_min_confidence, 0.55)

        late_world = build_world(seed=79, material_count=120)
        late_split = split_observations(late_world.observations)
        support_ramped = build_pipeline_examples(
            "support_ramped_compact_induction",
            late_split.train,
            late_world.rules,
        )
        late_confidence = build_pipeline_examples(
            "late_confidence_ramped_compact_induction",
            late_split.train,
            late_world.rules,
        )
        source_kinds = {example.source_kind for example in late_confidence.examples}

        self.assertEqual(late_confidence.external_event_count, support_ramped.external_event_count)
        self.assertEqual(late_confidence.ranked_synthetic_budget_ratio, 1.0)
        self.assertEqual(late_confidence.ranked_induction_min_support, 4)
        self.assertEqual(late_confidence.ranked_induction_min_confidence, 0.60)
        self.assertLessEqual(late_confidence.ranked_kept_candidate_count, support_ramped.ranked_kept_candidate_count)
        self.assertLessEqual(late_confidence.internal_example_count, support_ramped.internal_example_count)
        self.assertLessEqual(late_confidence.internal_token_count, support_ramped.internal_token_count)
        self.assertIn("raw", source_kinds)
        self.assertNotIn("qa", source_kinds)
        self.assertIn("late_confidence_ramped_compact_sample_aware_self_ranked_induced", source_kinds)

    def test_agreement_gated_self_ranked_induction_uses_train_only_source_agreement(self) -> None:
        world = build_world(seed=59, material_count=32)
        split = split_observations(world.observations)
        sample_aware = build_pipeline_examples("sample_aware_self_ranked_induction", split.train, world.rules)
        agreement_gated = build_pipeline_examples("agreement_gated_self_ranked_induction", split.train, world.rules)

        self.assertEqual(agreement_gated.external_event_count, sample_aware.external_event_count)
        self.assertEqual(agreement_gated.train_calibration_event_count, 0)
        self.assertEqual(agreement_gated.validation_calibration_event_count, 0)
        self.assertGreater(agreement_gated.ranked_candidate_count, agreement_gated.ranked_kept_candidate_count)
        self.assertGreater(agreement_gated.ranked_kept_candidate_count, 0)
        self.assertLessEqual(agreement_gated.ranked_kept_candidate_count, sample_aware.ranked_kept_candidate_count)
        self.assertGreater(agreement_gated.candidate_ranking_cost_tokens, 0)
        self.assertEqual(agreement_gated.ranked_induction_min_support, 2)
        self.assertEqual(agreement_gated.ranked_induction_min_confidence, 0.55)
        self.assertIn("agreement_gated_self_ranked_induced", {example.source_kind for example in agreement_gated.examples})


if __name__ == "__main__":
    unittest.main()
