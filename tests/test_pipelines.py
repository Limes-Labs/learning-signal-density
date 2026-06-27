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

        for condition in ("raw_text", "qa_expansion", "counterfactual_expansion", "prioritized_replay"):
            examples = build_pipeline_examples(condition, split.train, world.rules)
            produced_pairs = {example.pair_key for example in examples.examples}
            self.assertFalse(produced_pairs & heldout_pairs, condition)


if __name__ == "__main__":
    unittest.main()
