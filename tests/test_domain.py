import unittest

from learning_signal_density.domain import build_world, split_observations


class DomainSplitTests(unittest.TestCase):
    def test_split_keeps_train_validation_and_heldout_pairs_disjoint(self) -> None:
        world = build_world(seed=17, material_count=36)
        split = split_observations(world.observations, validation_fraction=0.2, heldout_fraction=0.2)

        train_pairs = {item.pair_key for item in split.train}
        validation_pairs = {item.pair_key for item in split.validation}
        heldout_pairs = {item.pair_key for item in split.heldout}

        self.assertTrue(train_pairs)
        self.assertTrue(validation_pairs)
        self.assertTrue(heldout_pairs)
        self.assertFalse(train_pairs & validation_pairs)
        self.assertFalse(train_pairs & heldout_pairs)
        self.assertFalse(validation_pairs & heldout_pairs)

    def test_world_observations_have_deterministic_labels(self) -> None:
        world_a = build_world(seed=23, material_count=24)
        world_b = build_world(seed=23, material_count=24)

        labels_a = [item.label for item in world_a.observations]
        labels_b = [item.label for item in world_b.observations]

        self.assertEqual(labels_a, labels_b)
        self.assertIn(True, labels_a)
        self.assertIn(False, labels_a)


if __name__ == "__main__":
    unittest.main()
