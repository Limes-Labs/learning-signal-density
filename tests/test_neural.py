import unittest

from learning_signal_density.neural import TinyMlpClassifier
from learning_signal_density.pipelines import TrainingExample


def _example(text: str, label: bool) -> TrainingExample:
    return TrainingExample(
        text=text,
        label=label,
        pair_key=("material", text),
        source_observation_id=text,
        source_kind="unit",
    )


class TinyMlpClassifierTests(unittest.TestCase):
    def test_tiny_mlp_learns_deterministic_interaction_pattern(self) -> None:
        train = tuple(
            _example(
                f"Question family={family} stimulus={stimulus} modifier=plain. Will brittle?",
                family == "auren" and stimulus == "blue_radiation",
            )
            for _ in range(12)
            for family in ("auren", "belith")
            for stimulus in ("blue_radiation", "amber_heat")
        )
        model = TinyMlpClassifier(feature_dimension=64, hidden_units=12, learning_rate=0.04)

        updates = model.fit(train, epochs=35, seed=123)
        evaluation = model.evaluate(train)

        self.assertEqual(updates, len(train) * 35)
        self.assertGreaterEqual(evaluation.accuracy, 0.95)
        self.assertEqual(model.parameter_count, 64 * 12 + 12 + 12 + 1)
        self.assertGreater(model.estimated_training_multiply_adds, 0)


if __name__ == "__main__":
    unittest.main()
