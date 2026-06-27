from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import random

from .learner import Evaluation, featurize
from .pipelines import TrainingExample


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _stable_index(name: str, dimension: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


@dataclass(frozen=True)
class NeuralTrainingProfile:
    parameter_count: int
    training_step_count: int
    estimated_training_multiply_adds: int


class TinyMlpClassifier:
    def __init__(
        self,
        feature_dimension: int = 128,
        hidden_units: int = 16,
        learning_rate: float = 0.03,
    ) -> None:
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        self.feature_dimension = feature_dimension
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.input_hidden: list[list[float]] = [
            [0.0 for _ in range(hidden_units)]
            for _ in range(feature_dimension)
        ]
        self.hidden_bias: list[float] = [0.0 for _ in range(hidden_units)]
        self.hidden_output: list[float] = [0.0 for _ in range(hidden_units)]
        self.output_bias = 0.0
        self.training_step_count = 0
        self.estimated_training_multiply_adds = 0

    @property
    def parameter_count(self) -> int:
        return self.feature_dimension * self.hidden_units + self.hidden_units + self.hidden_units + 1

    @property
    def training_profile(self) -> NeuralTrainingProfile:
        return NeuralTrainingProfile(
            parameter_count=self.parameter_count,
            training_step_count=self.training_step_count,
            estimated_training_multiply_adds=self.estimated_training_multiply_adds,
        )

    def _initialize(self, seed: int) -> None:
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(self.feature_dimension)
        self.input_hidden = [
            [rng.uniform(-scale, scale) for _ in range(self.hidden_units)]
            for _ in range(self.feature_dimension)
        ]
        hidden_scale = 1.0 / math.sqrt(self.hidden_units)
        self.hidden_bias = [0.0 for _ in range(self.hidden_units)]
        self.hidden_output = [
            rng.uniform(-hidden_scale, hidden_scale)
            for _ in range(self.hidden_units)
        ]
        self.output_bias = 0.0
        self.training_step_count = 0
        self.estimated_training_multiply_adds = 0

    def _vectorize(self, text: str) -> dict[int, float]:
        hashed: Counter[int] = Counter()
        for name, value in featurize(text).items():
            hashed[_stable_index(name, self.feature_dimension)] += value
        norm = math.sqrt(sum(value * value for value in hashed.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in hashed.items()}

    def _forward(self, features: dict[int, float]) -> tuple[list[float], float]:
        hidden: list[float] = []
        for unit in range(self.hidden_units):
            activation = self.hidden_bias[unit]
            for index, value in features.items():
                activation += self.input_hidden[index][unit] * value
            hidden.append(math.tanh(activation))
        logit = self.output_bias + sum(
            hidden[unit] * self.hidden_output[unit]
            for unit in range(self.hidden_units)
        )
        return hidden, _sigmoid(logit)

    def score(self, text: str) -> float:
        _, probability = self._forward(self._vectorize(text))
        return probability

    def predict(self, text: str) -> bool:
        return self.score(text) >= 0.5

    def fit(self, examples: tuple[TrainingExample, ...], epochs: int, seed: int) -> int:
        self._initialize(seed)
        rng = random.Random(seed)
        indexed = list(examples)
        for _ in range(epochs):
            rng.shuffle(indexed)
            for example in indexed:
                features = self._vectorize(example.text)
                hidden, probability = self._forward(features)
                target = 1.0 if example.label else 0.0
                output_error = probability - target

                old_hidden_output = list(self.hidden_output)
                for unit in range(self.hidden_units):
                    self.hidden_output[unit] -= self.learning_rate * output_error * hidden[unit]
                self.output_bias -= self.learning_rate * output_error

                for unit in range(self.hidden_units):
                    hidden_error = output_error * old_hidden_output[unit] * (1.0 - hidden[unit] * hidden[unit])
                    self.hidden_bias[unit] -= self.learning_rate * hidden_error
                    for index, value in features.items():
                        self.input_hidden[index][unit] -= self.learning_rate * hidden_error * value

                self.training_step_count += 1
                self.estimated_training_multiply_adds += self._estimated_step_multiply_adds(len(features))
        return self.training_step_count

    def _estimated_step_multiply_adds(self, nonzero_features: int) -> int:
        forward_hidden = 2 * nonzero_features * self.hidden_units
        forward_output = 2 * self.hidden_units
        backward_output = 2 * self.hidden_units
        backward_hidden = 3 * nonzero_features * self.hidden_units
        return forward_hidden + forward_output + backward_output + backward_hidden

    def evaluate(self, examples: tuple[TrainingExample, ...]) -> Evaluation:
        if not examples:
            return Evaluation(accuracy=0.0, correct=0, total=0)
        correct = sum(1 for example in examples if self.predict(example.text) == example.label)
        return Evaluation(accuracy=correct / len(examples), correct=correct, total=len(examples))
