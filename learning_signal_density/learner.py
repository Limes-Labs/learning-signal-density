from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
import re

from .pipelines import TrainingExample


TOKEN_RE = re.compile(r"[A-Za-z0-9_=.]+")


def featurize(text: str) -> Counter[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    features: Counter[str] = Counter(tokens)
    keyed = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            keyed[key] = value
    for left, right in (("family", "stimulus"), ("modifier", "stimulus"), ("family", "modifier")):
        if left in keyed and right in keyed:
            features[f"{left}={keyed[left]}|{right}={keyed[right]}"] += 1
    return features


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    correct: int
    total: int


class PerceptronClassifier:
    def __init__(self) -> None:
        self.weights: defaultdict[str, float] = defaultdict(float)
        self.bias = 0.0

    def score(self, text: str) -> float:
        return self.bias + sum(self.weights[name] * value for name, value in featurize(text).items())

    def predict(self, text: str) -> bool:
        return self.score(text) >= 0.0

    def fit(self, examples: tuple[TrainingExample, ...], epochs: int, seed: int) -> int:
        updates = 0
        rng = random.Random(seed)
        indexed = list(examples)
        for _ in range(epochs):
            rng.shuffle(indexed)
            for example in indexed:
                label = 1.0 if example.label else -1.0
                prediction = 1.0 if self.predict(example.text) else -1.0
                if prediction != label:
                    features = featurize(example.text)
                    for name, value in features.items():
                        self.weights[name] += label * value
                    self.bias += label
                    updates += 1
        return updates

    def evaluate(self, examples: tuple[TrainingExample, ...]) -> Evaluation:
        if not examples:
            return Evaluation(accuracy=0.0, correct=0, total=0)
        correct = sum(1 for example in examples if self.predict(example.text) == example.label)
        return Evaluation(accuracy=correct / len(examples), correct=correct, total=len(examples))


def majority_baseline(train: tuple[TrainingExample, ...], eval_examples: tuple[TrainingExample, ...]) -> Evaluation:
    positive = sum(1 for example in train if example.label)
    majority_label = positive >= (len(train) - positive)
    correct = sum(1 for example in eval_examples if example.label == majority_label)
    return Evaluation(accuracy=correct / len(eval_examples), correct=correct, total=len(eval_examples))
