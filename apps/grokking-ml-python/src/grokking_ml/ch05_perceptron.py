"""第 5 章: パーセプトロン。

点を直線で 2 クラスに分ける。予測が外れた点だけを使って境界線を動かす
パーセプトロントリックを実装する。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

Point = Sequence[float]


@dataclass(frozen=True)
class Perceptron:
    """線形分類器 score = w・x + bias。score >= 0 なら 1、そうでなければ 0。"""

    weights: tuple[float, ...]
    bias: float

    def score(self, point: Point) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, point))

    def predict(self, point: Point) -> int:
        return 1 if self.score(point) >= 0 else 0


def perceptron_trick(
    model: Perceptron,
    point: Point,
    label: int,
    learning_rate: float = 0.01,
) -> Perceptron:
    """パーセプトロントリック。誤分類した点だけモデルを動かす。"""
    error = label - model.predict(point)
    if error == 0:
        return model
    weights = tuple(w + learning_rate * error * x for w, x in zip(model.weights, point))
    return Perceptron(weights, model.bias + learning_rate * error)


def perceptron_error(model: Perceptron, point: Point, label: int) -> float:
    """1 点分の誤差。正しく分類していれば 0、誤っていればスコアの絶対値。"""
    if model.predict(point) == label:
        return 0.0
    return abs(model.score(point))


def mean_perceptron_error(
    model: Perceptron,
    points: Sequence[Point],
    labels: Sequence[int],
) -> float:
    """全点の平均誤差。"""
    total = sum(perceptron_error(model, point, label) for point, label in zip(points, labels))
    return total / len(points)


def accuracy(model: Perceptron, points: Sequence[Point], labels: Sequence[int]) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)


def perceptron_algorithm(
    points: Sequence[Point],
    labels: Sequence[int],
    learning_rate: float = 0.01,
    epochs: int = 1000,
    seed: int = 0,
) -> tuple[Perceptron, list[float]]:
    """パーセプトロンアルゴリズム。モデルとエポックごとの平均誤差を返す。"""
    rng = random.Random(seed)
    dimensions = len(points[0])
    model = Perceptron(tuple(0.0 for _ in range(dimensions)), 0.0)
    errors: list[float] = []
    for _ in range(epochs):
        errors.append(mean_perceptron_error(model, points, labels))
        i = rng.randrange(len(points))
        model = perceptron_trick(model, points[i], labels[i], learning_rate)
    return model, errors
