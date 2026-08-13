"""第 6 章: ロジスティック回帰。

パーセプトロンの「0 か 1 か」という硬い予測を、シグモイド関数で
「0 から 1 の確率」という連続的な予測に置き換える。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

Point = Sequence[float]


def sigmoid(x: float) -> float:
    """シグモイド関数。実数を 0 から 1 の範囲へ押し込む。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # x が大きな負の数のとき exp(-x) が溢れるため、数学的に等価な式へ切り替える
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class LogisticClassifier:
    """ロジスティック分類器。予測は 0 から 1 の確率。"""

    weights: tuple[float, ...]
    bias: float

    def score(self, point: Point) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, point))

    def predict_probability(self, point: Point) -> float:
        return sigmoid(self.score(point))

    def predict(self, point: Point, threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(point) >= threshold else 0


def log_loss(model: LogisticClassifier, point: Point, label: int) -> float:
    """1 点分の対数損失。予測確率が正解から離れるほど大きくなる。"""
    probability = model.predict_probability(point)
    # log(0) を避けるためにごくわずかに内側へ丸める
    epsilon = 1e-15
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    if label == 1:
        return -math.log(probability)
    return -math.log(1.0 - probability)


def mean_log_loss(
    model: LogisticClassifier,
    points: Sequence[Point],
    labels: Sequence[int],
) -> float:
    """全点の平均対数損失。"""
    total = sum(log_loss(model, point, label) for point, label in zip(points, labels))
    return total / len(points)


def logistic_trick(
    model: LogisticClassifier,
    point: Point,
    label: int,
    learning_rate: float = 0.01,
) -> LogisticClassifier:
    """ロジスティックトリック。すべての点を、確率の外れ具合に比例して動かす。"""
    error = label - model.predict_probability(point)
    weights = tuple(w + learning_rate * error * x for w, x in zip(model.weights, point))
    return LogisticClassifier(weights, model.bias + learning_rate * error)


def accuracy(model: LogisticClassifier, points: Sequence[Point], labels: Sequence[int]) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)


def logistic_regression(
    points: Sequence[Point],
    labels: Sequence[int],
    learning_rate: float = 0.1,
    epochs: int = 1000,
    seed: int = 0,
) -> tuple[LogisticClassifier, list[float]]:
    """ロジスティック回帰。モデルとエポックごとの平均対数損失を返す。"""
    rng = random.Random(seed)
    dimensions = len(points[0])
    model = LogisticClassifier(tuple(0.0 for _ in range(dimensions)), 0.0)
    losses: list[float] = []
    for _ in range(epochs):
        losses.append(mean_log_loss(model, points, labels))
        i = rng.randrange(len(points))
        model = logistic_trick(model, points[i], labels[i], learning_rate)
    return model, losses
