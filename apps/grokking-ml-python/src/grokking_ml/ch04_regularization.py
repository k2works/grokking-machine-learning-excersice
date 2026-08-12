"""第 4 章: 過学習・未学習と正則化。

多項式回帰で次数を変えながら、訓練データとテストデータの誤差を比べる。
さらに L1 / L2 正則化で係数を抑え、複雑すぎるモデルを緩める。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Regularization(Enum):
    """正則化の種類。"""

    NONE = "none"
    L1 = "l1"
    L2 = "l2"


@dataclass(frozen=True)
class PolynomialModel:
    """多項式モデル y = bias + w1*x + w2*x^2 + ... + wn*x^n。"""

    weights: tuple[float, ...]
    bias: float

    @property
    def degree(self) -> int:
        return len(self.weights)

    def predict(self, x: float) -> float:
        return self.bias + sum(w * x**power for power, w in enumerate(self.weights, start=1))


def polynomial_features(x: float, degree: int) -> list[float]:
    """x から [x, x^2, ..., x^degree] を作る。"""
    return [x**power for power in range(1, degree + 1)]


def regularization_gradient(weight: float, kind: Regularization, strength: float) -> float:
    """正則化項の勾配。重みを 0 に引き戻す向きの力を返す。"""
    if kind is Regularization.L1:
        return strength * (1.0 if weight > 0 else -1.0 if weight < 0 else 0.0)
    if kind is Regularization.L2:
        return strength * 2.0 * weight
    return 0.0


def square_trick(
    model: PolynomialModel,
    x: float,
    y: float,
    learning_rate: float,
    kind: Regularization = Regularization.NONE,
    strength: float = 0.0,
) -> PolynomialModel:
    """二乗トリックに正則化項を加えた 1 点分の更新。"""
    error = y - model.predict(x)
    features = polynomial_features(x, model.degree)
    weights = tuple(
        w + learning_rate * (error * feature - regularization_gradient(w, kind, strength))
        for w, feature in zip(model.weights, features)
    )
    return PolynomialModel(weights, model.bias + learning_rate * error)


def rmse(labels: Sequence[float], predictions: Sequence[float]) -> float:
    """二乗平均平方根誤差。"""
    n = len(labels)
    total = sum((label - prediction) ** 2 for label, prediction in zip(labels, predictions))
    return math.sqrt(total / n)


def model_rmse(model: PolynomialModel, features: Sequence[float], labels: Sequence[float]) -> float:
    return rmse(labels, [model.predict(x) for x in features])


def train_test_split(
    features: Sequence[float],
    labels: Sequence[float],
    test_ratio: float = 0.3,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """データを訓練用とテスト用に分割する。"""
    rng = random.Random(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)
    test_size = int(len(features) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        [features[i] for i in train_indices],
        [labels[i] for i in train_indices],
        [features[i] for i in test_indices],
        [labels[i] for i in test_indices],
    )


def polynomial_regression(
    features: Sequence[float],
    labels: Sequence[float],
    degree: int,
    learning_rate: float = 0.001,
    epochs: int = 5000,
    kind: Regularization = Regularization.NONE,
    strength: float = 0.0,
    seed: int = 0,
) -> PolynomialModel:
    """確率的勾配降下法で多項式回帰を学習する。"""
    rng = random.Random(seed)
    model = PolynomialModel(tuple(0.0 for _ in range(degree)), 0.0)
    for _ in range(epochs):
        i = rng.randrange(len(features))
        model = square_trick(model, features[i], labels[i], learning_rate, kind, strength)
    return model


def weight_magnitude(model: PolynomialModel) -> float:
    """重みの絶対値の合計。モデルの複雑さの目安。"""
    return sum(abs(w) for w in model.weights)
