"""第 3 章: 線形回帰。

部屋数から住宅価格を予測する 1 次元の線形回帰を、
simple / absolute / square の 3 つのトリックで学習する。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """1 次元の線形モデル price = slope * rooms + intercept。"""

    slope: float
    intercept: float

    def predict(self, rooms: float) -> float:
        return self.slope * rooms + self.intercept


def simple_trick(model: Model, rooms: float, price: float, rng: random.Random) -> Model:
    """単純なトリック。予測の上下だけを見て、ランダムな微小量だけ動かす。"""
    step_slope = rng.random() * 0.1
    step_intercept = rng.random() * 0.1
    predicted = model.predict(rooms)
    if price > predicted:
        slope = model.slope + step_slope if rooms > 0 else model.slope - step_slope
        intercept = model.intercept + step_intercept
    else:
        slope = model.slope - step_slope if rooms > 0 else model.slope + step_slope
        intercept = model.intercept - step_intercept
    return Model(slope, intercept)


def absolute_trick(model: Model, rooms: float, price: float, learning_rate: float) -> Model:
    """絶対トリック。誤差の符号のみを使い、特徴量に比例した量だけ動かす。"""
    predicted = model.predict(rooms)
    sign = 1.0 if price > predicted else -1.0
    return Model(
        model.slope + sign * learning_rate * rooms,
        model.intercept + sign * learning_rate,
    )


def square_trick(model: Model, rooms: float, price: float, learning_rate: float) -> Model:
    """二乗トリック。誤差の大きさに比例した量だけ動かす（二乗誤差の勾配降下法）。"""
    error = price - model.predict(rooms)
    return Model(
        model.slope + learning_rate * rooms * error,
        model.intercept + learning_rate * error,
    )


def rmse(labels: Sequence[float], predictions: Sequence[float]) -> float:
    """二乗平均平方根誤差。"""
    n = len(labels)
    total = sum((label - prediction) ** 2 for label, prediction in zip(labels, predictions))
    return math.sqrt(total / n)


def model_rmse(model: Model, features: Sequence[float], labels: Sequence[float]) -> float:
    return rmse(labels, [model.predict(x) for x in features])


def linear_regression(
    features: Sequence[float],
    labels: Sequence[float],
    learning_rate: float = 0.01,
    epochs: int = 1000,
    seed: int = 0,
) -> tuple[Model, list[float]]:
    """確率的勾配降下法で線形回帰を学習し、モデルとエポックごとの RMSE を返す。"""
    rng = random.Random(seed)
    model = Model(rng.random(), rng.random())
    errors: list[float] = []
    for _ in range(epochs):
        errors.append(model_rmse(model, features, labels))
        i = rng.randrange(len(features))
        model = square_trick(model, features[i], labels[i], learning_rate)
    return model, errors
