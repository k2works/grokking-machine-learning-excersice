"""第 11 章: サポートベクターマシンとカーネル法。

第 5 章のパーセプトロンは「分ければどこでもよい」だった。SVM は
2 クラスの間にできるだけ広い余白（マージン）を空ける境界線を選ぶ。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Point = Sequence[float]
Kernel = Callable[[Point, Point], float]


def dot(a: Point, b: Point) -> float:
    """内積。"""
    return sum(x * y for x, y in zip(a, b))


def linear_kernel(a: Point, b: Point) -> float:
    """線形カーネル。ただの内積。"""
    return dot(a, b)


def polynomial_kernel(degree: int = 2, constant: float = 1.0) -> Kernel:
    """多項式カーネル。特徴量の積を暗黙のうちに作る。"""

    def kernel(a: Point, b: Point) -> float:
        return (dot(a, b) + constant) ** degree

    return kernel


def rbf_kernel(gamma: float = 1.0) -> Kernel:
    """RBF（ガウシアン）カーネル。距離が近いほど 1 に近づく。"""

    def kernel(a: Point, b: Point) -> float:
        squared_distance = sum((x - y) ** 2 for x, y in zip(a, b))
        return math.exp(-gamma * squared_distance)

    return kernel


@dataclass(frozen=True)
class SupportVectorMachine:
    """線形 SVM。ラベルは +1 と -1 を使う。"""

    weights: tuple[float, ...]
    bias: float

    def score(self, point: Point) -> float:
        return self.bias + dot(self.weights, point)

    def predict(self, point: Point) -> int:
        return 1 if self.score(point) >= 0 else -1

    def margin(self, points: Sequence[Point], labels: Sequence[int]) -> float:
        """マージン幅。境界線からもっとも近い点までの距離の 2 倍。"""
        norm = math.sqrt(sum(w * w for w in self.weights))
        if norm == 0.0:
            return 0.0
        closest = min(abs(self.score(point)) for point in points)
        return 2.0 * closest / norm


def hinge_loss(model: SupportVectorMachine, point: Point, label: int) -> float:
    """ヒンジ損失。マージンの内側に入った分だけ罰する。"""
    return max(0.0, 1.0 - label * model.score(point))


def svm_error(
    model: SupportVectorMachine,
    points: Sequence[Point],
    labels: Sequence[int],
    regularization: float = 0.1,
) -> float:
    """SVM の目的関数。ヒンジ損失の平均 + 重みの大きさへの罰。"""
    losses = sum(
        hinge_loss(model, point, label) for point, label in zip(points, labels)
    ) / len(points)
    penalty = regularization * sum(w * w for w in model.weights)
    return losses + penalty


def svm_step(
    model: SupportVectorMachine,
    point: Point,
    label: int,
    learning_rate: float = 0.01,
    regularization: float = 0.1,
) -> SupportVectorMachine:
    """1 点分の更新。マージンの内側なら押し返し、常に重みを縮める。"""
    inside_margin = label * model.score(point) < 1.0
    weights = []
    for w, x in zip(model.weights, point):
        gradient = 2.0 * regularization * w
        if inside_margin:
            gradient -= label * x
        weights.append(w - learning_rate * gradient)
    bias = model.bias + (learning_rate * label if inside_margin else 0.0)
    return SupportVectorMachine(tuple(weights), bias)


def train_svm(
    points: Sequence[Point],
    labels: Sequence[int],
    learning_rate: float = 0.01,
    epochs: int = 5000,
    regularization: float = 0.1,
    seed: int = 0,
) -> tuple[SupportVectorMachine, list[float]]:
    """SVM を学習する。モデルとエポックごとの目的関数値を返す。"""
    rng = random.Random(seed)
    model = SupportVectorMachine(tuple(0.0 for _ in points[0]), 0.0)
    errors: list[float] = []
    for _ in range(epochs):
        errors.append(svm_error(model, points, labels, regularization))
        i = rng.randrange(len(points))
        model = svm_step(model, points[i], labels[i], learning_rate, regularization)
    return model, errors


def accuracy(
    model: SupportVectorMachine,
    points: Sequence[Point],
    labels: Sequence[int],
) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)


@dataclass(frozen=True)
class KernelClassifier:
    """カーネル分類器。訓練点そのものを重み付きで覚えておく。"""

    points: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    weights: tuple[float, ...]
    bias: float
    kernel: Kernel

    def score(self, point: Point) -> float:
        return self.bias + sum(
            weight * label * self.kernel(support, point)
            for weight, label, support in zip(self.weights, self.labels, self.points)
        )

    def predict(self, point: Point) -> int:
        return 1 if self.score(point) >= 0 else -1


def train_kernel_classifier(
    points: Sequence[Point],
    labels: Sequence[int],
    kernel: Kernel = linear_kernel,
    learning_rate: float = 0.1,
    epochs: int = 2000,
    seed: int = 0,
) -> KernelClassifier:
    """カーネル版パーセプトロン。誤分類した点の重みだけを増やす。"""
    rng = random.Random(seed)
    weights = [0.0] * len(points)
    bias = 0.0
    frozen_points = tuple(tuple(point) for point in points)
    for _ in range(epochs):
        i = rng.randrange(len(points))
        model = KernelClassifier(
            points=frozen_points,
            labels=tuple(labels),
            weights=tuple(weights),
            bias=bias,
            kernel=kernel,
        )
        if labels[i] * model.score(points[i]) <= 0:
            weights[i] += learning_rate
            bias += learning_rate * labels[i]
    return KernelClassifier(
        points=frozen_points,
        labels=tuple(labels),
        weights=tuple(weights),
        bias=bias,
        kernel=kernel,
    )


def kernel_accuracy(
    model: KernelClassifier,
    points: Sequence[Point],
    labels: Sequence[int],
) -> float:
    """カーネル分類器の正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)
