"""第 12 章: アンサンブル学習。

弱い学習器を集めて強い学習器を作る。多数決（バギング・ランダムフォレスト）と
逐次的な重み付け（AdaBoost）の 2 つの流儀を実装する。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from grokking_ml.ch09_decision_trees import (
    Impurity,
    Leaf,
    Point,
    Tree,
    build_tree,
    gini_impurity,
    majority_label,
)


def bootstrap_sample(
    points: Sequence[Point],
    labels: Sequence[int],
    rng: random.Random,
) -> tuple[list[Point], list[int]]:
    """復元抽出で元と同じ大きさの標本を作る。"""
    indices = [rng.randrange(len(points)) for _ in range(len(points))]
    return [points[i] for i in indices], [labels[i] for i in indices]


@dataclass(frozen=True)
class Forest:
    """多数決で予測する木の集まり。"""

    trees: tuple[Tree, ...]

    def votes(self, point: Point) -> list[int]:
        return [tree.predict(point) for tree in self.trees]

    def predict(self, point: Point) -> int:
        return majority_label(self.votes(point))


def train_forest(
    points: Sequence[Point],
    labels: Sequence[int],
    tree_count: int = 10,
    max_depth: int = 1,
    impurity: Impurity = gini_impurity,
    seed: int = 0,
) -> Forest:
    """バギング。復元抽出した標本ごとに木を育て、多数決で予測する。"""
    rng = random.Random(seed)
    trees = []
    for _ in range(tree_count):
        sample_points, sample_labels = bootstrap_sample(points, labels, rng)
        trees.append(build_tree(sample_points, sample_labels, max_depth=max_depth, impurity=impurity))
    return Forest(trees=tuple(trees))


@dataclass(frozen=True)
class WeightedTree:
    """AdaBoost の弱学習器。発言権（weight）を持つ。"""

    tree: Tree
    weight: float


@dataclass(frozen=True)
class AdaBoost:
    """重み付き多数決で予測する学習器の列。ラベルは +1 / -1。"""

    learners: tuple[WeightedTree, ...]

    def score(self, point: Point) -> float:
        return sum(learner.weight * learner.tree.predict(point) for learner in self.learners)

    def predict(self, point: Point) -> int:
        return 1 if self.score(point) >= 0 else -1


def weighted_error(
    tree: Tree,
    points: Sequence[Point],
    labels: Sequence[int],
    weights: Sequence[float],
) -> float:
    """重み付き誤り率。重みの大きい点を間違えるほど大きくなる。"""
    total = sum(weights)
    if total == 0.0:
        return 0.0
    wrong = sum(
        weight
        for point, label, weight in zip(points, labels, weights)
        if tree.predict(point) != label
    )
    return wrong / total


def learner_weight(error: float) -> float:
    """弱学習器の発言権。誤り率が小さいほど大きい。"""
    epsilon = 1e-10
    clamped = min(max(error, epsilon), 1.0 - epsilon)
    return 0.5 * math.log((1.0 - clamped) / clamped)


def train_adaboost(
    points: Sequence[Point],
    labels: Sequence[int],
    rounds: int = 5,
    max_depth: int = 1,
    impurity: Impurity = gini_impurity,
) -> AdaBoost:
    """AdaBoost。間違えた点の重みを上げながら弱学習器を足していく。"""
    weights = [1.0 / len(points)] * len(points)
    learners: list[WeightedTree] = []
    for _ in range(rounds):
        tree = build_tree_with_weights(points, labels, weights, max_depth, impurity)
        error = weighted_error(tree, points, labels, weights)
        if error >= 0.5:
            # 当てずっぽう以下の学習器は採用しない
            break
        alpha = learner_weight(error)
        learners.append(WeightedTree(tree=tree, weight=alpha))
        weights = [
            weight * math.exp(-alpha * label * tree.predict(point))
            for point, label, weight in zip(points, labels, weights)
        ]
        total = sum(weights)
        weights = [weight / total for weight in weights]
    return AdaBoost(learners=tuple(learners))


def build_tree_with_weights(
    points: Sequence[Point],
    labels: Sequence[int],
    weights: Sequence[float],
    max_depth: int,
    impurity: Impurity,
) -> Tree:
    """重みを反映した木。重みに比例して点を複製してから学習する。"""
    replicated_points: list[Point] = []
    replicated_labels: list[int] = []
    scale = 100
    for point, label, weight in zip(points, labels, weights):
        count = max(1, round(weight * scale))
        replicated_points.extend([point] * count)
        replicated_labels.extend([label] * count)
    if len(set(replicated_labels)) == 1:
        return Leaf(replicated_labels[0])
    return build_tree(replicated_points, replicated_labels, max_depth=max_depth, impurity=impurity)


def accuracy(model: Forest | AdaBoost, points: Sequence[Point], labels: Sequence[int]) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)


def tree_accuracy(tree: Tree, points: Sequence[Point], labels: Sequence[int]) -> float:
    """1 本の木の正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if tree.predict(point) == label)
    return correct / len(points)
