"""第 9 章: 決定木。

「どの質問をすれば、もっともよくデータが分かれるか」を貪欲に選び続けて
木を育てる。分割の良さはジニ不純度またはエントロピーで測る。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

Point = Sequence[float]
Impurity = Callable[[Sequence[int]], float]


def gini_impurity(labels: Sequence[int]) -> float:
    """ジニ不純度。ランダムに 2 つ選んだとき、ラベルが食い違う確率。"""
    if not labels:
        return 0.0
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in Counter(labels).values())


def entropy(labels: Sequence[int]) -> float:
    """エントロピー。ラベルの散らばりを情報量で測る。"""
    if not labels:
        return 0.0
    total = len(labels)
    return -sum(
        (count / total) * math.log2(count / total) for count in Counter(labels).values()
    )


@dataclass(frozen=True)
class Split:
    """1 つの質問による分割。「feature 番目の特徴量が threshold 未満か」を問う。"""

    feature: int
    threshold: float

    def matches(self, point: Point) -> bool:
        """左（True）へ進むか。"""
        return point[self.feature] < self.threshold


def weighted_impurity(
    left_labels: Sequence[int],
    right_labels: Sequence[int],
    impurity: Impurity = gini_impurity,
) -> float:
    """分割後の不純度。左右の大きさで重み付けして平均する。"""
    total = len(left_labels) + len(right_labels)
    if total == 0:
        return 0.0
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    return left_weight * impurity(left_labels) + right_weight * impurity(right_labels)


def information_gain(
    labels: Sequence[int],
    left_labels: Sequence[int],
    right_labels: Sequence[int],
    impurity: Impurity = gini_impurity,
) -> float:
    """情報利得。分割によって不純度がどれだけ下がったか。"""
    return impurity(labels) - weighted_impurity(left_labels, right_labels, impurity)


def apply_split(
    points: Sequence[Point],
    labels: Sequence[int],
    split: Split,
) -> tuple[list[Point], list[int], list[Point], list[int]]:
    """分割を適用して左右に振り分ける。"""
    left_points, left_labels, right_points, right_labels = [], [], [], []
    for point, label in zip(points, labels):
        if split.matches(point):
            left_points.append(point)
            left_labels.append(label)
        else:
            right_points.append(point)
            right_labels.append(label)
    return left_points, left_labels, right_points, right_labels


def candidate_splits(points: Sequence[Point]) -> list[Split]:
    """試す価値のある分割の候補。隣り合う値の中点を閾値にする。"""
    splits: list[Split] = []
    for feature in range(len(points[0])):
        values = sorted({point[feature] for point in points})
        for low, high in pairwise(values):
            splits.append(Split(feature=feature, threshold=(low + high) / 2))
    return splits


def best_split(
    points: Sequence[Point],
    labels: Sequence[int],
    impurity: Impurity = gini_impurity,
) -> tuple[Split, float] | None:
    """情報利得がもっとも大きい分割。改善しないなら None。"""
    best: tuple[Split, float] | None = None
    for split in candidate_splits(points):
        _, left_labels, _, right_labels = apply_split(points, labels, split)
        if not left_labels or not right_labels:
            continue
        gain = information_gain(labels, left_labels, right_labels, impurity)
        if best is None or gain > best[1]:
            best = (split, gain)
    if best is None or best[1] <= 0.0:
        return None
    return best


@dataclass(frozen=True)
class Leaf:
    """葉。多数決で決めたラベルを返す。"""

    label: int

    def predict(self, point: Point) -> int:
        return self.label


@dataclass(frozen=True)
class Node:
    """内部ノード。質問に応じて左右の枝へ進む。"""

    split: Split
    left: Tree
    right: Tree

    def predict(self, point: Point) -> int:
        branch = self.left if self.split.matches(point) else self.right
        return branch.predict(point)


Tree = Leaf | Node


def majority_label(labels: Sequence[int]) -> int:
    """多数決。同数なら小さいラベルを選ぶ。"""
    counts = Counter(labels)
    top = max(counts.values())
    return min(label for label, count in counts.items() if count == top)


def build_tree(
    points: Sequence[Point],
    labels: Sequence[int],
    max_depth: int = 5,
    min_samples: int = 1,
    impurity: Impurity = gini_impurity,
) -> Tree:
    """決定木を再帰的に構築する。"""
    if max_depth <= 0 or len(labels) <= min_samples or len(set(labels)) == 1:
        return Leaf(majority_label(labels))
    found = best_split(points, labels, impurity)
    if found is None:
        return Leaf(majority_label(labels))
    split, _ = found
    left_points, left_labels, right_points, right_labels = apply_split(points, labels, split)
    return Node(
        split=split,
        left=build_tree(left_points, left_labels, max_depth - 1, min_samples, impurity),
        right=build_tree(right_points, right_labels, max_depth - 1, min_samples, impurity),
    )


def depth(tree: Tree) -> int:
    """木の深さ。葉だけなら 0。"""
    if isinstance(tree, Leaf):
        return 0
    return 1 + max(depth(tree.left), depth(tree.right))


def leaf_count(tree: Tree) -> int:
    """葉の数。"""
    if isinstance(tree, Leaf):
        return 1
    return leaf_count(tree.left) + leaf_count(tree.right)


def accuracy(tree: Tree, points: Sequence[Point], labels: Sequence[int]) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if tree.predict(point) == label)
    return correct / len(points)
