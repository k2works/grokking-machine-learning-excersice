"""原著ノートブック #12 `Chapter_09_Decision_Trees/Regression_decision_tree.ipynb`。

決定木を **回帰** に使う回。年齢からアプリの利用日数を予測する 8 点のデータに、
深さ 2 の回帰木を当てはめる。

分類木がジニ不純度を最小にする分割を探したのに対し、回帰木は
**平均二乗誤差（MSE）を最小にする分割** を探す。原著は探索の過程を手で書き下し、
9 通りの分割位置それぞれで重み付き MSE を並べて見せている。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor

#: 原著が使う 8 点。年齢と、週あたりの利用日数
FEATURES = [[10], [20], [30], [40], [50], [60], [70], [80]]
LABELS = [7, 5, 7, 1, 2, 1, 5, 4]


@dataclass
class SplitMse:
    """ある分割位置での、左右の平均と重み付き MSE。"""

    index: int
    left: list[int]
    right: list[int]
    #: 左の平均。左が空なら NaN
    left_mean: float
    right_mean: float
    #: 全体を分母にした重み付き MSE
    weighted_mse: float


def _mean_or_nan(values: NDArray[np.float64]) -> float:
    """空の配列の平均は NaN。原著も NumPy の警告つきで NaN を出している。"""
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def split_mses(labels: list[int] = LABELS) -> list[SplitMse]:
    """分割位置を 0 から n まで動かし、それぞれの重み付き MSE を求める。

    原著は `range(0, 9)` と、要素数 8 に対して **9 通り** 回している。
    先頭（左が空）と末尾（右が空）の両方が含まれるので、
    分類木の回（[#08][nb08]）が `range(len(elements))` だったのと 1 つ違う。

    重みは「左右それぞれの二乗誤差の合計を、全体の件数で割る」形。
    各群の平均で割らないので、件数の多い群ほど強く効く。
    """
    values = np.array(labels, dtype=float)
    total = len(values)
    results = []

    for index in range(total + 1):
        left, right = values[:index], values[index:]
        left_errors = left - _mean_or_nan(left) if len(left) else left
        right_errors = right - _mean_or_nan(right) if len(right) else right
        weighted = (
            np.dot(left_errors, left_errors) + np.dot(right_errors, right_errors)
        ) / total

        results.append(
            SplitMse(
                index=index,
                left=[int(v) for v in left],
                right=[int(v) for v in right],
                left_mean=_mean_or_nan(left),
                right_mean=_mean_or_nan(right),
                weighted_mse=float(weighted),
            )
        )
    return results


def best_split(labels: list[int] = LABELS) -> SplitMse:
    """重み付き MSE がもっとも小さい分割を返す。"""
    return min(split_mses(labels), key=lambda split: split.weighted_mse)


def fit(max_depth: int = 2, random_state: int | None = 0) -> DecisionTreeRegressor:
    """回帰木を学習する。原著は深さ 2 に制限している。"""
    return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state).fit(
        FEATURES, LABELS
    )


def split_conditions(model: DecisionTreeRegressor) -> list[float]:
    """分割に使われたしきい値を、節の順に並べる。特徴量は 1 つしかない。"""
    tree = model.tree_
    return [
        float(threshold)
        for feature, threshold in zip(tree.feature, tree.threshold)
        if feature >= 0
    ]


def leaf_values(model: DecisionTreeRegressor) -> list[float]:
    """葉が返す予測値を、節の順に並べる。

    回帰木の葉は **その葉に落ちた点の平均** を返す。分類木が多数決だったのと違う。
    """
    tree = model.tree_
    return [
        float(tree.value[node][0][0])
        for node in range(tree.node_count)
        if tree.feature[node] < 0
    ]
