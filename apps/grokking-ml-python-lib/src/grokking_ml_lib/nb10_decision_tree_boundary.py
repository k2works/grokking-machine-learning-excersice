"""原著ノートブック #10 `Chapter_09_Decision_Trees/Graphical_example.ipynb`。

2 次元の 12 点を決定木で分け、**決定境界を図で見る** 回。
原著は 3 つのモデルを並べる。

1. ジニ不純度で分割した木
2. エントロピーで分割した木
3. 深さ 1 に制限した木（1 本の直線になる）

決定木の境界は必ず **軸に平行な長方形の集まり** になる。そこが線形モデルとの違いで、
原著が図で見せたいところでもある。図そのものは記事の対象外なので、
**境界を格子上の予測ラベルとして取り出し**、性質をテストで確かめる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier

#: 原著が使う 12 点
DATASET = pd.DataFrame(
    {
        "x_0": [7, 3, 2, 1, 2, 4, 1, 8, 6, 7, 8, 9],
        "x_1": [1, 2, 3, 5, 6, 7, 9, 10, 5, 8, 4, 6],
        "y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    }
)

FEATURE_NAMES = ["x_0", "x_1"]

#: 原著の `plot_model` が使う格子の刻み幅
PLOT_STEP = 0.2


def features() -> pd.DataFrame:
    return DATASET[FEATURE_NAMES]


def labels() -> pd.Series:
    return DATASET["y"]


def fit(
    criterion: str = "gini",
    max_depth: int | None = None,
    random_state: int | None = 0,
) -> DecisionTreeClassifier:
    """決定木を学習する。原著は 3 通りの設定で呼び分けている。"""
    return DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=random_state
    ).fit(features(), labels())


@dataclass
class DecisionGrid:
    """決定境界を格子上の予測ラベルとして持ったもの。"""

    #: x 軸の目盛
    x_values: NDArray[np.float64]
    #: y 軸の目盛
    y_values: NDArray[np.float64]
    #: 予測ラベル。`predictions[row][column]` が `(x_values[column], y_values[row])` に対応
    predictions: NDArray[np.int_]

    @property
    def shape(self) -> tuple[int, int]:
        return self.predictions.shape


def decision_grid(
    model: DecisionTreeClassifier, step: float = PLOT_STEP
) -> DecisionGrid:
    """原著の `plot_model` と同じ格子を作り、各点の予測ラベルを返す。

    原著は `X[:, 0].min() - 1` から `X[:, 0].max() + 1` までを `np.arange` で
    刻んでいる。`arange` は終端を含まないので、格子の右端・上端は
    最大値 + 1 の手前で止まる。
    """
    values = features().to_numpy()
    x_values = np.arange(values[:, 0].min() - 1, values[:, 0].max() + 1, step)
    y_values = np.arange(values[:, 1].min() - 1, values[:, 1].max() + 1, step)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)

    points = pd.DataFrame(
        np.c_[mesh_x.ravel(), mesh_y.ravel()], columns=FEATURE_NAMES
    )
    predictions = model.predict(points).reshape(mesh_x.shape)
    return DecisionGrid(x_values, y_values, predictions)


def split_conditions(model: DecisionTreeClassifier) -> list[tuple[str, float]]:
    """木のなかで実際に分割に使われた条件を、節の順に並べる。"""
    tree = model.tree_
    return [
        (FEATURE_NAMES[feature], float(threshold))
        for feature, threshold in zip(tree.feature, tree.threshold)
        if feature >= 0
    ]


def boundary_columns(grid: DecisionGrid) -> set[float]:
    """左右で予測が変わる x 座標を集める。

    決定木の境界は軸に平行なので、変わる位置は分割しきい値の近くに限られる。
    """
    changes: set[float] = set()
    for row in grid.predictions:
        for column in range(1, len(row)):
            if row[column] != row[column - 1]:
                changes.add(float(grid.x_values[column]))
    return changes
