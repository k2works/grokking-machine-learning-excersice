"""原著ノートブック #16 `Chapter_10_Neural_Networks/Plotting_Boundaries.ipynb`。

**学習をしない回** である。重みを手で決めたネットワークを 2 つ作り、
その境界を描いて「1 層目は直線、2 層目は曲がる」ことを見せる。

題材は [#13](nb13.md) と同じ「エイリアンが幸せかどうか」の 8 点。
`aack` と `beep` の 2 語の出現数から `happy` を当てる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: 原著が図を描く範囲。境界の比較もこの範囲で行う
GRID_MIN = -0.5
GRID_MAX = 3.0
GRID_STEP = 0.005

#: 出力を 1 と見なすしきい値。原著の `f(x, y) >= 0.5`
DECISION_THRESHOLD = 0.5


def alien_dataset() -> pd.DataFrame:
    """原著の 8 件。`aack` と `beep` の出現数、`happy` が正解ラベル。"""
    return pd.DataFrame(
        {
            "aack": [1, 2, 0, 0, 1, 1, 2, 2],
            "beep": [0, 0, 1, 2, 1, 2, 1, 2],
            "happy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def features(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[["aack", "beep"]]


def labels(dataset: pd.DataFrame) -> pd.Series:
    return dataset["happy"]


def step(x: float) -> int:
    """階段関数。0 以上なら 1、そうでなければ 0。"""
    return 1 if x >= 0 else 0


def sigmoid(x: float) -> float:
    """原著の書き方 `exp(x) / (1 + exp(x))` をそのまま使う。

    数学的には `1 / (1 + exp(-x))` と同じだが、**桁あふれの向きが逆**になる。
    こちらは x が大きいときに `exp(x)` が無限大になり、
    `inf / inf` で NaN を返す。原著が図を描く範囲（-0.5〜3）では
    最大でも `exp(45)` なので問題は起きない。
    """
    return float(np.exp(x) / (1.0 + np.exp(x)))


def line_1(a: float, b: float) -> int:
    """1 層目の 1 つ目のニューロン。重み (6, 10)、バイアス -15。"""
    return step(6 * a + 10 * b - 15)


def line_2(a: float, b: float) -> int:
    """1 層目の 2 つ目のニューロン。重み (10, 6)、バイアス -15。"""
    return step(10 * a + 6 * b - 15)


def bias(a: float, b: float) -> int:
    """常に 1 を返すニューロン。入力を一切見ない。"""
    return 1


def nn_with_step(a: float, b: float) -> int:
    """階段関数だけで組んだネットワーク。

    2 層目は `line_1 + line_2 - 1.5 >= 0`。1 層目の出力は 0 か 1 なので、
    和が 1.5 以上になるのは **両方とも 1 のときだけ**。つまり AND である。
    しきい値が 0.5 でも 1.5 でもなく **1.5 であることが AND を作っている**。
    """
    return step(step(6 * a + 10 * b - 15) + step(10 * a + 6 * b - 15) - 1.5)


def nn_with_sigmoid(a: float, b: float) -> float:
    """同じ重みでシグモイドに置き換えたネットワーク。

    階段関数と違って **出力が 0 か 1 の間の連続値** になる。
    """
    return sigmoid(1.0 * sigmoid(6 * a + 10 * b - 15) + 1.0 * sigmoid(10 * a + 6 * b - 15) - 1.5)


def classify(f, a: float, b: float) -> int:
    """原著の `h(x, y) = f(x, y) >= 0.5`。境界はこの判定で決まる。"""
    return 1 if f(a, b) >= DECISION_THRESHOLD else 0


def predictions(f, dataset: pd.DataFrame) -> list[int]:
    """8 点それぞれの予測。"""
    points = features(dataset)
    return [classify(f, row.aack, row.beep) for row in points.itertuples()]


def accuracy(f, dataset: pd.DataFrame) -> float:
    """8 点に対する正解率。"""
    return float(np.mean(np.array(predictions(f, dataset)) == labels(dataset).to_numpy()))


def grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """原著の `np.meshgrid` と同じ格子。"""
    axis = np.arange(GRID_MIN, GRID_MAX, GRID_STEP)
    return np.meshgrid(axis, axis)


def region_ratio(f) -> float:
    """格子のうち、1 と判定される点の割合。

    図を見なくても「境界がどこにあるか」を数値で比べられる。
    """
    xx, yy = grid()
    values = np.array([classify(f, x, y) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)])
    return float(values.mean())


def disagreement_ratio(f, g) -> float:
    """2 つの関数の判定が食い違う格子点の割合。"""
    xx, yy = grid()
    left = np.array([classify(f, x, y) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)])
    right = np.array([classify(g, x, y) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)])
    return float(np.mean(left != right))
