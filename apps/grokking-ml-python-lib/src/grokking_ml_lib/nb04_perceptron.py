"""原著ノートブック #04 `Chapter_05_Perceptron_Algorithm/Coding_perceptron_algorithm.ipynb`。

2 次元の 8 点を直線で 2 クラスに分ける。パーセプトロンのトリックを手で書いてから、
同じ問題を scikit-learn の `Perceptron` に解かせる。

原著はトリックを 2 通り書いており、**2 つ目は 1 つ目と挙動が違う**。
バイアスの更新が重みのループの内側にあり、特徴量の数だけ繰り返し適用されるためである。
原著のセル出力もその挙動を前提にしているので、両方を実装して差が見えるようにした。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Perceptron

#: 原著が使う 8 点。aack と beep の出現回数を模した 2 次元の特徴量
FEATURES = np.array([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2]])
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def score(weights: list[float], bias: float, features: NDArray[np.int_]) -> float:
    """重み付き和にバイアスを足したもの。直線からの符号つき距離に比例する。"""
    return float(features.dot(weights) + bias)


def step(x: float) -> int:
    """ステップ関数。0 以上なら 1、そうでなければ 0。"""
    return 1 if x >= 0 else 0


def prediction(weights: list[float], bias: float, features: NDArray[np.int_]) -> int:
    """スコアをステップ関数に通した予測ラベル。"""
    return step(score(weights, bias, features))


def error(
    weights: list[float], bias: float, features: NDArray[np.int_], label: int
) -> float:
    """パーセプトロン誤差。当たっていれば 0、外れていればスコアの絶対値。

    「外れたぶんだけ罰する」ので、境界のすぐ近くで間違えた点より、
    大きく間違えた点のほうが強く効く。
    """
    if prediction(weights, bias, features) == label:
        return 0.0
    return float(np.abs(score(weights, bias, features)))


def mean_perceptron_error(
    weights: list[float], bias: float, features: NDArray[np.int_], labels: NDArray[np.int_]
) -> float:
    """全点のパーセプトロン誤差の平均。"""
    total = sum(error(weights, bias, features[i], labels[i]) for i in range(len(features)))
    return total / len(features)


def perceptron_trick_explicit(
    weights: list[float],
    bias: float,
    features: NDArray[np.int_],
    label: int,
    learning_rate: float = 0.05,
) -> tuple[list[float], float]:
    """原著が最初に示すトリック。当たっていれば何もせず、外れたら向きを見て動かす。

    バイアスの更新はループの **外側** にあり、1 回だけ適用される。
    """
    updated = list(weights)
    pred = prediction(weights, bias, features)
    if pred == label:
        return updated, bias

    if label == 1 and pred == 0:
        for i in range(len(updated)):
            updated[i] += features[i] * learning_rate
        bias += learning_rate
    elif label == 0 and pred == 1:
        for i in range(len(updated)):
            updated[i] -= features[i] * learning_rate
        bias -= learning_rate
    return updated, bias


def perceptron_trick(
    weights: list[float],
    bias: float,
    features: NDArray[np.int_],
    label: int,
    learning_rate: float = 0.05,
) -> tuple[list[float], float]:
    """原著が「短く書いた版」として示すトリック。以降の学習ループはこちらを使う。

    `label - pred` が符号を持つので分岐が要らなくなる。ただし原著のコードでは
    **バイアスの更新が重みのループの内側にある**。特徴量が 2 つなら学習率が
    2 回足され、`perceptron_trick_explicit` の 2 倍動く。

    原著のセル出力（`[0.9, 1.85], -4.1`）はこの挙動を前提にしているので、
    そのまま写している。
    """
    updated = list(weights)
    pred = prediction(weights, bias, features)
    for i in range(len(updated)):
        updated[i] += (label - pred) * features[i] * learning_rate
        bias += (label - pred) * learning_rate
    return updated, bias


@dataclass
class TrainingLog:
    """学習の結果と途中経過。"""

    weights: list[float]
    bias: float
    #: 各エポック開始時点の平均パーセプトロン誤差
    errors: list[float] = field(default_factory=list)


def perceptron_algorithm(
    features: NDArray[np.int_] = FEATURES,
    labels: NDArray[np.int_] = LABELS,
    learning_rate: float = 0.01,
    epochs: int = 200,
    seed: int | None = 0,
) -> TrainingLog:
    """トリックを繰り返して分離直線を学習する。

    原著は `np.random.seed(42)` を呼んでいるが、点の選択に使っているのは
    **標準ライブラリの `random.randint`** である。こちらは種を与えていないので、
    原著の出力 `([0.55, 0.25], -1.1)` は実行のたびに変わる値であり再現できない。
    ここでは `random.seed` を引数で受け取り、決定的に走らせる。
    """
    if seed is not None:
        random.seed(seed)

    weights = [1.0 for _ in range(len(features[0]))]
    bias = 0.0
    errors: list[float] = []

    for _ in range(epochs):
        errors.append(mean_perceptron_error(weights, bias, features, labels))
        i = random.randint(0, len(features) - 1)
        weights, bias = perceptron_trick(
            weights, bias, features[i], labels[i], learning_rate=learning_rate
        )

    return TrainingLog(weights, bias, errors)


def fit_with_scikit_learn(
    features: NDArray[np.int_] = FEATURES,
    labels: NDArray[np.int_] = LABELS,
) -> Perceptron:
    """同じ問題を scikit-learn の `Perceptron` に解かせる。"""
    return Perceptron().fit(features, labels)
