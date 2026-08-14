"""原著ノートブック #05 `Chapter_06_Logistic_Regression/Coding_logistic_regression.ipynb`。

#04 と同じ形の 8 点を、今度はロジスティック回帰で分ける。
ステップ関数がシグモイドに、パーセプトロン誤差が対数損失に置き換わる。

原著は対数損失の「別の書き方」も示しているが、**その式は対数損失と一致しない**。
正しい形と並べて、両方を実装してある。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

#: 原著が使う 8 点。#04 と似ているが最後の 2 点が入れ替わっている
FEATURES = np.array([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [3, 2], [2, 3]])
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def sigmoid(x: float) -> float:
    """シグモイド関数。

    原著のコメントどおり `exp(x) / (1 + exp(x))` で書く。教科書によくある
    `1 / (1 + exp(-x))` と数学的には同じだが、x が大きな負の数のときに
    `exp(-x)` が溢れない形になっている。
    """
    return float(np.exp(x) / (1 + np.exp(x)))


def score(weights: list[float], bias: float, features: NDArray[np.int_]) -> float:
    """重み付き和にバイアスを足したもの。"""
    return float(np.dot(weights, features) + bias)


def prediction(weights: list[float], bias: float, features: NDArray[np.int_]) -> float:
    """予測確率。パーセプトロンと違い 0 / 1 ではなく 0〜1 の連続値を返す。"""
    return sigmoid(score(weights, bias, features))


def log_loss(
    weights: list[float], bias: float, features: NDArray[np.int_], label: int
) -> float:
    """対数損失。当たっていても 0 にはならず、確信の度合いで連続的に変わる。"""
    pred = prediction(weights, bias, features)
    return float(-label * np.log(pred) - (1 - label) * np.log(1 - pred))


def total_log_loss(
    weights: list[float], bias: float, features: NDArray[np.int_], labels: NDArray[np.int_]
) -> float:
    """全点の対数損失の合計。原著は平均ではなく合計を取っている。"""
    return sum(log_loss(weights, bias, features[i], labels[i]) for i in range(len(features)))


def soft_relu(x: float) -> float:
    """ソフト ReLU。`log(1 + exp(x))` で、ReLU をなめらかにしたもの。"""
    return float(np.log(1 + np.exp(x)))


def alternate_log_loss_original(
    weights: list[float], bias: float, features: NDArray[np.int_], label: int
) -> float:
    """原著が「対数損失の別の書き方」として示す式。

    **実際には対数損失と一致しない。** `pred` は 0〜1 の確率なので、
    `(pred - label)` は -1 か +1 ではなく中間の値になる。
    スコアが 0 のときだけ両者が一致する。詳しくは記事を参照。
    """
    pred = prediction(weights, bias, features)
    return soft_relu((pred - label) * score(weights, bias, features))


def alternate_log_loss(
    weights: list[float], bias: float, features: NDArray[np.int_], label: int
) -> float:
    """対数損失と厳密に等しい「別の書き方」。

    ラベルが 0 なら +1、1 なら -1 を掛ける。つまり `(1 - 2 * label)`。
    こうすると `soft_relu` の中身が `log_loss` の定義そのものになる。
    """
    return soft_relu((1 - 2 * label) * score(weights, bias, features))


def logistic_trick(
    weights: list[float],
    bias: float,
    features: NDArray[np.int_],
    label: int,
    learning_rate: float = 0.05,
) -> tuple[list[float], float]:
    """ロジスティックトリック。パーセプトロンのトリックと同じ形をしている。

    違いは `pred` が 0 / 1 ではなく 0〜1 の連続値であること。おかげで
    「少しだけ間違えた」点は少しだけ動かす、という調整が入る。

    #04 の「短く書いた版」と違い、**バイアスの更新はループの外側にある**。
    原著もここでは正しく書いている。
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
    #: 各エポック開始時点の対数損失の合計
    errors: list[float] = field(default_factory=list)


def logistic_regression_algorithm(
    features: NDArray[np.int_] = FEATURES,
    labels: NDArray[np.int_] = LABELS,
    learning_rate: float = 0.01,
    epochs: int = 500,
    seed: int | None = 0,
) -> TrainingLog:
    """トリックを繰り返して分離直線を学習する。

    #04 と同じく、原著は点の選択に種を与えていない標準ライブラリの `random` を
    使っている。原著の出力 `([1.2019, 0.7009], -2.7884)` は再現できない。
    """
    if seed is not None:
        random.seed(seed)

    weights = [1.0 for _ in range(len(features[0]))]
    bias = 0.0
    errors: list[float] = []

    for _ in range(epochs):
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features) - 1)
        weights, bias = logistic_trick(
            weights, bias, features[j], labels[j], learning_rate=learning_rate
        )

    return TrainingLog(weights, bias, errors)


def fit_with_scikit_learn(
    features: NDArray[np.int_] = FEATURES,
    labels: NDArray[np.int_] = LABELS,
) -> LogisticRegression:
    """同じ問題を scikit-learn の `LogisticRegression` に解かせる。"""
    return LogisticRegression().fit(features, labels)
