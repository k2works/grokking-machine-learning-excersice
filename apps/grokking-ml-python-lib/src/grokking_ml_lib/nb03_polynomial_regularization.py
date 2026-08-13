"""原著ノートブック #03 `Chapter_04_Testing_Overfitting_Underfitting/Polynomial_regression_regularization.ipynb`。

二次関数 -x^2 + 2 の周りに散らした 40 点へ、**次数 20** の多項式を当てはめる。
正則化なしでは激しく過学習し、L1（Lasso）・L2（Ridge）を入れると収まる。

多項式特徴量を作る `PolynomialFeatures` と、正則化つき線形回帰の `Lasso` / `Ridge` が
この章の主役である。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#: 元にした多項式 -x^2 + 2 の係数。添字が次数に対応する
POLYNOMIAL_COEFFICIENTS = [2, 0, -1]

#: 原著が使う点の数とノイズの大きさ
SAMPLE_SIZE = 40
NOISE_STD = 0.1

#: 原著が当てはめる多項式の次数。40 点に対して 20 次は明らかに過剰
DEGREE = 20


class Regularization(Enum):
    """正則化の種類。原著は文字列 'L1' / 'L2' / None で切り替えていた。"""

    NONE = "none"
    L1 = "L1"
    L2 = "L2"


def polynomial(coefficients: list[int], x: float) -> float:
    """多項式の値を求める。`coefficients[i]` が x^i の係数。"""
    return sum(coefficients[i] * x**i for i in range(len(coefficients)))


@dataclass
class Dataset:
    """生成したデータと、その訓練／テスト分割。"""

    x: list[float]
    y: list[float]
    x_train: list[float]
    x_test: list[float]
    y_train: list[float]
    y_test: list[float]


def generate_dataset(
    size: int = SAMPLE_SIZE,
    seed: int | None = 0,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Dataset:
    """-x^2 + 2 の周りにガウスノイズを載せた点を生成し、訓練とテストに分ける。

    乱数の消費順序は原著と同じ（1 点につき `uniform` → `gauss` の順）。
    順序が変わると生成される点がずれる。
    """
    if seed is not None:
        random.seed(seed)

    x: list[float] = []
    y: list[float] = []
    for _ in range(size):
        sampled = random.uniform(-1, 1)
        x.append(sampled)
        y.append(polynomial(POLYNOMIAL_COEFFICIENTS, sampled) + random.gauss(0, NOISE_STD))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return Dataset(x, y, x_train, x_test, y_train, y_test)


def polynomial_features(x: list[float], degree: int) -> NDArray[np.float64]:
    """x を x, x^2, ..., x^degree の列に展開する。

    `include_bias=False` にしているのは、切片を線形回帰側に任せるためである。
    True にすると定数 1 の列が特徴量として入り、切片と二重になる。
    """
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(
        np.array(x).reshape(-1, 1)
    )


def train_polynomial_regression(
    x: list[float],
    y: list[float],
    degree: int = DEGREE,
    regularization: Regularization = Regularization.NONE,
    alpha: float = 1.0,
) -> RegressorMixin:
    """多項式回帰を学習する。正則化は L1（Lasso）・L2（Ridge）から選ぶ。"""
    features = polynomial_features(x, degree)
    match regularization:
        case Regularization.L1:
            model = Lasso(alpha=alpha)
        case Regularization.L2:
            model = Ridge(alpha=alpha)
        case Regularization.NONE:
            model = LinearRegression()

    return model.fit(features, np.array(y))


def evaluate_model(
    model: RegressorMixin,
    x: list[float],
    y: list[float],
    degree: int = DEGREE,
) -> float:
    """テストセットに対する RMSE を返す。"""
    predictions = model.predict(polynomial_features(x, degree))
    return float(np.sqrt(mean_squared_error(np.array(y), predictions)))
