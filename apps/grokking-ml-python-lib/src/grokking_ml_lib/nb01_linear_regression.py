"""原著ノートブック #01 `Chapter_03_Linear_Regression/Coding_linear_regression.ipynb`。

部屋数から住宅価格を予測する線形回帰を、3 つのトリック（simple / absolute / square）で
学習したあと、同じ問題を scikit-learn の `LinearRegression` に解かせて突き合わせる。

原著は関数の中で `matplotlib` を呼びながら学習していた。ここでは描画と学習を分け、
学習の途中経過（各エポックの直線と誤差）を戻り値として返す。ノートブック側で
それを描画すれば、原著と同じ図が得られる。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

#: 原著が使うデータセット。部屋数と価格
FEATURES = np.array([1, 2, 3, 5, 6, 7])
LABELS = np.array([155, 197, 244, 356, 407, 448])


@dataclass
class Line:
    """学習された直線。原著の `price_per_room` と `base_price` に対応する。"""

    price_per_room: float
    base_price: float

    def predict(self, num_rooms: float) -> float:
        return self.base_price + self.price_per_room * num_rooms


@dataclass
class TrainingLog:
    """学習の途中経過。原著が学習ループの中で描いていたものを記録する。"""

    line: Line
    #: 各エポック開始時点の直線。原著の `draw_line` に渡していた値
    history: list[Line] = field(default_factory=list)
    #: 各エポック開始時点の RMSE
    errors: list[float] = field(default_factory=list)


def simple_trick(
    base_price: float,
    price_per_room: float,
    num_rooms: float,
    price: float,
) -> tuple[float, float]:
    """シンプルトリック。予測が外れた向きに、小さな乱数だけ直線を動かす。

    原著の実装をそのまま写している。第 3 の分岐（`price < predicted` かつ
    `num_rooms > 0`）だけ `base_price` を減らし、他は増やす非対称な書き方も、
    原著のコードに合わせてある。
    """
    small_random_1 = random.random() * 0.1
    small_random_2 = random.random() * 0.1
    predicted_price = base_price + price_per_room * num_rooms
    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random_1
        base_price += small_random_2
    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    if price < predicted_price and num_rooms > 0:
        price_per_room -= small_random_1
        base_price -= small_random_2
    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    return price_per_room, base_price


def absolute_trick(
    base_price: float,
    price_per_room: float,
    num_rooms: float,
    price: float,
    learning_rate: float,
) -> tuple[float, float]:
    """絶対トリック。外れた向きへ、学習率と部屋数に比例した幅で動かす。"""
    predicted_price = base_price + price_per_room * num_rooms
    if price > predicted_price:
        price_per_room += learning_rate * num_rooms
        base_price += learning_rate
    else:
        price_per_room -= learning_rate * num_rooms
        base_price -= learning_rate
    return price_per_room, base_price


def square_trick(
    base_price: float,
    price_per_room: float,
    num_rooms: float,
    price: float,
    learning_rate: float,
) -> tuple[float, float]:
    """二乗トリック。誤差の大きさにも比例して動かす。分岐が要らなくなる。"""
    predicted_price = base_price + price_per_room * num_rooms
    price_per_room += learning_rate * num_rooms * (price - predicted_price)
    base_price += learning_rate * (price - predicted_price)
    return price_per_room, base_price


def rmse(labels: NDArray[np.float64], predictions: NDArray[np.float64]) -> float:
    """二乗平均平方根誤差。原著の実装どおり内積で二乗和を取る。"""
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return float(np.sqrt(1.0 / n * np.dot(differences, differences)))


def linear_regression(
    features: NDArray[np.int_],
    labels: NDArray[np.int_],
    learning_rate: float = 0.01,
    epochs: int = 1000,
    trick: str = "square",
    seed: int | None = 0,
) -> TrainingLog:
    """トリックを繰り返して直線を学習する。

    原著は `random.seed(0)` をノートブックのセルで一度だけ呼んでいた。ここでは
    引数で受け取り、既定でも同じ 0 を使う。乱数の消費順序（重みの初期化 2 回 →
    毎エポックの `randint` 1 回）も原著と同じにしてある。順序が変わると
    生成される乱数列がずれ、学習結果の数値が一致しなくなる。
    """
    if seed is not None:
        random.seed(seed)

    price_per_room = random.random()
    base_price = random.random()
    history: list[Line] = []
    errors: list[float] = []

    for _ in range(epochs):
        history.append(Line(price_per_room, base_price))
        # 原著は features[0] だけを使って誤差を測っている（スカラーの予測値と
        # ラベル全体の差を取る）。ここも原著の式のまま写している
        predictions = features[0] * price_per_room + base_price
        errors.append(rmse(labels, predictions))

        i = random.randint(0, len(features) - 1)
        num_rooms = features[i]
        price = labels[i]

        if trick == "square":
            price_per_room, base_price = square_trick(
                base_price, price_per_room, num_rooms, price, learning_rate=learning_rate
            )
        elif trick == "absolute":
            price_per_room, base_price = absolute_trick(
                base_price, price_per_room, num_rooms, price, learning_rate=learning_rate
            )
        elif trick == "simple":
            price_per_room, base_price = simple_trick(
                base_price, price_per_room, num_rooms, price
            )
        else:
            raise ValueError(f"未知のトリックです: {trick}")

    return TrainingLog(Line(price_per_room, base_price), history, errors)


def fit_with_scikit_learn(
    features: NDArray[np.int_] = FEATURES,
    labels: NDArray[np.int_] = LABELS,
) -> LinearRegression:
    """同じ問題を scikit-learn に解かせる。

    scikit-learn は特徴量を 2 次元配列で受け取るので `reshape(-1, 1)` が要る。
    こちらは反復ではなく閉じた式で解くため、乱数にも学習率にも依存しない。
    """
    return LinearRegression().fit(np.asarray(features).reshape(-1, 1), labels)
