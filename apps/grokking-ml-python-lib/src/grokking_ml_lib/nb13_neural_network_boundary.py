"""原著ノートブック #13 `Chapter_10_Neural_Networks/Graphical_example.ipynb`。

ニューラルネットワークの章に入る。円形に分布した 110 点を、
**2 層の隠れ層を持つネットワーク** で分類し、決定境界を見る。

決定木（[#10](nb10.md)）の境界が軸に平行な長方形だったのに対し、
ニューラルネットワークの境界は **曲線** になる。そこが原著の見せ場である。

原著のネットワークは Keras で組まれている。
`Dense(128, relu)` → `Dropout(0.2)` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(2, softmax)`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from grokking_ml_lib.datasets import dataset_path

#: 原著が使う乱数の種
NUMPY_SEED = 0
TENSORFLOW_SEED = 1

#: 原著のネットワークの形
HIDDEN_UNITS = (128, 64)
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 10

#: 原著の `plot_model` が使う格子の刻み幅
PLOT_STEP = 0.2


def load_circle() -> pd.DataFrame:
    """円形に分布したデータを読み込む。110 点。"""
    return pd.read_csv(dataset_path("one_circle.csv"), index_col=0)


def features(data: pd.DataFrame) -> NDArray[np.float64]:
    return np.array(data[["x_1", "x_2"]])


def labels(data: pd.DataFrame) -> NDArray[np.int_]:
    return np.array(data["y"]).astype(int)


def build_model():
    """原著と同じ形のネットワークを組む。

    原著は `Dense(128, input_shape=(2,))` と書いているが、いまの Keras は
    `Input` 層を使うよう警告を出す。層の構成は変えずに `Input` へ寄せた。

    `Dropout` は学習中だけ働き、指定した割合のユニットを無効にする。
    過学習を抑える仕組みで、**推論時は何もしない**。
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input((2,)),
            keras.layers.Dense(HIDDEN_UNITS[0], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(HIDDEN_UNITS[1], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )


def compile_model(model):
    """原著と同じ損失・最適化・指標で組み立てる。"""
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


@dataclass
class TrainedModel:
    """学習済みモデルと、学習中の履歴。"""

    model: object
    #: 各エポックの正解率
    accuracies: list[float]
    #: 各エポックの損失
    losses: list[float]


def fit(data: pd.DataFrame, epochs: int = EPOCHS) -> TrainedModel:
    """原著と同じ手順で学習する。

    ラベルは `to_categorical` で 2 列の one-hot に変える。
    出力層が `softmax` で 2 ユニットなので、ラベルも 2 列にそろえる必要がある。
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TENSORFLOW_SEED)

    model = compile_model(build_model())
    history = model.fit(
        features(data),
        np.array(to_categorical(labels(data), 2)),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return TrainedModel(model, history.history["accuracy"], history.history["loss"])


def predicted_classes(trained: TrainedModel, points: NDArray[np.float64]) -> NDArray[np.int_]:
    """softmax の出力からクラスを決める。2 列のうち大きいほうの添字。"""
    return np.argmax(trained.model.predict(points, verbose=0), axis=1)


def decision_grid(
    trained: TrainedModel, data: pd.DataFrame, step: float = PLOT_STEP
) -> NDArray[np.int_]:
    """原著の `plot_model` と同じ格子を作り、各点の予測クラスを返す。"""
    values = features(data)
    x_values = np.arange(values[:, 0].min() - 1, values[:, 0].max() + 1, step)
    y_values = np.arange(values[:, 1].min() - 1, values[:, 1].max() + 1, step)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    points = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    return predicted_classes(trained, points).reshape(mesh_x.shape)


def boundary_changes_per_row(grid: NDArray[np.int_]) -> list[int]:
    """各行で予測が切り替わった回数を返す。

    決定木なら軸に平行な境界なので、切り替わる位置は行によらず同じだった。
    ニューラルネットワークは曲線を引けるので、行ごとに変わる。
    """
    return [int(np.sum(row[1:] != row[:-1])) for row in grid]
