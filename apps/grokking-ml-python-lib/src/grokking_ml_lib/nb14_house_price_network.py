"""原著ノートブック #14 `Chapter_10_Neural_Networks/House_price_predictions_neural_network.ipynb`。

ニューラルネットワークを **回帰** に使う回。[#02](nb02.md) と同じ
ハイデラバードの住宅データを、今度は 3 層のネットワークで予測する。

分類（[#13](nb13.md)）との違いは 2 つだけ。
- 出力層に活性化関数を付けない（そのままの値を出す）
- 損失を平均二乗誤差にする

**特徴量を標準化していない** のが原著の特徴で、そこが結果に効く。
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

#: 原著のネットワークの形。入力 38 に対して 38 → 128 → 64 → 1
HIDDEN_UNITS = (38, 128, 64)
DROPOUT_RATE = 0.2
EPOCHS = 10
BATCH_SIZE = 10


def load_housing() -> pd.DataFrame:
    """ハイデラバードの住宅データを読み込む。2518 行 × 40 列。"""
    return pd.read_csv(dataset_path("Hyderabad.csv"))


def features(housing: pd.DataFrame) -> pd.DataFrame:
    """`Location`（文字列）と `Price`（目的変数）を落とした 38 列。

    [#02](nb02.md) は `Location` を one-hot 符号化して 277 列にしたが、
    ここでは **捨てている**。ニューラルネットワークは特徴量が増えると
    パラメータも増えるので、原著は簡単なほうを選んだと思われる。
    """
    return housing.drop(["Location", "Price"], axis=1)


def labels(housing: pd.DataFrame) -> pd.Series:
    return housing["Price"]


def build_model(input_dim: int = 38):
    """原著と同じ形のネットワークを組む。

    出力層 `Dense(1)` に活性化関数を指定していないのが回帰の書き方。
    指定しないと恒等関数になり、**値をそのまま出す**。
    分類（[#13](nb13.md)）では `softmax` を付けて確率にしていた。
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input((input_dim,)),
            keras.layers.Dense(HIDDEN_UNITS[0], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(HIDDEN_UNITS[1], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(HIDDEN_UNITS[2], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(1),
        ]
    )


def compile_model(model):
    """損失を平均二乗誤差にし、指標に RMSE を足す。

    原著のコメントどおり、`metrics` に RMSE を入れると
    エポックごとに「平均どれくらい外したか」が円単位で読める。
    損失（MSE）は 2 乗なので桁が大きすぎて読めない。
    """
    from tensorflow import keras

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    return model


@dataclass
class TrainedModel:
    """学習済みモデルと、学習中の履歴。"""

    model: object
    #: 各エポックの RMSE
    rmses: list[float]
    losses: list[float]


def fit(housing: pd.DataFrame, epochs: int = EPOCHS) -> TrainedModel:
    """原著と同じ手順で学習する。"""
    import tensorflow as tf

    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TENSORFLOW_SEED)

    model = compile_model(build_model())
    history = model.fit(
        features(housing),
        labels(housing),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return TrainedModel(
        model, history.history["root_mean_squared_error"], history.history["loss"]
    )


def evaluate_rmse(trained: TrainedModel, housing: pd.DataFrame) -> float:
    """学習データに対する RMSE。

    `Dropout` は推論時に働かないので、`evaluate` の値は
    学習中の履歴より良くなることが多い。
    """
    return float(trained.model.evaluate(features(housing), labels(housing), verbose=0)[1])


def predict(trained: TrainedModel, housing: pd.DataFrame) -> NDArray[np.float64]:
    """全物件の価格を予測する。出力は (件数, 1) の 2 次元配列。"""
    return trained.model.predict(features(housing), verbose=0)


def baseline_rmse(housing: pd.DataFrame) -> float:
    """常に平均価格を答えたときの RMSE。

    これを下回れなければ、ネットワークは何も学習できていない。
    値は価格の標準偏差（母分散のほう）と一致する。
    """
    prices = labels(housing).to_numpy(dtype=float)
    return float(np.sqrt(np.mean((prices - prices.mean()) ** 2)))
