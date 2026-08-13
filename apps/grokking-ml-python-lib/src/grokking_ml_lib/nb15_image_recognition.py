"""原著ノートブック #15 `Chapter_10_Neural_Networks/Image_recognition.ipynb`。

MNIST の手書き数字 7 万枚を、[#13](nb13.md) と同じ形のネットワークで分類する。
違いは入力が 784 次元（28 × 28 画素）で、出力が 10 クラスになること。

原著はテストセットで **正解率 0.942** を出している。
このシリーズで初めて **訓練とテストを分けて評価する** 回でもある。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from grokking_ml_lib.datasets import dataset_path

#: 原著が使う乱数の種
NUMPY_SEED = 0
TENSORFLOW_SEED = 1

#: 画像の大きさ
IMAGE_SIZE = 28
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE

#: 原著のネットワークの形
HIDDEN_UNITS = (128, 64)
DROPOUT_RATE = 0.2
CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 10


@dataclass
class Mnist:
    """MNIST の訓練セットとテストセット。"""

    x_train: NDArray[np.uint8]
    y_train: NDArray[np.uint8]
    x_test: NDArray[np.uint8]
    y_test: NDArray[np.uint8]

    def reshaped_train(self) -> NDArray[np.uint8]:
        """28 × 28 の画像を 784 次元のベクトルに直す。

        原著は `-1` を「件数はそのまま」の意味で使っている。
        """
        return self.x_train.reshape(-1, INPUT_DIM)

    def reshaped_test(self) -> NDArray[np.uint8]:
        return self.x_test.reshape(-1, INPUT_DIM)


def load_mnist() -> Mnist:
    """MNIST を読み込む。

    原著は `keras.datasets.mnist.load_data()` を呼ぶ。それは
    `mnist.npz` を Google のサーバから取ってくるだけなので、
    ここでは共有データセット経由で同じファイルを読む。
    Kotlin 版・F# 版も同じファイルを使う。
    """
    with np.load(dataset_path("mnist.npz")) as archive:
        return Mnist(
            archive["x_train"],
            archive["y_train"],
            archive["x_test"],
            archive["y_test"],
        )


def build_model():
    """原著と同じ形のネットワークを組む。

    [#13](nb13.md) と層の数も Dropout の割合も同じで、
    入力が 2 から 784 に、出力が 2 から 10 に変わっただけである。
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input((INPUT_DIM,)),
            keras.layers.Dense(HIDDEN_UNITS[0], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(HIDDEN_UNITS[1], activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(CLASSES, activation="softmax"),
        ]
    )


def fit(mnist: Mnist, epochs: int = EPOCHS, sample_size: int | None = None):
    """原著と同じ手順で学習する。

    `sample_size` を渡すと訓練データの先頭だけを使う。
    6 万枚を 10 エポック回すと数分かかるので、テストでは小さくする。
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TENSORFLOW_SEED)

    x = mnist.reshaped_train()
    y = mnist.y_train
    if sample_size is not None:
        x, y = x[:sample_size], y[:sample_size]

    model = build_model()
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x, to_categorical(y, CLASSES), epochs=epochs, batch_size=BATCH_SIZE, verbose=0
    )
    return model


def predict(model, images: NDArray[np.uint8]) -> NDArray[np.int_]:
    """softmax の 10 次元出力から、もっとも高いクラスを選ぶ。

    原著は `[np.argmax(pred) for pred in predictions_vector]` と
    リスト内包表記で書いている。`axis=1` を指定すれば一度で済む。
    """
    return np.argmax(model.predict(images, verbose=0), axis=1)


def evaluate_test_accuracy(model, mnist: Mnist) -> float:
    """テストセットに対する正解率。

    原著は for 文で数えているが、比較して平均を取れば同じ値になる。
    """
    predictions = predict(model, mnist.reshaped_test())
    return float(np.mean(predictions == mnist.y_test))
