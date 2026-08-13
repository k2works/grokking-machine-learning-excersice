"""原著ノートブック #15 の再現テスト。

データの規模とパラメータ数（109386）は原著と完全に一致する。
6 万枚を 10 エポック回すと数分かかるので、学習を伴うテストは
**訓練データの一部だけ** を使う。
"""

import numpy as np
import pytest

from grokking_ml_lib.datasets import datasets_dir
from grokking_ml_lib.nb15_image_recognition import (
    CLASSES,
    HIDDEN_UNITS,
    INPUT_DIM,
    build_model,
    evaluate_test_accuracy,
    fit,
    load_mnist,
    predict,
)


def _mnist_available() -> bool:
    return (datasets_dir() / "mnist.npz").exists()


mnist_required = pytest.mark.skipif(
    not _mnist_available(),
    reason="mnist.npz（約 11 MB）が未取得です。datasets の README を参照してください",
)


@pytest.fixture(scope="module")
def mnist():
    return load_mnist()


@pytest.fixture(scope="module")
def trained(mnist):
    # 6 万枚 × 10 エポックは数分かかる。1 万枚 × 5 エポックで済ませる
    return fit(mnist, epochs=5, sample_size=10000)


def test_ネットワークの形は原著と同じ() -> None:
    # Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(10)
    assert INPUT_DIM == 784
    assert HIDDEN_UNITS == (128, 64)
    assert CLASSES == 10


def test_パラメータ数は原著と同じ109386() -> None:
    # 原著の model.summary() が出す Total params: 109,386
    assert build_model().count_params() == 109386
    assert (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10) == 109386


@mnist_required
def test_データセットの規模は原著と同じ(mnist) -> None:
    # 原著の出力
    #   Size of the training set 60000
    #   Size of the testing set 10000
    assert len(mnist.x_train) == 60000
    assert len(mnist.x_test) == 10000


@mnist_required
def test_画像は28かける28の8ビット整数(mnist) -> None:
    assert mnist.x_train.shape == (60000, 28, 28)
    assert mnist.x_train.dtype == np.uint8
    # 画素値は 0 から 255
    assert mnist.x_train.min() == 0
    assert mnist.x_train.max() == 255


@mnist_required
def test_6番目の訓練画像のラベルは2(mnist) -> None:
    # 原著の出力: The label is 2
    assert mnist.y_train[5] == 2


@mnist_required
def test_784次元に直すと件数は変わらない(mnist) -> None:
    # 原著は reshape(-1, 28*28) と書く。-1 は「件数はそのまま」の意味
    assert mnist.reshaped_train().shape == (60000, 784)
    assert mnist.reshaped_test().shape == (10000, 784)


@mnist_required
def test_ラベルは10クラスすべて現れる(mnist) -> None:
    assert sorted(np.unique(mnist.y_train)) == list(range(10))


@mnist_required
def test_5番目のテスト画像のラベルは4(mnist) -> None:
    # 原著の出力: The label is 4 / The prediction is 4
    assert mnist.y_test[4] == 4


@mnist_required
def test_19番目のテスト画像のラベルは3(mnist) -> None:
    # 原著はここで予測を外して 8 と答える例に使っている
    assert mnist.y_test[18] == 3


@mnist_required
def test_学習後の予測は10クラスに収まる(mnist, trained) -> None:
    predictions = predict(trained, mnist.reshaped_test()[:100])

    assert set(predictions) <= set(range(10))


@mnist_required
def test_学習後のテスト正解率は7割を超える(mnist, trained) -> None:
    # 原著は 6 万枚 × 10 エポックで 0.942。
    # ここは 1 万枚 × 5 エポックなので、それより低い 0.77〜0.86 になる。
    # 幅があるのは、TensorFlow の乱数が **プロセス全体で共有** されるため。
    # このファイル単体で走らせると 0.86、#13・#14 のあとだと 0.77 になった。
    # 種を固定しても、先に別のモデルを作ると状態が進む
    accuracy = evaluate_test_accuracy(trained, mnist)

    assert accuracy > 0.7


@mnist_required
def test_でたらめに答えるより大幅に良い(mnist, trained) -> None:
    # 10 クラスなので、でたらめなら 0.1。常に同じ数字を答えても 0.12 程度
    accuracy = evaluate_test_accuracy(trained, mnist)
    most_common_rate = float(np.max(np.bincount(mnist.y_test)) / len(mnist.y_test))

    assert accuracy > most_common_rate * 5
