"""原著ノートブック #13 の再現テスト。

パラメータ数（8770）は原著と完全に一致する。学習結果は Dropout の乱数と
Keras の版差で変わるので、**収束と境界の形** で検証する。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb13_neural_network_boundary import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    HIDDEN_UNITS,
    boundary_changes_per_row,
    build_model,
    decision_grid,
    features,
    fit,
    labels,
    load_circle,
    predicted_classes,
)


@pytest.fixture(scope="module")
def data():
    return load_circle()


@pytest.fixture(scope="module")
def trained(data):
    return fit(data)


def test_データセットは110点(data) -> None:
    assert len(data) == 110
    assert features(data).shape == (110, 2)
    assert set(labels(data)) == {0, 1}


def test_ラベルは偏っている(data) -> None:
    # 84 対 26。円の内側が少数派になる
    counts = np.bincount(labels(data))

    assert counts.tolist() == [84, 26]


def test_ネットワークの形は原著と同じ() -> None:
    # Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(2)
    assert HIDDEN_UNITS == (128, 64)
    assert DROPOUT_RATE == 0.2
    assert EPOCHS == 100
    assert BATCH_SIZE == 10


def test_パラメータ数は原著と同じ8770() -> None:
    # 原著の model.summary() が出す Total params: 8,770
    # (2*128 + 128) + (128*64 + 64) + (64*2 + 2) = 384 + 8256 + 130
    assert build_model().count_params() == 8770
    assert (2 * 128 + 128) + (128 * 64 + 64) + (64 * 2 + 2) == 8770


def test_Dropoutは学習可能なパラメータを持たない() -> None:
    # Dropout はユニットを無効にするだけで、重みを持たない
    model = build_model()
    dropout_layers = [
        layer for layer in model.layers if layer.__class__.__name__ == "Dropout"
    ]

    assert len(dropout_layers) == 2
    assert all(len(layer.weights) == 0 for layer in dropout_layers)


def test_学習は損失を下げる(trained) -> None:
    assert len(trained.losses) == EPOCHS
    assert trained.losses[-1] < trained.losses[0]


def test_学習後の正解率は8割を超える(trained) -> None:
    # 原著は最終的な正解率を出力していないが、図では大半が正しく塗られている
    assert trained.accuracies[-1] > 0.8


def test_学習後は多数派より良い予測をする(data, trained) -> None:
    # 常に 0 と答えるだけで 84/110 = 0.764 になる。それを上回る必要がある
    predictions = predicted_classes(trained, features(data))
    accuracy = float(np.mean(predictions == labels(data)))

    assert accuracy > 84 / 110


def test_出力は2列のsoftmaxになる(data, trained) -> None:
    probabilities = trained.model.predict(features(data)[:5], verbose=0)

    assert probabilities.shape == (5, 2)
    # 各行の合計は 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_境界は行ごとに変わる(data, trained) -> None:
    # 決定木（#10）の境界は軸に平行で、切り替わる位置が行によらなかった。
    # ニューラルネットワークは曲線を引けるので、行ごとに変わる
    grid = decision_grid(trained, data)
    changes = boundary_changes_per_row(grid)

    assert len(set(changes)) > 1


def test_境界は閉じた形になる(data, trained) -> None:
    # 円形のデータなので、内側を囲む境界ができる。
    # 少なくとも 1 行は「外・内・外」と 2 回切り替わる
    grid = decision_grid(trained, data)
    changes = boundary_changes_per_row(grid)

    assert max(changes) >= 2


def test_格子の予測は0か1しかない(data, trained) -> None:
    grid = decision_grid(trained, data)

    assert set(np.unique(grid)) <= {0, 1}
