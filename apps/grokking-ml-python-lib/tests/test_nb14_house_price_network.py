"""原著ノートブック #14 の再現テスト。

パラメータ数（14795）は原著と完全に一致する。学習結果は Dropout の乱数と
Keras の版差で変わるので、**「平均を答えるだけ」の基準を超えるか** で検証する。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb14_house_price_network import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_UNITS,
    baseline_rmse,
    build_model,
    evaluate_rmse,
    features,
    fit,
    labels,
    load_housing,
    predict,
)


@pytest.fixture(scope="module")
def housing():
    return load_housing()


@pytest.fixture(scope="module")
def trained(housing):
    return fit(housing)


def test_データセットは2518件40列(housing) -> None:
    assert housing.shape == (2518, 40)


def test_特徴量は38列になる(housing) -> None:
    # Location（文字列）と Price（目的変数）を落とす。
    # #02 は Location を one-hot にして 277 列にしたが、ここでは捨てている
    assert features(housing).shape == (2518, 38)
    assert "Location" not in features(housing).columns
    assert "Price" not in features(housing).columns


def test_ネットワークの形は原著と同じ() -> None:
    # Dense(38) -> Dense(128) -> Dense(64) -> Dense(1)
    assert HIDDEN_UNITS == (38, 128, 64)
    assert EPOCHS == 10
    assert BATCH_SIZE == 10


def test_パラメータ数は原著と同じ14795() -> None:
    # 原著の model.summary() が出す Total params: 14,795
    assert build_model().count_params() == 14795
    assert (38 * 38 + 38) + (38 * 128 + 128) + (128 * 64 + 64) + (64 * 1 + 1) == 14795


def test_出力層は活性化関数を持たない() -> None:
    # 回帰なので値をそのまま出す。分類（#13）は softmax を付けていた
    output_layer = build_model().layers[-1]

    assert output_layer.units == 1
    assert output_layer.activation.__name__ == "linear"


def test_平均を答えるだけの基準は約877万(housing) -> None:
    # 価格の標準偏差。ネットワークはこれを下回る必要がある
    assert baseline_rmse(housing) == pytest.approx(8_775_370, rel=1e-3)


def test_学習はRMSEを下げる(trained) -> None:
    assert len(trained.rmses) == EPOCHS
    assert trained.rmses[-1] < trained.rmses[0]


def test_学習後のRMSEは平均を答えるより良い(housing, trained) -> None:
    # 常に平均価格を答えると約 877 万円外す。それより良くなければ意味がない
    assert evaluate_rmse(trained, housing) < baseline_rmse(housing)


def test_予測は件数分の2次元配列になる(housing, trained) -> None:
    # Dense(1) の出力なので (2518, 1) になる
    predictions = predict(trained, housing)

    assert predictions.shape == (2518, 1)


def test_予測はおおむね正の値になる(housing, trained) -> None:
    # 価格なので負にはならないはず。ただし出力層に制約は無いので保証はされない
    predictions = predict(trained, housing).ravel()

    assert float(np.mean(predictions > 0)) > 0.95


def test_予測の平均は実際の平均に近い(housing, trained) -> None:
    # 回帰なので、全体の水準は合ってくる
    predictions = predict(trained, housing).ravel()
    actual = labels(housing).to_numpy(dtype=float)

    assert float(np.mean(predictions)) == pytest.approx(float(np.mean(actual)), rel=0.5)


def test_特徴量を標準化していない(housing) -> None:
    # 原著は前処理をしていない。Area は 4 桁、Resale は 0 か 1 と桁が違う。
    # #02 の線形回帰では標準化していたのに、ここではしていない
    numeric = features(housing)

    assert numeric["Area"].max() > 1000
    assert set(numeric["Resale"].unique()) <= {0, 1}
