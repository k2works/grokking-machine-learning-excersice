"""原著ノートブック #01 の再現テスト。

原著が出力した数値をそのまま期待値に置く。ここが一致していれば、記事に載せる
コードと数値が原著と食い違っていないと言える。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb01_linear_regression import (
    FEATURES,
    LABELS,
    absolute_trick,
    fit_with_scikit_learn,
    linear_regression,
    rmse,
    simple_trick,
    square_trick,
)


def test_データセットは原著と同じ() -> None:
    assert FEATURES.tolist() == [1, 2, 3, 5, 6, 7]
    assert LABELS.tolist() == [155, 197, 244, 356, 407, 448]


def test_二乗トリックは誤差に比例して動く() -> None:
    # 予測 0 + 1 * 2 = 2、実測 10 なので誤差は 8
    price_per_room, base_price = square_trick(0.0, 1.0, 2.0, 10.0, learning_rate=0.01)

    assert price_per_room == pytest.approx(1.0 + 0.01 * 2 * 8)
    assert base_price == pytest.approx(0.0 + 0.01 * 8)


def test_絶対トリックは誤差の大きさに依存しない() -> None:
    # 誤差が 8 でも 800 でも、動く幅は学習率と部屋数だけで決まる
    small = absolute_trick(0.0, 1.0, 2.0, 10.0, learning_rate=0.01)
    large = absolute_trick(0.0, 1.0, 2.0, 802.0, learning_rate=0.01)

    assert small == large


def test_シンプルトリックは予測を実測へ近づける() -> None:
    import random

    random.seed(0)
    # 予測 0 + 1 * 2 = 2 に対し実測 10。部屋数が正なので両方が増える
    price_per_room, base_price = simple_trick(0.0, 1.0, 2.0, 10.0)

    assert price_per_room > 1.0
    assert base_price > 0.0


def test_rmseは誤差の二乗平均平方根() -> None:
    labels = np.array([1.0, 2.0, 3.0])
    predictions = np.array([2.0, 2.0, 2.0])

    # 差は -1, 0, 1 なので二乗和 2、平均 2/3
    assert rmse(labels, predictions) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_二乗トリック1000エポックは原著と同じ数値になる() -> None:
    # 原著 `Coding_linear_regression.ipynb` の出力
    #   Price per room: 51.04430678220095
    #   Base price: 91.59448307644864
    result = linear_regression(FEATURES, LABELS, learning_rate=0.01, epochs=1000, seed=0)

    assert result.line.price_per_room == pytest.approx(51.04430678220095)
    assert result.line.base_price == pytest.approx(91.59448307644864)


def test_学習の途中経過をエポック数だけ記録する() -> None:
    result = linear_regression(FEATURES, LABELS, epochs=50, seed=0)

    assert len(result.history) == 50
    assert len(result.errors) == 50
    # 誤差は単調減少ではないが、最初より最後のほうが小さい
    assert result.errors[-1] < result.errors[0]


def test_エポックを増やすと最小二乗解に近づく() -> None:
    # 原著のノートブックは 10000 エポック版で 50.67 / 99.87 を出しているが、
    # あれは 1000 エポック版と乱数列を共有した状態の値で、単独では再現しない。
    # ここでは「閉じた式の解に近づく」ことを確認する
    exact = fit_with_scikit_learn()
    result = linear_regression(FEATURES, LABELS, learning_rate=0.01, epochs=10000, seed=0)

    assert result.line.price_per_room == pytest.approx(exact.coef_[0], abs=1.0)
    assert result.line.base_price == pytest.approx(float(exact.intercept_), abs=1.0)


def test_scikit_learnの解は原著と同じ数値になる() -> None:
    # 原著の出力
    #   Coefficient: [50.39285714]
    #   Intercept: 99.59523809523819
    model = fit_with_scikit_learn()

    assert model.coef_[0] == pytest.approx(50.39285714, abs=1e-8)
    assert float(model.intercept_) == pytest.approx(99.59523809523819)


def test_scikit_learnの4部屋の予測は原著と同じ数値になる() -> None:
    # 原著の出力
    #   Predicted label for feature 4: [301.16666667]
    model = fit_with_scikit_learn()

    assert model.predict(np.array([[4]]))[0] == pytest.approx(301.16666667, abs=1e-7)


def test_未知のトリック名はエラーになる() -> None:
    with pytest.raises(ValueError, match="未知のトリック"):
        linear_regression(FEATURES, LABELS, trick="unknown")
