"""原著ノートブック #03 の再現テスト。

L1・L2 正則化つきの結果は原著と完全に一致する。正則化なしの結果だけは一致しない。
次数 20 の多項式特徴量は条件数が 9 億に達し、解が数値的に不安定になるためである。
**正則化は過学習を抑えるだけでなく、解を再現可能にもしている。**
"""

import numpy as np
import pytest

from grokking_ml_lib.nb03_polynomial_regularization import (
    DEGREE,
    POLYNOMIAL_COEFFICIENTS,
    Regularization,
    evaluate_model,
    generate_dataset,
    polynomial,
    polynomial_features,
    train_polynomial_regression,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


def test_多項式は係数の添字が次数に対応する() -> None:
    # -x^2 + 2 なので x = 0 で 2、x = 1 で 1、x = 2 で -2
    assert polynomial(POLYNOMIAL_COEFFICIENTS, 0) == 2
    assert polynomial(POLYNOMIAL_COEFFICIENTS, 1) == 1
    assert polynomial(POLYNOMIAL_COEFFICIENTS, 2) == -2


def test_データセットは40点で訓練32テスト8に分かれる(dataset) -> None:
    # 原著の出力
    #   Shape of X_train: (32,)  Shape of X_test: (8,)
    assert len(dataset.x) == 40
    assert len(dataset.x_train) == 32
    assert len(dataset.x_test) == 8
    assert len(dataset.y_train) == 32
    assert len(dataset.y_test) == 8


def test_生成される点は原著と同じ乱数列になる(dataset) -> None:
    # random.seed(0) のあと uniform → gauss の順で消費する
    assert dataset.x[:4] == pytest.approx([0.688844, -0.482166, 0.022549, -0.393375], abs=1e-6)


def test_多項式特徴量は次数の数だけ列を作る(dataset) -> None:
    features = polynomial_features(dataset.x_train, DEGREE)

    # include_bias=False なので定数列は入らない
    assert features.shape == (32, DEGREE)
    # 1 列目は x そのもの、2 列目は x の 2 乗
    assert features[0, 0] == pytest.approx(dataset.x_train[0])
    assert features[0, 1] == pytest.approx(dataset.x_train[0] ** 2)


def test_L1正則化のテスト誤差は原著と同じ数値になる(dataset) -> None:
    # 原著の出力: Square loss on the test set (degree 20): 0.15277291798691608
    model = train_polynomial_regression(
        dataset.x_train, dataset.y_train, DEGREE, Regularization.L1, alpha=0.01
    )

    assert evaluate_model(model, dataset.x_test, dataset.y_test) == pytest.approx(
        0.15277291798691608
    )


def test_L2正則化のテスト誤差は原著と同じ数値になる(dataset) -> None:
    # 原著の出力: Square loss on the test set (degree 20): 0.10370797950325954
    model = train_polynomial_regression(
        dataset.x_train, dataset.y_train, DEGREE, Regularization.L2, alpha=0.01
    )

    assert evaluate_model(model, dataset.x_test, dataset.y_test) == pytest.approx(
        0.10370797950325954
    )


def test_L1正則化の係数は原著と同じくほとんどゼロになる(dataset) -> None:
    # 原著の出力
    #   1.9373689727450651
    #   [0. -0.83079676 0. -0. ... ]
    model = train_polynomial_regression(
        dataset.x_train, dataset.y_train, DEGREE, Regularization.L1, alpha=0.01
    )

    assert float(model.intercept_) == pytest.approx(1.9373689727450651)
    assert np.count_nonzero(model.coef_) == 1
    # 残った 1 つは x^2 の係数。元の多項式 -x^2 + 2 の -1 に向かっている
    assert model.coef_[1] == pytest.approx(-0.83079676, abs=1e-8)


def test_L2正則化は係数をゼロにはしないが小さく保つ(dataset) -> None:
    model = train_polynomial_regression(
        dataset.x_train, dataset.y_train, DEGREE, Regularization.L2, alpha=0.01
    )

    # L1 と違いスパースにはならない
    assert np.count_nonzero(model.coef_) == DEGREE
    # それでも正則化なしの数千倍という係数にはならない。実測で最大 0.82
    assert np.abs(model.coef_).max() < 1.0


def test_正則化なしは係数が爆発する(dataset) -> None:
    model = train_polynomial_regression(dataset.x_train, dataset.y_train, DEGREE)

    # 元の多項式の係数は -1 と 2 だけなのに、実測で最大 8633 まで膨らむ。
    # 原著の当時の scikit-learn では 1e7 に達していた（ソルバの実装差）
    assert np.abs(model.coef_).max() > 1_000


def test_正則化なしの多項式特徴量は条件数が悪い(dataset) -> None:
    # 原著の「正則化なし 1862.044」が再現しない理由。行列はフルランクだが
    # 条件数が 9 億に達し、ソルバの実装差がそのまま結果に出る
    features = polynomial_features(dataset.x_train, DEGREE)
    design = np.c_[np.ones(len(features)), features]
    singular_values = np.linalg.svd(design, compute_uv=False)

    assert np.linalg.matrix_rank(design) == design.shape[1]
    assert singular_values[0] / singular_values[-1] > 1e8


def test_正則化はテスト誤差を大きく下げる(dataset) -> None:
    # 原著の出力では 1862.044 -> 0.1528 / 0.1037。scikit-learn のソルバが
    # 変わったため正則化なしの値は再現しないが、桁違いに悪いことは変わらない
    no_reg = evaluate_model(
        train_polynomial_regression(dataset.x_train, dataset.y_train, DEGREE),
        dataset.x_test,
        dataset.y_test,
    )
    l1 = evaluate_model(
        train_polynomial_regression(
            dataset.x_train, dataset.y_train, DEGREE, Regularization.L1, alpha=0.01
        ),
        dataset.x_test,
        dataset.y_test,
    )
    l2 = evaluate_model(
        train_polynomial_regression(
            dataset.x_train, dataset.y_train, DEGREE, Regularization.L2, alpha=0.01
        ),
        dataset.x_test,
        dataset.y_test,
    )

    assert no_reg > 50 * l1
    assert l2 < l1
