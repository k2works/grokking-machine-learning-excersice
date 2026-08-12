import pytest

from grokking_ml.ch04_regularization import (
    PolynomialModel,
    Regularization,
    model_rmse,
    polynomial_features,
    polynomial_regression,
    regularization_gradient,
    square_trick,
    train_test_split,
    weight_magnitude,
)

# y = 2x + 3 に小さなノイズを乗せたデータ
FEATURES = [-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2]
LABELS = [0.08, 0.32, 1.07, 1.63, 2.54, 3.11, 3.84, 3.95, 4.75, 5.12]


def test_polynomial_features():
    assert polynomial_features(2.0, 3) == [2.0, 4.0, 8.0]


def test_predict_uses_every_power():
    model = PolynomialModel(weights=(2.0, 3.0), bias=1.0)
    # 1 + 2*2 + 3*4 = 17
    assert model.predict(2.0) == pytest.approx(17.0)


def test_regularization_gradient_l1_uses_the_sign():
    assert regularization_gradient(5.0, Regularization.L1, 0.1) == pytest.approx(0.1)
    assert regularization_gradient(-5.0, Regularization.L1, 0.1) == pytest.approx(-0.1)
    assert regularization_gradient(0.0, Regularization.L1, 0.1) == pytest.approx(0.0)


def test_regularization_gradient_l2_is_proportional_to_the_weight():
    assert regularization_gradient(5.0, Regularization.L2, 0.1) == pytest.approx(1.0)
    assert regularization_gradient(-5.0, Regularization.L2, 0.1) == pytest.approx(-1.0)


def test_regularization_gradient_none_is_zero():
    assert regularization_gradient(5.0, Regularization.NONE, 0.1) == pytest.approx(0.0)


def test_square_trick_without_regularization():
    model = PolynomialModel(weights=(2.0,), bias=1.0)
    # 予測 = 1 + 2*3 = 7、誤差 = 10 - 7 = 3
    moved = square_trick(model, x=3.0, y=10.0, learning_rate=0.01)
    assert moved.weights[0] == pytest.approx(2.09)  # 2 + 0.01 * 3 * 3
    assert moved.bias == pytest.approx(1.03)  # 1 + 0.01 * 3


def test_square_trick_with_l2_pulls_the_weight_toward_zero():
    model = PolynomialModel(weights=(2.0,), bias=1.0)
    plain = square_trick(model, x=3.0, y=10.0, learning_rate=0.01)
    regularized = square_trick(
        model, x=3.0, y=10.0, learning_rate=0.01, kind=Regularization.L2, strength=0.1
    )
    assert regularized.weights[0] < plain.weights[0]
    # 2 + 0.01 * (3*3 - 0.1*2*2)
    assert regularized.weights[0] == pytest.approx(2.086)


def test_square_trick_does_not_regularize_the_bias():
    model = PolynomialModel(weights=(2.0,), bias=1.0)
    plain = square_trick(model, x=3.0, y=10.0, learning_rate=0.01)
    regularized = square_trick(
        model, x=3.0, y=10.0, learning_rate=0.01, kind=Regularization.L2, strength=0.1
    )
    assert regularized.bias == pytest.approx(plain.bias)


def test_train_test_split_partitions_without_loss():
    train_x, train_y, test_x, test_y = train_test_split(FEATURES, LABELS, test_ratio=0.3, seed=0)
    assert len(test_x) == 3
    assert len(train_x) == 7
    assert sorted(train_x + test_x) == sorted(FEATURES)
    assert sorted(train_y + test_y) == sorted(LABELS)


def test_train_test_split_keeps_pairs_together():
    train_x, train_y, test_x, test_y = train_test_split(FEATURES, LABELS, test_ratio=0.3, seed=0)
    for x, y in zip(train_x + test_x, train_y + test_y):
        assert LABELS[FEATURES.index(x)] == pytest.approx(y)


def test_degree_1_model_fits_a_linear_dataset():
    model = polynomial_regression(FEATURES, LABELS, degree=1, learning_rate=0.01, epochs=20000)
    assert model.weights[0] == pytest.approx(2.0, abs=0.3)
    assert model.bias == pytest.approx(3.0, abs=0.3)
    assert model_rmse(model, FEATURES, LABELS) < 0.3


def test_high_degree_model_overfits():
    train_x, train_y, test_x, test_y = train_test_split(FEATURES, LABELS, test_ratio=0.3, seed=0)
    simple = polynomial_regression(train_x, train_y, degree=1, learning_rate=0.01, epochs=20000)
    complex_ = polynomial_regression(train_x, train_y, degree=5, learning_rate=0.01, epochs=20000)

    # 次数を上げると訓練誤差は下がる
    assert model_rmse(complex_, train_x, train_y) < model_rmse(simple, train_x, train_y)
    # しかし汎化ギャップ（テスト誤差 - 訓練誤差）は広がる
    simple_gap = model_rmse(simple, test_x, test_y) - model_rmse(simple, train_x, train_y)
    complex_gap = model_rmse(complex_, test_x, test_y) - model_rmse(complex_, train_x, train_y)
    assert complex_gap > simple_gap


def test_l2_regularization_improves_the_test_error():
    train_x, train_y, test_x, test_y = train_test_split(FEATURES, LABELS, test_ratio=0.3, seed=0)
    plain = polynomial_regression(train_x, train_y, degree=5, learning_rate=0.01, epochs=20000)
    regularized = polynomial_regression(
        train_x,
        train_y,
        degree=5,
        learning_rate=0.01,
        epochs=20000,
        kind=Regularization.L2,
        strength=0.01,
    )
    assert model_rmse(regularized, test_x, test_y) < model_rmse(plain, test_x, test_y)


def test_l1_regularization_drives_weights_to_zero():
    train_x, train_y, _, _ = train_test_split(FEATURES, LABELS, test_ratio=0.3, seed=0)
    plain = polynomial_regression(train_x, train_y, degree=5, learning_rate=0.01, epochs=20000)
    regularized = polynomial_regression(
        train_x,
        train_y,
        degree=5,
        learning_rate=0.01,
        epochs=20000,
        kind=Regularization.L1,
        strength=0.01,
    )
    assert weight_magnitude(regularized) < weight_magnitude(plain)
    # L1 は不要な次数の重みをほぼ 0 まで押し下げる
    assert sum(1 for w in regularized.weights if abs(w) < 5e-3) >= 2
    assert sum(1 for w in plain.weights if abs(w) < 5e-3) == 0
