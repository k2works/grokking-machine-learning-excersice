import pytest

from grokking_ml.ch05_perceptron import (
    Perceptron,
    accuracy,
    mean_perceptron_error,
    perceptron_algorithm,
    perceptron_error,
    perceptron_trick,
)

# 原著と同じ「悲しい／楽しい」文の分類データ
# 特徴量 = (aack の出現数, beep の出現数)、ラベル = 1 が楽しい
POINTS = [
    (1.0, 0.0),
    (0.0, 2.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3.0),
    (2.0, 2.0),
    (2.0, 3.0),
    (3.0, 2.0),
]
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


def test_score_is_the_linear_combination():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    # -4 + 1*1 + 2*2 = 1
    assert model.score((1.0, 2.0)) == pytest.approx(1.0)


def test_predict_uses_the_sign_of_the_score():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    assert model.predict((1.0, 2.0)) == 1
    assert model.predict((1.0, 1.0)) == 0


def test_predict_treats_zero_as_positive():
    model = Perceptron(weights=(1.0, 1.0), bias=-2.0)
    assert model.score((1.0, 1.0)) == pytest.approx(0.0)
    assert model.predict((1.0, 1.0)) == 1


def test_perceptron_trick_leaves_correct_points_untouched():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    # 予測 1、ラベル 1 なので動かない
    assert perceptron_trick(model, (1.0, 2.0), label=1) == model


def test_perceptron_trick_moves_toward_a_misclassified_positive_point():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    # 予測 0、ラベル 1 なので誤差 +1
    moved = perceptron_trick(model, (1.0, 1.0), label=1, learning_rate=0.1)
    assert moved.weights[0] == pytest.approx(1.1)  # 1 + 0.1 * 1 * 1
    assert moved.weights[1] == pytest.approx(2.1)  # 2 + 0.1 * 1 * 1
    assert moved.bias == pytest.approx(-3.9)  # -4 + 0.1 * 1


def test_perceptron_trick_moves_away_from_a_misclassified_negative_point():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    # 予測 1、ラベル 0 なので誤差 -1
    moved = perceptron_trick(model, (1.0, 2.0), label=0, learning_rate=0.1)
    assert moved.weights[0] == pytest.approx(0.9)  # 1 - 0.1 * 1
    assert moved.weights[1] == pytest.approx(1.8)  # 2 - 0.1 * 2
    assert moved.bias == pytest.approx(-4.1)  # -4 - 0.1


def test_perceptron_error_is_zero_when_correct():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    assert perceptron_error(model, (1.0, 2.0), label=1) == pytest.approx(0.0)


def test_perceptron_error_is_the_distance_when_wrong():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    # スコア -1 で予測 0、ラベル 1 なので誤差は |−1| = 1
    assert perceptron_error(model, (1.0, 1.0), label=1) == pytest.approx(1.0)


def test_mean_perceptron_error_averages_over_points():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    points = [(1.0, 2.0), (1.0, 1.0)]
    labels = [1, 1]
    assert mean_perceptron_error(model, points, labels) == pytest.approx(0.5)


def test_accuracy_counts_correct_predictions():
    model = Perceptron(weights=(1.0, 2.0), bias=-4.0)
    assert accuracy(model, [(1.0, 2.0), (1.0, 1.0)], [1, 1]) == pytest.approx(0.5)


def test_perceptron_algorithm_separates_the_dataset():
    model, errors = perceptron_algorithm(POINTS, LABELS, learning_rate=0.01, epochs=1000, seed=0)
    assert accuracy(model, POINTS, LABELS) == pytest.approx(1.0)
    assert len(errors) == 1000
    # 初期モデル（重みもバイアスもすべて 0）はすべての点が境界線上にあるため誤差 0
    assert errors[0] == pytest.approx(0.0)
    # 学習の途中では誤差が生じ、最終的に 0 へ戻る
    assert max(errors) > 0.0
    assert errors[-1] == pytest.approx(0.0)


def test_perceptron_algorithm_reaches_zero_error_on_separable_data():
    model, _ = perceptron_algorithm(POINTS, LABELS, learning_rate=0.01, epochs=1000, seed=0)
    assert mean_perceptron_error(model, POINTS, LABELS) == pytest.approx(0.0)
