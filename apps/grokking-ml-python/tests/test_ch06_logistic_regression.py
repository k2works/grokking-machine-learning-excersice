import math

import pytest

from grokking_ml.ch06_logistic_regression import (
    LogisticClassifier,
    accuracy,
    log_loss,
    logistic_regression,
    logistic_trick,
    mean_log_loss,
    sigmoid,
)

# 第 5 章と同じ「悲しい／楽しい」文の分類データ
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


def test_sigmoid_at_zero_is_a_half():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_is_monotonic():
    assert sigmoid(-1.0) < sigmoid(0.0) < sigmoid(1.0)


def test_sigmoid_stays_inside_the_unit_interval():
    for x in (-1000.0, -10.0, 0.0, 10.0, 1000.0):
        assert 0.0 <= sigmoid(x) <= 1.0


def test_sigmoid_does_not_overflow_for_large_negative_input():
    # 素朴な 1/(1+exp(-x)) 実装は exp のオーバーフローで落ちる
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    assert sigmoid(1000.0) == pytest.approx(1.0)


def test_sigmoid_is_symmetric_around_zero():
    assert sigmoid(2.0) + sigmoid(-2.0) == pytest.approx(1.0)


def test_predict_probability_applies_the_sigmoid_to_the_score():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    # スコア = -4 + 1*1 + 2*2 = 1
    assert model.score((1.0, 2.0)) == pytest.approx(1.0)
    assert model.predict_probability((1.0, 2.0)) == pytest.approx(sigmoid(1.0))


def test_predict_uses_the_threshold():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    assert model.predict((1.0, 2.0)) == 1
    assert model.predict((1.0, 2.0), threshold=0.8) == 0


def test_log_loss_is_small_when_confident_and_correct():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    confident = LogisticClassifier(weights=(10.0, 20.0), bias=-40.0)
    assert log_loss(confident, (1.0, 2.0), label=1) < log_loss(model, (1.0, 2.0), label=1)


def test_log_loss_is_large_when_confident_and_wrong():
    confident = LogisticClassifier(weights=(10.0, 20.0), bias=-40.0)
    assert log_loss(confident, (1.0, 2.0), label=0) > 5.0


def test_log_loss_matches_the_definition():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    probability = sigmoid(1.0)
    assert log_loss(model, (1.0, 2.0), label=1) == pytest.approx(-math.log(probability))
    assert log_loss(model, (1.0, 2.0), label=0) == pytest.approx(-math.log(1.0 - probability))


def test_mean_log_loss_averages_over_points():
    model = LogisticClassifier(weights=(0.0, 0.0), bias=0.0)
    # すべての予測が 0.5 なので、どのラベルでも損失は -log(0.5)
    assert mean_log_loss(model, POINTS, LABELS) == pytest.approx(-math.log(0.5))


def test_logistic_trick_moves_even_correctly_classified_points():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    # 予測は 1（正解）だが、確率は 0.73 なのでまだ動く
    moved = logistic_trick(model, (1.0, 2.0), label=1, learning_rate=0.1)
    assert moved != model
    assert moved.weights[0] > model.weights[0]


def test_logistic_trick_matches_the_gradient():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    error = 1 - sigmoid(1.0)
    moved = logistic_trick(model, (1.0, 2.0), label=1, learning_rate=0.1)
    assert moved.weights[0] == pytest.approx(1.0 + 0.1 * error * 1.0)
    assert moved.weights[1] == pytest.approx(2.0 + 0.1 * error * 2.0)
    assert moved.bias == pytest.approx(-4.0 + 0.1 * error)


def test_logistic_trick_pushes_away_from_a_wrong_positive_prediction():
    model = LogisticClassifier(weights=(1.0, 2.0), bias=-4.0)
    # 予測確率 0.73 に対しラベル 0 なので誤差は負
    moved = logistic_trick(model, (1.0, 2.0), label=0, learning_rate=0.1)
    assert moved.weights[0] < model.weights[0]
    assert moved.bias < model.bias


def test_logistic_regression_separates_the_dataset():
    model, losses = logistic_regression(POINTS, LABELS, learning_rate=0.1, epochs=1000, seed=0)
    assert accuracy(model, POINTS, LABELS) == pytest.approx(1.0)
    assert len(losses) == 1000
    # パーセプトロン誤差と違い、対数損失は初期状態でも 0 にならない
    assert losses[0] == pytest.approx(-math.log(0.5))
    assert losses[-1] < losses[0]


def test_logistic_regression_produces_calibrated_probabilities():
    model, _ = logistic_regression(POINTS, LABELS, learning_rate=0.1, epochs=1000, seed=0)
    # 明確に楽しい文は高い確率、明確に悲しい文は低い確率になる
    assert model.predict_probability((3.0, 2.0)) > 0.5
    assert model.predict_probability((1.0, 0.0)) < 0.5
