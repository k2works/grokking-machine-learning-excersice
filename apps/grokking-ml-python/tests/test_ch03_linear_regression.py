import math
import random

import pytest

from grokking_ml.ch03_linear_regression import (
    Model,
    absolute_trick,
    linear_regression,
    model_rmse,
    rmse,
    simple_trick,
    square_trick,
)

FEATURES = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
LABELS = [155.0, 197.0, 244.0, 356.0, 407.0, 448.0]


def test_predict():
    assert Model(50.0, 100.0).predict(3.0) == pytest.approx(250.0)


def test_absolute_trick_moves_toward_the_point():
    model = Model(50.0, 100.0)
    moved = absolute_trick(model, rooms=3.0, price=300.0, learning_rate=0.01)
    assert moved.slope == pytest.approx(50.03)
    assert moved.intercept == pytest.approx(100.01)


def test_absolute_trick_moves_away_when_prediction_is_too_high():
    model = Model(50.0, 100.0)
    moved = absolute_trick(model, rooms=3.0, price=200.0, learning_rate=0.01)
    assert moved.slope == pytest.approx(49.97)
    assert moved.intercept == pytest.approx(99.99)


def test_square_trick_is_proportional_to_the_error():
    model = Model(50.0, 100.0)
    moved = square_trick(model, rooms=3.0, price=300.0, learning_rate=0.01)
    assert moved.slope == pytest.approx(51.5)
    assert moved.intercept == pytest.approx(100.5)


def test_simple_trick_moves_in_the_same_direction_as_the_absolute_trick():
    model = Model(50.0, 100.0)
    moved = simple_trick(model, rooms=3.0, price=300.0, rng=random.Random(0))
    assert moved.slope > model.slope
    assert moved.intercept > model.intercept


def test_rmse():
    assert rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert rmse([1.0, 2.0], [2.0, 4.0]) == pytest.approx(math.sqrt(2.5))


def test_linear_regression_fits_the_housing_dataset():
    model, errors = linear_regression(FEATURES, LABELS, learning_rate=0.01, epochs=1000, seed=0)
    assert model.slope == pytest.approx(50.0, abs=5.0)
    assert model.intercept == pytest.approx(100.0, abs=20.0)
    assert model_rmse(model, FEATURES, LABELS) < 15.0
    assert len(errors) == 1000
    assert errors[-1] < errors[0]
