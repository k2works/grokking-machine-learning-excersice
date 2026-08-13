import pytest

from grokking_ml.ch06_logistic_regression import accuracy as logistic_accuracy
from grokking_ml.ch06_logistic_regression import logistic_regression
from grokking_ml.ch10_neural_networks import (
    Layer,
    NeuralNetwork,
    accuracy,
    backpropagate,
    initial_network,
    log_loss,
    mean_log_loss,
    sigmoid,
    sigmoid_derivative,
    train,
)

# XOR。直線では分けられない代表例
XOR_POINTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_LABELS = [0, 1, 1, 0]


def test_sigmoid_derivative_is_computed_from_the_output():
    # s'(x) = s(x) * (1 - s(x))。出力が 0.5 のとき最大の 0.25
    assert sigmoid_derivative(0.5) == pytest.approx(0.25)
    assert sigmoid_derivative(sigmoid(0.0)) == pytest.approx(0.25)


def test_sigmoid_derivative_vanishes_at_the_extremes():
    # 出力が 0 や 1 に近いと勾配がほぼ消える（勾配消失）
    assert sigmoid_derivative(0.999) < 0.002
    assert sigmoid_derivative(0.001) < 0.002


def test_layer_forward_applies_weights_and_sigmoid():
    layer = Layer(weights=[[1.0, 2.0]], biases=[-4.0])
    # -4 + 1*1 + 2*2 = 1
    assert layer.forward([1.0, 2.0]) == [pytest.approx(sigmoid(1.0))]


def test_layer_reports_its_shape():
    layer = Layer(weights=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], biases=[0.0, 0.0])
    assert layer.input_size == 3
    assert layer.output_size == 2


def test_forward_all_records_every_layer():
    model = initial_network([2, 3, 1])
    activations = model.forward_all([0.5, 0.5])
    assert len(activations) == 3  # 入力 + 隠れ層 + 出力層
    assert len(activations[0]) == 2
    assert len(activations[1]) == 3
    assert len(activations[2]) == 1


def test_initial_network_has_the_requested_shape():
    model = initial_network([2, 4, 1])
    assert len(model.layers) == 2
    assert model.layers[0].input_size == 2
    assert model.layers[0].output_size == 4
    assert model.layers[1].input_size == 4
    assert model.layers[1].output_size == 1


def test_initial_network_is_reproducible():
    assert initial_network([2, 3, 1], seed=7) == initial_network([2, 3, 1], seed=7)
    assert initial_network([2, 3, 1], seed=7) != initial_network([2, 3, 1], seed=8)


def test_predict_probability_stays_inside_the_unit_interval():
    model = initial_network([2, 4, 1])
    for point in XOR_POINTS:
        assert 0.0 <= model.predict_probability(point) <= 1.0


def test_backpropagate_reduces_the_loss_for_that_point():
    model = initial_network([2, 4, 1], seed=1)
    point, label = (1.0, 0.0), 1
    before = log_loss(model, point, label)
    after = log_loss(backpropagate(model, point, label, learning_rate=0.5), point, label)
    assert after < before


def test_backpropagate_keeps_the_shape():
    model = initial_network([2, 4, 1])
    updated = backpropagate(model, (1.0, 0.0), 1, learning_rate=0.5)
    assert len(updated.layers) == len(model.layers)
    for original, layer in zip(model.layers, updated.layers):
        assert layer.input_size == original.input_size
        assert layer.output_size == original.output_size


def test_backpropagate_returns_a_new_network():
    model = initial_network([2, 4, 1])
    updated = backpropagate(model, (1.0, 0.0), 1, learning_rate=0.5)
    assert updated != model


def test_a_single_layer_model_cannot_solve_xor():
    # 第 6 章のロジスティック回帰（隠れ層なし）は XOR を解けない
    model, _ = logistic_regression(XOR_POINTS, XOR_LABELS, learning_rate=0.5, epochs=20000, seed=0)
    assert logistic_accuracy(model, XOR_POINTS, XOR_LABELS) < 1.0


def test_a_network_with_one_hidden_neuron_cannot_solve_xor_either():
    # 隠れ層があっても、ニューロンが 1 つでは足りない
    model, _ = train(XOR_POINTS, XOR_LABELS, hidden_size=1, epochs=20000, seed=0)
    assert accuracy(model, XOR_POINTS, XOR_LABELS) < 1.0


def test_a_hidden_layer_solves_xor():
    model, losses = train(XOR_POINTS, XOR_LABELS, hidden_size=4, epochs=20000, seed=0)
    assert accuracy(model, XOR_POINTS, XOR_LABELS) == pytest.approx(1.0)
    assert losses[-1] < losses[0]


def test_training_produces_confident_predictions():
    model, _ = train(XOR_POINTS, XOR_LABELS, hidden_size=4, epochs=20000, seed=0)
    assert model.predict_probability((0.0, 0.0)) < 0.1
    assert model.predict_probability((0.0, 1.0)) > 0.9
    assert model.predict_probability((1.0, 0.0)) > 0.9
    assert model.predict_probability((1.0, 1.0)) < 0.1


def test_mean_log_loss_starts_near_the_uninformed_value():
    model = initial_network([2, 4, 1], seed=0)
    # 学習前は五分五分に近いので -log(0.5) = 0.693 あたりから始まる
    assert 0.5 < mean_log_loss(model, XOR_POINTS, XOR_LABELS) < 1.2


def test_network_is_immutable_across_training():
    original = initial_network([2, 4, 1], seed=3)
    snapshot = NeuralNetwork(layers=list(original.layers))
    backpropagate(original, (1.0, 0.0), 1, learning_rate=0.5)
    assert original == snapshot
