import pytest

from grokking_ml.ch05_perceptron import perceptron_algorithm
from grokking_ml.ch11_svm import (
    KernelClassifier,
    SupportVectorMachine,
    accuracy,
    hinge_loss,
    kernel_accuracy,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    svm_error,
    svm_step,
    train_kernel_classifier,
    train_svm,
)

# 第 5 章と同じデータ。ただし SVM ではラベルを +1 / -1 で表す
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
LABELS = [-1, -1, -1, -1, 1, 1, 1, 1]
PERCEPTRON_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]

# XOR。線形カーネルでは分けられない
XOR_POINTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_LABELS = [-1, 1, 1, -1]


def test_linear_kernel_is_the_dot_product():
    assert linear_kernel((1.0, 2.0), (3.0, 4.0)) == pytest.approx(11.0)


def test_polynomial_kernel_matches_the_definition():
    kernel = polynomial_kernel(degree=2, constant=1.0)
    # (1*3 + 2*4 + 1)^2 = 144
    assert kernel((1.0, 2.0), (3.0, 4.0)) == pytest.approx(144.0)


def test_rbf_kernel_is_one_at_the_same_point():
    kernel = rbf_kernel(gamma=1.0)
    assert kernel((1.0, 2.0), (1.0, 2.0)) == pytest.approx(1.0)


def test_rbf_kernel_decays_with_distance():
    kernel = rbf_kernel(gamma=1.0)
    near = kernel((0.0, 0.0), (0.5, 0.0))
    far = kernel((0.0, 0.0), (3.0, 0.0))
    assert 0.0 < far < near < 1.0


def test_predict_uses_plus_and_minus_one():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    assert model.predict((2.0, 2.0)) == 1
    assert model.predict((1.0, 1.0)) == -1


def test_hinge_loss_is_zero_outside_the_margin():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    # スコア 2、ラベル +1 なので余裕をもって正解
    assert hinge_loss(model, (3.0, 2.0), 1) == pytest.approx(0.0)


def test_hinge_loss_punishes_points_inside_the_margin():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    # スコア 0.5、ラベル +1。正解だがマージンの内側なので 1 - 0.5 = 0.5
    assert hinge_loss(model, (1.5, 2.0), 1) == pytest.approx(0.5)


def test_hinge_loss_grows_with_the_error():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    # スコア -1、ラベル +1。1 - (-1) = 2
    assert hinge_loss(model, (1.0, 1.0), 1) == pytest.approx(2.0)


def test_svm_step_leaves_confident_points_almost_alone():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    moved = svm_step(model, (3.0, 2.0), 1, learning_rate=0.01, regularization=0.1)
    # マージンの外なので、重みを縮める力しか働かない
    assert moved.bias == pytest.approx(model.bias)
    assert moved.weights[0] < model.weights[0]


def test_svm_step_pushes_back_points_inside_the_margin():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    moved = svm_step(model, (1.5, 2.0), 1, learning_rate=0.01, regularization=0.0)
    # 正則化なしなら、押し返す力だけが働く
    assert moved.weights[0] > model.weights[0]
    assert moved.bias > model.bias


def test_svm_separates_the_dataset():
    model, errors = train_svm(POINTS, LABELS, epochs=20000)
    assert accuracy(model, POINTS, LABELS) == pytest.approx(1.0)
    assert errors[-1] < errors[0]


def test_svm_leaves_a_wider_margin_than_the_perceptron():
    perceptron, _ = perceptron_algorithm(POINTS, PERCEPTRON_LABELS)
    as_svm = SupportVectorMachine(tuple(perceptron.weights), perceptron.bias)
    svm, _ = train_svm(POINTS, LABELS, epochs=20000, regularization=0.01)

    # どちらも完全に分離できている
    assert accuracy(as_svm, POINTS, LABELS) == pytest.approx(1.0)
    assert accuracy(svm, POINTS, LABELS) == pytest.approx(1.0)
    # しかしパーセプトロンは分離できた時点で止まるため、余白がない
    assert as_svm.margin(POINTS, LABELS) == pytest.approx(0.0)
    assert svm.margin(POINTS, LABELS) > 0.5


def test_weaker_regularization_widens_the_margin():
    loose, _ = train_svm(POINTS, LABELS, epochs=20000, regularization=0.01)
    tight, _ = train_svm(POINTS, LABELS, epochs=20000, regularization=0.1)
    assert loose.margin(POINTS, LABELS) > tight.margin(POINTS, LABELS)


def test_svm_error_combines_loss_and_penalty():
    model = SupportVectorMachine(weights=(1.0, 1.0), bias=-3.0)
    # 正則化 0 なら目的関数はヒンジ損失の平均そのもの
    losses = [hinge_loss(model, point, label) for point, label in zip(POINTS, LABELS)]
    assert svm_error(model, POINTS, LABELS, regularization=0.0) == pytest.approx(
        sum(losses) / len(losses)
    )
    # 正則化を足すと重みの二乗和の分だけ増える
    assert svm_error(model, POINTS, LABELS, regularization=0.1) == pytest.approx(
        sum(losses) / len(losses) + 0.2
    )


def test_kernel_classifier_with_a_linear_kernel_cannot_solve_xor():
    model = train_kernel_classifier(XOR_POINTS, XOR_LABELS, kernel=linear_kernel)
    assert kernel_accuracy(model, XOR_POINTS, XOR_LABELS) < 1.0


def test_polynomial_kernel_solves_xor():
    model = train_kernel_classifier(XOR_POINTS, XOR_LABELS, kernel=polynomial_kernel(degree=2))
    assert kernel_accuracy(model, XOR_POINTS, XOR_LABELS) == pytest.approx(1.0)


def test_rbf_kernel_solves_xor():
    model = train_kernel_classifier(XOR_POINTS, XOR_LABELS, kernel=rbf_kernel(gamma=1.0))
    assert kernel_accuracy(model, XOR_POINTS, XOR_LABELS) == pytest.approx(1.0)


def test_kernel_classifier_keeps_the_training_points():
    model = train_kernel_classifier(XOR_POINTS, XOR_LABELS, kernel=rbf_kernel())
    # モデルは重みだけでなく訓練点そのものを保持する
    assert len(model.points) == len(XOR_POINTS)
    assert len(model.weights) == len(XOR_POINTS)


def test_only_misclassified_points_gain_weight():
    model = train_kernel_classifier(XOR_POINTS, XOR_LABELS, kernel=rbf_kernel(), epochs=200)
    # 重みが増えた点だけがサポートベクターとして効く
    assert any(weight > 0.0 for weight in model.weights)
    assert all(weight >= 0.0 for weight in model.weights)


def test_kernel_classifier_is_a_pure_function_of_its_parts():
    kernel = rbf_kernel(gamma=1.0)
    a = KernelClassifier(
        points=((0.0, 0.0),), labels=(1,), weights=(1.0,), bias=0.0, kernel=kernel
    )
    b = KernelClassifier(
        points=((0.0, 0.0),), labels=(1,), weights=(1.0,), bias=0.0, kernel=kernel
    )
    assert a.score((0.0, 0.0)) == pytest.approx(b.score((0.0, 0.0)))
