"""第 10 章: ニューラルネットワーク。

パーセプトロンを積み重ねて、直線では分けられないデータを分ける。
学習は誤差逆伝播法（連鎖律による勾配の伝播）で行う。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

Point = Sequence[float]
Vector = list[float]
Matrix = list[list[float]]


def sigmoid(x: float) -> float:
    """シグモイド関数。第 6 章と同じ数値的に安定な実装。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def sigmoid_derivative(output: float) -> float:
    """シグモイドの微分。出力そのものから計算できる。"""
    return output * (1.0 - output)


@dataclass(frozen=True)
class Layer:
    """全結合層。weights[j][i] は入力 i から出力 j への重み。"""

    weights: Matrix
    biases: Vector

    @property
    def input_size(self) -> int:
        return len(self.weights[0])

    @property
    def output_size(self) -> int:
        return len(self.weights)

    def forward(self, inputs: Sequence[float]) -> Vector:
        """順伝播。重み付き和にシグモイドを適用する。"""
        return [
            sigmoid(bias + sum(w * x for w, x in zip(row, inputs)))
            for row, bias in zip(self.weights, self.biases)
        ]


@dataclass(frozen=True)
class NeuralNetwork:
    """多層パーセプトロン。層を順に適用する。"""

    layers: list[Layer]

    def forward_all(self, inputs: Sequence[float]) -> list[Vector]:
        """各層の出力を順に記録する。逆伝播で必要になる。"""
        activations = [list(inputs)]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations

    def predict_probability(self, inputs: Sequence[float]) -> float:
        """出力層の最初のニューロンの値を確率として返す。"""
        return self.forward_all(inputs)[-1][0]

    def predict(self, inputs: Sequence[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(inputs) >= threshold else 0


def initial_network(sizes: Sequence[int], seed: int = 0) -> NeuralNetwork:
    """指定した層構成のネットワークを乱数で初期化する。"""
    rng = random.Random(seed)
    layers = []
    for input_size, output_size in pairwise(sizes):
        weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(input_size)] for _ in range(output_size)
        ]
        biases = [rng.uniform(-1.0, 1.0) for _ in range(output_size)]
        layers.append(Layer(weights=weights, biases=biases))
    return NeuralNetwork(layers=layers)


def log_loss(model: NeuralNetwork, inputs: Sequence[float], label: int) -> float:
    """1 点分の対数損失。第 6 章と同じ。"""
    probability = model.predict_probability(inputs)
    epsilon = 1e-15
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    return -math.log(probability) if label == 1 else -math.log(1.0 - probability)


def mean_log_loss(
    model: NeuralNetwork,
    points: Sequence[Point],
    labels: Sequence[int],
) -> float:
    """全点の平均対数損失。"""
    total = sum(log_loss(model, point, label) for point, label in zip(points, labels))
    return total / len(points)


def backpropagate(
    model: NeuralNetwork,
    inputs: Sequence[float],
    label: int,
    learning_rate: float,
) -> NeuralNetwork:
    """誤差逆伝播法による 1 点分の更新。"""
    activations = model.forward_all(inputs)
    # 出力層の誤差（対数損失 × シグモイドの微分が predicted - label に簡約される）
    deltas: list[Vector] = [[activations[-1][0] - label]]
    # 出力層から入力側へ、連鎖律で誤差を遡らせる
    for index in range(len(model.layers) - 1, 0, -1):
        layer = model.layers[index]
        downstream = deltas[0]
        outputs = activations[index]
        deltas.insert(
            0,
            [
                sum(layer.weights[j][i] * downstream[j] for j in range(layer.output_size))
                * sigmoid_derivative(outputs[i])
                for i in range(layer.input_size)
            ],
        )
    updated = []
    for index, layer in enumerate(model.layers):
        delta = deltas[index]
        previous = activations[index]
        weights = [
            [w - learning_rate * delta[j] * previous[i] for i, w in enumerate(row)]
            for j, row in enumerate(layer.weights)
        ]
        biases = [bias - learning_rate * delta[j] for j, bias in enumerate(layer.biases)]
        updated.append(Layer(weights=weights, biases=biases))
    return NeuralNetwork(layers=updated)


def train(
    points: Sequence[Point],
    labels: Sequence[int],
    hidden_size: int = 4,
    learning_rate: float = 0.5,
    epochs: int = 5000,
    seed: int = 0,
) -> tuple[NeuralNetwork, list[float]]:
    """ネットワークを学習する。モデルとエポックごとの平均損失を返す。"""
    rng = random.Random(seed)
    model = initial_network([len(points[0]), hidden_size, 1], seed=seed)
    losses: list[float] = []
    for _ in range(epochs):
        losses.append(mean_log_loss(model, points, labels))
        i = rng.randrange(len(points))
        model = backpropagate(model, points[i], labels[i], learning_rate)
    return model, losses


def accuracy(model: NeuralNetwork, points: Sequence[Point], labels: Sequence[int]) -> float:
    """正解率。"""
    correct = sum(1 for point, label in zip(points, labels) if model.predict(point) == label)
    return correct / len(points)
