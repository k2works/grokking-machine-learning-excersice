/**
 * 第 10 章: ニューラルネットワーク。
 * パーセプトロンを積み重ねて、直線では分けられないデータを分ける。
 * 学習は誤差逆伝播法（連鎖律による勾配の伝播）で行う。
 */
package ch10

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** シグモイド関数。第 6 章と同じ数値的に安定な実装。 */
fun sigmoid(x: Double): Double =
    if (x >= 0) {
        1.0 / (1.0 + exp(-x))
    } else {
        val exponential = exp(x)
        exponential / (1.0 + exponential)
    }

/** シグモイドの微分。出力そのものから計算できる。 */
fun sigmoidDerivative(output: Double): Double = output * (1.0 - output)

/** 全結合層。weights[j][i] は入力 i から出力 j への重み。 */
data class Layer(val weights: List<List<Double>>, val biases: List<Double>) {
    val inputSize: Int get() = weights.first().size

    val outputSize: Int get() = weights.size

    /** 順伝播。重み付き和にシグモイドを適用する。 */
    fun forward(inputs: List<Double>): List<Double> =
        weights.zip(biases) { row, bias ->
            sigmoid(bias + row.zip(inputs) { w, x -> w * x }.sum())
        }
}

/** 多層パーセプトロン。層を順に適用する。 */
data class NeuralNetwork(val layers: List<Layer>) {
    /** 各層の出力を順に記録する。逆伝播で必要になる。 */
    fun forwardAll(inputs: List<Double>): List<List<Double>> =
        layers.fold(listOf(inputs)) { activations, layer ->
            activations + listOf(layer.forward(activations.last()))
        }

    /** 出力層の最初のニューロンの値を確率として返す。 */
    fun predictProbability(inputs: List<Double>): Double = forwardAll(inputs).last().first()

    fun predict(inputs: List<Double>, threshold: Double = 0.5): Int =
        if (predictProbability(inputs) >= threshold) 1 else 0
}

/** 指定した層構成のネットワークを乱数で初期化する。 */
fun initialNetwork(sizes: List<Int>, seed: Int = 0): NeuralNetwork {
    val random = Random(seed)
    return NeuralNetwork(
        sizes.zipWithNext { inputSize, outputSize ->
            Layer(
                weights = List(outputSize) { List(inputSize) { random.nextDouble(-1.0, 1.0) } },
                biases = List(outputSize) { random.nextDouble(-1.0, 1.0) },
            )
        },
    )
}

/** 1 点分の対数損失。第 6 章と同じ。 */
fun logLoss(model: NeuralNetwork, inputs: List<Double>, label: Int): Double {
    val epsilon = 1e-15
    val probability = min(max(model.predictProbability(inputs), epsilon), 1.0 - epsilon)
    return if (label == 1) -ln(probability) else -ln(1.0 - probability)
}

/** 全点の平均対数損失。 */
fun meanLogLoss(model: NeuralNetwork, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).sumOf { (point, label) -> logLoss(model, point, label) } / points.size

/** 誤差逆伝播法による 1 点分の更新。 */
fun backpropagate(
    model: NeuralNetwork,
    inputs: List<Double>,
    label: Int,
    learningRate: Double,
): NeuralNetwork {
    val activations = model.forwardAll(inputs)
    // 出力層の誤差（対数損失 × シグモイドの微分が predicted - label に簡約される）
    val deltas = ArrayDeque(listOf(listOf(activations.last().first() - label)))
    // 出力層から入力側へ、連鎖律で誤差を遡らせる
    for (index in model.layers.size - 1 downTo 1) {
        val layer = model.layers[index]
        val downstream = deltas.first()
        val outputs = activations[index]
        deltas.addFirst(
            List(layer.inputSize) { i ->
                (0 until layer.outputSize).sumOf { j -> layer.weights[j][i] * downstream[j] } *
                    sigmoidDerivative(outputs[i])
            },
        )
    }
    val updated = model.layers.mapIndexed { index, layer ->
        val delta = deltas[index]
        val previous = activations[index]
        Layer(
            weights = layer.weights.mapIndexed { j, row ->
                row.mapIndexed { i, w -> w - learningRate * delta[j] * previous[i] }
            },
            biases = layer.biases.mapIndexed { j, bias -> bias - learningRate * delta[j] },
        )
    }
    return NeuralNetwork(updated)
}

/** ネットワークを学習する。モデルとエポックごとの平均損失を返す。 */
fun train(
    points: List<Point>,
    labels: List<Int>,
    hiddenSize: Int = 4,
    learningRate: Double = 0.5,
    epochs: Int = 5000,
    seed: Int = 0,
): Pair<NeuralNetwork, List<Double>> {
    val random = Random(seed)
    var model = initialNetwork(listOf(points.first().size, hiddenSize, 1), seed)
    val losses = mutableListOf<Double>()
    repeat(epochs) {
        losses += meanLogLoss(model, points, labels)
        val i = random.nextInt(points.size)
        model = backpropagate(model, points[i], labels[i], learningRate)
    }
    return model to losses
}

/** 正解率。 */
fun accuracy(model: NeuralNetwork, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }
        .toDouble() / points.size
