/**
 * 第 11 章: サポートベクターマシンとカーネル法。
 * 第 5 章のパーセプトロンは「分ければどこでもよい」だった。SVM は
 * 2 クラスの間にできるだけ広い余白（マージン）を空ける境界線を選ぶ。
 */
package ch11

import kotlin.math.exp
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** 2 点の類似度を測る関数。 */
typealias Kernel = (Point, Point) -> Double

/** 内積。 */
fun dot(a: Point, b: Point): Double = a.zip(b) { x, y -> x * y }.sum()

/** 線形カーネル。ただの内積。 */
val linearKernel: Kernel = { a, b -> dot(a, b) }

/** 多項式カーネル。特徴量の積を暗黙のうちに作る。 */
fun polynomialKernel(degree: Int = 2, constant: Double = 1.0): Kernel =
    { a, b -> (dot(a, b) + constant).pow(degree) }

/** RBF（ガウシアン）カーネル。距離が近いほど 1 に近づく。 */
fun rbfKernel(gamma: Double = 1.0): Kernel =
    { a, b ->
        val squaredDistance = a.zip(b) { x, y -> (x - y) * (x - y) }.sum()
        exp(-gamma * squaredDistance)
    }

/** 線形 SVM。ラベルは +1 と -1 を使う。 */
data class SupportVectorMachine(val weights: List<Double>, val bias: Double) {
    fun score(point: Point): Double = bias + dot(weights, point)

    fun predict(point: Point): Int = if (score(point) >= 0) 1 else -1

    /** マージン幅。境界線からもっとも近い点までの距離の 2 倍。 */
    fun margin(points: List<Point>): Double {
        val norm = sqrt(weights.sumOf { it * it })
        if (norm == 0.0) return 0.0
        return 2.0 * points.minOf { abs(score(it)) } / norm
    }
}

/** ヒンジ損失。マージンの内側に入った分だけ罰する。 */
fun hingeLoss(model: SupportVectorMachine, point: Point, label: Int): Double =
    max(0.0, 1.0 - label * model.score(point))

/** SVM の目的関数。ヒンジ損失の平均 + 重みの大きさへの罰。 */
fun svmError(
    model: SupportVectorMachine,
    points: List<Point>,
    labels: List<Int>,
    regularization: Double = 0.1,
): Double {
    val losses = points.zip(labels).sumOf { (point, label) -> hingeLoss(model, point, label) } / points.size
    return losses + regularization * model.weights.sumOf { it * it }
}

/** 1 点分の更新。マージンの内側なら押し返し、常に重みを縮める。 */
fun svmStep(
    model: SupportVectorMachine,
    point: Point,
    label: Int,
    learningRate: Double = 0.01,
    regularization: Double = 0.1,
): SupportVectorMachine {
    val insideMargin = label * model.score(point) < 1.0
    val weights = model.weights.zip(point) { w, x ->
        val gradient = 2.0 * regularization * w - if (insideMargin) label * x else 0.0
        w - learningRate * gradient
    }
    val bias = model.bias + if (insideMargin) learningRate * label else 0.0
    return SupportVectorMachine(weights, bias)
}

/** SVM を学習する。モデルとエポックごとの目的関数値を返す。 */
fun trainSvm(
    points: List<Point>,
    labels: List<Int>,
    learningRate: Double = 0.01,
    epochs: Int = 5000,
    regularization: Double = 0.1,
    seed: Int = 0,
): Pair<SupportVectorMachine, List<Double>> {
    val random = Random(seed)
    var model = SupportVectorMachine(List(points.first().size) { 0.0 }, 0.0)
    val errors = mutableListOf<Double>()
    repeat(epochs) {
        errors += svmError(model, points, labels, regularization)
        val i = random.nextInt(points.size)
        model = svmStep(model, points[i], labels[i], learningRate, regularization)
    }
    return model to errors
}

/** 正解率。 */
fun accuracy(model: SupportVectorMachine, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }
        .toDouble() / points.size

/** カーネル分類器。訓練点そのものを重み付きで覚えておく。 */
data class KernelClassifier(
    val points: List<Point>,
    val labels: List<Int>,
    val weights: List<Double>,
    val bias: Double,
    val kernel: Kernel,
) {
    fun score(point: Point): Double =
        bias + weights.indices.sumOf { i -> weights[i] * labels[i] * kernel(points[i], point) }

    fun predict(point: Point): Int = if (score(point) >= 0) 1 else -1
}

/** カーネル版パーセプトロン。誤分類した点の重みだけを増やす。 */
fun trainKernelClassifier(
    points: List<Point>,
    labels: List<Int>,
    kernel: Kernel = linearKernel,
    learningRate: Double = 0.1,
    epochs: Int = 2000,
    seed: Int = 0,
): KernelClassifier {
    val random = Random(seed)
    val weights = MutableList(points.size) { 0.0 }
    var bias = 0.0
    repeat(epochs) {
        val i = random.nextInt(points.size)
        val model = KernelClassifier(points, labels, weights.toList(), bias, kernel)
        if (labels[i] * model.score(points[i]) <= 0) {
            weights[i] += learningRate
            bias += learningRate * labels[i]
        }
    }
    return KernelClassifier(points, labels, weights.toList(), bias, kernel)
}

/** カーネル分類器の正解率。 */
fun kernelAccuracy(model: KernelClassifier, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }
        .toDouble() / points.size
