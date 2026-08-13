/**
 * 第 5 章: パーセプトロン。
 * 点を直線で 2 クラスに分ける。予測が外れた点だけを使って境界線を動かす
 * パーセプトロントリックを実装する。
 */
package ch05

import kotlin.math.abs
import kotlin.random.Random

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** 線形分類器 score = w・x + bias。score >= 0 なら 1、そうでなければ 0。 */
data class Perceptron(val weights: List<Double>, val bias: Double) {
    fun score(point: Point): Double =
        bias + weights.zip(point) { w, x -> w * x }.sum()

    fun predict(point: Point): Int = if (score(point) >= 0) 1 else 0
}

/** パーセプトロントリック。誤分類した点だけモデルを動かす。 */
fun perceptronTrick(
    model: Perceptron,
    point: Point,
    label: Int,
    learningRate: Double = 0.01,
): Perceptron {
    val error = label - model.predict(point)
    if (error == 0) return model
    val weights = model.weights.zip(point) { w, x -> w + learningRate * error * x }
    return Perceptron(weights, model.bias + learningRate * error)
}

/** 1 点分の誤差。正しく分類していれば 0、誤っていればスコアの絶対値。 */
fun perceptronError(model: Perceptron, point: Point, label: Int): Double =
    if (model.predict(point) == label) 0.0 else abs(model.score(point))

/** 全点の平均誤差。 */
fun meanPerceptronError(model: Perceptron, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).sumOf { (point, label) -> perceptronError(model, point, label) } / points.size

/** 正解率。 */
fun accuracy(model: Perceptron, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }.toDouble() / points.size

/** パーセプトロンアルゴリズム。モデルとエポックごとの平均誤差を返す。 */
fun perceptronAlgorithm(
    points: List<Point>,
    labels: List<Int>,
    learningRate: Double = 0.01,
    epochs: Int = 1000,
    seed: Int = 0,
): Pair<Perceptron, List<Double>> {
    val random = Random(seed)
    var model = Perceptron(List(points.first().size) { 0.0 }, 0.0)
    val errors = mutableListOf<Double>()
    repeat(epochs) {
        errors += meanPerceptronError(model, points, labels)
        val i = random.nextInt(points.size)
        model = perceptronTrick(model, points[i], labels[i], learningRate)
    }
    return model to errors
}
