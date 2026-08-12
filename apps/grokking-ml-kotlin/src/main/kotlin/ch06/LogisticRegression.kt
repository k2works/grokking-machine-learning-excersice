/**
 * 第 6 章: ロジスティック回帰。
 * パーセプトロンの「0 か 1 か」という硬い予測を、シグモイド関数で
 * 「0 から 1 の確率」という連続的な予測に置き換える。
 */
package ch06

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** シグモイド関数。実数を 0 から 1 の範囲へ押し込む。 */
fun sigmoid(x: Double): Double =
    if (x >= 0) {
        1.0 / (1.0 + exp(-x))
    } else {
        // x が大きな負の数のとき exp(-x) が溢れるため、数学的に等価な式へ切り替える
        val exponential = exp(x)
        exponential / (1.0 + exponential)
    }

/** ロジスティック分類器。予測は 0 から 1 の確率。 */
data class LogisticClassifier(val weights: List<Double>, val bias: Double) {
    fun score(point: Point): Double = bias + weights.zip(point) { w, x -> w * x }.sum()

    fun predictProbability(point: Point): Double = sigmoid(score(point))

    fun predict(point: Point, threshold: Double = 0.5): Int =
        if (predictProbability(point) >= threshold) 1 else 0
}

/** 1 点分の対数損失。予測確率が正解から離れるほど大きくなる。 */
fun logLoss(model: LogisticClassifier, point: Point, label: Int): Double {
    // log(0) を避けるためにごくわずかに内側へ丸める
    val epsilon = 1e-15
    val probability = min(max(model.predictProbability(point), epsilon), 1.0 - epsilon)
    return if (label == 1) -ln(probability) else -ln(1.0 - probability)
}

/** 全点の平均対数損失。 */
fun meanLogLoss(model: LogisticClassifier, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).sumOf { (point, label) -> logLoss(model, point, label) } / points.size

/** ロジスティックトリック。すべての点を、確率の外れ具合に比例して動かす。 */
fun logisticTrick(
    model: LogisticClassifier,
    point: Point,
    label: Int,
    learningRate: Double = 0.01,
): LogisticClassifier {
    val error = label - model.predictProbability(point)
    val weights = model.weights.zip(point) { w, x -> w + learningRate * error * x }
    return LogisticClassifier(weights, model.bias + learningRate * error)
}

/** 正解率。 */
fun accuracy(model: LogisticClassifier, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }.toDouble() / points.size

/** ロジスティック回帰。モデルとエポックごとの平均対数損失を返す。 */
fun logisticRegression(
    points: List<Point>,
    labels: List<Int>,
    learningRate: Double = 0.1,
    epochs: Int = 1000,
    seed: Int = 0,
): Pair<LogisticClassifier, List<Double>> {
    val random = Random(seed)
    var model = LogisticClassifier(List(points.first().size) { 0.0 }, 0.0)
    val losses = mutableListOf<Double>()
    repeat(epochs) {
        losses += meanLogLoss(model, points, labels)
        val i = random.nextInt(points.size)
        model = logisticTrick(model, points[i], labels[i], learningRate)
    }
    return model to losses
}
