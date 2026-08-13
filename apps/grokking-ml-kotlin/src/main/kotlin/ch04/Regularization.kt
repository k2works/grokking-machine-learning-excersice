/**
 * 第 4 章: 過学習・未学習と正則化。
 * 多項式回帰で次数を変えながら、訓練データとテストデータの誤差を比べる。
 * さらに L1 / L2 正則化で係数を抑え、複雑すぎるモデルを緩める。
 */
package ch04

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/** 正則化の種類。 */
enum class Regularization {
    NONE,
    L1,
    L2,
}

/** 多項式モデル y = bias + w1*x + w2*x^2 + ... + wn*x^n。 */
data class PolynomialModel(val weights: List<Double>, val bias: Double) {
    val degree: Int get() = weights.size

    fun predict(x: Double): Double =
        bias + weights.withIndex().sumOf { (index, w) -> w * x.pow(index + 1) }
}

/** x から [x, x^2, ..., x^degree] を作る。 */
fun polynomialFeatures(x: Double, degree: Int): List<Double> =
    (1..degree).map { power -> x.pow(power) }

/** 正則化項の勾配。重みを 0 に引き戻す向きの力を返す。 */
fun regularizationGradient(weight: Double, kind: Regularization, strength: Double): Double =
    when (kind) {
        Regularization.NONE -> 0.0
        Regularization.L1 -> strength * when {
            weight > 0 -> 1.0
            weight < 0 -> -1.0
            else -> 0.0
        }
        Regularization.L2 -> strength * 2.0 * weight
    }

/** 二乗トリックに正則化項を加えた 1 点分の更新。 */
fun squareTrick(
    model: PolynomialModel,
    x: Double,
    y: Double,
    learningRate: Double,
    kind: Regularization = Regularization.NONE,
    strength: Double = 0.0,
): PolynomialModel {
    val error = y - model.predict(x)
    val features = polynomialFeatures(x, model.degree)
    val weights = model.weights.zip(features) { w, feature ->
        w + learningRate * (error * feature - regularizationGradient(w, kind, strength))
    }
    return PolynomialModel(weights, model.bias + learningRate * error)
}

/** 二乗平均平方根誤差。 */
fun rmse(labels: List<Double>, predictions: List<Double>): Double {
    val total = labels.zip(predictions).sumOf { (label, prediction) ->
        val difference = label - prediction
        difference * difference
    }
    return sqrt(total / labels.size)
}

/** モデルの RMSE。 */
fun modelRmse(model: PolynomialModel, features: List<Double>, labels: List<Double>): Double =
    rmse(labels, features.map { model.predict(it) })

/** 訓練用とテスト用に分割したデータセット。 */
data class Split(
    val trainFeatures: List<Double>,
    val trainLabels: List<Double>,
    val testFeatures: List<Double>,
    val testLabels: List<Double>,
)

/** データを訓練用とテスト用に分割する。 */
fun trainTestSplit(
    features: List<Double>,
    labels: List<Double>,
    testRatio: Double = 0.3,
    seed: Int = 0,
): Split {
    val indices = features.indices.shuffled(Random(seed))
    val testSize = (features.size * testRatio).toInt()
    val testIndices = indices.take(testSize)
    val trainIndices = indices.drop(testSize)
    return Split(
        trainFeatures = trainIndices.map { features[it] },
        trainLabels = trainIndices.map { labels[it] },
        testFeatures = testIndices.map { features[it] },
        testLabels = testIndices.map { labels[it] },
    )
}

/** 確率的勾配降下法で多項式回帰を学習する。 */
fun polynomialRegression(
    features: List<Double>,
    labels: List<Double>,
    degree: Int,
    learningRate: Double = 0.01,
    epochs: Int = 20000,
    kind: Regularization = Regularization.NONE,
    strength: Double = 0.0,
    seed: Int = 0,
): PolynomialModel {
    val random = Random(seed)
    var model = PolynomialModel(List(degree) { 0.0 }, 0.0)
    repeat(epochs) {
        val i = random.nextInt(features.size)
        model = squareTrick(model, features[i], labels[i], learningRate, kind, strength)
    }
    return model
}

/** 重みの絶対値の合計。モデルの複雑さの目安。 */
fun weightMagnitude(model: PolynomialModel): Double = model.weights.sumOf { abs(it) }
