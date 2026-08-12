/**
 * 第 3 章: 線形回帰。
 * 部屋数から住宅価格を予測する 1 次元の線形回帰を、
 * simple / absolute / square の 3 つのトリックで学習する。
 */
package ch03

import kotlin.math.sqrt
import kotlin.random.Random

/** 1 次元の線形モデル price = slope * rooms + intercept。 */
data class Model(val slope: Double, val intercept: Double) {
    fun predict(rooms: Double): Double = slope * rooms + intercept
}

/** 単純なトリック。予測の上下だけを見て、ランダムな微小量だけ動かす。 */
fun simpleTrick(model: Model, rooms: Double, price: Double, random: Random): Model {
    val stepSlope = random.nextDouble() * 0.1
    val stepIntercept = random.nextDouble() * 0.1
    return if (price > model.predict(rooms)) {
        Model(
            slope = if (rooms > 0) model.slope + stepSlope else model.slope - stepSlope,
            intercept = model.intercept + stepIntercept,
        )
    } else {
        Model(
            slope = if (rooms > 0) model.slope - stepSlope else model.slope + stepSlope,
            intercept = model.intercept - stepIntercept,
        )
    }
}

/** 絶対トリック。誤差の符号のみを使い、特徴量に比例した量だけ動かす。 */
fun absoluteTrick(model: Model, rooms: Double, price: Double, learningRate: Double): Model {
    val sign = if (price > model.predict(rooms)) 1.0 else -1.0
    return Model(
        slope = model.slope + sign * learningRate * rooms,
        intercept = model.intercept + sign * learningRate,
    )
}

/** 二乗トリック。誤差の大きさに比例した量だけ動かす（二乗誤差の勾配降下法）。 */
fun squareTrick(model: Model, rooms: Double, price: Double, learningRate: Double): Model {
    val error = price - model.predict(rooms)
    return Model(
        slope = model.slope + learningRate * rooms * error,
        intercept = model.intercept + learningRate * error,
    )
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
fun modelRmse(model: Model, features: List<Double>, labels: List<Double>): Double =
    rmse(labels, features.map { model.predict(it) })

/** 確率的勾配降下法で線形回帰を学習し、モデルとエポックごとの RMSE を返す。 */
fun linearRegression(
    features: List<Double>,
    labels: List<Double>,
    learningRate: Double = 0.01,
    epochs: Int = 1000,
    seed: Int = 0,
): Pair<Model, List<Double>> {
    val random = Random(seed)
    var model = Model(random.nextDouble(), random.nextDouble())
    val errors = mutableListOf<Double>()
    repeat(epochs) {
        errors += modelRmse(model, features, labels)
        val i = random.nextInt(features.size)
        model = squareTrick(model, features[i], labels[i], learningRate)
    }
    return model to errors
}
