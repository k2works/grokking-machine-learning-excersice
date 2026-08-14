package lib

import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.LASSO
import smile.regression.OLS
import smile.regression.RidgeRegression

/**
 * 原著ノートブック #03
 * `Chapter_04_Testing_Overfitting_Underfitting/Polynomial_regression_regularization.ipynb`。
 *
 * 二次関数 -x^2 + 2 の周りに散らした 40 点へ、**次数 20** の多項式を当てはめる。
 * 正則化なしでは激しく過学習し、L1（LASSO）・L2（Ridge）を入れると収まる。
 *
 * scikit-learn の `PolynomialFeatures` にあたるものが Smile には無いので、
 * 多項式特徴量の展開は自前で書く。
 */
object Nb03PolynomialRegularization {

    /** 元にした多項式 -x^2 + 2 の係数。添字が次数に対応する */
    val POLYNOMIAL_COEFFICIENTS = intArrayOf(2, 0, -1)

    /** 原著が使う点の数とノイズの大きさ */
    const val SAMPLE_SIZE = 40
    const val NOISE_STD = 0.1

    /** 原著が当てはめる多項式の次数。40 点に対して 20 次は明らかに過剰 */
    const val DEGREE = 20

    /** 正則化の種類。原著は文字列 'L1' / 'L2' / None で切り替えていた */
    enum class Regularization { NONE, L1, L2 }

    /** 生成したデータと、その訓練／テスト分割 */
    data class Dataset(
        val x: DoubleArray,
        val y: DoubleArray,
        val xTrain: DoubleArray,
        val yTrain: DoubleArray,
        val xTest: DoubleArray,
        val yTest: DoubleArray,
    )

    /** 学習済みモデル。切片と各次数の係数を持つ */
    data class PolynomialModel(val intercept: Double, val coefficients: DoubleArray) {
        /** x に対する予測値。coefficients[i] は x^(i+1) の係数 */
        fun predict(x: Double): Double =
            intercept + coefficients.withIndex().sumOf { (i, c) -> c * x.pow(i + 1) }
    }

    /** 多項式の値を求める。`coefficients[i]` が x^i の係数 */
    fun polynomial(coefficients: IntArray, x: Double): Double =
        coefficients.withIndex().sumOf { (i, c) -> c * x.pow(i) }

    /**
     * -x^2 + 2 の周りにガウスノイズを載せた点を生成し、訓練とテストに分ける。
     *
     * Kotlin の [Random] は Python の Mersenne Twister と別物なので、同じ種を
     * 渡しても原著と同じ点にはならない。テストは数値の一致ではなく性質で検証する。
     */
    fun generateDataset(
        size: Int = SAMPLE_SIZE,
        seed: Int = 0,
        testRatio: Double = 0.2,
    ): Dataset {
        val random = Random(seed)
        val x = DoubleArray(size)
        val y = DoubleArray(size)
        for (i in 0 until size) {
            val sampled = random.nextDouble(-1.0, 1.0)
            x[i] = sampled
            y[i] = polynomial(POLYNOMIAL_COEFFICIENTS, sampled) + random.nextGaussian() * NOISE_STD
        }

        // train_test_split と同じく、添字をシャッフルしてから分ける
        val indices = (0 until size).shuffled(random)
        val testSize = (size * testRatio).toInt()
        val testIndices = indices.take(testSize)
        val trainIndices = indices.drop(testSize)

        return Dataset(
            x = x,
            y = y,
            xTrain = DoubleArray(trainIndices.size) { x[trainIndices[it]] },
            yTrain = DoubleArray(trainIndices.size) { y[trainIndices[it]] },
            xTest = DoubleArray(testIndices.size) { x[testIndices[it]] },
            yTest = DoubleArray(testIndices.size) { y[testIndices[it]] },
        )
    }

    /**
     * kotlin.random.Random は正規分布を持たないので、Box-Muller 法で作る。
     *
     * java.util.Random には `nextGaussian()` があるが、乱数列を 1 つの生成器に
     * 揃えたいのでここで定義する。
     */
    private fun Random.nextGaussian(): Double {
        val u1 = nextDouble()
        val u2 = nextDouble()
        return sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)
    }

    /**
     * x を x, x^2, ..., x^degree の列に展開する。
     *
     * scikit-learn の `PolynomialFeatures(include_bias=False)` にあたる。
     * 定数 1 の列を作らないのは、切片を回帰側に任せるためである。
     */
    fun polynomialFeatures(x: DoubleArray, degree: Int = DEGREE): Array<DoubleArray> =
        Array(x.size) { row -> DoubleArray(degree) { column -> x[row].pow(column + 1) } }

    private fun toSmileDataFrame(features: Array<DoubleArray>, y: DoubleArray): DataFrame {
        val degree = features[0].size
        val vectors = buildList {
            add(DoubleVector.of("y", y))
            for (column in 0 until degree) {
                add(DoubleVector.of("x$column", DoubleArray(features.size) { features[it][column] }))
            }
        }
        return DataFrame.of(*vectors.toTypedArray())
    }

    /** 多項式回帰を学習する。正則化は L1（LASSO）・L2（Ridge）から選ぶ */
    fun trainPolynomialRegression(
        x: DoubleArray,
        y: DoubleArray,
        degree: Int = DEGREE,
        regularization: Regularization = Regularization.NONE,
        alpha: Double = 1.0,
    ): PolynomialModel {
        val df = toSmileDataFrame(polynomialFeatures(x, degree), y)
        val formula = Formula.lhs("y")
        val model = when (regularization) {
            // Smile は正則化なしの OLS だけ Properties でソルバを選ぶ。
            // 次数 20 の特徴量は条件数が悪いので、ここでも SVD を使う
            Regularization.NONE -> OLS.fit(
                formula,
                df,
                java.util.Properties().apply { setProperty("smile.ols.method", "svd") },
            )
            Regularization.L1 -> LASSO.fit(formula, df, alpha)
            Regularization.L2 -> RidgeRegression.fit(formula, df, alpha)
        }
        return PolynomialModel(model.intercept(), model.coefficients())
    }

    /** テストセットに対する RMSE を返す */
    fun evaluateModel(model: PolynomialModel, x: DoubleArray, y: DoubleArray): Double {
        var squaredSum = 0.0
        for (i in x.indices) {
            val difference = y[i] - model.predict(x[i])
            squaredSum += difference * difference
        }
        return sqrt(squaredSum / x.size)
    }
}
