package lib

import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.GradientTreeBoost
import smile.regression.RegressionTree

/**
 * 原著ノートブック #21 `Chapter_12_Ensemble_Methods/Gradient_boosting_and_XGBoost.ipynb`。
 *
 * 8 人の年齢から「週に何日アプリを使うか」を当てる回帰の回である。
 *
 * 前半は **勾配ブースティングを手で追う**。弱学習器を 1 本ずつ取り出し、
 * 残差がどう縮んでいくかを段ごとに見る。
 *
 * 後半は **XGBoost の類似度スコアを手で計算する**。
 * ここはライブラリに依存しないので、原著の式をそのまま写せる。
 */
object Nb21GradientBoosting {

    /** 原著の 8 人。年齢 */
    val AGES = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

    /** 週あたりの利用日数 */
    val DAYS = doubleArrayOf(7.0, 5.0, 7.0, 1.0, 2.0, 1.0, 5.0, 4.0)

    /** 原著の勾配ブースティングの設定 */
    const val MAX_DEPTH = 2
    const val N_ESTIMATORS = 4
    const val LEARNING_RATE = 0.8

    /** XGBoost 側の設定。最初の予測はラベルの平均 */
    const val INITIAL_PREDICTION = 4.0
    const val XGB_LAMBDA = 3.0

    /** 目的変数の列名 */
    const val LABEL = "Days"

    /** Smile のデータフレームに直す */
    fun frame(): DataFrame =
        DataFrame.of(DoubleVector.of("Age", AGES), DoubleVector.of(LABEL, DAYS))

    private val formula = Formula.lhs(LABEL)

    /**
     * 比較用の回帰木 1 本。
     *
     * [#12][Nb12RegressionTree] と同じく `RegressionTree.fit` は
     * `SplitRule` を取らない（回帰では分散の減少で分割が決まるため）。
     * 深さは [#10][Nb10DecisionTreeBoundary] の対応どおり
     * **scikit-learn より 1 大きく** 渡す。
     */
    fun fitRegressionTree(maxDepth: Int = MAX_DEPTH + 1): RegressionTree =
        RegressionTree.fit(formula, frame(), maxDepth, 10000, 1)

    /** 原著と同じ設定の勾配ブースティング */
    fun fitGradientBoosting(nEstimators: Int = N_ESTIMATORS): GradientTreeBoost {
        smile.math.MathEx.setSeed(0L)
        // loss, ntrees, maxDepth, maxNodes, nodeSize, shrinkage, subsample。
        // scikit-learn の既定は二乗誤差なので Loss.ls() を選ぶ
        return GradientTreeBoost.fit(
            formula, frame(), smile.base.cart.Loss.ls(),
            nEstimators, MAX_DEPTH + 1, 10000, 1, LEARNING_RATE, 1.0,
        )
    }

    /** ラベルから平均を引いたもの。勾配ブースティングが最初に予測する対象 */
    fun centeredLabels(): DoubleArray {
        val mean = DAYS.average()
        return DoubleArray(DAYS.size) { DAYS[it] - mean }
    }

    /** 全員ぶんの予測 */
    fun predict(model: smile.regression.Regression<smile.data.Tuple>): DoubleArray {
        val data = frame()
        return DoubleArray(AGES.size) { model.predict(data[it]) }
    }

    /** 予測との残差 */
    fun residuals(model: smile.regression.Regression<smile.data.Tuple>): DoubleArray {
        val predicted = predict(model)
        return DoubleArray(DAYS.size) { DAYS[it] - predicted[it] }
    }

    /**
     * XGBoost の類似度スコア。原著の実装をそのまま写した。
     *
     * `sum(l)^2 / (len(l) + lambda)`。**分子は和の 2 乗であって
     * 2 乗の和ではない。** だから符号がばらけるほど値が小さくなり、
     * 「似ている点が集まっているか」の尺度になる。
     */
    fun similarityScore(values: DoubleArray, lambda: Double = 0.0): Double {
        if (values.isEmpty()) return 0.0
        val total = values.sum()
        return total * total / (values.size + lambda)
    }

    /** 分割の結果 */
    data class Split(val left: DoubleArray, val right: DoubleArray, val score: Double) {
        override fun equals(other: Any?): Boolean =
            other is Split &&
                left.contentEquals(other.left) &&
                right.contentEquals(other.right) &&
                score == other.score

        override fun hashCode(): Int =
            (left.contentHashCode() * 31 + right.contentHashCode()) * 31 + score.hashCode()
    }

    /** 切れ目ごとの類似度スコアの和。原著が 1 行ずつ印刷しているもの */
    fun splitScores(residuals: DoubleArray, lambda: Double): List<Double> =
        (1 until residuals.size).map { index ->
            similarityScore(residuals.copyOfRange(0, index), lambda) +
                similarityScore(residuals.copyOfRange(index, residuals.size), lambda)
        }

    /**
     * 原著の `find_best_split`。
     *
     * **並び順のまま前から切る** だけで、しきい値を探しているわけではない。
     * データが年齢順に並んでいるから成り立つ簡略版である。
     *
     * なお **原著の実装にはバグがある。** 返り値の 3 つ目に `best_score`
     * ではなく `new_score`（ループの最後の切れ目のスコア）を渡している。
     * ここでは最良のスコアを返す。詳しくは記事を参照。
     */
    fun findBestSplit(residuals: DoubleArray, lambda: Double): Split {
        var bestScore = 0.0
        var bestLeft = DoubleArray(0)
        var bestRight = residuals

        for (index in 1 until residuals.size) {
            val left = residuals.copyOfRange(0, index)
            val right = residuals.copyOfRange(index, residuals.size)
            val score = similarityScore(left, lambda) + similarityScore(right, lambda)
            if (score > bestScore) {
                bestScore = score
                bestLeft = left
                bestRight = right
            }
        }

        return Split(bestLeft, bestRight, bestScore)
    }

    /** 最初の予測（平均 4.0）からの残差 */
    fun xgboostResiduals(): DoubleArray = DoubleArray(DAYS.size) { DAYS[it] - INITIAL_PREDICTION }
}
