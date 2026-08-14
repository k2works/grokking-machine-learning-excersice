package lib

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random
import smile.classification.LogisticRegression

/**
 * 原著ノートブック #05 `Chapter_06_Logistic_Regression/Coding_logistic_regression.ipynb`。
 *
 * #04 と同じ形の 8 点を、今度はロジスティック回帰で分ける。
 * ステップ関数がシグモイドに、パーセプトロン誤差が対数損失に置き換わる。
 *
 * 原著は対数損失の「別の書き方」も示しているが、**その式は対数損失と一致しない**。
 * 正しい形と並べて、両方を実装してある。
 */
object Nb05LogisticRegression {

    /** 原著が使う 8 点。#04 と似ているが最後の 2 点が入れ替わっている */
    val FEATURES = arrayOf(
        intArrayOf(1, 0), intArrayOf(0, 2), intArrayOf(1, 1), intArrayOf(1, 2),
        intArrayOf(1, 3), intArrayOf(2, 2), intArrayOf(3, 2), intArrayOf(2, 3),
    )
    val LABELS = intArrayOf(0, 0, 0, 0, 1, 1, 1, 1)

    /** 分離直線。重みとバイアスの組 */
    data class Boundary(val weights: DoubleArray, val bias: Double) {
        /** 重み付き和にバイアスを足したもの */
        fun score(features: IntArray): Double =
            bias + weights.indices.sumOf { weights[it] * features[it] }

        /** 予測確率。0 / 1 ではなく 0〜1 の連続値を返す */
        fun prediction(features: IntArray): Double = sigmoid(score(features))

        /** 対数損失。当たっていても 0 にはならず、確信の度合いで連続的に変わる */
        fun logLoss(features: IntArray, label: Int): Double {
            val pred = prediction(features)
            return -label * ln(pred) - (1 - label) * ln(1 - pred)
        }

        override fun equals(other: Any?): Boolean =
            other is Boundary && weights.contentEquals(other.weights) && bias == other.bias

        override fun hashCode(): Int = 31 * weights.contentHashCode() + bias.hashCode()
    }

    /** 学習の結果と途中経過 */
    data class TrainingLog(
        val boundary: Boundary,
        /** 各エポック開始時点の対数損失の合計 */
        val errors: List<Double>,
    )

    /**
     * シグモイド関数。
     *
     * 原著のコメントどおり `exp(x) / (1 + exp(x))` で書く。教科書によくある
     * `1 / (1 + exp(-x))` と数学的には同じだが、x が大きな負の数のときに
     * `exp(-x)` が溢れない形になっている。
     */
    fun sigmoid(x: Double): Double = exp(x) / (1 + exp(x))

    /** ソフト ReLU。`log(1 + exp(x))` で、ReLU をなめらかにしたもの */
    fun softRelu(x: Double): Double = ln(1 + exp(x))

    /** 全点の対数損失の合計。原著は平均ではなく合計を取っている */
    fun totalLogLoss(
        boundary: Boundary,
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
    ): Double = features.indices.sumOf { boundary.logLoss(features[it], labels[it]) }

    /**
     * 原著が「対数損失の別の書き方」として示す式。
     *
     * **実際には対数損失と一致しない。** `pred` は 0〜1 の確率なので、
     * `(pred - label)` は -1 か +1 ではなく中間の値になる。
     * スコアが 0 のときだけ両者が一致する。詳しくは記事を参照。
     */
    fun alternateLogLossOriginal(boundary: Boundary, features: IntArray, label: Int): Double =
        softRelu((boundary.prediction(features) - label) * boundary.score(features))

    /**
     * 対数損失と厳密に等しい「別の書き方」。
     *
     * ラベルが 0 なら +1、1 なら -1 を掛ける。つまり `(1 - 2 * label)`。
     */
    fun alternateLogLoss(boundary: Boundary, features: IntArray, label: Int): Double =
        softRelu((1 - 2 * label) * boundary.score(features))

    /**
     * ロジスティックトリック。パーセプトロンのトリックと同じ形をしている。
     *
     * 違いは `pred` が 0 / 1 ではなく 0〜1 の連続値であること。
     * #04 の「短く書いた版」と違い、**バイアスの更新はループの外側にある**。
     */
    fun logisticTrick(
        boundary: Boundary,
        features: IntArray,
        label: Int,
        learningRate: Double = 0.05,
    ): Boundary {
        val pred = boundary.prediction(features)
        val weights = boundary.weights.copyOf()
        for (i in weights.indices) {
            weights[i] += (label - pred) * features[i] * learningRate
        }
        return Boundary(weights, boundary.bias + (label - pred) * learningRate)
    }

    /**
     * トリックを繰り返して分離直線を学習する。
     *
     * #04 と同じく、原著は点の選択に種を与えていない標準ライブラリの `random` を
     * 使っている。原著の出力 `([1.2019, 0.7009], -2.7884)` は再現できない。
     */
    fun logisticRegressionAlgorithm(
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
        learningRate: Double = 0.01,
        epochs: Int = 500,
        seed: Int = 0,
    ): TrainingLog {
        val random = Random(seed)
        var boundary = Boundary(DoubleArray(features[0].size) { 1.0 }, 0.0)
        val errors = ArrayList<Double>(epochs)

        repeat(epochs) {
            errors.add(totalLogLoss(boundary, features, labels))
            val j = random.nextInt(features.size)
            boundary = logisticTrick(boundary, features[j], labels[j], learningRate)
        }

        return TrainingLog(boundary, errors)
    }

    /**
     * 同じ問題を Smile の [LogisticRegression] に解かせる。
     *
     * scikit-learn は既定で L2 正則化を掛けるが、Smile の `lambda` の既定は 0 である。
     * 原著と比較できるよう、scikit-learn の既定 `C = 1.0` に対応する
     * `lambda = 1.0 / データ数` を明示的に渡す。
     */
    fun fitWithSmile(
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
        lambda: Double = 1.0 / FEATURES.size,
    ): LogisticRegression {
        val x = Array(features.size) { row ->
            DoubleArray(features[row].size) { features[row][it].toDouble() }
        }
        // Smile は二値と多値でメソッドが分かれている。二値は binomial
        return LogisticRegression.binomial(x, labels, lambda, 1e-6, 500)
    }

    /** Smile のモデルで全点を予測する */
    fun predictWithSmile(
        model: LogisticRegression,
        features: Array<IntArray> = FEATURES,
    ): List<Int> = features.map { row ->
        model.predict(DoubleArray(row.size) { row[it].toDouble() })
    }
}
