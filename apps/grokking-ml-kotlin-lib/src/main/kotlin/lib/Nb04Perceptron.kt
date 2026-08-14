package lib

import kotlin.math.abs
import kotlin.random.Random
import smile.classification.Classifier
import smile.classification.SVM

/**
 * 原著ノートブック #04 `Chapter_05_Perceptron_Algorithm/Coding_perceptron_algorithm.ipynb`。
 *
 * 2 次元の 8 点を直線で 2 クラスに分ける。パーセプトロンのトリックを手で書いてから、
 * 同じ問題を Smile の [Perceptron] に解かせる。
 *
 * 原著はトリックを 2 通り書いており、**2 つ目は 1 つ目と挙動が違う**。
 * バイアスの更新が重みのループの内側にあり、特徴量の数だけ繰り返し適用されるためである。
 * 原著のセル出力もその挙動を前提にしているので、両方を実装して差が見えるようにした。
 */
object Nb04Perceptron {

    /** 原著が使う 8 点。aack と beep の出現回数を模した 2 次元の特徴量 */
    val FEATURES = arrayOf(
        intArrayOf(1, 0), intArrayOf(0, 2), intArrayOf(1, 1), intArrayOf(1, 2),
        intArrayOf(1, 3), intArrayOf(2, 2), intArrayOf(2, 3), intArrayOf(3, 2),
    )
    val LABELS = intArrayOf(0, 0, 0, 0, 1, 1, 1, 1)

    /** 分離直線。重みとバイアスの組 */
    data class Boundary(val weights: DoubleArray, val bias: Double) {
        /** 重み付き和にバイアスを足したもの。直線からの符号つき距離に比例する */
        fun score(features: IntArray): Double =
            bias + weights.indices.sumOf { weights[it] * features[it] }

        /** スコアをステップ関数に通した予測ラベル */
        fun predict(features: IntArray): Int = step(score(features))

        /** パーセプトロン誤差。当たっていれば 0、外れていればスコアの絶対値 */
        fun error(features: IntArray, label: Int): Double =
            if (predict(features) == label) 0.0 else abs(score(features))

        // data class に配列を持たせると equals が参照比較になるので明示的に実装する
        override fun equals(other: Any?): Boolean =
            other is Boundary && weights.contentEquals(other.weights) && bias == other.bias

        override fun hashCode(): Int = 31 * weights.contentHashCode() + bias.hashCode()
    }

    /** 学習の結果と途中経過 */
    data class TrainingLog(
        val boundary: Boundary,
        /** 各エポック開始時点の平均パーセプトロン誤差 */
        val errors: List<Double>,
    )

    /** ステップ関数。0 以上なら 1、そうでなければ 0 */
    fun step(x: Double): Int = if (x >= 0) 1 else 0

    /** 全点のパーセプトロン誤差の平均 */
    fun meanPerceptronError(
        boundary: Boundary,
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
    ): Double = features.indices.sumOf { boundary.error(features[it], labels[it]) } / features.size

    /**
     * 原著が最初に示すトリック。当たっていれば何もせず、外れたら向きを見て動かす。
     *
     * バイアスの更新はループの **外側** にあり、1 回だけ適用される。
     */
    fun perceptronTrickExplicit(
        boundary: Boundary,
        features: IntArray,
        label: Int,
        learningRate: Double = 0.05,
    ): Boundary {
        val pred = boundary.predict(features)
        if (pred == label) return boundary

        val weights = boundary.weights.copyOf()
        val bias = when {
            label == 1 && pred == 0 -> {
                for (i in weights.indices) weights[i] += features[i] * learningRate
                boundary.bias + learningRate
            }
            else -> {
                for (i in weights.indices) weights[i] -= features[i] * learningRate
                boundary.bias - learningRate
            }
        }
        return Boundary(weights, bias)
    }

    /**
     * 原著が「短く書いた版」として示すトリック。以降の学習ループはこちらを使う。
     *
     * `label - pred` が符号を持つので分岐が要らなくなる。ただし原著のコードでは
     * **バイアスの更新が重みのループの内側にある**。特徴量が 2 つなら学習率が
     * 2 回足され、[perceptronTrickExplicit] の 2 倍動く。
     *
     * 原著のセル出力（`[0.9, 1.85], -4.1`）はこの挙動を前提にしているので、
     * そのまま写している。
     */
    fun perceptronTrick(
        boundary: Boundary,
        features: IntArray,
        label: Int,
        learningRate: Double = 0.05,
    ): Boundary {
        val pred = boundary.predict(features)
        val weights = boundary.weights.copyOf()
        var bias = boundary.bias
        for (i in weights.indices) {
            weights[i] += (label - pred) * features[i] * learningRate
            bias += (label - pred) * learningRate
        }
        return Boundary(weights, bias)
    }

    /**
     * トリックを繰り返して分離直線を学習する。
     *
     * 原著は `np.random.seed(42)` を呼んでいるが、点の選択に使っているのは
     * **標準ライブラリの `random.randint`** で、こちらに種を与えていない。
     * 原著の出力 `([0.55, 0.25], -1.1)` は実行のたびに変わる値であり再現できない。
     */
    fun perceptronAlgorithm(
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
        learningRate: Double = 0.01,
        epochs: Int = 200,
        seed: Int = 0,
    ): TrainingLog {
        val random = Random(seed)
        var boundary = Boundary(DoubleArray(features[0].size) { 1.0 }, 0.0)
        val errors = ArrayList<Double>(epochs)

        repeat(epochs) {
            errors.add(meanPerceptronError(boundary, features, labels))
            val i = random.nextInt(features.size)
            boundary = perceptronTrick(boundary, features[i], labels[i], learningRate)
        }

        return TrainingLog(boundary, errors)
    }

    /**
     * 同じ問題を Smile に解かせる。
     *
     * **Smile にはパーセプトロンの実装が無い**（`smile.classification` に
     * `Perceptron` は存在しない）。線形分離できるデータに対して分離超平面を
     * 求めるという役割がいちばん近いのは線形カーネルの [SVM] なので、
     * それで代替する。得られる直線はパーセプトロンとは違うが、
     * 「ライブラリに任せると全点が正しく分類できる」という原著の主張は確かめられる。
     *
     * Smile の SVM はラベルを **-1 / +1** で要求する。scikit-learn は 0 / 1 を
     * そのまま受け取るので、ここで変換が要る。
     */
    fun fitWithSmile(
        features: Array<IntArray> = FEATURES,
        labels: IntArray = LABELS,
        regularization: Double = 1.0,
        tolerance: Double = 1e-4,
    ): Classifier<DoubleArray> {
        val x = Array(features.size) { row ->
            DoubleArray(features[row].size) { features[row][it].toDouble() }
        }
        val signed = IntArray(labels.size) { if (labels[it] == 1) 1 else -1 }
        return SVM.fit(x, signed, regularization, tolerance)
    }

    /** Smile のモデルで全点を予測し、ラベルを 0 / 1 に戻す */
    fun predictWithSmile(
        model: Classifier<DoubleArray>,
        features: Array<IntArray> = FEATURES,
    ): List<Int> = features.map { row ->
        val predicted = model.predict(DoubleArray(row.size) { row[it].toDouble() })
        if (predicted == 1) 1 else 0
    }
}
