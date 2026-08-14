package lib

import kotlin.math.exp
import smile.classification.Classifier
import smile.classification.SVM

/**
 * 原著ノートブック #18 `Chapter_11_Support_Vector_Machines/Calculating_similarities.ipynb`。
 *
 * RBF カーネルの正体を **手で計算して見せる** 回である。
 *
 * 7 点の小さなデータに対して全対の類似度を計算し、7 列の特徴量を足してから
 * **線形** SVM を当てる。カーネルトリックを使わず、
 * 「カーネルとは特徴量を増やすことだ」を目に見える形にしている。
 */
object Nb18CalculatingSimilarities {

    /** 原著が予測に使う符号。ラベル 0 を -1 に読み替えたもの */
    val PREDICTION_SIGNS = intArrayOf(-1, -1, -1, 1, 1, 1, 1)

    /** 原著の 7 点。原点とその周りに 6 点が並ぶ */
    val X1 = doubleArrayOf(0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0)
    val X2 = doubleArrayOf(0.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0)
    val Y = intArrayOf(0, 0, 0, 1, 1, 1, 1)

    /** 点の数 */
    const val SIZE = 7

    /** 元の 2 列に類似度 7 列を足した特徴量の数 */
    const val FEATURE_COUNT = 9

    /** 座標の並び。原著の `data[['x1','x2']]` */
    val POINTS: Array<DoubleArray> = Array(SIZE) { doubleArrayOf(X1[it], X2[it]) }

    /**
     * 原著の類似度。RBF（ガウス）カーネルそのもの。
     *
     * `exp(-(x1-y1)^2 - (x2-y2)^2)` は `exp(-||x-y||^2)`、
     * つまり γ = 1 の RBF カーネルである。
     * 同じ点なら 1、離れるほど急速に 0 に近づく。
     */
    fun similarity(a: DoubleArray, b: DoubleArray): Double {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        return exp(-dx * dx - dy * dy)
    }

    /** 全対の類似度を並べた 7 × 7 の行列 */
    fun similarityMatrix(): Array<DoubleArray> =
        Array(SIZE) { i -> DoubleArray(SIZE) { j -> similarity(POINTS[i], POINTS[j]) } }

    /** SVM に渡す特徴量。x1・x2 に類似度 7 列を足した 9 列 */
    fun features(): Array<DoubleArray> {
        val matrix = similarityMatrix()
        return Array(SIZE) { row ->
            DoubleArray(FEATURE_COUNT) { column ->
                when (column) {
                    0 -> X1[row]
                    1 -> X2[row]
                    else -> matrix[column - 2][row]
                }
            }
        }
    }

    /**
     * **線形** SVM を、類似度を足した特徴量に当てる。
     *
     * `kernel='rbf'` を使わないのが要点である。カーネルは既に特徴量として
     * 展開済みなので、線形で足りる。
     *
     * Smile の `SVM.fit` は **ラベルを ±1 で要求する**（[#04][Nb04Perceptron] と同じ）。
     * 0 のままだと `Invalid label: 0` で落ちるので、-1 に読み替えて渡す。
     */
    fun fit(c: Double = 1.0, tolerance: Double = 1e-3): Classifier<DoubleArray> {
        val labels = IntArray(SIZE) { if (Y[it] == 1) 1 else -1 }
        return SVM.fit(features(), labels, c, tolerance)
    }

    /**
     * 原著が手で書いた予測式。
     *
     * 学習した SVM の係数ではなく、**ラベルの符号をそのまま重みにする**。
     * `similarity` の重み付き和が正なら 1、負なら 0 と読む。
     */
    fun svmRbfPrediction(newPoint: DoubleArray): Double =
        POINTS.indices.sumOf { similarity(newPoint, POINTS[it]) * PREDICTION_SIGNS[it] }

    /** 7 点それぞれに対する [svmRbfPrediction] の値 */
    fun trainingPredictions(): List<Double> = POINTS.map { svmRbfPrediction(it) }
}
