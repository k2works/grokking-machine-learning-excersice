package lib

import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb18CalculatingSimilarities.FEATURE_COUNT
import lib.Nb18CalculatingSimilarities.POINTS
import lib.Nb18CalculatingSimilarities.SIZE
import lib.Nb18CalculatingSimilarities.Y
import lib.Nb18CalculatingSimilarities.features
import lib.Nb18CalculatingSimilarities.fit
import lib.Nb18CalculatingSimilarities.similarity
import lib.Nb18CalculatingSimilarities.similarityMatrix
import lib.Nb18CalculatingSimilarities.svmRbfPrediction
import lib.Nb18CalculatingSimilarities.trainingPredictions

/**
 * 原著ノートブック #18 の再現テスト。
 *
 * 類似度も手書きの予測式も、**原著の出力と 16 桁まで一致する**。
 * 一方、SVM の係数は Smile が公開しないので比べられない（後述）。
 */
class Nb18CalculatingSimilaritiesTest {

    @Test
    fun `データセットは7点`() {
        // 原点と、その上下左右と斜め 2 点
        assertEquals(7, SIZE)
        assertEquals(listOf(0, 0, 0, 1, 1, 1, 1), Y.toList())
    }

    @Test
    fun `同じ点の類似度は1`() {
        // exp(0) = 1
        assertEquals(1.0, similarity(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 0.0)))
        assertEquals(1.0, similarity(doubleArrayOf(3.0, -2.0), doubleArrayOf(3.0, -2.0)))
    }

    @Test
    fun `類似度は距離の2乗で決まる`() {
        val origin = doubleArrayOf(0.0, 0.0)
        // exp(-||x-y||^2)。距離 1 なら exp(-1)
        assertEquals(exp(-1.0), similarity(origin, doubleArrayOf(1.0, 0.0)), 1e-15)
        // 距離 √2 なら exp(-2)
        assertEquals(exp(-2.0), similarity(origin, doubleArrayOf(1.0, 1.0)), 1e-15)
        // 原著のセルに残っている出力 1.522997974471263e-08 は (0,0) と (3,3) の類似度
        assertEquals(1.522997974471263e-08, similarity(origin, doubleArrayOf(3.0, 3.0)), 1e-23)
    }

    @Test
    fun `類似度は対称`() {
        val matrix = similarityMatrix()
        for (i in 0 until SIZE) {
            for (j in 0 until SIZE) {
                assertEquals(matrix[i][j], matrix[j][i], 0.0, "($i, $j)")
            }
        }
    }

    @Test
    fun `対角成分はすべて1`() {
        val matrix = similarityMatrix()
        for (i in 0 until SIZE) assertEquals(1.0, matrix[i][i])
    }

    @Test
    fun `類似度行列は原著の表と一致する`() {
        val matrix = similarityMatrix()

        // 原著の表の 1 行目: 1.000000 0.367879 0.367879 0.367879 0.367879 0.135335 ...
        assertEquals(0.36787944117144233, matrix[0][1], 1e-15)
        assertEquals(0.1353352832366127, matrix[0][5], 1e-15)
        // 2 行目の Sim4: 0.018316（距離の 2 乗が 4）
        assertEquals(0.01831563888873418, matrix[1][4], 1e-15)
        // 3 行目の Sim5: 0.006738（距離の 2 乗が 5）
        assertEquals(0.006737946999085467, matrix[2][5], 1e-15)
    }

    @Test
    fun `特徴量は9列になる`() {
        // 元の x1, x2 に類似度 7 列を足す
        assertEquals(9, FEATURE_COUNT)
        features().forEach { assertEquals(9, it.size) }
    }

    @Test
    fun `SVMは7点すべてを正しく分類する`() {
        // Smile の線形 SVM は **重みベクトルを公開しない**。
        // `SVM.fit` が返すのは無名の Classifier で、predict しかできない。
        // 原著の svm.coef_（9 個の係数）とは突き合わせられないので、
        // 代わりに分類結果で確かめる
        val model = fit()
        val predicted = features().map { if (model.predict(it) == 1) 1 else 0 }

        assertEquals(Y.toList(), predicted)
    }

    @Test
    fun `手書きの予測式は原著と桁まで一致する`() {
        // 原著の出力をそのまま並べたもの
        val expected = listOf(
            -0.7293294335267746,
            -0.9749464141121803,
            -0.9749464141121804,
            0.9884223081103513,
            0.9884223081103514,
            0.8650001793912898,
            0.8650001793912898,
        )

        assertEquals(expected, trainingPredictions())
    }

    @Test
    fun `予測の符号は正解ラベルと合う`() {
        // 原著のコメント「ラベルが 1 なら正、0 なら負になるはず」
        trainingPredictions().forEachIndexed { index, value ->
            assertEquals(Y[index] == 1, value > 0, "point $index")
        }
    }

    @Test
    fun `対称な2点は同じ値になる`() {
        // (0,1) と (1,0) は x1 と x2 を入れ替えた関係。データ全体もその入れ替えで
        // 不変なので、予測値も一致する
        assertEquals(
            svmRbfPrediction(doubleArrayOf(0.0, 1.0)),
            svmRbfPrediction(doubleArrayOf(1.0, 0.0)),
            1e-15,
        )
        assertEquals(
            svmRbfPrediction(doubleArrayOf(-1.0, 1.0)),
            svmRbfPrediction(doubleArrayOf(1.0, -1.0)),
        )
    }

    @Test
    fun `遠く離れた点の予測は0に近づく`() {
        // RBF は距離とともに指数的に減る。データから離れると判断できなくなる
        assertTrue(kotlin.math.abs(svmRbfPrediction(doubleArrayOf(10.0, 10.0))) < 1e-30)
    }

    @Test
    fun `点の並びは原著と同じ`() {
        assertEquals(listOf(0.0, 0.0), POINTS[0].toList())
        assertEquals(listOf(1.0, -1.0), POINTS[6].toList())
    }
}
