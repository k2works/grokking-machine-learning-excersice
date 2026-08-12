package ch05

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PerceptronTest {
    // 原著と同じ「悲しい／楽しい」文の分類データ
    // 特徴量 = (aack の出現数, beep の出現数)、ラベル = 1 が楽しい
    private val points = listOf(
        listOf(1.0, 0.0),
        listOf(0.0, 2.0),
        listOf(1.0, 1.0),
        listOf(1.0, 2.0),
        listOf(1.0, 3.0),
        listOf(2.0, 2.0),
        listOf(2.0, 3.0),
        listOf(3.0, 2.0),
    )
    private val labels = listOf(0, 0, 0, 0, 1, 1, 1, 1)

    @Test
    fun `score は重み付き和`() {
        // -4 + 1*1 + 2*2 = 1
        assertEquals(1.0, Perceptron(listOf(1.0, 2.0), -4.0).score(listOf(1.0, 2.0)), 1e-9)
    }

    @Test
    fun `predict はスコアの符号を使う`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        assertEquals(1, model.predict(listOf(1.0, 2.0)))
        assertEquals(0, model.predict(listOf(1.0, 1.0)))
    }

    @Test
    fun `predict は境界線上を正のクラスとして扱う`() {
        val model = Perceptron(listOf(1.0, 1.0), -2.0)
        assertEquals(0.0, model.score(listOf(1.0, 1.0)), 1e-9)
        assertEquals(1, model.predict(listOf(1.0, 1.0)))
    }

    @Test
    fun `perceptronTrick は正しく分類できた点では動かない`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        assertEquals(model, perceptronTrick(model, listOf(1.0, 2.0), label = 1))
    }

    @Test
    fun `perceptronTrick は誤分類した正の点に近づく`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        // 予測 0、ラベル 1 なので誤差 +1
        val moved = perceptronTrick(model, listOf(1.0, 1.0), label = 1, learningRate = 0.1)
        assertEquals(1.1, moved.weights[0], 1e-9) // 1 + 0.1 * 1 * 1
        assertEquals(2.1, moved.weights[1], 1e-9) // 2 + 0.1 * 1 * 1
        assertEquals(-3.9, moved.bias, 1e-9) // -4 + 0.1
    }

    @Test
    fun `perceptronTrick は誤分類した負の点から離れる`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        // 予測 1、ラベル 0 なので誤差 -1
        val moved = perceptronTrick(model, listOf(1.0, 2.0), label = 0, learningRate = 0.1)
        assertEquals(0.9, moved.weights[0], 1e-9) // 1 - 0.1 * 1
        assertEquals(1.8, moved.weights[1], 1e-9) // 2 - 0.1 * 2
        assertEquals(-4.1, moved.bias, 1e-9) // -4 - 0.1
    }

    @Test
    fun `perceptronError は正解のとき 0`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        assertEquals(0.0, perceptronError(model, listOf(1.0, 2.0), label = 1), 1e-9)
    }

    @Test
    fun `perceptronError は誤りのときスコアの絶対値`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        // スコア -1 で予測 0、ラベル 1 なので誤差は |-1| = 1
        assertEquals(1.0, perceptronError(model, listOf(1.0, 1.0), label = 1), 1e-9)
    }

    @Test
    fun `meanPerceptronError は全点の平均`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        val sample = listOf(listOf(1.0, 2.0), listOf(1.0, 1.0))
        assertEquals(0.5, meanPerceptronError(model, sample, listOf(1, 1)), 1e-9)
    }

    @Test
    fun `accuracy は正解した割合`() {
        val model = Perceptron(listOf(1.0, 2.0), -4.0)
        val sample = listOf(listOf(1.0, 2.0), listOf(1.0, 1.0))
        assertEquals(0.5, accuracy(model, sample, listOf(1, 1)), 1e-9)
    }

    @Test
    fun `perceptronAlgorithm はデータを分離する`() {
        val (model, errors) = perceptronAlgorithm(points, labels, learningRate = 0.01, epochs = 1000, seed = 0)
        assertEquals(1.0, accuracy(model, points, labels), 1e-9)
        assertEquals(1000, errors.size)
        // 初期モデル（重みもバイアスもすべて 0）はすべての点が境界線上にあるため誤差 0
        assertEquals(0.0, errors.first(), 1e-9)
        // 学習の途中では誤差が生じ、最終的に 0 へ戻る
        assertTrue(errors.max() > 0.0)
        assertEquals(0.0, errors.last(), 1e-9)
    }

    @Test
    fun `perceptronAlgorithm は分離可能なデータで誤差 0 に到達する`() {
        val (model, _) = perceptronAlgorithm(points, labels, learningRate = 0.01, epochs = 1000, seed = 0)
        assertEquals(0.0, meanPerceptronError(model, points, labels), 1e-9)
    }
}
