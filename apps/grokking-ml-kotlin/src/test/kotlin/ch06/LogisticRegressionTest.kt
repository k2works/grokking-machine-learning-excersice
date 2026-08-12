package ch06

import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class LogisticRegressionTest {
    // 第 5 章と同じ「悲しい／楽しい」文の分類データ
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
    private val model = LogisticClassifier(listOf(1.0, 2.0), -4.0)

    @Test
    fun `sigmoid は 0 で 0_5 を返す`() {
        assertEquals(0.5, sigmoid(0.0), 1e-9)
    }

    @Test
    fun `sigmoid は単調増加`() {
        assertTrue(sigmoid(-1.0) < sigmoid(0.0))
        assertTrue(sigmoid(0.0) < sigmoid(1.0))
    }

    @Test
    fun `sigmoid の値域は 0 から 1`() {
        listOf(-1000.0, -10.0, 0.0, 10.0, 1000.0).forEach {
            assertTrue(sigmoid(it) in 0.0..1.0, "sigmoid($it)=${sigmoid(it)}")
        }
    }

    @Test
    fun `sigmoid は大きな負の入力でも溢れない`() {
        // 素朴な 1/(1+exp(-x)) 実装は exp のオーバーフローで壊れる
        assertEquals(0.0, sigmoid(-1000.0), 1e-9)
        assertEquals(1.0, sigmoid(1000.0), 1e-9)
    }

    @Test
    fun `sigmoid は 0 を中心に対称`() {
        assertEquals(1.0, sigmoid(2.0) + sigmoid(-2.0), 1e-9)
    }

    @Test
    fun `predictProbability はスコアにシグモイドを適用する`() {
        // スコア = -4 + 1*1 + 2*2 = 1
        assertEquals(1.0, model.score(listOf(1.0, 2.0)), 1e-9)
        assertEquals(sigmoid(1.0), model.predictProbability(listOf(1.0, 2.0)), 1e-9)
    }

    @Test
    fun `predict は閾値を使う`() {
        assertEquals(1, model.predict(listOf(1.0, 2.0)))
        assertEquals(0, model.predict(listOf(1.0, 2.0), threshold = 0.8))
    }

    @Test
    fun `logLoss は自信を持って正解したとき小さい`() {
        val confident = LogisticClassifier(listOf(10.0, 20.0), -40.0)
        assertTrue(
            logLoss(confident, listOf(1.0, 2.0), label = 1) < logLoss(model, listOf(1.0, 2.0), label = 1),
        )
    }

    @Test
    fun `logLoss は自信を持って間違えたとき大きい`() {
        val confident = LogisticClassifier(listOf(10.0, 20.0), -40.0)
        assertTrue(logLoss(confident, listOf(1.0, 2.0), label = 0) > 5.0)
    }

    @Test
    fun `logLoss は定義どおり`() {
        val probability = sigmoid(1.0)
        assertEquals(-ln(probability), logLoss(model, listOf(1.0, 2.0), label = 1), 1e-9)
        assertEquals(-ln(1.0 - probability), logLoss(model, listOf(1.0, 2.0), label = 0), 1e-9)
    }

    @Test
    fun `meanLogLoss は全点の平均`() {
        val zero = LogisticClassifier(listOf(0.0, 0.0), 0.0)
        // すべての予測が 0.5 なので、どのラベルでも損失は -ln(0.5)
        assertEquals(-ln(0.5), meanLogLoss(zero, points, labels), 1e-9)
    }

    @Test
    fun `logisticTrick は正しく分類できた点でも動く`() {
        // 予測は 1（正解）だが、確率は 0.73 なのでまだ動く
        val moved = logisticTrick(model, listOf(1.0, 2.0), label = 1, learningRate = 0.1)
        assertNotEquals(model, moved)
        assertTrue(moved.weights[0] > model.weights[0])
    }

    @Test
    fun `logisticTrick は勾配どおりに動く`() {
        val error = 1 - sigmoid(1.0)
        val moved = logisticTrick(model, listOf(1.0, 2.0), label = 1, learningRate = 0.1)
        assertEquals(1.0 + 0.1 * error * 1.0, moved.weights[0], 1e-9)
        assertEquals(2.0 + 0.1 * error * 2.0, moved.weights[1], 1e-9)
        assertEquals(-4.0 + 0.1 * error, moved.bias, 1e-9)
    }

    @Test
    fun `logisticTrick は誤った正の予測から離れる`() {
        // 予測確率 0.73 に対しラベル 0 なので誤差は負
        val moved = logisticTrick(model, listOf(1.0, 2.0), label = 0, learningRate = 0.1)
        assertTrue(moved.weights[0] < model.weights[0])
        assertTrue(moved.bias < model.bias)
    }

    @Test
    fun `logisticRegression はデータを分離する`() {
        val (trained, losses) = logisticRegression(points, labels, learningRate = 0.1, epochs = 1000, seed = 0)
        assertEquals(1.0, accuracy(trained, points, labels), 1e-9)
        assertEquals(1000, losses.size)
        // パーセプトロン誤差と違い、対数損失は初期状態でも 0 にならない
        assertEquals(-ln(0.5), losses.first(), 1e-9)
        assertTrue(losses.last() < losses.first())
    }

    @Test
    fun `logisticRegression は確率として使える出力を返す`() {
        val (trained, _) = logisticRegression(points, labels, learningRate = 0.1, epochs = 1000, seed = 0)
        assertTrue(trained.predictProbability(listOf(3.0, 2.0)) > 0.5)
        assertTrue(trained.predictProbability(listOf(1.0, 0.0)) < 0.5)
    }
}
