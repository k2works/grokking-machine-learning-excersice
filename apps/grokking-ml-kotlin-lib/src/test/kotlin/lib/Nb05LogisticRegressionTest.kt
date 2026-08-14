package lib

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb05LogisticRegression.Boundary
import lib.Nb05LogisticRegression.FEATURES
import lib.Nb05LogisticRegression.LABELS
import lib.Nb05LogisticRegression.alternateLogLoss
import lib.Nb05LogisticRegression.alternateLogLossOriginal
import lib.Nb05LogisticRegression.fitWithSmile
import lib.Nb05LogisticRegression.logisticRegressionAlgorithm
import lib.Nb05LogisticRegression.logisticTrick
import lib.Nb05LogisticRegression.predictWithSmile
import lib.Nb05LogisticRegression.sigmoid
import lib.Nb05LogisticRegression.softRelu
import lib.Nb05LogisticRegression.totalLogLoss

/**
 * 原著ノートブック #05 の再現テスト。
 *
 * 手書きの学習ループは #04 と同じく種が与えられていないため再現できない。
 * あわせて、原著が示す「対数損失の別の書き方」が実際には対数損失と一致しないことを
 * テストで固定してある。
 */
class Nb05LogisticRegressionTest {

    private val start = Boundary(doubleArrayOf(1.0, 1.0), 0.0)

    @Test
    fun `データセットは原著と同じ`() {
        // #04 と似ているが、最後の 2 点が入れ替わっている
        assertEquals(
            listOf(listOf(1, 0), listOf(0, 2), listOf(1, 1), listOf(1, 2),
                listOf(1, 3), listOf(2, 2), listOf(3, 2), listOf(2, 3)),
            FEATURES.map { it.toList() },
        )
        assertEquals(listOf(0, 0, 0, 0, 1, 1, 1, 1), LABELS.toList())
    }

    @Test
    fun `シグモイドは0で0_5を返す`() {
        assertEquals(0.5, sigmoid(0.0), 1e-12)
        assertEquals(1.0, sigmoid(100.0), 1e-12)
        assertEquals(0.0, sigmoid(-100.0), 1e-12)
    }

    @Test
    fun `シグモイドの2つの書き方は一致する`() {
        // 原著が使う exp(x)/(1+exp(x)) と、教科書の 1/(1+exp(-x))
        for (x in listOf(-3.0, -0.5, 0.0, 1.7, 4.2)) {
            assertEquals(1.0 / (1.0 + exp(-x)), sigmoid(x), 1e-12)
        }
    }

    @Test
    fun `ソフト relu は relu をなめらかにしたもの`() {
        assertEquals(20.0, softRelu(20.0), 1e-8)
        assertEquals(0.0, softRelu(-20.0), 1e-8)
        assertEquals(ln(2.0), softRelu(0.0), 1e-12)
    }

    @Test
    fun `対数損失は当たっていても0にはならない`() {
        // FEATURES[4] = [1, 3] はラベル 1。スコア 4 なのでよく当たっている
        val confident = start.logLoss(FEATURES[4], 1)
        // FEATURES[0] = [1, 0] はラベル 0。スコア 1 なので外している
        val wrong = start.logLoss(FEATURES[0], 0)

        assertTrue(confident in 0.0..0.1)
        assertTrue(wrong > 1.0)
    }

    @Test
    fun `原著の別の書き方は対数損失と一致しない`() {
        // pred は 0〜1 の確率なので (pred - label) は ±1 にならない
        for (i in FEATURES.indices) {
            if (abs(start.score(FEATURES[i])) > 1e-9) {
                assertTrue(
                    abs(
                        alternateLogLossOriginal(start, FEATURES[i], LABELS[i]) -
                            start.logLoss(FEATURES[i], LABELS[i]),
                    ) > 1e-6,
                )
            }
        }
    }

    @Test
    fun `正しい別の書き方は対数損失と厳密に一致する`() {
        // ラベルが 0 なら +1、1 なら -1 を掛ける。つまり (1 - 2 * label)
        for (i in FEATURES.indices) {
            for (label in listOf(0, 1)) {
                assertEquals(
                    start.logLoss(FEATURES[i], label),
                    alternateLogLoss(start, FEATURES[i], label),
                    1e-12,
                )
            }
        }
    }

    @Test
    fun `ロジスティックトリックはバイアスを1回だけ動かす`() {
        // #04 の「短く書いた版」はバイアスをループの内側で更新していたが、
        // こちらは外側にあるので 1 回だけ適用される
        val updated = logisticTrick(start, FEATURES[0], 0, learningRate = 0.05)

        assertEquals((0 - start.prediction(FEATURES[0])) * 0.05, updated.bias, 1e-12)
    }

    @Test
    fun `確信を持って間違えた点ほど大きく動く`() {
        val close = logisticTrick(Boundary(doubleArrayOf(1.0, 1.0), -3.0), intArrayOf(1, 2), 1)
        val far = logisticTrick(Boundary(doubleArrayOf(1.0, 1.0), -10.0), intArrayOf(1, 2), 1)

        assertTrue(far.weights[0] > close.weights[0])
    }

    @Test
    fun `合計対数損失は各点の合計`() {
        val expected = FEATURES.indices.sumOf { start.logLoss(FEATURES[it], LABELS[it]) }

        assertEquals(expected, totalLogLoss(start), 1e-12)
    }

    @Test
    fun `学習は誤差を下げる`() {
        val result = logisticRegressionAlgorithm(epochs = 500)

        assertEquals(500, result.errors.size)
        assertTrue(result.errors.last() < result.errors.first())
    }

    @Test
    fun `学習後は全点を正しく分類できる`() {
        // 対数損失は当たっていても 0 にならないので、パーセプトロンより
        // 収束に時間がかかる。この乱数列では 20000 エポックで揃う
        val result = logisticRegressionAlgorithm(epochs = 20000)

        assertEquals(
            LABELS.toList(),
            FEATURES.map { if (result.boundary.prediction(it) >= 0.5) 1 else 0 },
        )
    }

    @Test
    fun `Smile のロジスティック回帰も全点を正しく分類する`() {
        // 原著 scikit-learn の出力: Logistic Regression Predictions: [0 0 0 0 1 1 1 1]
        assertEquals(LABELS.toList(), predictWithSmile(fitWithSmile()))
    }
}
