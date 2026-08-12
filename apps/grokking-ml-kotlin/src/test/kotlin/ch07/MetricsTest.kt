package ch07

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MetricsTest {
    // 1000 人に 10 人が罹る病気を、全員「陰性」と判定するモデル
    private val sickLabels = List(10) { 1 } + List(990) { 0 }
    private val alwaysHealthy = List(1000) { 0 }
    private val sample = ConfusionMatrix(
        truePositives = 3,
        falsePositives = 1,
        falseNegatives = 2,
        trueNegatives = 4,
    )

    @Test
    fun `confusionMatrix は 4 つの場合を数える`() {
        val matrix = confusionMatrix(listOf(1, 1, 0, 0), listOf(1, 0, 1, 0))
        assertEquals(1, matrix.truePositives)
        assertEquals(1, matrix.falseNegatives)
        assertEquals(1, matrix.falsePositives)
        assertEquals(1, matrix.trueNegatives)
        assertEquals(4, matrix.total)
    }

    @Test
    fun `正解率が高くてもモデルが役立たずなことがある`() {
        val matrix = confusionMatrix(sickLabels, alwaysHealthy)
        // 990/1000 を当てているので正解率は 99%
        assertEquals(0.99, accuracy(matrix), 1e-9)
        // しかし病人を 1 人も見つけられていない
        assertEquals(0.0, recall(matrix), 1e-9)
    }

    @Test
    fun `precision は陽性予測の信頼度`() {
        // 陽性と予測した 4 件のうち 3 件が当たり
        assertEquals(0.75, precision(sample), 1e-9)
    }

    @Test
    fun `recall は陽性をどれだけ拾えたか`() {
        // 実際の陽性 5 件のうち 3 件を拾えた
        assertEquals(0.6, recall(sample), 1e-9)
    }

    @Test
    fun `precision と recall はトレードオフ`() {
        // 全員を陽性と予測すると再現率は 1.0 だが適合率は下がる
        val aggressive = confusionMatrix(sickLabels, List(1000) { 1 })
        assertEquals(1.0, recall(aggressive), 1e-9)
        assertEquals(0.01, precision(aggressive), 1e-9)
    }

    @Test
    fun `f1Score は調和平均`() {
        assertEquals(2 * 0.75 * 0.6 / (0.75 + 0.6), f1Score(sample), 1e-9)
    }

    @Test
    fun `f1Score は偏りを算術平均より強く罰する`() {
        // 適合率 1.0、再現率 0.1 のモデル
        val unbalanced = ConfusionMatrix(1, 0, 9, 90)
        val arithmeticMean = (precision(unbalanced) + recall(unbalanced)) / 2
        assertTrue(f1Score(unbalanced) < arithmeticMean)
    }

    @Test
    fun `fBetaScore は beta が大きいほど再現率を重視する`() {
        // 適合率 0.75、再現率 0.6
        assertTrue(fBetaScore(sample, beta = 2.0) < f1Score(sample))
        assertTrue(fBetaScore(sample, beta = 0.5) > f1Score(sample))
    }

    @Test
    fun `定義できない指標は 0 を返す`() {
        val empty = ConfusionMatrix(0, 0, 0, 0)
        assertEquals(0.0, accuracy(empty), 1e-9)
        assertEquals(0.0, precision(empty), 1e-9)
        assertEquals(0.0, recall(empty), 1e-9)
        assertEquals(0.0, f1Score(empty), 1e-9)
    }

    @Test
    fun `predictionsAtThreshold は閾値で 0 と 1 に分ける`() {
        val probabilities = listOf(0.9, 0.6, 0.4, 0.1)
        assertEquals(listOf(1, 1, 0, 0), predictionsAtThreshold(probabilities, 0.5))
        assertEquals(listOf(0, 0, 0, 0), predictionsAtThreshold(probabilities, 0.95))
        assertEquals(listOf(1, 1, 1, 1), predictionsAtThreshold(probabilities, 0.05))
    }

    @Test
    fun `閾値を下げると再現率が上がり適合率が下がる`() {
        val labels = listOf(1, 1, 0, 0)
        val probabilities = listOf(0.9, 0.4, 0.6, 0.1)
        val strict = confusionMatrix(labels, predictionsAtThreshold(probabilities, 0.8))
        val loose = confusionMatrix(labels, predictionsAtThreshold(probabilities, 0.3))
        assertTrue(recall(loose) > recall(strict))
        assertTrue(precision(loose) < precision(strict))
    }

    @Test
    fun `rocPoints は両端の角から始まり角で終わる`() {
        val points = rocPoints(listOf(1, 1, 0, 0), listOf(0.9, 0.6, 0.4, 0.1))
        assertEquals(0.0 to 0.0, points.first())
        assertEquals(1.0 to 1.0, points.last())
    }

    @Test
    fun `auc は完全な順位付けで 1`() {
        assertEquals(1.0, auc(listOf(1, 1, 0, 0), listOf(0.9, 0.8, 0.2, 0.1)), 1e-9)
    }

    @Test
    fun `auc は情報のない順位付けで 0_5`() {
        // 陽性が最上位と最下位に 1 つずつ
        assertEquals(0.5, auc(listOf(1, 0, 0, 1), listOf(0.8, 0.6, 0.4, 0.2)), 1e-9)
    }

    @Test
    fun `auc は正しく並んだ組の割合`() {
        // 陽性 2 件と陰性 2 件の組 4 通りのうち、3 通りで陽性が上位
        assertEquals(0.75, auc(listOf(1, 0, 1, 0), listOf(0.8, 0.6, 0.4, 0.2)), 1e-9)
    }

    @Test
    fun `auc は完全に逆転した順位付けで 0`() {
        assertEquals(0.0, auc(listOf(0, 0, 1, 1), listOf(0.9, 0.8, 0.2, 0.1)), 1e-9)
    }

    @Test
    fun `auc は閾値に依存しない`() {
        val labels = listOf(1, 1, 0, 0)
        val scaled = listOf(0.99, 0.98, 0.02, 0.01)
        val compressed = listOf(0.55, 0.54, 0.46, 0.45)
        assertEquals(auc(labels, scaled), auc(labels, compressed), 1e-9)
    }
}
