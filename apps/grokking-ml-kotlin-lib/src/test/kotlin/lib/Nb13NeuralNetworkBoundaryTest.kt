package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb13NeuralNetworkBoundary.EPOCHS
import lib.Nb13NeuralNetworkBoundary.HIDDEN_UNITS
import lib.Nb13NeuralNetworkBoundary.ORIGINAL_EPOCHS
import lib.Nb13NeuralNetworkBoundary.accuracy
import lib.Nb13NeuralNetworkBoundary.arange
import lib.Nb13NeuralNetworkBoundary.boundaryChangesPerRow
import lib.Nb13NeuralNetworkBoundary.decisionGrid
import lib.Nb13NeuralNetworkBoundary.fit
import lib.Nb13NeuralNetworkBoundary.loadCircle

/**
 * 原著ノートブック #13 の再現テスト。
 *
 * Smile の [smile.classification.MLP] には Dropout が無く、最適化も単純な
 * 確率的勾配降下なので、原著の 100 エポックでは学習が進まない。
 * 2000 エポックで原著と同程度の正解率に届く。
 */
class Nb13NeuralNetworkBoundaryTest {

    private val data = loadCircle()

    @Test
    fun `データセットは110点`() {
        assertEquals(110, data.size)
        assertEquals(2, data.x[0].size)
    }

    @Test
    fun `ラベルは偏っている`() {
        // 84 対 26。円の内側が少数派になる
        assertEquals(26, data.y.sum())
        assertEquals(84, data.size - data.y.sum())
    }

    @Test
    fun `ネットワークの隠れ層は原著と同じ`() {
        // Dense(128) と Dense(64)
        assertEquals(listOf(128, 64), HIDDEN_UNITS.toList())
    }

    @Test
    fun `原著の100エポックでは学習が進まない`() {
        // 常に多数派（0）を答えるだけの 84/110 = 0.7636 にとどまる。
        // Keras の Adam に対し Smile は単純な確率的勾配降下なので収束が遅い
        val shallow = fit(data, ORIGINAL_EPOCHS)

        assertEquals(84.0 / 110.0, accuracy(shallow, data), 1e-12)
    }

    @Test
    fun `2000エポックなら原著と同程度の正解率になる`() {
        // 原著（Keras・100 エポック）は 0.88 前後。Smile は 2000 エポックで 0.89
        assertTrue(accuracy(fit(data, EPOCHS), data) > 0.85)
    }

    @Test
    fun `学習後は多数派より良い予測をする`() {
        // 常に 0 と答えるだけで 84/110 = 0.764 になる。それを上回る必要がある
        assertTrue(accuracy(fit(data, EPOCHS), data) > 84.0 / 110.0)
    }

    @Test
    fun `arange は終端を含まない`() {
        assertEquals(listOf(0.0, 0.5, 1.0, 1.5), arange(0.0, 2.0, 0.5).toList())
    }

    @Test
    fun `格子の予測は0か1しかない`() {
        val grid = decisionGrid(fit(data, EPOCHS), data)

        assertEquals(setOf(0, 1), grid.flatMap { it.toList() }.toSet())
    }

    @Test
    fun `境界は行ごとに変わる`() {
        // 決定木（#10）の境界は軸に平行で、切り替わる位置が行によらなかった。
        // ニューラルネットワークは曲線を引けるので、行ごとに変わる
        val changes = boundaryChangesPerRow(decisionGrid(fit(data, EPOCHS), data))

        assertTrue(changes.toSet().size > 1, "changes=$changes")
    }

    @Test
    fun `境界は閉じた形になる`() {
        // 円形のデータなので、内側を囲む境界ができる。
        // 少なくとも 1 行は「外・内・外」と 2 回切り替わる
        val changes = boundaryChangesPerRow(decisionGrid(fit(data, EPOCHS), data))

        assertTrue(changes.max() >= 2, "changes=$changes")
    }

    @Test
    fun `種を固定すれば同じ結果になる`() {
        // Smile の MLP は重みの初期化に乱数を使う。MathEx.setSeed で固定できる
        assertEquals(accuracy(fit(data, 200, seed = 7), data), accuracy(fit(data, 200, seed = 7), data), 1e-12)
    }
}
