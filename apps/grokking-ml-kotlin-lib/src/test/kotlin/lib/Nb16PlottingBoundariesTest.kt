package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb16PlottingBoundaries.ALIEN_DATASET
import lib.Nb16PlottingBoundaries.accuracy
import lib.Nb16PlottingBoundaries.axis
import lib.Nb16PlottingBoundaries.bias
import lib.Nb16PlottingBoundaries.classify
import lib.Nb16PlottingBoundaries.disagreementRatio
import lib.Nb16PlottingBoundaries.line1
import lib.Nb16PlottingBoundaries.line2
import lib.Nb16PlottingBoundaries.nnWithSigmoid
import lib.Nb16PlottingBoundaries.nnWithStep
import lib.Nb16PlottingBoundaries.predictions
import lib.Nb16PlottingBoundaries.regionRatio
import lib.Nb16PlottingBoundaries.sigmoid
import lib.Nb16PlottingBoundaries.step

/**
 * 原著ノートブック #16 の再現テスト。
 *
 * 原著は図しか出さないので、突き合わせる数値が無い。
 * そこで **境界を数値に直して** 検証する。8 点の判定は Python 版と完全に一致する。
 * 格子上の割合だけは 5 桁目でずれる（`ANDの領域は各直線より狭い` のコメント参照）。
 */
class Nb16PlottingBoundariesTest {

    @Test
    fun `データセットは8点`() {
        assertEquals(8, ALIEN_DATASET.size)
        // 幸せなエイリアンと不幸せなエイリアンが半分ずつ
        assertEquals(listOf(0, 0, 0, 0, 1, 1, 1, 1), ALIEN_DATASET.map { it.happy })
    }

    @Test
    fun `階段関数は0で1になる`() {
        // 原著は x >= 0。0 ちょうどは 1 の側に入る
        assertEquals(1, step(0.0))
        assertEquals(0, step(-1e-15))
    }

    @Test
    fun `シグモイドは0で半分`() {
        assertEquals(0.5, sigmoid(0.0), 1e-15)
        // 原著の書き方 exp(x)/(1+exp(x))。1/(1+exp(-x)) と同じ値になる
        assertEquals(0.8807970779778823, sigmoid(2.0), 1e-15)
    }

    @Test
    fun `1層目の2つの直線は対称`() {
        // 重みが (6, 10) と (10, 6) なので、aack と beep を入れ替えた関係
        assertEquals(listOf(0, 0, 0, 1, 1, 1, 1, 1), predictions(::line1))
        assertEquals(listOf(0, 1, 0, 0, 1, 1, 1, 1), predictions(::line2))
        // 領域の広さは同じ
        assertEquals(regionRatio(::line1), regionRatio(::line2), 1e-15)
    }

    @Test
    fun `1層目だけでは8点を分けられない`() {
        // 直線 1 本では 1 点ずつ間違える
        assertEquals(0.875, accuracy(::line1))
        assertEquals(0.875, accuracy(::line2))
    }

    @Test
    fun `バイアスは入力を見ない`() {
        assertEquals(List(8) { 1 }, predictions(::bias))
        assertEquals(1.0, regionRatio(::bias))
    }

    @Test
    fun `2層目はANDになっている`() {
        // 1 層目の出力は 0 か 1。その和が 1.5 以上になるのは両方 1 のときだけ
        assertEquals(1.0, nnWithStep(1.0, 1.0))
        assertEquals(0.0, nnWithStep(2.0, 0.0)) // line2 だけが 1
        assertEquals(0.0, nnWithStep(0.0, 2.0)) // line1 だけが 1
    }

    @Test
    fun `階段関数のネットワークは8点すべて正解`() {
        assertEquals(listOf(0, 0, 0, 0, 1, 1, 1, 1), predictions(::nnWithStep))
        assertEquals(1.0, accuracy(::nnWithStep))
    }

    @Test
    fun `シグモイド版は1点だけ外す`() {
        // aack=1 beep=1 の点だけ 0 と答える。正解は 1
        assertEquals(listOf(0, 0, 0, 0, 0, 1, 1, 1), predictions(::nnWithSigmoid))
        assertEquals(0.875, accuracy(::nnWithSigmoid))
    }

    @Test
    fun `外した点は判定の境目にある`() {
        // 0.4905 で、しきい値 0.5 をわずかに下回る
        assertEquals(0.4905304218, nnWithSigmoid(1.0, 1.0), 1e-9)
        assertEquals(0, classify(::nnWithSigmoid, 1.0, 1.0))
    }

    @Test
    fun `シグモイドは出力が飽和する`() {
        // 内側のシグモイドが 1 に近づくので、外側は sigmoid(0.5) で頭打ちになる
        assertEquals(0.6224593117, nnWithSigmoid(2.0, 2.0), 1e-9)
        assertEquals(0.6224593312, nnWithSigmoid(3.0, 3.0), 1e-9)
        assertTrue(nnWithSigmoid(3.0, 3.0) - nnWithSigmoid(2.0, 2.0) < 1e-7)
    }

    @Test
    fun `格子は原著と同じ700刻み`() {
        // np.arange(-0.5, 3, 0.005) と同じ
        assertEquals(700, axis().size)
        assertEquals(-0.5, axis().first(), 1e-15)
        // こちらは `min + i * step` なのでちょうど 0.0 になる。
        // NumPy の `arange` は 4.44e-16 になる
        assertEquals(0.0, axis()[100], 0.0)
    }

    @Test
    fun `2つのネットワークの境界はほぼ重なる`() {
        // 図では見分けが付かないが、格子の 0.48% だけ判定が食い違う
        assertEquals(0.0048, disagreementRatio(::nnWithStep, ::nnWithSigmoid), 1e-4)
    }

    @Test
    fun `ANDの領域は各直線より狭い`() {
        // Python 版は 0.5544327、Kotlin 版は 0.5543918。**格子の作り方が違う**。
        // NumPy の `arange` は誤差を累積するので、`-0.5 + 100 * 0.005` が
        // 0.0 ではなく 4.44e-16 になる。境界ちょうどの点の判定がそこで分かれ、
        // 49 万点のうち約 20 点だけ食い違う
        assertEquals(0.5544, regionRatio(::nnWithStep), 1e-4)
        assertTrue(regionRatio(::nnWithStep) < regionRatio(::line1))
    }
}
