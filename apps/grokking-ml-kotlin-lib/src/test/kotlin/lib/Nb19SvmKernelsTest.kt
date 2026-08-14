package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb19SvmKernels.accuracy
import lib.Nb19SvmKernels.fitKernel
import lib.Nb19SvmKernels.fitLinear
import lib.Nb19SvmKernels.load
import lib.Nb19SvmKernels.polynomialKernel
import lib.Nb19SvmKernels.rbfKernel

/**
 * 原著ノートブック #19 の再現テスト。
 *
 * 原著は 9 つの正解率を印刷する。Smile で再現できたのは **4 つ** だった。
 * 残りは解き方の違いで外れる。どこがどれだけ違うかを表にして記録する。
 */
class Nb19SvmKernelsTest {

    private val linear = load("linear")
    private val oneCircle = load("one_circle")
    private val twoCircles = load("two_circles")

    @Test
    fun `データセットは17で作ったもの`() {
        assertEquals(60, linear.size)
        assertEquals(110, oneCircle.size)
        assertEquals(220, twoCircles.size)
    }

    @Test
    fun `線形カーネルの正解率は原著と完全に一致する`() {
        // 原著の出力: Accuracy: 0.9333333333333333
        assertEquals(0.9333333333333333, accuracy(fitLinear(linear), linear), 1e-15)
    }

    @Test
    fun `極端なCでは原著から大きく外れる`() {
        // 原著は C=0.01 で 0.867、C=100 で 0.917。Smile はどちらも大きく下回る。
        // Smile の SVM は LASVM（オンライン近似解法）で、C が極端だと
        // 収束しきらない。scikit-learn は libsvm の SMO で厳密解に到達する
        assertEquals(0.5, accuracy(fitLinear(linear, c = 0.01), linear), 1e-15)
        assertEquals(0.6666666666666666, accuracy(fitLinear(linear, c = 100.0), linear), 1e-15)
    }

    @Test
    fun `多項式カーネルは原著に近いが一致しない`() {
        // 原著は degree=2 で 0.891、degree=4 で 0.900
        val degree2 = accuracy(fitKernel(oneCircle, polynomialKernel(oneCircle, 2)), oneCircle)
        val degree4 = accuracy(fitKernel(oneCircle, polynomialKernel(oneCircle, 4)), oneCircle)

        assertEquals(0.9, degree2, 1e-15)
        assertEquals(0.8181818181818182, degree4, 1e-15)
    }

    @Test
    fun `RBFのgammaを変えると原著の4つ中3つが完全に一致する`() {
        val scores = listOf(0.1, 1.0, 10.0, 100.0).map { gamma ->
            accuracy(fitKernel(twoCircles, rbfKernel(gamma)), twoCircles)
        }

        // 原著: 0.8772727272727273 / 0.9045454545454545 / 0.9636363636363636 / 0.990909090909091
        assertEquals(0.8636363636363636, scores[0], 1e-15) // ここだけ外れる
        assertEquals(0.9045454545454545, scores[1], 1e-15)
        assertEquals(0.9636363636363636, scores[2], 1e-15)
        assertEquals(0.990909090909091, scores[3], 1e-15)
    }

    @Test
    fun `gammaを上げるほど正解率が単調に上がる`() {
        // 原著と同じ傾向。値がずれても、章が示したい性質は再現できている
        val scores = listOf(0.1, 1.0, 10.0, 100.0).map { gamma ->
            accuracy(fitKernel(twoCircles, rbfKernel(gamma)), twoCircles)
        }

        assertEquals(scores.sorted(), scores)
    }

    @Test
    fun `種を固定すれば結果は再現する`() {
        // fitKernel は毎回 MathEx.setSeed を呼ぶ。呼ばないと実行ごとに変わる
        val first = accuracy(fitKernel(oneCircle, polynomialKernel(oneCircle, 2)), oneCircle)
        val second = accuracy(fitKernel(oneCircle, polynomialKernel(oneCircle, 2)), oneCircle)

        assertEquals(first, second, 0.0)
    }

    @Test
    fun `scaleはscikit-learnのgamma_scaleと同じ式`() {
        // 1 / (特徴量数 * 全体の分散)。特徴量は 2 列なので 1 / (2 * var)
        val scale = Nb19SvmKernels.scaleOf(oneCircle)
        val values = oneCircle.x.flatMap { it.asIterable() }
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size

        assertEquals(1.0 / (2 * variance), scale, 1e-15)
        assertTrue(scale > 0)
    }
}
