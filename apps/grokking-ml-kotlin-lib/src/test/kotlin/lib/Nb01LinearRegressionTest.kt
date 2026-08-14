package lib

import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb01LinearRegression.FEATURES
import lib.Nb01LinearRegression.LABELS
import lib.Nb01LinearRegression.Trick
import lib.Nb01LinearRegression.absoluteTrick
import lib.Nb01LinearRegression.fitWithSmile
import lib.Nb01LinearRegression.linearRegression
import lib.Nb01LinearRegression.rmse
import lib.Nb01LinearRegression.simpleTrick
import lib.Nb01LinearRegression.squareTrick

/**
 * 原著ノートブック #01 の再現テスト。
 *
 * 閉じた式で解く [fitWithSmile] は原著の数値と完全に一致する。トリックによる
 * 学習は乱数列が Python と違うので、収束先で検証する。
 */
class Nb01LinearRegressionTest {

    @Test
    fun `データセットは原著と同じ`() {
        assertEquals(listOf(1, 2, 3, 5, 6, 7), FEATURES.toList())
        assertEquals(listOf(155, 197, 244, 356, 407, 448), LABELS.toList())
    }

    @Test
    fun `二乗トリックは誤差に比例して動く`() {
        // 予測 0 + 1 * 2 = 2、実測 10 なので誤差は 8
        val line = squareTrick(0.0, 1.0, 2.0, 10.0, learningRate = 0.01)

        assertEquals(1.0 + 0.01 * 2 * 8, line.pricePerRoom, 1e-12)
        assertEquals(0.0 + 0.01 * 8, line.basePrice, 1e-12)
    }

    @Test
    fun `絶対トリックは誤差の大きさに依存しない`() {
        val small = absoluteTrick(0.0, 1.0, 2.0, 10.0, learningRate = 0.01)
        val large = absoluteTrick(0.0, 1.0, 2.0, 802.0, learningRate = 0.01)

        // data class なので構造的等価性でそのまま比べられる
        assertEquals(small, large)
    }

    @Test
    fun `シンプルトリックは予測を実測へ近づける`() {
        val line = simpleTrick(0.0, 1.0, 2.0, 10.0, Random(0))

        assertTrue(line.pricePerRoom > 1.0)
        assertTrue(line.basePrice > 0.0)
    }

    @Test
    fun `rmse は誤差の二乗平均平方根`() {
        // ラベル 1, 2, 3 に対して予測は一律 2。差は -1, 0, 1
        val labels = intArrayOf(1, 2, 3)

        assertEquals(sqrt(2.0 / 3.0), rmse(labels, 2.0), 1e-12)
    }

    @Test
    fun `学習の途中経過をエポック数だけ記録する`() {
        val result = linearRegression(epochs = 50)

        assertEquals(50, result.history.size)
        assertEquals(50, result.errors.size)
        assertTrue(result.errors.last() < result.errors.first())
    }

    @Test
    fun `二乗トリックは最小二乗解へ収束する`() {
        // 乱数列が Python と違うので原著の 51.044 / 91.594 には一致しない。
        // 閉じた式の解に十分近づくことで検証する
        val exact = fitWithSmile()
        val result = linearRegression(learningRate = 0.01, epochs = 10000, trick = Trick.SQUARE)

        assertEquals(exact.pricePerRoom, result.line.pricePerRoom, 1.0)
        assertEquals(exact.basePrice, result.line.basePrice, 1.0)
    }

    @Test
    fun `Smile の解は原著と同じ数値になる`() {
        // 原著 scikit-learn の出力
        //   Coefficient: [50.39285714]
        //   Intercept: 99.59523809523819
        val line = fitWithSmile()

        assertEquals(50.39285714, line.pricePerRoom, 1e-8)
        assertEquals(99.59523809523819, line.basePrice, 1e-8)
    }

    @Test
    fun `Smile の4部屋の予測は原著と同じ数値になる`() {
        // 原著の出力: Predicted label for feature 4: [301.16666667]
        assertEquals(301.16666667, fitWithSmile().predict(4.0), 1e-7)
    }
}
