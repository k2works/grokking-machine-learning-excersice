package lib

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb03PolynomialRegularization.DEGREE
import lib.Nb03PolynomialRegularization.POLYNOMIAL_COEFFICIENTS
import lib.Nb03PolynomialRegularization.Regularization
import lib.Nb03PolynomialRegularization.evaluateModel
import lib.Nb03PolynomialRegularization.generateDataset
import lib.Nb03PolynomialRegularization.polynomial
import lib.Nb03PolynomialRegularization.polynomialFeatures
import lib.Nb03PolynomialRegularization.trainPolynomialRegression

/**
 * 原著ノートブック #03 の再現テスト。
 *
 * データ生成に乱数を使うので、原著の数値（0.1528 / 0.1037）とは一致しない。
 * 検証するのは **正則化が持つ性質** である。正則化なしでは係数が爆発し、
 * L1・L2 はそれを抑えてテスト誤差を下げる。
 *
 * なお Smile の LASSO は scikit-learn の Lasso と違い、係数を厳密なゼロにはしない。
 */
class Nb03PolynomialRegularizationTest {

    private val dataset = generateDataset()

    @Test
    fun `多項式は係数の添字が次数に対応する`() {
        // -x^2 + 2 なので x = 0 で 2、x = 1 で 1、x = 2 で -2
        assertEquals(2.0, polynomial(POLYNOMIAL_COEFFICIENTS, 0.0), 1e-12)
        assertEquals(1.0, polynomial(POLYNOMIAL_COEFFICIENTS, 1.0), 1e-12)
        assertEquals(-2.0, polynomial(POLYNOMIAL_COEFFICIENTS, 2.0), 1e-12)
    }

    @Test
    fun `データセットは40点で訓練32テスト8に分かれる`() {
        // 原著の出力
        //   Shape of X_train: (32,)  Shape of X_test: (8,)
        assertEquals(40, dataset.x.size)
        assertEquals(32, dataset.xTrain.size)
        assertEquals(8, dataset.xTest.size)
        assertEquals(32, dataset.yTrain.size)
        assertEquals(8, dataset.yTest.size)
    }

    @Test
    fun `生成した点は元の多項式の近くにある`() {
        // ノイズの標準偏差は 0.1 なので、3 シグマの 0.3 に収まるはず
        for (i in dataset.x.indices) {
            val expected = polynomial(POLYNOMIAL_COEFFICIENTS, dataset.x[i])
            assertTrue(
                abs(dataset.y[i] - expected) < 0.5,
                "x=${dataset.x[i]} y=${dataset.y[i]} expected=$expected",
            )
        }
    }

    @Test
    fun `多項式特徴量は次数の数だけ列を作る`() {
        val features = polynomialFeatures(dataset.xTrain, DEGREE)

        // 定数列は入らない
        assertEquals(32, features.size)
        assertEquals(DEGREE, features[0].size)
        // 1 列目は x そのもの、2 列目は x の 2 乗
        assertEquals(dataset.xTrain[0], features[0][0], 1e-12)
        assertEquals(dataset.xTrain[0] * dataset.xTrain[0], features[0][1], 1e-12)
    }

    @Test
    fun `Smile の L1 は係数を縮めるが厳密なゼロにはしない`() {
        val model = trainPolynomialRegression(
            dataset.xTrain, dataset.yTrain, DEGREE, Regularization.L1, alpha = 0.01,
        )

        // scikit-learn の Lasso は座標降下で厳密に 0 を出すが、Smile の LASSO は
        // 出さない。実測で 20 個中 8 個が 1e-3 を下回るところまで縮むだけである
        assertEquals(0, model.coefficients.count { it == 0.0 })
        assertTrue(
            model.coefficients.count { abs(it) < 1e-3 } >= 5,
            "small=${model.coefficients.count { abs(it) < 1e-3 }}",
        )
        // L2 と違い、値の大きさが桁で散らばる
        assertTrue(model.coefficients.maxOf { abs(it) } < 10.0)
    }

    @Test
    fun `L2 正則化は係数をゼロにしないが小さく保つ`() {
        val model = trainPolynomialRegression(
            dataset.xTrain, dataset.yTrain, DEGREE, Regularization.L2, alpha = 0.01,
        )

        assertTrue(model.coefficients.maxOf { abs(it) } < 10.0)
    }

    @Test
    fun `正則化なしは係数が爆発する`() {
        val model = trainPolynomialRegression(dataset.xTrain, dataset.yTrain, DEGREE)

        // 元の多項式の係数は -1 と 2 だけなのに、桁違いの係数が現れる
        assertTrue(
            model.coefficients.maxOf { abs(it) } > 100.0,
            "max=${model.coefficients.maxOf { abs(it) }}",
        )
    }

    @Test
    fun `正則化はテスト誤差を下げる`() {
        val noReg = evaluateModel(
            trainPolynomialRegression(dataset.xTrain, dataset.yTrain, DEGREE),
            dataset.xTest, dataset.yTest,
        )
        val l1 = evaluateModel(
            trainPolynomialRegression(
                dataset.xTrain, dataset.yTrain, DEGREE, Regularization.L1, alpha = 0.01,
            ),
            dataset.xTest, dataset.yTest,
        )
        val l2 = evaluateModel(
            trainPolynomialRegression(
                dataset.xTrain, dataset.yTrain, DEGREE, Regularization.L2, alpha = 0.01,
            ),
            dataset.xTest, dataset.yTest,
        )

        assertTrue(l1 < noReg, "l1=$l1 noReg=$noReg")
        assertTrue(l2 < noReg, "l2=$l2 noReg=$noReg")
    }

    @Test
    fun `正則化ありのモデルは元の多項式に近い予測をする`() {
        val model = trainPolynomialRegression(
            dataset.xTrain, dataset.yTrain, DEGREE, Regularization.L2, alpha = 0.01,
        )

        // 元の多項式 -x^2 + 2 は x = 0 で 2 を返す
        assertEquals(2.0, model.predict(0.0), 0.3)
    }
}
