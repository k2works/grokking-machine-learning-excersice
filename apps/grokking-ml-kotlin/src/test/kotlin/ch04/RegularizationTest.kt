package ch04

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RegularizationTest {
    // y = 2x + 3 に小さなノイズを乗せたデータ
    private val features = listOf(-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2)
    private val labels = listOf(0.08, 0.32, 1.07, 1.63, 2.54, 3.11, 3.84, 3.95, 4.75, 5.12)

    @Test
    fun `polynomialFeatures は x の累乗を並べる`() {
        assertEquals(listOf(2.0, 4.0, 8.0), polynomialFeatures(2.0, 3))
    }

    @Test
    fun `predict はすべての次数を使う`() {
        // 1 + 2*2 + 3*4 = 17
        assertEquals(17.0, PolynomialModel(listOf(2.0, 3.0), 1.0).predict(2.0), 1e-9)
    }

    @Test
    fun `L1 の勾配は重みの符号を使う`() {
        assertEquals(0.1, regularizationGradient(5.0, Regularization.L1, 0.1), 1e-9)
        assertEquals(-0.1, regularizationGradient(-5.0, Regularization.L1, 0.1), 1e-9)
        assertEquals(0.0, regularizationGradient(0.0, Regularization.L1, 0.1), 1e-9)
    }

    @Test
    fun `L2 の勾配は重みに比例する`() {
        assertEquals(1.0, regularizationGradient(5.0, Regularization.L2, 0.1), 1e-9)
        assertEquals(-1.0, regularizationGradient(-5.0, Regularization.L2, 0.1), 1e-9)
    }

    @Test
    fun `正則化なしの勾配は 0`() {
        assertEquals(0.0, regularizationGradient(5.0, Regularization.NONE, 0.1), 1e-9)
    }

    @Test
    fun `squareTrick は正則化なしで誤差に比例して動く`() {
        // 予測 = 1 + 2*3 = 7、誤差 = 10 - 7 = 3
        val moved = squareTrick(PolynomialModel(listOf(2.0), 1.0), x = 3.0, y = 10.0, learningRate = 0.01)
        assertEquals(2.09, moved.weights[0], 1e-9) // 2 + 0.01 * 3 * 3
        assertEquals(1.03, moved.bias, 1e-9) // 1 + 0.01 * 3
    }

    @Test
    fun `squareTrick は L2 で重みを 0 に引き戻す`() {
        val model = PolynomialModel(listOf(2.0), 1.0)
        val plain = squareTrick(model, x = 3.0, y = 10.0, learningRate = 0.01)
        val regularized = squareTrick(
            model, x = 3.0, y = 10.0, learningRate = 0.01,
            kind = Regularization.L2, strength = 0.1,
        )
        assertTrue(regularized.weights[0] < plain.weights[0])
        // 2 + 0.01 * (3*3 - 0.1*2*2)
        assertEquals(2.086, regularized.weights[0], 1e-9)
    }

    @Test
    fun `squareTrick はバイアスを正則化しない`() {
        val model = PolynomialModel(listOf(2.0), 1.0)
        val plain = squareTrick(model, x = 3.0, y = 10.0, learningRate = 0.01)
        val regularized = squareTrick(
            model, x = 3.0, y = 10.0, learningRate = 0.01,
            kind = Regularization.L2, strength = 0.1,
        )
        assertEquals(plain.bias, regularized.bias, 1e-9)
    }

    @Test
    fun `trainTestSplit はデータを失わずに分割する`() {
        val split = trainTestSplit(features, labels, testRatio = 0.3, seed = 0)
        assertEquals(3, split.testFeatures.size)
        assertEquals(7, split.trainFeatures.size)
        assertEquals(features.sorted(), (split.trainFeatures + split.testFeatures).sorted())
        assertEquals(labels.sorted(), (split.trainLabels + split.testLabels).sorted())
    }

    @Test
    fun `trainTestSplit は特徴量とラベルの対応を保つ`() {
        val split = trainTestSplit(features, labels, testRatio = 0.3, seed = 0)
        val xs = split.trainFeatures + split.testFeatures
        val ys = split.trainLabels + split.testLabels
        xs.zip(ys).forEach { (x, y) ->
            assertEquals(labels[features.indexOf(x)], y, 1e-9)
        }
    }

    @Test
    fun `1 次モデルは直線的なデータに適合する`() {
        val model = polynomialRegression(features, labels, degree = 1)
        assertTrue(abs(model.weights[0] - 2.0) < 0.3, "weight=${model.weights[0]}")
        assertTrue(abs(model.bias - 3.0) < 0.3, "bias=${model.bias}")
        assertTrue(modelRmse(model, features, labels) < 0.3)
    }

    @Test
    fun `高次モデルは過学習する`() {
        val split = trainTestSplit(features, labels, testRatio = 0.3, seed = 0)
        val simple = polynomialRegression(split.trainFeatures, split.trainLabels, degree = 1)
        val complex = polynomialRegression(split.trainFeatures, split.trainLabels, degree = 5)

        // 次数を上げると訓練誤差は下がる
        assertTrue(
            modelRmse(complex, split.trainFeatures, split.trainLabels) <
                modelRmse(simple, split.trainFeatures, split.trainLabels),
        )
        // しかし汎化ギャップ（テスト誤差 - 訓練誤差）は広がる
        val simpleGap = modelRmse(simple, split.testFeatures, split.testLabels) -
            modelRmse(simple, split.trainFeatures, split.trainLabels)
        val complexGap = modelRmse(complex, split.testFeatures, split.testLabels) -
            modelRmse(complex, split.trainFeatures, split.trainLabels)
        assertTrue(complexGap > simpleGap, "simpleGap=$simpleGap complexGap=$complexGap")
    }

    @Test
    fun `L2 正則化はテスト誤差を改善する`() {
        val split = trainTestSplit(features, labels, testRatio = 0.3, seed = 0)
        val plain = polynomialRegression(split.trainFeatures, split.trainLabels, degree = 5)
        val regularized = polynomialRegression(
            split.trainFeatures, split.trainLabels, degree = 5,
            kind = Regularization.L2, strength = 0.01,
        )
        assertTrue(
            modelRmse(regularized, split.testFeatures, split.testLabels) <
                modelRmse(plain, split.testFeatures, split.testLabels),
        )
    }

    @Test
    fun `L1 正則化は不要な重みをほぼ 0 にする`() {
        val split = trainTestSplit(features, labels, testRatio = 0.3, seed = 0)
        val plain = polynomialRegression(split.trainFeatures, split.trainLabels, degree = 5)
        val regularized = polynomialRegression(
            split.trainFeatures, split.trainLabels, degree = 5,
            kind = Regularization.L1, strength = 0.01,
        )
        assertTrue(weightMagnitude(regularized) < weightMagnitude(plain))
        assertTrue(regularized.weights.count { abs(it) < 5e-3 } >= 2)
        assertEquals(0, plain.weights.count { abs(it) < 5e-3 })
    }
}
