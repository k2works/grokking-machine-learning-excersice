package ch03

import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearRegressionTest {
    private val features = listOf(1.0, 2.0, 3.0, 5.0, 6.0, 7.0)
    private val labels = listOf(155.0, 197.0, 244.0, 356.0, 407.0, 448.0)

    @Test
    fun `predict は slope x rooms + intercept を返す`() {
        assertEquals(250.0, Model(50.0, 100.0).predict(3.0), 1e-9)
    }

    @Test
    fun `absoluteTrick は予測が低いとき点に近づく`() {
        val moved = absoluteTrick(Model(50.0, 100.0), rooms = 3.0, price = 300.0, learningRate = 0.01)
        assertEquals(50.03, moved.slope, 1e-9)
        assertEquals(100.01, moved.intercept, 1e-9)
    }

    @Test
    fun `absoluteTrick は予測が高いとき下げる`() {
        val moved = absoluteTrick(Model(50.0, 100.0), rooms = 3.0, price = 200.0, learningRate = 0.01)
        assertEquals(49.97, moved.slope, 1e-9)
        assertEquals(99.99, moved.intercept, 1e-9)
    }

    @Test
    fun `squareTrick の移動量は誤差に比例する`() {
        val moved = squareTrick(Model(50.0, 100.0), rooms = 3.0, price = 300.0, learningRate = 0.01)
        assertEquals(51.5, moved.slope, 1e-9)
        assertEquals(100.5, moved.intercept, 1e-9)
    }

    @Test
    fun `simpleTrick は absoluteTrick と同じ向きに動く`() {
        val model = Model(50.0, 100.0)
        val moved = simpleTrick(model, rooms = 3.0, price = 300.0, random = Random(0))
        assertTrue(moved.slope > model.slope)
        assertTrue(moved.intercept > model.intercept)
    }

    @Test
    fun `rmse は誤差ゼロのとき 0 を返す`() {
        assertEquals(0.0, rmse(listOf(1.0, 2.0), listOf(1.0, 2.0)), 1e-9)
    }

    @Test
    fun `rmse は二乗平均平方根を返す`() {
        assertEquals(sqrt(2.5), rmse(listOf(1.0, 2.0), listOf(2.0, 4.0)), 1e-9)
    }

    @Test
    fun `linearRegression は住宅データセットに適合する`() {
        val (model, errors) = linearRegression(features, labels, learningRate = 0.01, epochs = 1000, seed = 0)
        assertTrue(model.slope in 45.0..55.0, "slope=${model.slope}")
        assertTrue(model.intercept in 80.0..120.0, "intercept=${model.intercept}")
        assertTrue(modelRmse(model, features, labels) < 15.0)
        assertEquals(1000, errors.size)
        assertTrue(errors.last() < errors.first())
    }
}
