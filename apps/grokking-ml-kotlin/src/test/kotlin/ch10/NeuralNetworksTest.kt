package ch10

import ch06.logisticRegression
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import ch06.accuracy as logisticAccuracy

class NeuralNetworksTest {
    // XOR。直線では分けられない代表例
    private val xorPoints = listOf(
        listOf(0.0, 0.0),
        listOf(0.0, 1.0),
        listOf(1.0, 0.0),
        listOf(1.0, 1.0),
    )
    private val xorLabels = listOf(0, 1, 1, 0)

    @Test
    fun `sigmoidDerivative は出力から計算できる`() {
        // s'(x) = s(x) * (1 - s(x))。出力が 0.5 のとき最大の 0.25
        assertEquals(0.25, sigmoidDerivative(0.5), 1e-9)
        assertEquals(0.25, sigmoidDerivative(sigmoid(0.0)), 1e-9)
    }

    @Test
    fun `sigmoidDerivative は両端で消える`() {
        // 出力が 0 や 1 に近いと勾配がほぼ消える（勾配消失）
        assertTrue(sigmoidDerivative(0.999) < 0.002)
        assertTrue(sigmoidDerivative(0.001) < 0.002)
    }

    @Test
    fun `layer は重み付き和にシグモイドを適用する`() {
        val layer = Layer(listOf(listOf(1.0, 2.0)), listOf(-4.0))
        // -4 + 1*1 + 2*2 = 1
        assertEquals(sigmoid(1.0), layer.forward(listOf(1.0, 2.0)).first(), 1e-9)
    }

    @Test
    fun `layer は自身の形を報告する`() {
        val layer = Layer(listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), listOf(0.0, 0.0))
        assertEquals(3, layer.inputSize)
        assertEquals(2, layer.outputSize)
    }

    @Test
    fun `forwardAll はすべての層の出力を記録する`() {
        val activations = initialNetwork(listOf(2, 3, 1)).forwardAll(listOf(0.5, 0.5))
        assertEquals(3, activations.size) // 入力 + 隠れ層 + 出力層
        assertEquals(2, activations[0].size)
        assertEquals(3, activations[1].size)
        assertEquals(1, activations[2].size)
    }

    @Test
    fun `initialNetwork は指定した形を持つ`() {
        val model = initialNetwork(listOf(2, 4, 1))
        assertEquals(2, model.layers.size)
        assertEquals(2, model.layers[0].inputSize)
        assertEquals(4, model.layers[0].outputSize)
        assertEquals(4, model.layers[1].inputSize)
        assertEquals(1, model.layers[1].outputSize)
    }

    @Test
    fun `initialNetwork は再現可能`() {
        assertEquals(initialNetwork(listOf(2, 3, 1), seed = 7), initialNetwork(listOf(2, 3, 1), seed = 7))
        assertNotEquals(initialNetwork(listOf(2, 3, 1), seed = 7), initialNetwork(listOf(2, 3, 1), seed = 8))
    }

    @Test
    fun `predictProbability は 0 から 1 の範囲に収まる`() {
        val model = initialNetwork(listOf(2, 4, 1))
        xorPoints.forEach { assertTrue(model.predictProbability(it) in 0.0..1.0) }
    }

    @Test
    fun `backpropagate はその点の損失を下げる`() {
        val model = initialNetwork(listOf(2, 4, 1), seed = 1)
        val point = listOf(1.0, 0.0)
        val before = logLoss(model, point, 1)
        val after = logLoss(backpropagate(model, point, 1, learningRate = 0.5), point, 1)
        assertTrue(after < before)
    }

    @Test
    fun `backpropagate は形を保つ`() {
        val model = initialNetwork(listOf(2, 4, 1))
        val updated = backpropagate(model, listOf(1.0, 0.0), 1, learningRate = 0.5)
        assertEquals(model.layers.size, updated.layers.size)
        model.layers.zip(updated.layers).forEach { (original, layer) ->
            assertEquals(original.inputSize, layer.inputSize)
            assertEquals(original.outputSize, layer.outputSize)
        }
    }

    @Test
    fun `backpropagate は新しいネットワークを返す`() {
        val model = initialNetwork(listOf(2, 4, 1))
        assertNotEquals(model, backpropagate(model, listOf(1.0, 0.0), 1, learningRate = 0.5))
    }

    @Test
    fun `隠れ層のないモデルは XOR を解けない`() {
        // 第 6 章のロジスティック回帰
        val (model, _) = logisticRegression(xorPoints, xorLabels, learningRate = 0.5, epochs = 20000, seed = 0)
        assertTrue(logisticAccuracy(model, xorPoints, xorLabels) < 1.0)
    }

    @Test
    fun `隠れニューロンが 1 つでも XOR は解けない`() {
        val (model, _) = train(xorPoints, xorLabels, hiddenSize = 1, epochs = 20000, seed = 0)
        assertTrue(accuracy(model, xorPoints, xorLabels) < 1.0)
    }

    @Test
    fun `隠れ層があれば XOR を解ける`() {
        val (model, losses) = train(xorPoints, xorLabels, hiddenSize = 4, epochs = 20000, seed = 0)
        assertEquals(1.0, accuracy(model, xorPoints, xorLabels), 1e-9)
        assertTrue(losses.last() < losses.first())
    }

    @Test
    fun `学習後は自信のある予測になる`() {
        val (model, _) = train(xorPoints, xorLabels, hiddenSize = 4, epochs = 20000, seed = 0)
        assertTrue(model.predictProbability(listOf(0.0, 0.0)) < 0.1)
        assertTrue(model.predictProbability(listOf(0.0, 1.0)) > 0.9)
        assertTrue(model.predictProbability(listOf(1.0, 0.0)) > 0.9)
        assertTrue(model.predictProbability(listOf(1.0, 1.0)) < 0.1)
    }

    @Test
    fun `学習前の損失は五分五分あたりから始まる`() {
        val loss = meanLogLoss(initialNetwork(listOf(2, 4, 1), seed = 0), xorPoints, xorLabels)
        assertTrue(loss in 0.5..1.2, "loss=$loss")
    }

    @Test
    fun `学習はモデルを破壊的に変更しない`() {
        val original = initialNetwork(listOf(2, 4, 1), seed = 3)
        val snapshot = NeuralNetwork(original.layers.toList())
        backpropagate(original, listOf(1.0, 0.0), 1, learningRate = 0.5)
        assertEquals(snapshot, original)
    }
}
