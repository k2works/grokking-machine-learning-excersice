package ch11

import ch05.perceptronAlgorithm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SvmTest {
    // 第 5 章と同じデータ。ただし SVM ではラベルを +1 / -1 で表す
    private val points = listOf(
        listOf(1.0, 0.0),
        listOf(0.0, 2.0),
        listOf(1.0, 1.0),
        listOf(1.0, 2.0),
        listOf(1.0, 3.0),
        listOf(2.0, 2.0),
        listOf(2.0, 3.0),
        listOf(3.0, 2.0),
    )
    private val labels = listOf(-1, -1, -1, -1, 1, 1, 1, 1)
    private val perceptronLabels = listOf(0, 0, 0, 0, 1, 1, 1, 1)

    // XOR。線形カーネルでは分けられない
    private val xorPoints = listOf(
        listOf(0.0, 0.0),
        listOf(0.0, 1.0),
        listOf(1.0, 0.0),
        listOf(1.0, 1.0),
    )
    private val xorLabels = listOf(-1, 1, 1, -1)

    @Test
    fun `linearKernel は内積`() {
        assertEquals(11.0, linearKernel(listOf(1.0, 2.0), listOf(3.0, 4.0)), 1e-9)
    }

    @Test
    fun `polynomialKernel は定義どおり`() {
        // (1*3 + 2*4 + 1)^2 = 144
        assertEquals(144.0, polynomialKernel(2, 1.0)(listOf(1.0, 2.0), listOf(3.0, 4.0)), 1e-9)
    }

    @Test
    fun `rbfKernel は同じ点で 1`() {
        assertEquals(1.0, rbfKernel(1.0)(listOf(1.0, 2.0), listOf(1.0, 2.0)), 1e-9)
    }

    @Test
    fun `rbfKernel は距離とともに減衰する`() {
        val kernel = rbfKernel(1.0)
        val near = kernel(listOf(0.0, 0.0), listOf(0.5, 0.0))
        val far = kernel(listOf(0.0, 0.0), listOf(3.0, 0.0))
        assertTrue(0.0 < far)
        assertTrue(far < near)
        assertTrue(near < 1.0)
    }

    @Test
    fun `predict は +1 と -1 を返す`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        assertEquals(1, model.predict(listOf(2.0, 2.0)))
        assertEquals(-1, model.predict(listOf(1.0, 1.0)))
    }

    @Test
    fun `hingeLoss はマージンの外で 0`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        assertEquals(0.0, hingeLoss(model, listOf(3.0, 2.0), 1), 1e-9)
    }

    @Test
    fun `hingeLoss はマージンの内側を罰する`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        // スコア 0.5、ラベル +1。正解だがマージンの内側なので 1 - 0.5 = 0.5
        assertEquals(0.5, hingeLoss(model, listOf(1.5, 2.0), 1), 1e-9)
    }

    @Test
    fun `hingeLoss は誤差とともに増える`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        // スコア -1、ラベル +1。1 - (-1) = 2
        assertEquals(2.0, hingeLoss(model, listOf(1.0, 1.0), 1), 1e-9)
    }

    @Test
    fun `svmStep は自信のある点をほとんど動かさない`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        val moved = svmStep(model, listOf(3.0, 2.0), 1, learningRate = 0.01, regularization = 0.1)
        // マージンの外なので、重みを縮める力しか働かない
        assertEquals(model.bias, moved.bias, 1e-9)
        assertTrue(moved.weights[0] < model.weights[0])
    }

    @Test
    fun `svmStep はマージンの内側の点を押し返す`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        val moved = svmStep(model, listOf(1.5, 2.0), 1, learningRate = 0.01, regularization = 0.0)
        assertTrue(moved.weights[0] > model.weights[0])
        assertTrue(moved.bias > model.bias)
    }

    @Test
    fun `SVM はデータを分離する`() {
        val (model, errors) = trainSvm(points, labels, epochs = 20000)
        assertEquals(1.0, accuracy(model, points, labels), 1e-9)
        assertTrue(errors.last() < errors.first())
    }

    @Test
    fun `SVM はパーセプトロンより広いマージンを残す`() {
        val (perceptron, _) = perceptronAlgorithm(points, perceptronLabels)
        val asSvm = SupportVectorMachine(perceptron.weights, perceptron.bias)
        val (svm, _) = trainSvm(points, labels, epochs = 20000, regularization = 0.01)

        // どちらも完全に分離できている
        assertEquals(1.0, accuracy(asSvm, points, labels), 1e-9)
        assertEquals(1.0, accuracy(svm, points, labels), 1e-9)
        // しかしパーセプトロンは分離できた時点で止まるため、余白がない
        assertEquals(0.0, asSvm.margin(points), 1e-9)
        assertTrue(svm.margin(points) > 0.5)
    }

    @Test
    fun `正則化が弱いほどマージンは広がる`() {
        val (loose, _) = trainSvm(points, labels, epochs = 20000, regularization = 0.01)
        val (tight, _) = trainSvm(points, labels, epochs = 20000, regularization = 0.1)
        assertTrue(loose.margin(points) > tight.margin(points))
    }

    @Test
    fun `svmError は損失と罰則の和`() {
        val model = SupportVectorMachine(listOf(1.0, 1.0), -3.0)
        val meanLoss = points.zip(labels).sumOf { (point, label) -> hingeLoss(model, point, label) } / points.size
        assertEquals(meanLoss, svmError(model, points, labels, regularization = 0.0), 1e-9)
        assertEquals(meanLoss + 0.2, svmError(model, points, labels, regularization = 0.1), 1e-9)
    }

    @Test
    fun `線形カーネルでは XOR を分けられない`() {
        val model = trainKernelClassifier(xorPoints, xorLabels, kernel = linearKernel)
        assertTrue(kernelAccuracy(model, xorPoints, xorLabels) < 1.0)
    }

    @Test
    fun `多項式カーネルは XOR を解く`() {
        val model = trainKernelClassifier(xorPoints, xorLabels, kernel = polynomialKernel(2))
        assertEquals(1.0, kernelAccuracy(model, xorPoints, xorLabels), 1e-9)
    }

    @Test
    fun `RBF カーネルは XOR を解く`() {
        val model = trainKernelClassifier(xorPoints, xorLabels, kernel = rbfKernel(1.0))
        assertEquals(1.0, kernelAccuracy(model, xorPoints, xorLabels), 1e-9)
    }

    @Test
    fun `カーネル分類器は訓練点を保持する`() {
        val model = trainKernelClassifier(xorPoints, xorLabels, kernel = rbfKernel())
        assertEquals(xorPoints.size, model.points.size)
        assertEquals(xorPoints.size, model.weights.size)
    }

    @Test
    fun `重みが増えるのは誤分類した点だけ`() {
        val model = trainKernelClassifier(xorPoints, xorLabels, kernel = rbfKernel(), epochs = 200)
        assertTrue(model.weights.any { it > 0.0 })
        assertTrue(model.weights.all { it >= 0.0 })
    }
}
