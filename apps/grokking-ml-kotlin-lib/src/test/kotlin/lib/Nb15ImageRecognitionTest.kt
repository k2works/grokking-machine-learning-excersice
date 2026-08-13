package lib

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import lib.Nb15ImageRecognition.CLASSES
import lib.Nb15ImageRecognition.HIDDEN_UNITS
import lib.Nb15ImageRecognition.IMAGE_SIZE
import lib.Nb15ImageRecognition.INPUT_DIM
import lib.Nb15ImageRecognition.fit
import lib.Nb15ImageRecognition.loadMnist
import lib.Nb15ImageRecognition.parameterCount
import lib.Nb15ImageRecognition.testAccuracy

/**
 * 原著ノートブック #15 の再現テスト。
 *
 * `mnist.npz`（約 11 MB）はリポジトリに入れていない。未取得なら
 * ダウンロードせずスキップする（テストがネットワークに依存しないように）。
 */
class Nb15ImageRecognitionTest {

    private val available =
        Files.exists(Datasets.directory().resolve("mnist.npz"))

    private fun mnist(): Nb15ImageRecognition.Mnist {
        assumeTrue(available, "mnist.npz が未取得のためスキップ")
        return cached ?: loadMnist().also { cached = it }
    }

    @Test
    fun `画像は28かける28の784次元`() {
        assertEquals(28, IMAGE_SIZE)
        assertEquals(784, INPUT_DIM)
    }

    @Test
    fun `重みの総数は原著と同じ109386`() {
        // 原著の model.summary() が出す Total params
        assertEquals(109_386, parameterCount())
    }

    @Test
    fun `層の構成は原著と同じ`() {
        assertEquals(listOf(128, 64), HIDDEN_UNITS.toList())
        assertEquals(10, CLASSES)
    }

    @Test
    fun `データ件数は訓練6万テスト1万`() {
        val data = mnist()
        assertEquals(60_000, data.yTrain.size)
        assertEquals(10_000, data.yTest.size)
    }

    @Test
    fun `原著が例に挙げるラベルが一致する`() {
        val data = mnist()
        assertEquals(2, data.yTrain[5])
        assertEquals(4, data.yTest[4])
        // 原著が「モデルが間違える例」として挙げる 18 番目の正解
        assertEquals(3, data.yTest[18])
    }

    @Test
    fun `画素値は0から255の整数`() {
        val data = mnist()
        assertEquals(255.0, data.xTrain[0].max())
        assertEquals(0.0, data.xTrain[0].min())
    }

    @Test
    fun `1万件3エポックでも正解率は9割を超える`() {
        val data = mnist()
        val model = fit(data, epochs = 3, sampleSize = 10_000)

        // 原著は 6 万件 10 エポックで 0.942。小さく回しても近いところまで届く
        assertTrue(testAccuracy(model, data, sampleSize = 2_000) > 0.9)
    }

    private companion object {
        var cached: Nb15ImageRecognition.Mnist? = null
    }
}
