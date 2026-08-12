package ch08

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NaiveBayesTest {
    // 原著と同じ「lottery / sale / winning」を含むスパム判定
    private val documents = listOf(
        "lottery sale",
        "lottery winning",
        "winning lottery sale",
        "sale today",
        "meeting tomorrow",
        "project meeting",
        "lunch meeting today",
        "project deadline",
    )
    private val labels = listOf(1, 1, 1, 0, 0, 0, 0, 0)
    private val model = train(documents, labels)

    @Test
    fun `tokenize は小文字化して分割する`() {
        assertEquals(listOf("lottery", "winning", "today"), tokenize("Lottery WINNING today"))
    }

    @Test
    fun `train はクラスごとの文書数を数える`() {
        assertEquals(3, model.spamDocuments)
        assertEquals(5, model.hamDocuments)
        assertEquals(8, model.totalDocuments)
    }

    @Test
    fun `train は同じ単語を 1 文書につき 1 回だけ数える`() {
        val repeated = train(listOf("spam spam spam"), listOf(1))
        assertEquals(1, repeated.spamWordCounts["spam"])
    }

    @Test
    fun `事前確率はスパム文書の割合`() {
        assertEquals(3.0 / 8.0, priorSpamProbability(model), 1e-9)
    }

    @Test
    fun `wordSpamProbability はラプラス平滑化を使う`() {
        // lottery はスパム 3 件、ハム 0 件。平滑化なしなら 1.0 になってしまう
        assertEquals(3, model.spamWordCounts["lottery"])
        assertEquals(null, model.hamWordCounts["lottery"])
        // (3+1) / ((3+1) + (0+1)) = 0.8
        assertEquals(0.8, wordSpamProbability(model, "lottery"), 1e-9)
    }

    @Test
    fun `平滑化により確率が両端に張り付かない`() {
        model.vocabulary.forEach { word ->
            val probability = wordSpamProbability(model, word)
            assertTrue(probability > 0.0 && probability < 1.0, "$word=$probability")
        }
    }

    @Test
    fun `未知語は五分五分になる`() {
        assertEquals(0.5, wordSpamProbability(model, "unseen"), 1e-9)
    }

    @Test
    fun `未知語は予測を変えない`() {
        val known = predictProbability(model, "lottery")
        val withUnknown = predictProbability(model, "lottery zzzz qqqq")
        assertEquals(known, withUnknown, 1e-9)
    }

    @Test
    fun `スパム語は確率を上げる`() {
        assertTrue(predictProbability(model, "lottery winning") > 0.5)
        assertTrue(predictProbability(model, "project deadline") < 0.5)
    }

    @Test
    fun `スパム語が重なるほど確信が強まる`() {
        assertTrue(predictProbability(model, "lottery winning") > predictProbability(model, "lottery"))
    }

    @Test
    fun `確率は 0 から 1 の範囲に収まる`() {
        (documents + listOf("", "lottery lottery lottery winning sale")).forEach { document ->
            assertTrue(predictProbability(model, document) in 0.0..1.0)
        }
    }

    @Test
    fun `空の文書は事前確率を返す`() {
        assertEquals(priorSpamProbability(model), predictProbability(model, ""), 1e-9)
    }

    @Test
    fun `predict は閾値を使う`() {
        assertEquals(1, predict(model, "lottery winning"))
        assertEquals(0, predict(model, "lottery winning", threshold = 0.99))
    }

    @Test
    fun `分類器は訓練データを分離する`() {
        assertEquals(1.0, accuracy(model, documents, labels), 1e-9)
    }

    @Test
    fun `学習していないモデルは 0 を返す`() {
        val untrained = train(emptyList(), emptyList())
        assertEquals(0.0, priorSpamProbability(untrained), 1e-9)
        assertEquals(0.0, predictProbability(untrained, "lottery"), 1e-9)
    }
}
