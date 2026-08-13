package lib

import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb06SentimentImdb.CountVectorizer
import lib.Nb06SentimentImdb.MAX_FEATURES
import lib.Nb06SentimentImdb.Reviews
import lib.Nb06SentimentImdb.englishStopWords
import lib.Nb06SentimentImdb.fit
import lib.Nb06SentimentImdb.loadReviews

/**
 * 原著ノートブック #06 の再現テスト。
 *
 * IMDB のデータセットは約 63 MB あり、リポジトリに含めていない。
 * **テストからは自動ダウンロードしない**。未取得なら該当テストを飛ばす。
 *
 * 語彙の作り方そのものは、小さなコーパスで常に検証している。
 */
class Nb06SentimentImdbTest {

    private val tinyReviews = Reviews(
        listOf(
            "This movie was wonderful and the acting was superb",
            "A dreadful waste of time, truly awful acting",
            "Wonderful direction and a superb cast",
            "Awful script and a dreadful waste",
        ),
        intArrayOf(1, 0, 1, 0),
    )

    private fun imdbAvailable(): Boolean =
        Files.exists(Datasets.directory().resolve("IMDB_Dataset.csv"))

    @Test
    fun `ストップワードは318語ある`() {
        // scikit-learn 1.9.0 の ENGLISH_STOP_WORDS から書き出したもの
        val stopWords = englishStopWords()

        assertEquals(318, stopWords.size)
        assertTrue("the" in stopWords)
        assertTrue("and" in stopWords)
        assertTrue("movie" !in stopWords)
    }

    @Test
    fun `ベクトル化は2文字以上の単語だけを拾う`() {
        // トークン正規表現は \b\w\w+\b なので 1 文字の語は落ちる
        val vectorizer = CountVectorizer(maxFeatures = 50)
        vectorizer.fitTransform(listOf("a wonderful movie", "I saw it"))

        assertTrue("wonderful" in vectorizer.vocabulary)
        assertTrue("a" !in vectorizer.vocabulary)
    }

    @Test
    fun `ベクトル化は小文字に揃える`() {
        val vectorizer = CountVectorizer(maxFeatures = 50)
        vectorizer.fitTransform(listOf("Wonderful WONDERFUL wonderful"))

        assertEquals(listOf("wonderful"), vectorizer.vocabulary)
    }

    @Test
    fun `ストップワードは語彙から除かれる`() {
        val vectorizer = CountVectorizer(maxFeatures = 50)
        vectorizer.fitTransform(listOf("the movie and the acting"))

        assertTrue("movie" in vectorizer.vocabulary)
        assertTrue("acting" in vectorizer.vocabulary)
        assertTrue("the" !in vectorizer.vocabulary)
        assertTrue("and" !in vectorizer.vocabulary)
    }

    @Test
    fun `maxFeatures は出現回数の上位を採る`() {
        val vectorizer = CountVectorizer(maxFeatures = 2)
        // rare は 1 回、common は 3 回、middle は 2 回
        vectorizer.fitTransform(listOf("common middle rare", "common middle", "common"))

        assertEquals(listOf("common", "middle"), vectorizer.vocabulary)
    }

    @Test
    fun `語彙は辞書順に並ぶ`() {
        // scikit-learn の get_feature_names_out() と同じ。添字が特徴量の添字になる
        val vectorizer = CountVectorizer(maxFeatures = 50)
        vectorizer.fitTransform(listOf("zebra apple mango apple zebra apple"))

        assertEquals(vectorizer.vocabulary.sorted(), vectorizer.vocabulary)
    }

    @Test
    fun `出現回数の行列は語彙の順に並ぶ`() {
        val vectorizer = CountVectorizer(maxFeatures = 50)
        val matrix = vectorizer.fitTransform(listOf("apple apple zebra", "zebra"))
        val appleIndex = vectorizer.vocabulary.indexOf("apple")
        val zebraIndex = vectorizer.vocabulary.indexOf("zebra")

        assertEquals(2.0, matrix[0].get(appleIndex))
        assertEquals(1.0, matrix[0].get(zebraIndex))
        // 疎行列なので 0 の要素は保持されない
        assertEquals(0.0, matrix[1].get(appleIndex))
        assertEquals(1, matrix[1].size())
    }

    @Test
    fun `小さなコーパスでも単語の重みが感情を反映する`() {
        val trained = fit(tinyReviews, maxFeatures = 20)
        val weights = trained.wordSentiments().toMap()

        // 肯定的なレビューにだけ出る語は正、否定的なレビューにだけ出る語は負
        assertTrue(weights.getValue("wonderful") > 0)
        assertTrue(weights.getValue("superb") > 0)
        assertTrue(weights.getValue("dreadful") < 0)
        assertTrue(weights.getValue("waste") < 0)
        // 両方に出る語は 0 に近い
        assertTrue(abs(weights.getValue("acting")) < abs(weights.getValue("wonderful")))
    }

    @Test
    fun `係数の数は語彙の数と一致する`() {
        val trained = fit(tinyReviews, maxFeatures = 20)

        assertEquals(trained.vocabulary.size, trained.model.coefficients().size - 1)
        assertEquals(trained.vocabulary.size, trained.wordSentiments().size)
    }

    @Test
    fun `IMDB の語彙は原著と同じ添字になる`() {
        if (!imdbAvailable()) {
            println("IMDB_Dataset.csv が未取得のためスキップします")
            return
        }

        val trained = fit(loadReviews())

        assertEquals(MAX_FEATURES, trained.vocabulary.size)
        // 自前のベクトル化が scikit-learn と同じ語彙を作れている証拠。
        // 原著の出力に現れる添字（1964 wonderfully / 1921 waste）と一致する
        assertEquals(1964, trained.vocabulary.indexOf("wonderfully"))
        assertEquals(1921, trained.vocabulary.indexOf("waste"))
    }

    @Test
    fun `IMDB の上位感情語は原著と同じ集合になる`() {
        if (!imdbAvailable()) {
            println("IMDB_Dataset.csv が未取得のためスキップします")
            return
        }

        val trained = fit(loadReviews())

        // 係数の細かい順序は正則化の掛け方で入れ替わるが、集合は一致する
        assertEquals(
            listOf(
                "brilliantly", "delightful", "excellent", "finest", "funniest",
                "gem", "subtle", "superb", "underrated", "wonderfully",
            ),
            trained.mostPositiveWords(10).sorted(),
        )
        assertEquals(
            listOf(
                "awful", "disappointment", "dreadful", "laughable", "poorly",
                "redeeming", "tedious", "unfunny", "waste", "worst",
            ),
            trained.mostNegativeWords(10).sorted(),
        )
        // 首位は原著と同じ
        assertEquals("wonderfully", trained.mostPositiveWords(1).first())
        assertEquals("waste", trained.mostNegativeWords(1).first())
    }
}
