package lib

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb07NaiveBayes.Emails
import lib.Nb07NaiveBayes.WordCounts
import lib.Nb07NaiveBayes.loadEmails
import lib.Nb07NaiveBayes.predictBayes
import lib.Nb07NaiveBayes.predictNaiveBayes
import lib.Nb07NaiveBayes.processEmail
import lib.Nb07NaiveBayes.train

/**
 * 原著ノートブック #07 の再現テスト。
 *
 * この回は原著が scikit-learn を使わないので、**すべての数値を再現できる**。
 * `np.compat.long` による切り捨てまで写しているので 16 桁まで一致する。
 *
 * `emails.csv` は約 8.5 MB あり、リポジトリに含めていない。
 * テストからは自動ダウンロードせず、未取得ならスキップする。
 */
class Nb07NaiveBayesTest {

    private val tinyEmails = Emails(
        listOf("win lottery now", "lottery lottery lottery", "meeting at noon", "lunch meeting"),
        intArrayOf(1, 1, 0, 0),
    )

    private fun emailsAvailable(): Boolean =
        Files.exists(Datasets.directory().resolve("emails.csv"))

    @Test
    fun `メールは小文字の単語集合になる`() {
        // 原著は list(set(text.lower().split()))。同じ単語は 1 回しか数えない
        assertEquals(setOf("lottery", "the", "win"), processEmail("Win WIN the lottery"))
    }

    @Test
    fun `空白の連続はまとめて区切る`() {
        // Python の split() は空白の連続をまとめるが、Kotlin の split(" ") は
        // 空文字列を残す。Regex で分けて空文字列を落とす必要がある
        assertEquals(setOf("win", "lottery"), processEmail("win   lottery"))
    }

    @Test
    fun `出現数は1から数え始める`() {
        // ラプラス平滑化。一度も見ていない側の確率が 0 にならないようにする
        assertEquals(WordCounts(spam = 1, ham = 1), WordCounts())
    }

    @Test
    fun `小さなコーパスで出現通数を数える`() {
        val trained = train(tinyEmails)

        // lottery はスパム 2 通に出る。1 から数え始めるので spam は 3
        assertEquals(WordCounts(spam = 3, ham = 1), trained.words["lottery"])
        assertEquals(WordCounts(spam = 1, ham = 3), trained.words["meeting"])
    }

    @Test
    fun `同じ単語が何度出ても1通と数える`() {
        // 2 通目は "lottery lottery lottery" だが、1 通ぶんしか数えない
        assertEquals(3, train(tinyEmails).words.getValue("lottery").spam)
    }

    @Test
    fun `語彙にない単語は事前確率を返す`() {
        // スパム 2 通 / 全 4 通
        assertEquals(0.5, predictNaiveBayes(train(tinyEmails), "zzzz"), 1e-12)
    }

    @Test
    fun `データセットは原著と同じ規模`() {
        if (!emailsAvailable()) return
        val model = train(loadEmails())

        // 原著の出力
        //   Number of emails: 5728 / Number of spam emails: 1368
        //   Probability of spam: 0.2388268156424581
        assertEquals(5728, model.corpus.total)
        assertEquals(1368, model.corpus.spam)
        assertEquals(0.2388268156424581, model.corpus.spamProbability, 1e-16)
    }

    @Test
    fun `単語ごとの出現数と予測は原著と同じ`() {
        if (!emailsAvailable()) return
        val model = train(loadEmails())

        // 原著の出力
        //   model['lottery'] -> {'spam': 9, 'ham': 1}
        //   model['sale']    -> {'spam': 39, 'ham': 42}
        //   predict_bayes('lottery') -> 0.9 / ('sale') -> 0.48148148148148145
        assertEquals(WordCounts(spam = 9, ham = 1), model.words["lottery"])
        assertEquals(WordCounts(spam = 39, ham = 42), model.words["sale"])
        assertEquals(0.9, predictBayes(model, "lottery"), 1e-16)
        assertEquals(0.48148148148148145, predictBayes(model, "sale"), 1e-16)
    }

    @Test
    fun `メール全体の予測は原著と同じ`() {
        if (!emailsAvailable()) return
        val model = train(loadEmails())

        // 原著のセル出力をそのまま期待値にしている
        val expected = listOf(
            "lottery sale" to 0.9638144992048691,
            "Hi mom how are you" to 0.12554358867164464,
            "meet me at the lobby of the hotel at nine am" to 6.964603508395961e-05,
            "enter the lottery to win three million dollars" to 0.9995234218677428,
            "buy cheap lottery easy money now" to 0.999973472265966,
            "Grokking Machine Learning by Luis Serrano" to 0.4197107645488719,
            "asdfgh" to 0.2388268156424581,
        )

        for ((email, value) in expected) {
            assertEquals(value, predictNaiveBayes(model, email), value * 1e-14, email)
        }
    }

    @Test
    fun `知らない単語を足しても結果は変わらない`() {
        if (!emailsAvailable()) return
        val model = train(loadEmails())

        assertEquals(
            predictNaiveBayes(model, "Hi mom how are you"),
            predictNaiveBayes(model, "Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd"),
            1e-16,
        )
    }

    @Test
    fun `切り捨てを外すと原著と食い違う`() {
        if (!emailsAvailable()) return
        val model = train(loadEmails())
        val corpus = model.corpus

        // 切り捨てなしで同じ計算をすると 0.9638144470140118 になり、
        // 原著の 0.9638144992048691 と 8 桁目から分かれる
        var spams = corpus.spam.toDouble()
        var hams = corpus.ham.toDouble()
        for (word in listOf("lottery", "sale")) {
            val counts = model.words.getValue(word)
            spams *= counts.spam.toDouble() / corpus.spam * corpus.total
            hams *= counts.ham.toDouble() / corpus.ham * corpus.total
        }
        val withoutTruncation = spams / (spams + hams)

        assertEquals(0.9638144470140118, withoutTruncation, 1e-15)
        assertTrue(Math.abs(withoutTruncation - 0.9638144992048691) > 1e-9)
    }
}
