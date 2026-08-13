package lib

import java.math.BigDecimal
import java.math.MathContext
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV

/**
 * 原著ノートブック #07 `Chapter_08_Naive_Bayes/Coding_naive_Bayes.ipynb`。
 *
 * メール 5728 通からスパム判定器をナイーブベイズで作る。
 * **この回だけ原著が scikit-learn を使わず、pandas と辞書だけで書いている。**
 * だからライブラリの差が出ず、Kotlin でも原著の数値をそのまま再現できる。
 *
 * 原著は最後の確率計算で `np.compat.long` を使っており、これが浮動小数点数を
 * **整数に切り捨てる**。切り捨てないと原著の出力と 8 桁目から食い違うので、
 * その挙動もそのまま写してある。
 */
object Nb07NaiveBayes {

    /**
     * ある単語が、スパムとハムそれぞれ何通に現れたか。
     *
     * 原著は 1 から数え始める。ラプラス平滑化にあたり、
     * 「一度も見ていない側」の確率が 0 になるのを防ぐ。
     */
    data class WordCounts(var spam: Int = 1, var ham: Int = 1)

    /** 学習に使ったメール全体の統計 */
    data class Corpus(val total: Int, val spam: Int) {
        val ham: Int get() = total - spam

        /** 事前確率。何も情報が無いときにスパムと判断する確率 */
        val spamProbability: Double get() = spam.toDouble() / total
    }

    /** 単語ごとの出現数と、コーパス全体の統計 */
    data class NaiveBayesModel(val words: Map<String, WordCounts>, val corpus: Corpus)

    /** メールと、スパムかどうかのラベル */
    data class Emails(val texts: List<String>, val spam: IntArray)

    /**
     * メールのデータセットを読み込む。5728 通、うち 1368 通がスパム。
     *
     * このファイルは約 8.5 MB あるためリポジトリには含めていない。
     */
    fun loadEmails(): Emails {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("emails.csv").toFile())
        val texts = frame.getColumn("text").convertTo<String>().toList()
        val spam = frame.getColumn("spam").convertTo<Int>().toList()
        return Emails(texts, spam.toIntArray())
    }

    /**
     * メール本文を、重複を除いた小文字の単語集合にする。
     *
     * 原著は `list(set(text.lower().split()))`。**同じ単語が何度出ても 1 回** と
     * 数えるのがこの実装の前提である。
     *
     * `split()` は Python では空白の連続をまとめて区切るが、Kotlin の
     * `String.split(" ")` は空文字列を残す。`Regex("\\s+")` で分けたうえで
     * 空文字列を落とす必要がある。
     */
    fun processEmail(text: String): Set<String> =
        text.lowercase().split(WHITESPACE).filter { it.isNotEmpty() }.toSet()

    private val WHITESPACE = Regex("""\s+""")

    /** 単語ごとに、スパム・ハムそれぞれの出現通数を数える */
    fun train(emails: Emails): NaiveBayesModel {
        val words = HashMap<String, WordCounts>()
        for (i in emails.texts.indices) {
            for (word in processEmail(emails.texts[i])) {
                val counts = words.getOrPut(word) { WordCounts() }
                if (emails.spam[i] == 1) counts.spam += 1 else counts.ham += 1
            }
        }
        return NaiveBayesModel(
            words,
            Corpus(total = emails.texts.size, spam = emails.spam.count { it == 1 }),
        )
    }

    /**
     * 単語 1 つだけを見たときの、スパムである確率。
     *
     * ベイズの定理そのものではなく、単純に「その単語を含むメールのうち
     * スパムの割合」を返す。原著もそう書いている。
     */
    fun predictBayes(model: NaiveBayesModel, word: String): Double {
        val counts = model.words.getValue(word.lowercase())
        return counts.spam.toDouble() / (counts.spam + counts.ham)
    }

    /**
     * メール全体を見たときの、スパムである確率。
     *
     * 「単語の出現が互いに独立」と仮定して、単語ごとの尤度比を掛け合わせる。
     * 語彙にない単語は無視するので、知らない単語ばかりのメールは事前確率に落ちる。
     *
     * 最後に整数へ切り捨てているのは、原著の `np.compat.long` に合わせるため。
     * 切り捨てないと 8 桁目から数値が変わる。
     */
    fun predictNaiveBayes(model: NaiveBayesModel, email: String): Double {
        val corpus = model.corpus
        var productSpams = corpus.spam.toDouble()
        var productHams = corpus.ham.toDouble()

        for (word in processEmail(email)) {
            val counts = model.words[word] ?: continue
            productSpams *= counts.spam.toDouble() / corpus.spam * corpus.total
            productHams *= counts.ham.toDouble() / corpus.ham * corpus.total
        }

        // 原著の np.compat.long と同じ切り捨て。値は 10^20 を超えることがあるので
        // Long では溢れる。Python の int は多倍長なので BigInteger で受ける
        val truncatedSpams = BigDecimal(productSpams).toBigInteger()
        val truncatedHams = BigDecimal(productHams).toBigInteger()

        // Python は int / int を多倍長のまま計算してから float にする。
        // ここで toDouble() してから割ると桁が落ちるので、BigDecimal で割る
        return truncatedSpams.toBigDecimal()
            .divide((truncatedSpams + truncatedHams).toBigDecimal(), MathContext.DECIMAL128)
            .toDouble()
    }
}
