/**
 * 第 8 章: ナイーブベイズ。
 * 「単語が独立に出現する」という（実際には正しくない）仮定を置くことで、
 * ベイズの定理による分類を単なる掛け算に単純化する。
 */
package ch08

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/** 文章を小文字の単語列に分解する。 */
fun tokenize(text: String): List<String> =
    text.lowercase().split(" ").filter { it.isNotEmpty() }

/** ナイーブベイズ分類器。単語ごとの出現回数からスパム確率を求める。 */
data class NaiveBayesClassifier(
    val spamWordCounts: Map<String, Int>,
    val hamWordCounts: Map<String, Int>,
    val spamDocuments: Int,
    val hamDocuments: Int,
) {
    val totalDocuments: Int get() = spamDocuments + hamDocuments

    val vocabulary: Set<String> get() = spamWordCounts.keys + hamWordCounts.keys
}

/** 文書とラベル（1 がスパム）から分類器を学習する。 */
fun train(documents: List<String>, labels: List<Int>): NaiveBayesClassifier {
    // 同じ単語が何度出ても 1 文書につき 1 回だけ数える（ベルヌーイ型）
    val byLabel = documents.zip(labels).groupBy({ it.second }, { tokenize(it.first).toSet() })
    fun countWords(label: Int): Map<String, Int> =
        byLabel[label].orEmpty().flatten().groupingBy { it }.eachCount()
    return NaiveBayesClassifier(
        spamWordCounts = countWords(1),
        hamWordCounts = countWords(0),
        spamDocuments = byLabel[1].orEmpty().size,
        hamDocuments = byLabel[0].orEmpty().size,
    )
}

/** その単語を含む文書がスパムである確率。ラプラス平滑化つき。 */
fun wordSpamProbability(
    model: NaiveBayesClassifier,
    word: String,
    smoothing: Double = 1.0,
): Double {
    val spam = (model.spamWordCounts[word] ?: 0) + smoothing
    val ham = (model.hamWordCounts[word] ?: 0) + smoothing
    return spam / (spam + ham)
}

/** 事前確率。何も見ないときのスパム率。 */
fun priorSpamProbability(model: NaiveBayesClassifier): Double =
    if (model.totalDocuments == 0) 0.0
    else model.spamDocuments.toDouble() / model.totalDocuments

/** log(0) を避けるためにごくわずかに内側へ丸める。 */
private fun safe(probability: Double): Double {
    val epsilon = 1e-15
    return min(max(probability, epsilon), 1.0 - epsilon)
}

/** 文書がスパムである確率。対数空間で計算する。 */
fun predictProbability(
    model: NaiveBayesClassifier,
    document: String,
    smoothing: Double = 1.0,
): Double {
    if (model.totalDocuments == 0) return 0.0
    val prior = priorSpamProbability(model)
    var logSpam = ln(safe(prior))
    var logHam = ln(safe(1.0 - prior))
    // 学習時に見ていない単語は何も語らないので無視する
    tokenize(document).toSet().filter { it in model.vocabulary }.forEach { word ->
        val spamGivenWord = wordSpamProbability(model, word, smoothing)
        logSpam += ln(safe(spamGivenWord))
        logHam += ln(safe(1.0 - spamGivenWord))
    }
    // log の差から確率へ戻す（シグモイドと同じ形）
    return 1.0 / (1.0 + exp(min(max(logHam - logSpam, -700.0), 700.0)))
}

/** 閾値による 0 / 1 の分類。 */
fun predict(
    model: NaiveBayesClassifier,
    document: String,
    threshold: Double = 0.5,
    smoothing: Double = 1.0,
): Int = if (predictProbability(model, document, smoothing) >= threshold) 1 else 0

/** 正解率。 */
fun accuracy(model: NaiveBayesClassifier, documents: List<String>, labels: List<Int>): Double =
    documents.zip(labels).count { (document, label) -> predict(model, document) == label }
        .toDouble() / documents.size
