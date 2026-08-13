package lib

import java.nio.file.Files
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.classification.SparseLogisticRegression
import smile.data.SparseDataset
import smile.util.SparseArray

/**
 * 原著ノートブック #06 `Chapter_06_Logistic_Regression/Sentiment_analysis_IMDB.ipynb`。
 *
 * IMDB の映画レビュー 50000 件を、単語の出現回数だけからロジスティック回帰で
 * 肯定・否定に分類する。学習した係数がそのまま「単語の感情スコア」になる。
 *
 * **Kotlin には `CountVectorizer` に相当するものが無い。** Smile の
 * `smile.nlp` にもテキストの前処理はあるが、scikit-learn と同じ規則ではない。
 * 3 言語で同じ語彙を作るため、ベクトル化は [CountVectorizer] として自前で書く。
 */
object Nb06SentimentImdb {

    /** 原著が使う語彙の上限 */
    const val MAX_FEATURES = 2000

    /**
     * scikit-learn の `CountVectorizer` と同じ規則でテキストをベクトル化する。
     *
     * 揃えているのは次の 4 点である。
     * 1. 小文字化する
     * 2. トークンは正規表現 `\b\w\w+\b`（2 文字以上の単語）で切り出す
     * 3. ストップワードを除く（scikit-learn の 318 語を共有データセットから読む）
     * 4. 残った語をコーパス全体の出現回数で並べ、上位 `maxFeatures` 語を採る
     *
     * 語彙は最後に **辞書順** に並べ替える。scikit-learn も同じで、
     * `get_feature_names_out()` の順序が特徴量の添字になる。
     */
    class CountVectorizer(
        private val maxFeatures: Int = MAX_FEATURES,
        private val stopWords: Set<String> = englishStopWords(),
    ) {
        /** 学習した語彙。添字がそのまま特徴量の添字になる */
        lateinit var vocabulary: List<String>
            private set

        private lateinit var indexOf: Map<String, Int>

        /** コーパスから語彙を学習し、出現回数の疎行列を返す */
        fun fitTransform(documents: List<String>): List<SparseArray> {
            val counts = HashMap<String, Int>()
            for (document in documents) {
                for (token in tokenize(document)) {
                    counts[token] = (counts[token] ?: 0) + 1
                }
            }

            // 出現回数の多い順。同数なら辞書順にして結果を決定的にする。
            // scikit-learn の同数時の扱いは実装依存なので、そこだけは規則が違いうる
            vocabulary = counts.entries
                .sortedWith(compareByDescending<Map.Entry<String, Int>> { it.value }.thenBy { it.key })
                .take(maxFeatures)
                .map { it.key }
                .sorted()
            indexOf = vocabulary.withIndex().associate { (index, word) -> word to index }

            return transform(documents)
        }

        /**
         * 学習済みの語彙で出現回数の行列を作る。
         *
         * **疎行列で持つのが必須である。** IMDB は 50000 件 × 2000 語なので、
         * 密な `Array<DoubleArray>` にすると倍精度で 800 MB になり、
         * 既定のヒープでは `OutOfMemoryError` になる。実際の非ゼロ要素は
         * 全体の数パーセントしかない。scipy の疎行列を返す scikit-learn の
         * `CountVectorizer` と同じ考え方である。
         */
        fun transform(documents: List<String>): List<SparseArray> =
            documents.map { document ->
                val counts = HashMap<Int, Double>()
                for (token in tokenize(document)) {
                    indexOf[token]?.let { counts[it] = (counts[it] ?: 0.0) + 1.0 }
                }
                val array = SparseArray(counts.size)
                for ((index, value) in counts.entries.sortedBy { it.key }) {
                    array.set(index, value)
                }
                array
            }

        /** 小文字化して 2 文字以上の単語を取り出し、ストップワードを除く */
        fun tokenize(document: String): List<String> =
            TOKEN_PATTERN.findAll(document.lowercase())
                .map { it.value }
                .filter { it !in stopWords }
                .toList()

        companion object {
            /** scikit-learn の既定と同じトークン正規表現 */
            val TOKEN_PATTERN = Regex("""\b\w\w+\b""")
        }
    }

    /** scikit-learn 内蔵の英語ストップワード 318 語。共有データセットから読む */
    fun englishStopWords(): Set<String> =
        Files.readAllLines(Datasets.path("sklearn_english_stop_words.txt"))
            .filter { it.isNotBlank() }
            .toSet()

    /** 学習済みモデルと、それを作るのに使った語彙 */
    class SentimentModel(
        val vectorizer: CountVectorizer,
        val model: SparseLogisticRegression.Binomial,
        val features: List<SparseArray>,
    ) {
        val vocabulary: List<String> get() = vectorizer.vocabulary

        /** 単語と、その係数（感情スコア）の対応表 */
        fun wordSentiments(): List<Pair<String, Double>> =
            vocabulary.mapIndexed { index, word -> word to model.coefficients()[index] }

        /** 係数が大きい順に単語を返す */
        fun mostPositiveWords(count: Int = 10): List<String> =
            wordSentiments().sortedByDescending { it.second }.take(count).map { it.first }

        /** 係数が小さい順に単語を返す */
        fun mostNegativeWords(count: Int = 10): List<String> =
            wordSentiments().sortedBy { it.second }.take(count).map { it.first }
    }

    /** レビューと 0 / 1 のラベル */
    data class Reviews(val texts: List<String>, val sentiments: IntArray)

    /**
     * IMDB のレビューを読み込み、`sentiment` を 0 / 1 に置き換える。
     *
     * このファイルは約 63 MB あるためリポジトリには含めていない。
     * 未取得なら [Datasets.path] が原著リポジトリから取得する。
     */
    fun loadReviews(): Reviews {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("IMDB_Dataset.csv").toFile())
        val texts = frame.getColumn("review").convertTo<String>().toList()
        val sentiments = frame.getColumn("sentiment").convertTo<String>().toList()
        return Reviews(
            texts,
            IntArray(sentiments.size) { if (sentiments[it] == "positive") 1 else 0 },
        )
    }

    /**
     * レビューをベクトル化してロジスティック回帰を学習する。
     *
     * scikit-learn の既定 `C = 1.0` に対応させるため、Smile の `lambda` に
     * `1.0 / データ数` を渡す。#05 と同じ理由である。
     */
    fun fit(reviews: Reviews, maxFeatures: Int = MAX_FEATURES): SentimentModel {
        val vectorizer = CountVectorizer(maxFeatures)
        val features = vectorizer.fitTransform(reviews.texts)
        val model = SparseLogisticRegression.binomial(
            SparseDataset.of(features, vectorizer.vocabulary.size),
            reviews.sentiments,
            1.0 / reviews.texts.size,
            1e-5,
            1000,
        )
        return SentimentModel(vectorizer, model, features)
    }
}
