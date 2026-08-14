package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb22TitanicEndToEnd.AGE_BINS
import lib.Nb22TitanicEndToEnd.FEATURE_NAMES
import lib.Nb22TitanicEndToEnd.Split
import lib.Nb22TitanicEndToEnd.accuracy
import lib.Nb22TitanicEndToEnd.ageBin
import lib.Nb22TitanicEndToEnd.clean
import lib.Nb22TitanicEndToEnd.f1Score
import lib.Nb22TitanicEndToEnd.featuresOf
import lib.Nb22TitanicEndToEnd.fitAll
import lib.Nb22TitanicEndToEnd.loadRaw
import lib.Nb22TitanicEndToEnd.majorityBaseline
import lib.Nb22TitanicEndToEnd.medianAge
import lib.Nb22TitanicEndToEnd.missingCounts
import lib.Nb22TitanicEndToEnd.preprocess

/**
 * 原著ノートブック #22 の再現テスト。
 *
 * **前処理と分割の件数は原著と完全に一致** する。
 * モデルの成績は、分割の中身が違うので一致しない（`Split` の説明を参照）。
 */
class Nb22TitanicEndToEndTest {

    private val raw = loadRaw()
    private val prepared = preprocess(clean(raw))
    private val split = Split(prepared)

    @Test
    fun `データセットは891行`() {
        // 原著の出力: The dataset has 891 rows
        assertEquals(891, raw.size)
    }

    @Test
    fun `生存者は342人`() {
        // 原著の出力: 342 passengers survived out of 891
        assertEquals(342, raw.count { it.survived == 1 })
    }

    @Test
    fun `欠損の数は原著と一致する`() {
        // 原著の isna().sum() の出力
        val counts = missingCounts(raw)

        assertEquals(177, counts.getValue("Age"))
        assertEquals(687, counts.getValue("Cabin"))
        assertEquals(2, counts.getValue("Embarked"))
    }

    @Test
    fun `年齢の中央値は28`() {
        // 原著の出力: 28.0
        assertEquals(28.0, medianAge(raw))
    }

    @Test
    fun `前処理後は欠損がなくなる`() {
        val cleaned = clean(raw)

        assertTrue(cleaned.all { it.age != null })
        assertTrue(cleaned.all { it.embarked != null })
        // Embarked の欠損は U という新しい区分になる
        assertEquals(2, cleaned.count { it.embarked == "U" })
    }

    @Test
    fun `年齢の区切りは10歳刻みで8区間`() {
        assertEquals(listOf(0, 10, 20, 30, 40, 50, 60, 70, 80), AGE_BINS)
        assertEquals(8, AGE_BINS.size - 1)
    }

    @Test
    fun `年齢の区間は左を開き右を閉じる`() {
        // pandas.cut の既定。10 歳ちょうどは (0, 10] に入る
        assertEquals(0, ageBin(10.0))
        assertEquals(1, ageBin(10.1))
        assertEquals(2, ageBin(28.0))
    }

    @Test
    fun `特徴量は20列になる`() {
        // SibSp + Parch + Fare + Sex 2 + Embarked 4 + Pclass 3 + 年齢 8。
        // Python 版はここに Survived を含めて 21 列と数えている
        assertEquals(20, FEATURE_NAMES.size)
        assertEquals(20, prepared.featureCount)
    }

    @Test
    fun `one_hotは各群でちょうど1つだけ立つ`() {
        val row = featuresOf(clean(raw)[0])
        fun sumOf(prefix: String) =
            FEATURE_NAMES.indices.filter { FEATURE_NAMES[it].startsWith(prefix) }.sumOf { row[it] }

        assertEquals(1.0, sumOf("Sex_"))
        assertEquals(1.0, sumOf("Embarked_"))
        assertEquals(1.0, sumOf("Pclass_"))
        assertEquals(1.0, sumOf("Categorized_age_"))
    }

    @Test
    fun `分割の件数は原著と一致する`() {
        // 原著の出力: 534 / 178 / 179。
        // scikit-learn は test_size の側を切り上げる
        assertEquals(534, split.train.second.size)
        assertEquals(178, split.validation.second.size)
        assertEquals(179, split.test.second.size)
    }

    @Test
    fun `分割は全体を覆い重複しない`() {
        assertEquals(
            891,
            split.train.second.size + split.validation.second.size + split.test.second.size,
        )
    }

    @Test
    fun `7つのモデルすべてが学習できる`() {
        // 原著と同じ 7 種類。Smile に無い GaussianNB だけ自前で書いた
        assertEquals(
            listOf(
                "Logistic regression", "Decision tree", "Naive Bayes", "SVM",
                "Random forest", "Gradient boosting", "AdaBoost",
            ),
            fitAll(split).keys.toList(),
        )
    }

    @Test
    fun `全モデルが多数派を答えるより良い`() {
        // 「全員死亡」と答えると 0.669。それを全モデルが上回る。
        // **原著は基準を出していない**ので、この比較は補ったもの
        val baseline = majorityBaseline(split.validation.second)

        fitAll(split).forEach { (name, predict) ->
            val predicted = predict(split.validation.first)
            assertTrue(accuracy(predicted, split.validation.second) > baseline, name)
        }
    }

    @Test
    fun `勾配ブースティングは上位に入る`() {
        // 原著は勾配ブースティングが単独 1 位。こちらは実行のたびに
        // ランダムフォレストと入れ替わる（[#20][Nb20EnsembleSpam] で見たとおり
        // Smile の RandomForest は種を固定しても結果が変わる）。
        // **1 位を主張できないので、上位 3 つに入ることを確かめる**
        val scores = fitAll(split).mapValues { (_, predict) ->
            accuracy(predict(split.validation.first), split.validation.second)
        }
        val top3 = scores.entries.sortedByDescending { it.value }.take(3).map { it.key }

        assertTrue("Gradient boosting" in top3, "上位 3 つ: $top3")
    }

    @Test
    fun `SVMは正解率とF1の差が大きい`() {
        // 原著も同じ傾向（正解率 0.680 に対し F1 は 0.400）。
        // 多数派に寄せた予測をしているので、正解率だけでは見抜けない
        val predict = fitAll(split).getValue("SVM")
        val predicted = predict(split.validation.first)

        assertTrue(
            accuracy(predicted, split.validation.second) -
                f1Score(predicted, split.validation.second) > 0.25,
        )
    }

    @Test
    fun `F1スコアの定義を確かめる`() {
        // 適合率 1.0・再現率 0.5 なら F1 は 2/3
        val predicted = intArrayOf(1, 0, 0, 0)
        val actual = intArrayOf(1, 1, 0, 0)

        assertEquals(2.0 / 3.0, f1Score(predicted, actual), 1e-15)
        // 1 つも当てられなければ 0
        assertEquals(0.0, f1Score(intArrayOf(0, 0), intArrayOf(1, 1)))
    }

    @Test
    fun `多数派を答える基準はおよそ7割`() {
        // 検証データの死亡率。原著は 0.607、こちらは分割が違うので 0.669
        assertEquals(0.6685393258426966, majorityBaseline(split.validation.second), 1e-15)
    }
}
