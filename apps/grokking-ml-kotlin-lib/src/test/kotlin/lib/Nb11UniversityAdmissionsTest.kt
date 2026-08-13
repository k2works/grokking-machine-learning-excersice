package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb11UniversityAdmissions.ADMISSION_THRESHOLD
import lib.Nb11UniversityAdmissions.EXAM_FEATURES
import lib.Nb11UniversityAdmissions.FEATURE_NAMES
import lib.Nb11UniversityAdmissions.accuracy
import lib.Nb11UniversityAdmissions.fit
import lib.Nb11UniversityAdmissions.fitExams
import lib.Nb11UniversityAdmissions.fitFull
import lib.Nb11UniversityAdmissions.fitSameSizeAsOriginal
import lib.Nb11UniversityAdmissions.fitSmaller
import lib.Nb11UniversityAdmissions.loadData
import lib.Nb11UniversityAdmissions.nodeCount
import lib.Nb11UniversityAdmissions.predictApplicant
import lib.Nb11UniversityAdmissions.splitConditions

/**
 * 原著ノートブック #11 の再現テスト。
 *
 * 2 特徴量の木は scikit-learn と同じ分割・同じ正解率になった。
 * 一方で **深さの指定が対応しない** ため、原著の「小さい木」は実測で設定を選んでいる。
 */
class Nb11UniversityAdmissionsTest {

    private val data = loadData()

    @Test
    fun `データセットは400件7特徴量になる`() {
        assertEquals(400, data.size)
        assertEquals(7, FEATURE_NAMES.size)
        assertEquals("GRE Score", FEATURE_NAMES.first())
    }

    @Test
    fun `合格ラベルは合格確率0_75で切る`() {
        // 原著は Chance of Admit >= 0.75 を合格とする
        assertEquals(0.75, ADMISSION_THRESHOLD)
        // 400 件中 180 件が合格
        assertEquals(180, data.admitted.sum())
    }

    @Test
    fun `制限なしの木はほぼ完全に覚える`() {
        // 原著の scikit-learn は 1.0。Smile は 0.9875 で止まる。
        // 分割の停止条件が違うため、最後の数件を覚えきらない
        val full = fitFull(data)

        assertTrue(accuracy(full, data) > 0.98, "acc=${accuracy(full, data)}")
        assertTrue(nodeCount(full) > 100)
    }

    @Test
    fun `Smile の深さは飛び飛びに効く`() {
        // #10 の 12 点では「Smile の深さ = scikit-learn + 1」で対応したが、
        // このデータでは対応しない。2 から 4 まで結果が変わらず、5 で跳ぶ
        val nodes = (1..6).map { nodeCount(fit(data, FEATURE_NAMES, maxDepth = it, minLeafSize = 10)) }

        assertEquals(listOf(1, 3, 3, 3, 15, 15), nodes)
    }

    @Test
    fun `小さい木の正解率は原著とほぼ同じ`() {
        // 原著の 0.885 に対して 0.88。400 件中 2 件ぶんの差
        assertEquals(0.88, accuracy(fitSmaller(data), data), 1e-12)
    }

    @Test
    fun `節の数を原著に合わせると予測が変わる`() {
        // maxDepth=5 なら原著と同じ 15 節・正解率 0.89 になるが、木の中身は違う。
        // 原著が例に挙げた CGPA 8.9 の出願者を不合格と判定してしまう
        val sameSize = fitSameSizeAsOriginal(data)

        assertEquals(15, nodeCount(sameSize))
        assertEquals(0.89, accuracy(sameSize, data), 1e-12)
        assertTrue(!predictApplicant(sameSize, listOf(320.0, 110.0, 3.0, 4.0, 3.5, 8.9, 0.0)))
    }

    @Test
    fun `小さい木の根はCGPAで分割する`() {
        // 7 つの特徴量のうち、成績（CGPA）がもっとも効く。scikit-learn も同じ
        assertEquals("CGPA", splitConditions(fitSmaller(data)).first().feature)
    }

    @Test
    fun `CGPAが高い出願者は合格と予測される`() {
        // 原著の出力: dt_smaller.predict([[320, 110, 3, 4.0, 3.5, 8.9, 0]]) -> True
        assertTrue(predictApplicant(fitSmaller(data), listOf(320.0, 110.0, 3.0, 4.0, 3.5, 8.9, 0.0)))
    }

    @Test
    fun `CGPAだけ下げると不合格に変わる`() {
        // 原著の出力: 8.9 を 8.0 にすると False。
        // 他の 6 項目は同じ。根が CGPA なので、ここだけで判定が反転する
        assertTrue(!predictApplicant(fitSmaller(data), listOf(320.0, 110.0, 3.0, 4.0, 3.5, 8.0, 0.0)))
    }

    @Test
    fun `2特徴量の木は scikit-learn と同じ分割から始まる`() {
        // GRE Score <= 319.5。TOEFL より GRE のほうが効く
        for (depth in listOf(2, 3, 20)) {
            val root = splitConditions(fitExams(data, depth)).first()

            assertEquals("GRE Score", root.feature)
            assertEquals(319.5, root.threshold, 1e-9)
        }
    }

    @Test
    fun `2特徴量の浅い木は scikit-learn と同じ正解率になる`() {
        // scikit-learn の max_depth=1 が 0.8525、max_depth=2 が 0.8625。
        // Smile では深さが 1 つずれて同じ値になる
        assertEquals(0.8525, accuracy(fitExams(data, 2), data, EXAM_FEATURES), 1e-12)
        assertEquals(0.8625, accuracy(fitExams(data, 3), data, EXAM_FEATURES), 1e-12)
    }

    @Test
    fun `2特徴量では制限なしでも完全には分けられない`() {
        // 同じ GRE・TOEFL で合否が違う出願者がいるため
        val unbounded = fitExams(data, 20)

        assertTrue(accuracy(unbounded, data, EXAM_FEATURES) < 1.0)
        assertTrue(accuracy(unbounded, data, EXAM_FEATURES) > 0.85)
    }

    @Test
    fun `深さを増やすほど訓練データに当たるようになる`() {
        // 過学習の進み方。節が増えても正解率の伸びは鈍る
        val scores = listOf(2, 3, 20).map { accuracy(fitExams(data, it), data, EXAM_FEATURES) }

        assertTrue(scores[0] < scores[1])
        assertTrue(scores[1] < scores[2])
    }
}
