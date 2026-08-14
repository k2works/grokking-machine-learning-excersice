package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb09AppRecommendations.AGES
import lib.Nb09AppRecommendations.APPS
import lib.Nb09AppRecommendations.APP_CLASSES
import lib.Nb09AppRecommendations.PLATFORMS
import lib.Nb09AppRecommendations.fitCategorical
import lib.Nb09AppRecommendations.fitNumeric
import lib.Nb09AppRecommendations.leafClasses
import lib.Nb09AppRecommendations.splits
import lib.Nb09AppRecommendations.toDot

/**
 * 原著ノートブック #09 の再現テスト。
 *
 * 根の分割（年齢が先）と木の大きさは原著と一致する。
 * 2 段目の分割は原著が `Platform_iPhone`、Smile は `Platform_Android` を選ぶが、
 * 2 つの列は互いに裏返しなので木としては等価である。
 */
class Nb09AppRecommendationsTest {

    @Test
    fun `元データは6人ぶん`() {
        assertEquals(6, PLATFORMS.size)
        assertEquals(listOf(15, 25, 32, 35, 12, 14), AGES.toList())
        assertEquals(3, APPS.distinct().size)
    }

    @Test
    fun `クラスは辞書順に並ぶ`() {
        // Smile はラベルを Int で扱うので、名前を 0 始まりの番号に直す
        assertEquals(listOf("Atom Count", "Beehive Finder", "Check Mate Mate"), APP_CLASSES)
    }

    @Test
    fun `数値版の根は年齢20歳で分割する`() {
        // 原著の出力: X[0] <= 20.0（X[0] は Age）
        val rootSplit = splits(fitNumeric()).first()

        assertEquals("Age", rootSplit.feature)
        assertEquals(20.0, rootSplit.threshold, 1e-12)
    }

    @Test
    fun `しきい値は隣り合う年齢の中点になる`() {
        // 20 歳の人はいない。15（Atom Count）と 25（Check Mate Mate）の中点である
        assertTrue(20 !in AGES.toList())
        assertEquals((15 + 25) / 2.0, splits(fitNumeric()).first().threshold, 1e-12)
    }

    @Test
    fun `カテゴリ版の根も年齢で分割する`() {
        // 原著の出力: X[3] <= 0.5（X[3] は Age_Young）。
        // Smile は裏返しの Age_Adult を選ぶが、分割としては同じ
        val rootSplit = splits(fitCategorical()).first()

        assertTrue(rootSplit.feature in setOf("Age_Young", "Age_Adult"))
        assertEquals(0.5, rootSplit.threshold, 1e-12)
    }

    @Test
    fun `2段目は端末で分割する`() {
        // 原著は Platform_iPhone、Smile は Platform_Android。
        // 2 つの列は互いに裏返しなので、木としては等価
        for (model in listOf(fitNumeric(), fitCategorical())) {
            val second = splits(model)[1]

            assertTrue(second.feature in setOf("Platform_iPhone", "Platform_Android"))
            assertEquals(0.5, second.threshold, 1e-12)
        }
    }

    @Test
    fun `木は分割2つと葉3つになる`() {
        // 原著の木も 5 節（内部 2 + 葉 3）
        for (model in listOf(fitNumeric(), fitCategorical())) {
            assertEquals(2, splits(model).size)
            assertEquals(3, leafClasses(model).size)
        }
    }

    @Test
    fun `葉は3つのアプリをそれぞれ予測する`() {
        // 全問正解するので、3 つの葉が 3 つのアプリに 1 対 1 で対応する
        assertEquals(APP_CLASSES.sorted(), leafClasses(fitNumeric()).sorted())
    }

    @Test
    fun `Smile は同じ入力に対して同じ木を作る`() {
        // scikit-learn は同点の分割候補から無作為に選ぶため木の形が変わるが、
        // Smile は決定的なので毎回同じ木になる
        assertEquals(toDot(fitNumeric()), toDot(fitNumeric()))
        assertEquals(splits(fitNumeric()), splits(fitNumeric()))
    }

    @Test
    fun `DOT 出力は Graphviz の書式になる`() {
        // scikit-learn は plot_tree で図を描くが、Smile は DOT 文字列を返すだけ
        val dot = toDot(fitNumeric())

        assertTrue(dot.startsWith("digraph CART {"))
        assertTrue(dot.trimEnd().endsWith("}"))
        // 実体参照 &le; が <= を表す
        assertTrue(dot.contains("&le;"))
    }

    @Test
    fun `Smile は不純度ではなく不純度の減少量を表示する`() {
        // scikit-learn の plot_tree は各節の gini を出すが、
        // Smile の DOT はその節で得られた impurity reduction を出す
        val dot = toDot(fitNumeric())

        assertTrue(dot.contains("impurity reduction"))
        assertTrue(!dot.contains("gini"))
    }
}
