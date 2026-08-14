package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb10DecisionTreeBoundary.PLOT_STEP
import lib.Nb10DecisionTreeBoundary.Split
import lib.Nb10DecisionTreeBoundary.X0
import lib.Nb10DecisionTreeBoundary.Y
import lib.Nb10DecisionTreeBoundary.accuracy
import lib.Nb10DecisionTreeBoundary.arange
import lib.Nb10DecisionTreeBoundary.boundaryColumns
import lib.Nb10DecisionTreeBoundary.decisionGrid
import lib.Nb10DecisionTreeBoundary.fit
import lib.Nb10DecisionTreeBoundary.fitWithSklearnDepth
import lib.Nb10DecisionTreeBoundary.splitConditions
import smile.base.cart.SplitRule

/**
 * 原著ノートブック #10 の再現テスト。
 *
 * Smile の決定木は scikit-learn とまったく同じ 3 つの分割を選んだ。
 * ただし **深さの数え方が 1 つずれている** ので、そこは変換が要る。
 */
class Nb10DecisionTreeBoundaryTest {

    @Test
    fun `データセットは12点で半々に分かれる`() {
        assertEquals(12, X0.size)
        assertEquals(List(6) { 0 } + List(6) { 1 }, Y.toList())
    }

    @Test
    fun `ジニの木は全問正解する`() {
        // 原著の出力: decision_tree.score(features, labels) -> 1.0
        assertEquals(1.0, accuracy(fit(SplitRule.GINI)), 1e-12)
    }

    @Test
    fun `エントロピーの木も全問正解する`() {
        // 原著の出力: decision_tree_entropy.score(features, labels) -> 1.0
        assertEquals(1.0, accuracy(fit(SplitRule.ENTROPY)), 1e-12)
    }

    @Test
    fun `ジニとエントロピーは同じ木になる`() {
        // 分割の候補に同点が無いので、どちらの指標でも同じ順序で選ばれる。
        // #09 では同点があって木の形が変わったのと対照的
        assertEquals(
            splitConditions(fit(SplitRule.GINI)),
            splitConditions(fit(SplitRule.ENTROPY)),
        )
    }

    @Test
    fun `木は scikit-learn と同じ3つの条件で分割する`() {
        // scikit-learn も x_0 <= 5.0 / x_1 <= 8.0 / x_1 <= 2.5 を選ぶ
        assertEquals(
            listOf(Split("x_0", 5.0), Split("x_1", 8.0), Split("x_1", 2.5)),
            splitConditions(fit()),
        )
    }

    @Test
    fun `Smile の深さは根を1と数える`() {
        // scikit-learn は根を深さ 0 と数えるので max_depth=1 で分割が 1 つ入る。
        // Smile は根を 1 と数えるため maxDepth=1 では分割が入らない
        assertEquals(emptyList(), splitConditions(fit(SplitRule.GINI, maxDepth = 1)))
        assertEquals(0.5, accuracy(fit(SplitRule.GINI, maxDepth = 1)), 1e-12)
    }

    @Test
    fun `深さ1の木は1本の直線になる`() {
        // 原著の「1 本の縦線または横線」。Smile では maxDepth=2 に相当する
        val shallow = fitWithSklearnDepth(1)

        assertEquals(listOf(Split("x_0", 5.0)), splitConditions(shallow))
    }

    @Test
    fun `深さ1の木は12点中10点しか当てられない`() {
        // 原著と同じ 0.8333。直線 1 本では 2 点を取り違える
        assertEquals(10.0 / 12.0, accuracy(fitWithSklearnDepth(1)), 1e-12)
    }

    @Test
    fun `arange は終端を含まない`() {
        // NumPy の np.arange と同じ挙動にする
        assertEquals(listOf(0.0, 0.5, 1.0, 1.5), arange(0.0, 2.0, 0.5).toList())
    }

    @Test
    fun `格子は原著と同じ大きさになる`() {
        // x は 0 から 10 まで、y は 0 から 11 まで、刻みは 0.2。
        // 終端を含まないので 50 × 55 になる
        val grid = decisionGrid(fit())

        assertEquals(0.2, PLOT_STEP)
        assertEquals(0.0, grid.xValues[0], 1e-12)
        assertEquals(0.0, grid.yValues[0], 1e-12)
        assertEquals(55, grid.rowCount)
        assertEquals(50, grid.columnCount)
    }

    @Test
    fun `格子の予測は0か1しかない`() {
        val grid = decisionGrid(fit())

        assertEquals(setOf(0, 1), grid.predictions.flatMap { it.toList() }.toSet())
    }

    @Test
    fun `境界は軸に平行になる`() {
        // 決定木の境界は長方形の集まりなので、予測が変わる x 座標は
        // 分割しきい値の直後だけに限られる
        val changes = boundaryColumns(decisionGrid(fit()))

        assertEquals(1, changes.size)
        assertEquals(5.2, changes[0], 1e-9)
    }

    @Test
    fun `深さ1の境界は縦線なので全行で同じ`() {
        val grid = decisionGrid(fitWithSklearnDepth(1))

        assertTrue(grid.predictions.all { it.contentEquals(grid.predictions[0]) })
    }

    @Test
    fun `深い木の境界は行によって変わる`() {
        // x_1 での分割が入るので、行ごとにパターンが違う
        val grid = decisionGrid(fit())

        assertTrue(grid.predictions.map { it.toList() }.toSet().size > 1)
    }

    @Test
    fun `Smile は同じ入力に対して同じ格子を作る`() {
        assertEquals(decisionGrid(fit()), decisionGrid(fit()))
    }
}
