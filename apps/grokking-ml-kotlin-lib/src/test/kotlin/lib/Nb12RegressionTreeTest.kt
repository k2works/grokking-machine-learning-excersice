package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb12RegressionTree.AGES
import lib.Nb12RegressionTree.DAYS
import lib.Nb12RegressionTree.bestSplit
import lib.Nb12RegressionTree.fit
import lib.Nb12RegressionTree.predictAll
import lib.Nb12RegressionTree.splitMses
import lib.Nb12RegressionTree.splitThresholds

/**
 * 原著ノートブック #12 の再現テスト。
 *
 * 手計算の 9 通りの重み付き MSE、Smile の回帰木の分割としきい値、
 * 葉が返す平均値まで **すべて原著と一致** した。
 */
class Nb12RegressionTreeTest {

    /** Smile の深さは根を 1 と数えるので、scikit-learn の深さ 2 は 3 になる */
    private val sklearnDepth2 = 3

    @Test
    fun `データセットは8点`() {
        assertEquals(listOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0), AGES.toList())
        assertEquals(listOf(7.0, 5.0, 7.0, 1.0, 2.0, 1.0, 5.0, 4.0), DAYS.toList())
    }

    @Test
    fun `全体の平均は4`() {
        // 原著の出力: np.array([7,5,7,1,2,1,5,4]).mean() -> 4.0
        assertEquals(4.0, DAYS.average(), 1e-12)
    }

    @Test
    fun `分割は9通り試される`() {
        // 要素は 8 個だが range(0, 9) なので 9 通り
        val splits = splitMses()

        assertEquals(9, splits.size)
        assertEquals(emptyList(), splits[0].left)
        assertEquals(emptyList(), splits[8].right)
    }

    @Test
    fun `空の側の平均はNaNになる`() {
        // 原著も NumPy の RuntimeWarning つきで nan を出している
        assertTrue(splitMses()[0].leftMean.isNaN())
        assertTrue(splitMses()[8].rightMean.isNaN())
    }

    @Test
    fun `各分割の重み付きMSEは原著と同じ`() {
        // 原著のセル出力をそのまま期待値にしている
        val expected = listOf(
            5.25, 3.9642857142857144, 3.916666666666667, 1.9833333333333334,
            4.25, 4.983333333333333, 5.166666666666667, 5.25, 5.25,
        )
        val splits = splitMses()

        expected.forEachIndexed { index, value ->
            assertEquals(value, splits[index].weightedMse, 1e-14, "index $index")
        }
    }

    @Test
    fun `最良の分割は3番目`() {
        // 原著の一覧で 1.9833 がもっとも小さい
        val best = bestSplit()

        assertEquals(3, best.index)
        assertEquals(listOf(7.0, 5.0, 7.0), best.left)
        assertEquals(6.333333333333333, best.leftMean, 1e-12)
        assertEquals(2.6, best.rightMean, 1e-12)
    }

    @Test
    fun `Smile も同じ位置で分割する`() {
        // 手計算の最小値は 3 番目、つまり 30 歳と 40 歳の間。中点の 35.0 が選ばれる
        assertEquals(35.0, splitThresholds(fit(sklearnDepth2)).first(), 1e-12)
    }

    @Test
    fun `深さ2の木は scikit-learn と同じ3回の分割になる`() {
        // 根で 35.0、左で 15.0、右で 65.0
        assertEquals(listOf(35.0, 15.0, 65.0), splitThresholds(fit(sklearnDepth2)))
    }

    @Test
    fun `予測は scikit-learn と完全に一致する`() {
        // 10 歳 -> 7、20〜30 歳 -> 6、40〜60 歳 -> 1.333、70〜80 歳 -> 4.5
        val expected = listOf(7.0, 6.0, 6.0, 4.0 / 3, 4.0 / 3, 4.0 / 3, 4.5, 4.5)

        predictAll(fit(sklearnDepth2)).forEachIndexed { index, value ->
            assertEquals(expected[index], value, 1e-12, "index $index")
        }
    }

    @Test
    fun `葉は平均値を返す`() {
        // 40, 50, 60 歳のラベルは 1, 2, 1。その平均 4/3 が葉の値になる
        assertEquals((1.0 + 2.0 + 1.0) / 3, predictAll(fit(sklearnDepth2))[3], 1e-12)
    }

    @Test
    fun `予測は階段状になる`() {
        // 同じ葉に落ちる点は同じ値を返す。回帰木の予測は連続にならない
        val predictions = predictAll(fit(sklearnDepth2))

        assertEquals(predictions[1], predictions[2], 1e-12)
        assertEquals(predictions[3], predictions[5], 1e-12)
        assertEquals(predictions[6], predictions[7], 1e-12)
    }

    @Test
    fun `深さを増やすと1点ずつ覚えていく`() {
        // 深さ 3 まで許すと、8 点のうち 6 点を個別の値で覚える
        val deeper = predictAll(fit(4))

        assertEquals(7.0, deeper[0], 1e-12)
        assertEquals(5.0, deeper[1], 1e-12)
        assertEquals(7.0, deeper[2], 1e-12)
    }
}
