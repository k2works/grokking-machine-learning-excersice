package ch09

import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DecisionTreesTest {
    // 原著と同じアプリ推薦データ
    // 特徴量 = (性別: 0=女性 1=男性, 年齢)、ラベル = 1 が推薦
    private val points = listOf(
        listOf(1.0, 15.0),
        listOf(0.0, 25.0),
        listOf(0.0, 32.0),
        listOf(1.0, 35.0),
        listOf(0.0, 12.0),
        listOf(1.0, 14.0),
        listOf(1.0, 55.0),
        listOf(0.0, 40.0),
    )
    private val labels = listOf(1, 0, 0, 0, 1, 1, 0, 0)

    private fun log2(x: Double) = ln(x) / ln(2.0)

    @Test
    fun `純粋な集合のジニ不純度は 0`() {
        assertEquals(0.0, giniImpurity(listOf(1, 1, 1)), 1e-9)
        assertEquals(0.0, giniImpurity(listOf(0, 0)), 1e-9)
    }

    @Test
    fun `均等な集合のジニ不純度は 0_5`() {
        assertEquals(0.5, giniImpurity(listOf(0, 1)), 1e-9)
        assertEquals(0.5, giniImpurity(listOf(0, 0, 1, 1)), 1e-9)
    }

    @Test
    fun `空集合のジニ不純度は 0`() {
        assertEquals(0.0, giniImpurity(emptyList()), 1e-9)
    }

    @Test
    fun `純粋な集合のエントロピーは 0`() {
        assertEquals(0.0, entropy(listOf(1, 1, 1)), 1e-9)
    }

    @Test
    fun `均等な 2 クラスのエントロピーは 1 ビット`() {
        assertEquals(1.0, entropy(listOf(0, 1)), 1e-9)
    }

    @Test
    fun `均等な 4 クラスのエントロピーは 2 ビット`() {
        assertEquals(2.0, entropy(listOf(0, 1, 2, 3)), 1e-9)
    }

    @Test
    fun `エントロピーは定義どおり`() {
        val expected = -(0.75 * log2(0.75) + 0.25 * log2(0.25))
        assertEquals(expected, entropy(listOf(1, 1, 1, 0)), 1e-9)
    }

    @Test
    fun `ジニとエントロピーは同じ順序を与える`() {
        val pure = listOf(1, 1, 1, 1)
        val skewed = listOf(1, 1, 1, 0)
        val balanced = listOf(1, 1, 0, 0)
        assertTrue(giniImpurity(pure) < giniImpurity(skewed))
        assertTrue(giniImpurity(skewed) < giniImpurity(balanced))
        assertTrue(entropy(pure) < entropy(skewed))
        assertTrue(entropy(skewed) < entropy(balanced))
    }

    @Test
    fun `split は小さい値を左へ送る`() {
        val split = Split(feature = 1, threshold = 20.0)
        assertTrue(split.matches(listOf(0.0, 15.0)))
        assertTrue(!split.matches(listOf(0.0, 25.0)))
    }

    @Test
    fun `applySplit はデータを失わずに振り分ける`() {
        val partition = applySplit(points, labels, Split(feature = 1, threshold = 20.0))
        assertEquals(points.size, partition.leftPoints.size + partition.rightPoints.size)
        assertEquals(labels.size, partition.leftLabels.size + partition.rightLabels.size)
        assertTrue(partition.leftPoints.all { it[1] < 20.0 })
        assertTrue(partition.rightPoints.all { it[1] >= 20.0 })
    }

    @Test
    fun `weightedImpurity は純粋な子を好む`() {
        assertEquals(0.0, weightedImpurity(listOf(1, 1), listOf(0, 0)), 1e-9)
        assertEquals(0.5, weightedImpurity(listOf(1, 0), listOf(1, 0)), 1e-9)
    }

    @Test
    fun `役に立たない分割の情報利得は 0`() {
        assertEquals(0.0, informationGain(listOf(1, 1, 0, 0), listOf(1, 0), listOf(1, 0)), 1e-9)
    }

    @Test
    fun `完全な分割の情報利得は最大`() {
        assertEquals(0.5, informationGain(listOf(1, 1, 0, 0), listOf(1, 1), listOf(0, 0)), 1e-9)
    }

    @Test
    fun `candidateSplits は中点を使う`() {
        val sample = listOf(listOf(0.0, 10.0), listOf(0.0, 20.0), listOf(1.0, 30.0))
        val splits = candidateSplits(sample)
        assertEquals(listOf(15.0, 25.0), splits.filter { it.feature == 1 }.map { it.threshold }.sorted())
        assertEquals(listOf(0.5), splits.filter { it.feature == 0 }.map { it.threshold })
    }

    @Test
    fun `bestSplit は年齢の境界を見つける`() {
        val found = bestSplit(points, labels)
        assertTrue(found != null)
        val (split, gain) = found!!
        assertEquals(1, split.feature)
        assertTrue(split.threshold in 15.0..25.0)
        assertTrue(gain > 0.0)
    }

    @Test
    fun `改善しないとき bestSplit は null`() {
        assertNull(bestSplit(listOf(listOf(1.0), listOf(2.0), listOf(3.0)), listOf(1, 1, 1)))
    }

    @Test
    fun `majorityLabel は同数なら小さいラベルを選ぶ`() {
        assertEquals(1, majorityLabel(listOf(1, 1, 0)))
        assertEquals(0, majorityLabel(listOf(0, 1)))
    }

    @Test
    fun `buildTree は訓練データに適合する`() {
        assertEquals(1.0, accuracy(buildTree(points, labels), points, labels), 1e-9)
    }

    @Test
    fun `buildTree は maxDepth で止まる`() {
        val shallow = buildTree(points, labels, maxDepth = 1)
        assertEquals(1, depth(shallow))
        assertEquals(2, leafCount(shallow))
    }

    @Test
    fun `純粋なデータは 1 枚の葉になる`() {
        val tree = buildTree(listOf(listOf(1.0), listOf(2.0)), listOf(1, 1))
        assertTrue(tree is Leaf)
        assertEquals(1, (tree as Leaf).label)
        assertEquals(0, depth(tree))
    }

    @Test
    fun `深い木は浅い木より浅くならない`() {
        val shallow = buildTree(points, labels, maxDepth = 1)
        val deep = buildTree(points, labels, maxDepth = 5)
        assertTrue(depth(deep) >= depth(shallow))
        assertTrue(accuracy(deep, points, labels) >= accuracy(shallow, points, labels))
    }

    @Test
    fun `minSamples は木の成長を止める`() {
        assertTrue(buildTree(points, labels, minSamples = labels.size) is Leaf)
    }

    @Test
    fun `ジニとエントロピーはこのデータで同じ木を作る`() {
        assertEquals(
            buildTree(points, labels, impurity = giniImpurity),
            buildTree(points, labels, impurity = entropy),
        )
    }

    @Test
    fun `木の構造を検査できる`() {
        val tree = buildTree(points, labels, maxDepth = 1)
        assertTrue(tree is Node)
        assertTrue((tree as Node).left is Leaf)
        assertTrue(tree.right is Leaf)
    }
}
