package ch12

import ch09.Leaf
import ch09.buildTree
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EnsemblesTest {
    // 3 つの領域に分かれるデータ。深さ 1 の木（切り株）1 本では解けない
    private val points = listOf(
        listOf(1.0, 1.0),
        listOf(2.0, 1.0),
        listOf(3.0, 1.0),
        listOf(4.0, 1.0),
        listOf(5.0, 1.0),
        listOf(6.0, 1.0),
        listOf(7.0, 1.0),
        listOf(8.0, 1.0),
    )
    private val labels = listOf(1, 1, -1, -1, -1, -1, 1, 1)

    @Test
    fun `bootstrapSample は大きさを保つ`() {
        val (samplePoints, sampleLabels) = bootstrapSample(points, labels, Random(0))
        assertEquals(points.size, samplePoints.size)
        assertEquals(labels.size, sampleLabels.size)
    }

    @Test
    fun `bootstrapSample は特徴量とラベルの対応を保つ`() {
        val (samplePoints, sampleLabels) = bootstrapSample(points, labels, Random(0))
        samplePoints.zip(sampleLabels).forEach { (point, label) ->
            assertEquals(labels[points.indexOf(point)], label)
        }
    }

    @Test
    fun `bootstrapSample は復元抽出`() {
        val (samplePoints, _) = bootstrapSample(points, labels, Random(0))
        // 復元抽出なので、同じ点が複数回選ばれ、選ばれない点も出る
        assertTrue(samplePoints.distinct().size < points.size)
    }

    @Test
    fun `切り株 1 本では 3 領域を分けられない`() {
        assertTrue(treeAccuracy(buildTree(points, labels, maxDepth = 1), points, labels) < 1.0)
    }

    @Test
    fun `forest は多数決で予測する`() {
        val forest = Forest(listOf(Leaf(1), Leaf(1), Leaf(-1)))
        assertEquals(1, forest.predict(listOf(0.0)))
        assertEquals(listOf(1, 1, -1), forest.votes(listOf(0.0)))
    }

    @Test
    fun `切り株のバギングはこの問題では改善しない`() {
        val stump = buildTree(points, labels, maxDepth = 1)
        val forest = trainForest(points, labels, treeCount = 10, maxDepth = 1)
        // バギングは似た木ばかり作るので、この問題では改善しない
        assertTrue(accuracy(forest, points, labels) <= treeAccuracy(stump, points, labels) + 1e-9)
    }

    @Test
    fun `誤り率 0_5 の学習器には発言権がない`() {
        assertEquals(0.0, learnerWeight(0.5), 1e-9)
    }

    @Test
    fun `誤り率が小さいほど発言権は大きい`() {
        assertTrue(learnerWeight(0.4) < learnerWeight(0.2))
        assertTrue(learnerWeight(0.2) < learnerWeight(0.05))
    }

    @Test
    fun `当てずっぽうより悪い学習器の発言権は負`() {
        assertTrue(learnerWeight(0.7) < 0.0)
    }

    @Test
    fun `誤り率 0 でも発言権は有限`() {
        assertTrue(learnerWeight(0.0) > 0.0)
        assertTrue(learnerWeight(0.0) < 100.0)
    }

    @Test
    fun `weightedError は個数ではなく重みを数える`() {
        val tree = Leaf(1)
        val sample = listOf(listOf(0.0), listOf(1.0), listOf(2.0))
        val sampleLabels = listOf(1, 1, -1)
        assertEquals(0.8, weightedError(tree, sample, sampleLabels, listOf(0.1, 0.1, 0.8)), 1e-9)
        assertEquals(0.1, weightedError(tree, sample, sampleLabels, listOf(0.45, 0.45, 0.1)), 1e-9)
    }

    @Test
    fun `完璧な木の重み付き誤り率は 0`() {
        val tree = buildTree(points, labels, maxDepth = 5)
        val weights = List(points.size) { 1.0 / points.size }
        assertEquals(0.0, weightedError(tree, points, labels, weights), 1e-9)
    }

    @Test
    fun `adaBoost は重み付き投票で判定する`() {
        val model = AdaBoost(listOf(WeightedTree(Leaf(1), 2.0), WeightedTree(Leaf(-1), 1.0)))
        // 2*1 + 1*(-1) = 1 なので正のクラス
        assertEquals(1.0, model.score(listOf(0.0)), 1e-9)
        assertEquals(1, model.predict(listOf(0.0)))
    }

    @Test
    fun `AdaBoost は切り株 1 本では解けない問題を解く`() {
        val stump = buildTree(points, labels, maxDepth = 1)
        val boosted = trainAdaBoost(points, labels, rounds = 10, maxDepth = 1)
        assertTrue(treeAccuracy(stump, points, labels) < 1.0)
        assertEquals(1.0, accuracy(boosted, points, labels), 1e-9)
    }

    @Test
    fun `この問題では AdaBoost がバギングに勝つ`() {
        val forest = trainForest(points, labels, treeCount = 10, maxDepth = 1)
        val boosted = trainAdaBoost(points, labels, rounds = 10, maxDepth = 1)
        assertTrue(accuracy(boosted, points, labels) > accuracy(forest, points, labels))
    }

    @Test
    fun `AdaBoost の学習器は正の発言権を持つ`() {
        val boosted = trainAdaBoost(points, labels, rounds = 10, maxDepth = 1)
        assertTrue(boosted.learners.isNotEmpty())
        assertTrue(boosted.learners.all { it.weight > 0.0 })
    }

    @Test
    fun `ラウンドを増やしても訓練精度は下がらない`() {
        val few = trainAdaBoost(points, labels, rounds = 2, maxDepth = 1)
        val many = trainAdaBoost(points, labels, rounds = 10, maxDepth = 1)
        assertTrue(accuracy(many, points, labels) >= accuracy(few, points, labels))
    }
}
