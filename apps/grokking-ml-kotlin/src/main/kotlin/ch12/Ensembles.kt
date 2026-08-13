/**
 * 第 12 章: アンサンブル学習。
 * 弱い学習器を集めて強い学習器を作る。多数決（バギング）と
 * 逐次的な重み付け（AdaBoost）の 2 つの流儀を実装する。
 */
package ch12

import ch09.Impurity
import ch09.Leaf
import ch09.Point
import ch09.Tree
import ch09.buildTree
import ch09.giniImpurity
import ch09.majorityLabel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random

/** 復元抽出で元と同じ大きさの標本を作る。 */
fun bootstrapSample(
    points: List<Point>,
    labels: List<Int>,
    random: Random,
): Pair<List<Point>, List<Int>> {
    val indices = List(points.size) { random.nextInt(points.size) }
    return indices.map { points[it] } to indices.map { labels[it] }
}

/** 多数決で予測する木の集まり。 */
data class Forest(val trees: List<Tree>) {
    fun votes(point: Point): List<Int> = trees.map { it.predict(point) }

    fun predict(point: Point): Int = majorityLabel(votes(point))
}

/** バギング。復元抽出した標本ごとに木を育て、多数決で予測する。 */
fun trainForest(
    points: List<Point>,
    labels: List<Int>,
    treeCount: Int = 10,
    maxDepth: Int = 1,
    impurity: Impurity = giniImpurity,
    seed: Int = 0,
): Forest {
    val random = Random(seed)
    return Forest(
        List(treeCount) {
            val (samplePoints, sampleLabels) = bootstrapSample(points, labels, random)
            buildTree(samplePoints, sampleLabels, maxDepth = maxDepth, impurity = impurity)
        },
    )
}

/** AdaBoost の弱学習器。発言権（weight）を持つ。 */
data class WeightedTree(val tree: Tree, val weight: Double)

/** 重み付き多数決で予測する学習器の列。ラベルは +1 / -1。 */
data class AdaBoost(val learners: List<WeightedTree>) {
    fun score(point: Point): Double =
        learners.sumOf { it.weight * it.tree.predict(point) }

    fun predict(point: Point): Int = if (score(point) >= 0) 1 else -1
}

/** 重み付き誤り率。重みの大きい点を間違えるほど大きくなる。 */
fun weightedError(
    tree: Tree,
    points: List<Point>,
    labels: List<Int>,
    weights: List<Double>,
): Double {
    val total = weights.sum()
    if (total == 0.0) return 0.0
    val wrong = points.indices
        .filter { tree.predict(points[it]) != labels[it] }
        .sumOf { weights[it] }
    return wrong / total
}

/** 弱学習器の発言権。誤り率が小さいほど大きい。 */
fun learnerWeight(error: Double): Double {
    val epsilon = 1e-10
    val clamped = min(max(error, epsilon), 1.0 - epsilon)
    return 0.5 * ln((1.0 - clamped) / clamped)
}

/** 重みを反映した木。重みに比例して点を複製してから学習する。 */
fun buildTreeWithWeights(
    points: List<Point>,
    labels: List<Int>,
    weights: List<Double>,
    maxDepth: Int,
    impurity: Impurity,
): Tree {
    val scale = 100
    val replicated = points.indices.flatMap { i ->
        List(max(1, (weights[i] * scale).roundToInt())) { points[i] to labels[i] }
    }
    val replicatedLabels = replicated.map { it.second }
    if (replicatedLabels.distinct().size == 1) return Leaf(replicatedLabels.first())
    return buildTree(replicated.map { it.first }, replicatedLabels, maxDepth = maxDepth, impurity = impurity)
}

/** AdaBoost。間違えた点の重みを上げながら弱学習器を足していく。 */
fun trainAdaBoost(
    points: List<Point>,
    labels: List<Int>,
    rounds: Int = 5,
    maxDepth: Int = 1,
    impurity: Impurity = giniImpurity,
): AdaBoost {
    var weights = List(points.size) { 1.0 / points.size }
    val learners = mutableListOf<WeightedTree>()
    repeat(rounds) {
        val tree = buildTreeWithWeights(points, labels, weights, maxDepth, impurity)
        val error = weightedError(tree, points, labels, weights)
        // 当てずっぽう以下の学習器は採用しない
        if (error < 0.5) {
            val alpha = learnerWeight(error)
            learners += WeightedTree(tree, alpha)
            val updated = points.indices.map { i ->
                weights[i] * exp(-alpha * labels[i] * tree.predict(points[i]))
            }
            val total = updated.sum()
            weights = updated.map { it / total }
        }
    }
    return AdaBoost(learners)
}

/** 森の正解率。 */
fun accuracy(model: Forest, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }
        .toDouble() / points.size

/** AdaBoost の正解率。 */
fun accuracy(model: AdaBoost, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> model.predict(point) == label }
        .toDouble() / points.size

/** 1 本の木の正解率。 */
fun treeAccuracy(tree: Tree, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> tree.predict(point) == label }
        .toDouble() / points.size
