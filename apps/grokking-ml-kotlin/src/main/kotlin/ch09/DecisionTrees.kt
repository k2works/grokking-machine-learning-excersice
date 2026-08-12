/**
 * 第 9 章: 決定木。
 * 「どの質問をすれば、もっともよくデータが分かれるか」を貪欲に選び続けて
 * 木を育てる。分割の良さはジニ不純度またはエントロピーで測る。
 */
package ch09

import kotlin.math.ln

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** 不純度の測り方。 */
typealias Impurity = (List<Int>) -> Double

/** ジニ不純度。ランダムに 2 つ選んだとき、ラベルが食い違う確率。 */
val giniImpurity: Impurity = { labels ->
    if (labels.isEmpty()) {
        0.0
    } else {
        val total = labels.size.toDouble()
        1.0 - labels.groupingBy { it }.eachCount().values.sumOf { count ->
            val share = count / total
            share * share
        }
    }
}

/** エントロピー。ラベルの散らばりを情報量で測る。 */
val entropy: Impurity = { labels ->
    if (labels.isEmpty()) {
        0.0
    } else {
        val total = labels.size.toDouble()
        -labels.groupingBy { it }.eachCount().values.sumOf { count ->
            val share = count / total
            share * (ln(share) / ln(2.0))
        }
    }
}

/** 1 つの質問による分割。「feature 番目の特徴量が threshold 未満か」を問う。 */
data class Split(val feature: Int, val threshold: Double) {
    /** 左（true）へ進むか。 */
    fun matches(point: Point): Boolean = point[feature] < threshold
}

/** 分割で振り分けられた左右のデータ。 */
data class Partition(
    val leftPoints: List<Point>,
    val leftLabels: List<Int>,
    val rightPoints: List<Point>,
    val rightLabels: List<Int>,
)

/** 分割を適用して左右に振り分ける。 */
fun applySplit(points: List<Point>, labels: List<Int>, split: Split): Partition {
    val (left, right) = points.zip(labels).partition { (point, _) -> split.matches(point) }
    return Partition(
        leftPoints = left.map { it.first },
        leftLabels = left.map { it.second },
        rightPoints = right.map { it.first },
        rightLabels = right.map { it.second },
    )
}

/** 分割後の不純度。左右の大きさで重み付けして平均する。 */
fun weightedImpurity(
    leftLabels: List<Int>,
    rightLabels: List<Int>,
    impurity: Impurity = giniImpurity,
): Double {
    val total = (leftLabels.size + rightLabels.size).toDouble()
    if (total == 0.0) return 0.0
    return leftLabels.size / total * impurity(leftLabels) +
        rightLabels.size / total * impurity(rightLabels)
}

/** 情報利得。分割によって不純度がどれだけ下がったか。 */
fun informationGain(
    labels: List<Int>,
    leftLabels: List<Int>,
    rightLabels: List<Int>,
    impurity: Impurity = giniImpurity,
): Double = impurity(labels) - weightedImpurity(leftLabels, rightLabels, impurity)

/** 試す価値のある分割の候補。隣り合う値の中点を閾値にする。 */
fun candidateSplits(points: List<Point>): List<Split> =
    points.first().indices.flatMap { feature ->
        points.map { it[feature] }.distinct().sorted()
            .zipWithNext { low, high -> Split(feature, (low + high) / 2) }
    }

/** 情報利得がもっとも大きい分割。改善しないなら null。 */
fun bestSplit(
    points: List<Point>,
    labels: List<Int>,
    impurity: Impurity = giniImpurity,
): Pair<Split, Double>? =
    candidateSplits(points)
        .mapNotNull { split ->
            val partition = applySplit(points, labels, split)
            if (partition.leftLabels.isEmpty() || partition.rightLabels.isEmpty()) {
                null
            } else {
                split to informationGain(labels, partition.leftLabels, partition.rightLabels, impurity)
            }
        }
        .maxByOrNull { it.second }
        ?.takeIf { it.second > 0.0 }

/** 決定木。葉か内部ノードのいずれか。 */
sealed interface Tree {
    fun predict(point: Point): Int
}

/** 葉。多数決で決めたラベルを返す。 */
data class Leaf(val label: Int) : Tree {
    override fun predict(point: Point): Int = label
}

/** 内部ノード。質問に応じて左右の枝へ進む。 */
data class Node(val split: Split, val left: Tree, val right: Tree) : Tree {
    override fun predict(point: Point): Int =
        if (split.matches(point)) left.predict(point) else right.predict(point)
}

/** 多数決。同数なら小さいラベルを選ぶ。 */
fun majorityLabel(labels: List<Int>): Int {
    val counts = labels.groupingBy { it }.eachCount()
    val top = counts.values.max()
    return counts.filterValues { it == top }.keys.min()
}

/** 決定木を再帰的に構築する。 */
fun buildTree(
    points: List<Point>,
    labels: List<Int>,
    maxDepth: Int = 5,
    minSamples: Int = 1,
    impurity: Impurity = giniImpurity,
): Tree {
    if (maxDepth <= 0 || labels.size <= minSamples || labels.distinct().size == 1) {
        return Leaf(majorityLabel(labels))
    }
    val (split, _) = bestSplit(points, labels, impurity) ?: return Leaf(majorityLabel(labels))
    val partition = applySplit(points, labels, split)
    return Node(
        split = split,
        left = buildTree(partition.leftPoints, partition.leftLabels, maxDepth - 1, minSamples, impurity),
        right = buildTree(partition.rightPoints, partition.rightLabels, maxDepth - 1, minSamples, impurity),
    )
}

/** 木の深さ。葉だけなら 0。 */
fun depth(tree: Tree): Int = when (tree) {
    is Leaf -> 0
    is Node -> 1 + maxOf(depth(tree.left), depth(tree.right))
}

/** 葉の数。 */
fun leafCount(tree: Tree): Int = when (tree) {
    is Leaf -> 1
    is Node -> leafCount(tree.left) + leafCount(tree.right)
}

/** 正解率。 */
fun accuracy(tree: Tree, points: List<Point>, labels: List<Int>): Double =
    points.zip(labels).count { (point, label) -> tree.predict(point) == label }
        .toDouble() / points.size
