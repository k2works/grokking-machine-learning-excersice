package lib

import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.RegressionTree

/**
 * 原著ノートブック #12 `Chapter_09_Decision_Trees/Regression_decision_tree.ipynb`。
 *
 * 決定木を **回帰** に使う回。年齢からアプリの利用日数を予測する 8 点のデータに、
 * 深さ 2 の回帰木を当てはめる。
 *
 * 分類木がジニ不純度を最小にする分割を探したのに対し、回帰木は
 * **平均二乗誤差（MSE）を最小にする分割** を探す。原著は探索の過程を手で書き下している。
 */
object Nb12RegressionTree {

    /** 原著が使う 8 点。年齢と、週あたりの利用日数 */
    val AGES = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    val DAYS = doubleArrayOf(7.0, 5.0, 7.0, 1.0, 2.0, 1.0, 5.0, 4.0)

    /** ある分割位置での、左右の平均と重み付き MSE */
    data class SplitMse(
        val index: Int,
        val left: List<Double>,
        val right: List<Double>,
        /** 左の平均。左が空なら NaN */
        val leftMean: Double,
        val rightMean: Double,
        /** 全体を分母にした重み付き MSE */
        val weightedMse: Double,
    )

    /** 空の配列の平均は NaN。原著も NumPy の警告つきで NaN を出している */
    private fun meanOrNaN(values: List<Double>): Double =
        if (values.isEmpty()) Double.NaN else values.average()

    /**
     * 分割位置を 0 から n まで動かし、それぞれの重み付き MSE を求める。
     *
     * 原著は `range(0, 9)` と、要素数 8 に対して **9 通り** 回している。
     * 先頭（左が空）と末尾（右が空）の両方が含まれるので、
     * 分類木の回（[Nb08GiniEntropy]）が要素数ぶんだったのと 1 つ違う。
     */
    fun splitMses(labels: DoubleArray = DAYS): List<SplitMse> {
        val total = labels.size
        return (0..total).map { index ->
            val left = labels.take(index)
            val right = labels.drop(index)
            val leftMean = meanOrNaN(left)
            val rightMean = meanOrNaN(right)

            val squaredErrors =
                left.sumOf { (it - leftMean) * (it - leftMean) } +
                    right.sumOf { (it - rightMean) * (it - rightMean) }

            SplitMse(index, left, right, leftMean, rightMean, squaredErrors / total)
        }
    }

    /** 重み付き MSE がもっとも小さい分割を返す */
    fun bestSplit(labels: DoubleArray = DAYS): SplitMse =
        splitMses(labels).minBy { it.weightedMse }

    private fun frame(): DataFrame =
        DataFrame.of(DoubleVector.of("Age", AGES), DoubleVector.of("Days", DAYS))

    /**
     * 回帰木を学習する。
     *
     * 分類木は [smile.classification.DecisionTree] だったが、回帰木は
     * `smile.regression.RegressionTree` と **別のパッケージの別のクラス** である。
     * 分類木は `SplitRule` を取ったが、回帰木は取らない。
     * 回帰の分割基準は最小二乗しか無いので、選ぶ余地が無いためである。
     */
    fun fit(maxDepth: Int = 3): RegressionTree =
        RegressionTree.fit(Formula.lhs("Days"), frame(), maxDepth, 100, 1)

    /** DOT 文字列から、分割に使われたしきい値を根から順に取り出す */
    fun splitThresholds(model: RegressionTree): List<Double> =
        SPLIT_LABEL.findAll(model.dot()).map { it.groupValues[1].toDouble() }.toList()

    private val SPLIT_LABEL = Regex("""label=<Age &le; ([0-9.]+)<br/>""")

    /** 各点に対する予測 */
    fun predictAll(model: RegressionTree): List<Double> {
        val data = frame()
        return AGES.indices.map { model.predict(data[it]) }
    }
}
