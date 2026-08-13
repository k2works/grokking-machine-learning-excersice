package lib

import smile.base.cart.SplitRule
import smile.classification.DecisionTree
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.measure.NominalScale
import smile.data.type.DataTypes
import smile.data.type.StructField
import smile.data.vector.DoubleVector
import smile.data.vector.IntVector

/**
 * 原著ノートブック #10 `Chapter_09_Decision_Trees/Graphical_example.ipynb`。
 *
 * 2 次元の 12 点を決定木で分け、**決定境界を図で見る** 回。原著は 3 つのモデルを並べる。
 *
 * 1. ジニ不純度で分割した木
 * 2. エントロピーで分割した木
 * 3. 深さ 1 に制限した木（1 本の直線になる）
 *
 * 決定木の境界は必ず **軸に平行な長方形の集まり** になる。図そのものは記事の対象外なので、
 * **境界を格子上の予測ラベルとして取り出し**、性質をテストで確かめる。
 */
object Nb10DecisionTreeBoundary {

    /** 原著が使う 12 点 */
    val X0 = doubleArrayOf(7.0, 3.0, 2.0, 1.0, 2.0, 4.0, 1.0, 8.0, 6.0, 7.0, 8.0, 9.0)
    val X1 = doubleArrayOf(1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 5.0, 8.0, 4.0, 6.0)
    val Y = intArrayOf(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

    /** 原著の `plot_model` が使う格子の刻み幅 */
    const val PLOT_STEP = 0.2

    /** 分割に使われた特徴量としきい値 */
    data class Split(val feature: String, val threshold: Double)

    /** 決定境界を格子上の予測ラベルとして持ったもの */
    data class DecisionGrid(
        val xValues: DoubleArray,
        val yValues: DoubleArray,
        /** `predictions[row][column]` が `(xValues[column], yValues[row])` に対応 */
        val predictions: Array<IntArray>,
    ) {
        val rowCount: Int get() = predictions.size
        val columnCount: Int get() = predictions[0].size

        override fun equals(other: Any?): Boolean =
            other is DecisionGrid &&
                xValues.contentEquals(other.xValues) &&
                yValues.contentEquals(other.yValues) &&
                predictions.contentDeepEquals(other.predictions)

        override fun hashCode(): Int =
            31 * (31 * xValues.contentHashCode() + yValues.contentHashCode()) +
                predictions.contentDeepHashCode()
    }

    private fun frame(): DataFrame {
        val labelField = StructField("y", DataTypes.IntegerType, NominalScale("0", "1"))
        return DataFrame.of(
            DoubleVector.of("x_0", X0),
            DoubleVector.of("x_1", X1),
            IntVector.of(labelField, Y),
        )
    }

    /**
     * 決定木を学習する。原著は 3 通りの設定で呼び分けている。
     *
     * Smile は分割の基準を [SplitRule] で選ぶ。`GINI` と `ENTROPY` があり、
     * scikit-learn の `criterion='gini'` / `'entropy'` に対応する。
     */
    fun fit(rule: SplitRule = SplitRule.GINI, maxDepth: Int = 20): DecisionTree =
        DecisionTree.fit(Formula.lhs("y"), frame(), rule, maxDepth, 100, 1)

    /**
     * scikit-learn の `max_depth` と同じ意味で深さを制限する。
     *
     * **深さの数え方が 1 つずれている。** scikit-learn は根を深さ 0 と数えるので
     * `max_depth=1` で分割が 1 つ入るが、Smile は根を深さ 1 と数えるため
     * `maxDepth=1` では分割が 1 つも入らない（根だけの木になり正解率 0.5）。
     * 同じ木を得るには 1 を足す。
     */
    fun fitWithSklearnDepth(depth: Int, rule: SplitRule = SplitRule.GINI): DecisionTree =
        fit(rule, depth + 1)

    /** 学習データに対する正解率 */
    fun accuracy(model: DecisionTree): Double {
        val data = frame()
        val correct = Y.indices.count { model.predict(data[it]) == Y[it] }
        return correct.toDouble() / Y.size
    }

    /** DOT 文字列から、分割に使われた条件を根から順に取り出す */
    fun splitConditions(model: DecisionTree): List<Split> =
        SPLIT_LABEL.findAll(model.dot())
            .map { Split(it.groupValues[1], it.groupValues[2].toDouble()) }
            .toList()

    private val SPLIT_LABEL = Regex("""label=<([A-Za-z_0-9]+) &le; ([0-9.]+)<br/>""")

    /**
     * 原著の `plot_model` と同じ格子を作り、各点の予測ラベルを返す。
     *
     * 原著は `np.arange(min - 1, max + 1, step)` で刻む。`arange` は終端を含まないので、
     * 格子の右端・上端は最大値 + 1 の手前で止まる。Kotlin に `arange` は無いので、
     * 点数を計算してから生成する。
     */
    fun decisionGrid(model: DecisionTree, step: Double = PLOT_STEP): DecisionGrid {
        val xValues = arange(X0.min() - 1, X0.max() + 1, step)
        val yValues = arange(X1.min() - 1, X1.max() + 1, step)

        val predictions = Array(yValues.size) { row ->
            IntArray(xValues.size) { column ->
                val point = DataFrame.of(
                    DoubleVector.of("x_0", doubleArrayOf(xValues[column])),
                    DoubleVector.of("x_1", doubleArrayOf(yValues[row])),
                )
                model.predict(point[0])
            }
        }
        return DecisionGrid(xValues, yValues, predictions)
    }

    /** NumPy の `np.arange` と同じく、終端を含まない等差数列を作る */
    fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        val count = Math.ceil((stop - start) / step).toInt()
        return DoubleArray(count) { start + it * step }
    }

    /**
     * 左右で予測が変わる x 座標を集める。
     *
     * 決定木の境界は軸に平行なので、変わる位置は分割しきい値の近くに限られる。
     */
    fun boundaryColumns(grid: DecisionGrid): List<Double> {
        val changes = sortedSetOf<Double>()
        for (row in grid.predictions) {
            for (column in 1 until row.size) {
                if (row[column] != row[column - 1]) changes.add(grid.xValues[column])
            }
        }
        return changes.toList()
    }
}
