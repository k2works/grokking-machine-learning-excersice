package lib

import kotlin.random.Random
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV

/**
 * 原著ノートブック #17 `Chapter_11_Support_Vector_Machines/Building_the_datasets.ipynb`。
 *
 * 第 11 章（SVM）で使う 3 つのデータセットを **乱数で作る** 回である。
 *
 * 原著は種を固定していないので、**生成そのものは再現できない**。
 * できるのは 2 つ。
 *
 * 1. 同じ規則で生成器を書き、規則どおりの点が出ることを確かめる
 * 2. 原著が生成して配布した CSV を読み、**その規則で作られたことを検証する**
 *
 * 2 つ目が本命である。
 */
object Nb17BuildingDatasets {

    /** 座標の範囲。原著の `6 * random.random() - 3` */
    const val COORD_SCALE = 6.0
    const val COORD_OFFSET = -3.0

    /** 2 次元の点とラベル */
    data class Point(val x1: Double, val x2: Double, val y: Int)

    /** 1 つのデータセットの作り方 */
    data class Spec(
        val name: String,
        /** 規則どおりに作る点の数 */
        val points: Int,
        /** ラベルを乱数にする点（ノイズ）の数 */
        val noise: Int,
        /** ラベルを決める規則 */
        val rule: (Double, Double) -> Int,
    ) {
        val total: Int get() = points + noise
    }

    /** 直線 `x + y = 0.5` の上側なら 1 */
    fun linearRule(x: Double, y: Double): Int = if (x + y > 0.5) 1 else 0

    /** 原点を中心とする半径 √2.8 の円の内側なら 1 */
    fun oneCircleRule(x: Double, y: Double): Int = if (x * x + y * y < 2.8) 1 else 0

    /** (1, 0) と (-1, 0) を中心とする 2 つの円の **どちらか** の内側なら 1 */
    fun twoCirclesRule(x: Double, y: Double): Int {
        val left = (x - 1) * (x - 1) + y * y < 2
        val right = (x + 1) * (x + 1) + y * y < 2
        return if (left || right) 1 else 0
    }

    val LINEAR = Spec("linear", 50, 10, ::linearRule)
    val ONE_CIRCLE = Spec("one_circle", 100, 10, ::oneCircleRule)
    val TWO_CIRCLES = Spec("two_circles", 200, 20, ::twoCirclesRule)

    val SPECS = listOf(LINEAR, ONE_CIRCLE, TWO_CIRCLES)

    /**
     * 原著と同じ手順でデータセットを作る。
     *
     * 原著は種を固定していないので実行のたびに違うものが出る。
     * ここでは `seed` を渡せるようにして、テストで扱えるようにした。
     * **原著が配布している CSV とは一致しない**（一致しようがない）。
     */
    fun generate(spec: Spec, seed: Int): List<Point> {
        val random = Random(seed)
        fun coordinate() = COORD_SCALE * random.nextDouble() + COORD_OFFSET

        val points = List(spec.points) {
            val x = coordinate()
            val y = coordinate()
            Point(x, y, spec.rule(x, y))
        }
        val noise = List(spec.noise) {
            val x = coordinate()
            val y = coordinate()
            Point(x, y, random.nextInt(0, 2))
        }
        return points + noise
    }

    /**
     * 原著が生成して配布している CSV を読む。
     *
     * 先頭に無名の添字列が付いているが、列名で取り出すので気にしなくてよい。
     */
    fun load(spec: Spec): List<Point> {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("${spec.name}.csv").toFile())
        val x1 = frame.getColumn("x_1").convertTo<Double>().toList()
        val x2 = frame.getColumn("x_2").convertTo<Double>().toList()
        val y = frame.getColumn("y").convertTo<Int>().toList()
        return x1.indices.map { Point(x1[it], x2[it], y[it]) }
    }

    /**
     * 規則とラベルが食い違う行の添字。
     *
     * ノイズとして入れた点は規則と無関係にラベルを振っているので、
     * **約半分がここに現れる**。規則どおりに作った先頭の点は 1 つも現れない。
     */
    fun ruleViolations(spec: Spec, data: List<Point>): List<Int> =
        data.indices.filter { spec.rule(data[it].x1, data[it].x2) != data[it].y }
}
