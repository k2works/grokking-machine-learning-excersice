package lib

import kotlin.math.ceil
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.MLP

/**
 * 原著ノートブック #13 `Chapter_10_Neural_Networks/Graphical_example.ipynb`。
 *
 * ニューラルネットワークの章に入る。円形に分布した 110 点を、
 * **2 層の隠れ層を持つネットワーク** で分類し、決定境界を見る。
 *
 * 決定木（[#10][Nb10DecisionTreeBoundary]）の境界が軸に平行な長方形だったのに対し、
 * ニューラルネットワークの境界は **曲線** になる。
 *
 * 原著は Keras で `Dense(128, relu)` → `Dropout` → `Dense(64, relu)` → `Dropout`
 * → `Dense(2, softmax)` と組む。Smile の [MLP] には **Dropout が無い** ので、
 * 隠れ層 2 つだけの構成になる。
 */
object Nb13NeuralNetworkBoundary {

    /** 原著のネットワークの隠れ層 */
    val HIDDEN_UNITS = intArrayOf(128, 64)

    /**
     * 原著の学習回数は 100 だが、Smile では足りない。
     *
     * Keras は Adam（学習率を自動調整する最適化）を使うのに対し、Smile の [MLP] は
     * 単純な確率的勾配降下である。実測すると 100 回では多数派を答えるだけの
     * 正解率 0.764 にとどまり、2000 回で 0.891 に届いた。
     */
    const val EPOCHS = 2000

    /** 原著が指定している学習回数。比較用に残す */
    const val ORIGINAL_EPOCHS = 100

    /** 原著の `plot_model` が使う格子の刻み幅 */
    const val PLOT_STEP = 0.2

    /** 読み込んだ円形データ */
    class Circle(val x: Array<DoubleArray>, val y: IntArray) {
        val size: Int get() = y.size
    }

    /** 円形に分布したデータを読み込む。110 点 */
    fun loadCircle(): Circle {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("one_circle.csv").toFile())
        val x1 = frame.getColumn("x_1").convertTo<Double>().toList()
        val x2 = frame.getColumn("x_2").convertTo<Double>().toList()
        val y = frame.getColumn("y").convertTo<Int>().toList()
        return Circle(Array(x1.size) { doubleArrayOf(x1[it], x2[it]) }, y.toIntArray())
    }

    /**
     * 原著と同じ隠れ層を持つネットワークを学習する。
     *
     * Smile の [MLP] は層を [Layer] のビルダーで組む。
     * `Layer.input` が入力層、`Layer.rectifier` が ReLU、
     * `Layer.mle` が出力層である。
     *
     * **Dropout は無い。** 原著は過学習を抑えるために 2 か所へ入れているが、
     * Smile では代わりに学習回数で調整することになる。
     *
     * 出力は 2 クラスなので `OutputFunction.SIGMOID` を使う。
     * Keras の `softmax` は 2 クラスならシグモイドと等価な境界を与える。
     */
    fun fit(data: Circle, epochs: Int = EPOCHS, seed: Int = 0): MLP {
        // Smile の MLP は内部で乱数を使うので、種を固定して再現性を確保する
        smile.math.MathEx.setSeed(seed.toLong())

        val network = MLP(
            Layer.input(2),
            Layer.rectifier(HIDDEN_UNITS[0]),
            Layer.rectifier(HIDDEN_UNITS[1]),
            Layer.mle(1, OutputFunction.SIGMOID),
        )
        repeat(epochs) { network.update(data.x, data.y) }
        return network
    }

    /** 学習データに対する正解率 */
    fun accuracy(model: MLP, data: Circle): Double {
        val correct = data.x.indices.count { model.predict(data.x[it]) == data.y[it] }
        return correct.toDouble() / data.size
    }

    /** NumPy の `np.arange` と同じく、終端を含まない等差数列を作る */
    fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        val count = ceil((stop - start) / step).toInt()
        return DoubleArray(count) { start + it * step }
    }

    /** 原著の `plot_model` と同じ格子を作り、各点の予測クラスを返す */
    fun decisionGrid(model: MLP, data: Circle, step: Double = PLOT_STEP): Array<IntArray> {
        val xs = data.x.map { it[0] }
        val ys = data.x.map { it[1] }
        val xValues = arange(xs.min() - 1, xs.max() + 1, step)
        val yValues = arange(ys.min() - 1, ys.max() + 1, step)

        return Array(yValues.size) { row ->
            IntArray(xValues.size) { column ->
                model.predict(doubleArrayOf(xValues[column], yValues[row]))
            }
        }
    }

    /**
     * 各行で予測が切り替わった回数を返す。
     *
     * 決定木なら軸に平行な境界なので、切り替わる位置は行によらず同じだった。
     * ニューラルネットワークは曲線を引けるので、行ごとに変わる。
     */
    fun boundaryChangesPerRow(grid: Array<IntArray>): List<Int> =
        grid.map { row -> (1 until row.size).count { row[it] != row[it - 1] } }
}
