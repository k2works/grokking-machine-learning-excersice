package lib

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.classification.Classifier
import smile.classification.SVM
import smile.math.kernel.GaussianKernel
import smile.math.kernel.MercerKernel
import smile.math.kernel.PolynomialKernel

/**
 * 原著ノートブック #19 `Chapter_11_Support_Vector_Machines/SVM_graphical_example.ipynb`。
 *
 * [#17][Nb17BuildingDatasets] で作った 3 つのデータセットに、いろいろな SVM を当てる回。
 *
 * - 直線データに線形カーネル。`C` を変えて正則化の効き方を見る
 * - 円データに多項式カーネル。`degree` を変える
 * - 二重円データに RBF カーネル。`gamma` を変える
 *
 * 原著が印刷する 9 つの正解率が突き合わせの対象になる。
 */
object Nb19SvmKernels {

    /** 読み込んだデータ */
    class Dataset(val x: Array<DoubleArray>, val y: IntArray) {
        val size: Int get() = y.size

        /** Smile が要求する ±1 のラベル（[#04][Nb04Perceptron] と同じ） */
        val signedY: IntArray get() = IntArray(size) { if (y[it] == 1) 1 else -1 }
    }

    /** [#17][Nb17BuildingDatasets] が作った CSV を読む */
    fun load(name: String): Dataset {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("$name.csv").toFile())
        val x1 = frame.getColumn("x_1").convertTo<Double>().toList()
        val x2 = frame.getColumn("x_2").convertTo<Double>().toList()
        val y = frame.getColumn("y").convertTo<Int>().toList()
        return Dataset(Array(x1.size) { doubleArrayOf(x1[it], x2[it]) }, y.toIntArray())
    }

    /**
     * 既定の乱数の種。
     *
     * **Smile の SVM は種を固定しないと実行のたびに結果が変わる。**
     * 同じデータ・同じ引数で `one_circle` に 2 次の多項式カーネルを当てたところ、
     * 0.773 / 0.900 / 0.827 / 0.918 と毎回違う正解率が出た。
     * scikit-learn の `SVC` は決定的なので、この違いは移植で必ず踏む。
     */
    const val SEED = 0L

    /**
     * 線形 SVM。scikit-learn の `SVC(kernel='linear', C=...)` に対応する。
     *
     * Smile の `SVM.fit(x, y, C, tol)` は線形カーネル専用の入口である。
     */
    fun fitLinear(data: Dataset, c: Double = 1.0, tolerance: Double = 1e-3): Classifier<DoubleArray> {
        smile.math.MathEx.setSeed(SEED)
        return SVM.fit(data.x, data.signedY, c, tolerance)
    }

    /**
     * カーネルを指定した SVM。
     *
     * Smile は `MercerKernel` を渡す形になっていて、`kernel='poly'` のような
     * 文字列指定は無い。**そのぶん、どの式を使っているかが明示的になる。**
     */
    fun fitKernel(
        data: Dataset,
        kernel: MercerKernel<DoubleArray>,
        c: Double = 1.0,
        tolerance: Double = 1e-3,
    ): Classifier<DoubleArray> {
        smile.math.MathEx.setSeed(SEED)
        return SVM.fit(data.x, data.signedY, kernel, c, tolerance)
    }

    /**
     * scikit-learn の `kernel='poly'` と同じ式のカーネル。
     *
     * scikit-learn は `(gamma * <x, y> + coef0)^degree` で、`gamma='scale'` の
     * 既定は `1 / (特徴量数 * 分散)`、`coef0` の既定は 0。
     * Smile の `PolynomialKernel(degree, scale, offset)` はこの scale と offset に対応する。
     */
    fun polynomialKernel(data: Dataset, degree: Int): PolynomialKernel =
        PolynomialKernel(degree, scaleOf(data), 0.0)

    /**
     * scikit-learn の `gamma='scale'` を計算する。
     *
     * `1 / (n_features * X.var())`。**分散は列ごとではなく行列全体** で取る。
     */
    fun scaleOf(data: Dataset): Double {
        val values = data.x.flatMap { it.asIterable() }
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        return 1.0 / (2 * variance)
    }

    /** scikit-learn の `kernel='rbf', gamma=g` に対応する。σ は `1/√(2γ)` */
    fun rbfKernel(gamma: Double): GaussianKernel = GaussianKernel(Math.sqrt(1.0 / (2 * gamma)))

    /** 学習データに対する正解率 */
    fun accuracy(model: Classifier<DoubleArray>, data: Dataset): Double {
        val correct = (0 until data.size).count {
            val predicted = if (model.predict(data.x[it]) == 1) 1 else 0
            predicted == data.y[it]
        }
        return correct.toDouble() / data.size
    }
}
