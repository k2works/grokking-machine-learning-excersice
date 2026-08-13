package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import ml.dmlc.xgboost4j.java.DMatrix
import org.jetbrains.kotlinx.dataframe.api.count
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.MLP
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.OLS

/**
 * ライブラリ版の実行環境が揃っていることを確認するスモークテスト。
 *
 * 章の実装を書く前に、原著の scikit-learn / NumPy / pandas / XGBoost に対応する
 * 4 つのライブラリが実際に動くことをここで担保する。
 */
class EnvironmentTest {

    @Test
    fun `共有データセットを読み込める`() {
        val df = org.jetbrains.kotlinx.dataframe.DataFrame.readCSV(
            Datasets.path("Hyderabad.csv").toFile()
        )
        assertTrue(df.columnNames().contains("Price"))
        assertTrue(df.count() > 1000)
    }

    @Test
    fun `未登録のデータセットはエラーになる`() {
        assertTrue(
            runCatching { Datasets.path("does_not_exist.csv") }.exceptionOrNull()
                is java.io.FileNotFoundException
        )
    }

    @Test
    fun `Smile で線形回帰が学習できる`() {
        // y = 2x + 1 を完全に復元できる
        val df = DataFrame.of(
            DoubleVector.of("x", doubleArrayOf(1.0, 2.0, 3.0, 4.0)),
            DoubleVector.of("y", doubleArrayOf(3.0, 5.0, 7.0, 9.0)),
        )
        val model = OLS.fit(Formula.lhs("y"), df)

        // Smile は切片を coefficients() に含めず intercept() で返す
        assertEquals(2.0, model.coefficients()[0], 1e-6)
        assertEquals(1.0, model.intercept(), 1e-6)
    }

    @Test
    fun `Multik で多次元配列を扱える`() {
        val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        assertEquals(listOf(2, 2), a.shape.toList())
    }

    @Test
    fun `Smile の MLP で小さなネットワークが学習できる`() {
        val x = arrayOf(
            doubleArrayOf(0.0, 0.0),
            doubleArrayOf(1.0, 1.0),
            doubleArrayOf(0.0, 1.0),
            doubleArrayOf(1.0, 0.0),
        )
        val y = intArrayOf(0, 1, 0, 1)
        val net = MLP(Layer.input(2), Layer.sigmoid(4), Layer.mle(1, OutputFunction.SIGMOID))
        repeat(200) { net.update(x, y) }

        assertTrue(net.predict(x[1]) in 0..1)
    }

    @Test
    fun `XGBoost4J の行列を構築できる`() {
        val m = DMatrix(floatArrayOf(0f, 0f, 1f, 1f), 2, 2, Float.NaN)
        assertEquals(2L, m.rowNum())
    }
}
