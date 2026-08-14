package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import lib.Nb02HousePrices.VALID_ROWS
import lib.Nb02HousePrices.fitAllFeatures
import lib.Nb02HousePrices.fitAreaOnly
import lib.Nb02HousePrices.loadData
import lib.Nb02HousePrices.predictAll
import lib.Nb02HousePrices.predictNewHouse
import lib.Nb02HousePrices.preprocess
import lib.Nb02HousePrices.rmse
import org.jetbrains.kotlinx.dataframe.api.count

/**
 * 原著ノートブック #02 の再現テスト。
 *
 * 単回帰は原著の数値と完全に一致する。全特徴量の重回帰は one-hot 符号化によって
 * 設計行列がランク落ちしており係数が一意に決まらないので、当てはまりの良さで検証する。
 */
class Nb02HousePricesTest {

    // データの読み込みと前処理は重いので、クラスで 1 度だけ行う
    private val data = loadData()
    private val prepared = preprocess(data)
    private val fullModel = fitAllFeatures(prepared)

    @Test
    fun `データセットの形は原著と同じ`() {
        // 原著の出力: The dataset has 2518 rows, and 40 columns
        assertEquals(2518, data.count())
        assertEquals(40, data.columnNames().size)
    }

    @Test
    fun `単回帰の係数は原著と同じ数値になる`() {
        // 原著 scikit-learn の出力
        //   y-intercept: -6222669.083283698
        //   slope (coefficient of Area): 9753.940608184039
        val line = fitAreaOnly(data)

        assertEquals(-6222669.083283698, line.basePrice, 1e-6)
        assertEquals(9753.940608184039, line.pricePerRoom, 1e-9)
    }

    @Test
    fun `欠損を含む末尾の行を落とす`() {
        assertEquals(2434, VALID_ROWS)
        assertEquals(VALID_ROWS, prepared.rowCount)
        assertTrue(prepared.rowCount < data.count())
    }

    @Test
    fun `標準化した列は平均0分散1になる`() {
        for (name in listOf("Area", "No. of Bedrooms")) {
            val values = prepared.features.getValue(name)
            val mean = values.average()
            // 不偏分散（ddof = 1）なのでちょうど 1 になる
            val variance = values.sumOf { (it - mean) * (it - mean) } / (values.size - 1)

            assertEquals(0.0, mean, 1e-12)
            assertEquals(1.0, variance, 1e-12)
        }
    }

    @Test
    fun `標準化の統計量は Python 版と同じ`() {
        val standardizer = prepared.standardizer

        assertEquals(1644.1516023007396, standardizer.areaMean, 1e-9)
        assertEquals(748.1348121200747, standardizer.areaStd, 1e-9)
        assertEquals(2.6261298274445357, standardizer.bedroomsMean, 1e-12)
        assertEquals(0.6850461155463963, standardizer.bedroomsStd, 1e-12)
    }

    @Test
    fun `one-hot 符号化で277列になる`() {
        // 原著の出力: X_full.loc[0] ... Length: 277
        assertEquals(277, prepared.columnNames.size)

        val locationColumns = prepared.columnNames.filter { it.startsWith("Location_") }
        // 元の 40 列から Price と Location を除いた 38 列 + 地域の one-hot
        assertEquals(277 - 38, locationColumns.size)
    }

    @Test
    fun `全特徴量モデルの RMSE は原著とほぼ同じ`() {
        // 原著の出力: Root Mean Squared Error (RMSE) of the model: 3981401.4927888927
        // 係数は一意でないが、当てはまりの良さは解の取り方によらずほぼ同じになる
        val error = rmse(prepared.labels, predictAll(fullModel, prepared))

        assertEquals(3981401.4927888927, error, 3981401.4927888927 * 1e-4)
    }

    @Test
    fun `全特徴量モデルは単回帰より当てはまりが良い`() {
        val simple = fitAreaOnly(data)
        // 単回帰は標準化前の面積で学習しているので、標準化を戻してから予測する
        val standardizedArea = prepared.features.getValue("Area")
        val simplePredictions = DoubleArray(prepared.rowCount) {
            val rawArea = standardizedArea[it] * prepared.standardizer.areaStd +
                prepared.standardizer.areaMean
            simple.predict(rawArea)
        }

        val simpleRmse = rmse(prepared.labels, simplePredictions)
        val fullRmse = rmse(prepared.labels, predictAll(fullModel, prepared))

        assertTrue(fullRmse < simpleRmse, "full=$fullRmse simple=$simpleRmse")
    }

    @Test
    fun `新しい物件の予測は原著とほぼ同じ`() {
        // 原著の出力
        //   Predicted price for a house with size 1000 and 3 bedrooms: 6,006,016.00
        val predicted = predictNewHouse(fullModel, prepared, area = 1000.0, bedrooms = 3)

        assertEquals(6006016.00, predicted, 6006016.00 * 1e-2)
    }

    @Test
    fun `学習データに無い地域はエラーになる`() {
        val error = assertFailsWith<IllegalArgumentException> {
            predictNewHouse(fullModel, prepared, area = 1000.0, bedrooms = 3, location = "Atlantis")
        }

        assertTrue(error.message!!.contains("学習データに無い地域"))
    }
}
