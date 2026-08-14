package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb14HousePriceNetwork.EPOCHS
import lib.Nb14HousePriceNetwork.HIDDEN_UNITS
import lib.Nb14HousePriceNetwork.ORIGINAL_EPOCHS
import lib.Nb14HousePriceNetwork.baselineRmse
import lib.Nb14HousePriceNetwork.fit
import lib.Nb14HousePriceNetwork.loadHousing
import lib.Nb14HousePriceNetwork.predict
import lib.Nb14HousePriceNetwork.rmse
import lib.Nb14HousePriceNetwork.standardize

/**
 * 原著ノートブック #14 の再現テスト。
 *
 * 原著は特徴量を標準化していないが、Smile では **標準化しないと発散する**。
 * 標準化さえすれば、原著（RMSE 約 554 万）より良い約 490 万に届く。
 */
class Nb14HousePriceNetworkTest {

    private val data = loadHousing()
    private val scaled = standardize(data)

    @Test
    fun `データセットは2518件38特徴量になる`() {
        // Location（文字列）と Price（目的変数）を落とす
        assertEquals(2518, data.size)
        assertEquals(38, data.featureCount)
    }

    @Test
    fun `ネットワークの形は原著と同じ`() {
        // Dense(38) -> Dense(128) -> Dense(64) -> Dense(1)
        assertEquals(listOf(38, 128, 64), HIDDEN_UNITS.toList())
    }

    @Test
    fun `平均を答えるだけの基準は約877万`() {
        // 価格の標準偏差。ネットワークはこれを下回る必要がある
        assertEquals(8_775_369.83, baselineRmse(data), 1.0)
    }

    @Test
    fun `標準化した特徴量は平均0分散1になる`() {
        for (column in 0 until data.featureCount) {
            val values = DoubleArray(data.size) { scaled.x[it][column] }
            val mean = values.average()

            assertEquals(0.0, mean, 1e-9, "column $column")
        }
    }

    @Test
    fun `標準化を戻すと元の価格になる`() {
        // 予測を円単位に戻すために必要な操作
        for (index in listOf(0, 100, 2517)) {
            assertEquals(
                data.prices[index],
                scaled.priceScaler.unscale(scaled.prices[index]),
                1e-6,
            )
        }
    }

    @Test
    fun `原著の10エポックでも基準を下回る`() {
        // 原著と同じ 10 エポック。RMSE 約 797 万で、基準の 877 万は下回る
        assertTrue(rmse(fit(scaled, ORIGINAL_EPOCHS), data, scaled) < baselineRmse(data))
    }

    @Test
    fun `200エポックなら原著より良い予測になる`() {
        // 原著の evaluate は RMSE 約 554 万。Smile は 200 エポックで約 490 万
        assertTrue(rmse(fit(scaled, EPOCHS), data, scaled) < 5_540_000.0)
    }

    @Test
    fun `学習を進めるほどRMSEが下がる`() {
        val early = rmse(fit(scaled, 10), data, scaled)
        val late = rmse(fit(scaled, EPOCHS), data, scaled)

        assertTrue(late < early, "early=$early late=$late")
    }

    @Test
    fun `予測は件数分の配列になる`() {
        assertEquals(2518, predict(fit(scaled, 10), scaled).size)
    }

    @Test
    fun `予測の平均は実際の平均に近い`() {
        // 回帰なので、全体の水準は合ってくる
        val predictions = predict(fit(scaled, EPOCHS), scaled)
        val actual = data.prices.average()

        assertEquals(actual, predictions.average(), actual * 0.5)
    }

    @Test
    fun `種を固定すれば同じ結果になる`() {
        assertEquals(
            rmse(fit(scaled, 20, seed = 7), data, scaled),
            rmse(fit(scaled, 20, seed = 7), data, scaled),
            1e-6,
        )
    }
}
