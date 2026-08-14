package lib

import kotlin.math.sqrt
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.regression.MLP

/**
 * 原著ノートブック #14
 * `Chapter_10_Neural_Networks/House_price_predictions_neural_network.ipynb`。
 *
 * ニューラルネットワークを **回帰** に使う回。[#02][Nb02HousePrices] と同じ
 * ハイデラバードの住宅データを、今度は 3 層のネットワークで予測する。
 *
 * Smile は分類と回帰で **クラスが分かれている**。
 * 分類は `smile.classification.MLP`、回帰は `smile.regression.MLP` である。
 * 名前が同じで置き場所だけ違うので、import を間違えやすい。
 */
object Nb14HousePriceNetwork {

    /** 原著のネットワークの形。入力 38 に対して 38 → 128 → 64 → 1 */
    val HIDDEN_UNITS = intArrayOf(38, 128, 64)

    /**
     * 原著の学習回数は 10 だが、Smile では足りない。
     *
     * [#13][Nb13NeuralNetworkBoundary] と同じ理由（Adam が無い）に加え、
     * **価格が 1000 万の桁** なので勾配が大きく、学習が安定しない。
     */
    const val EPOCHS = 200

    /** 原著が指定している学習回数。比較用に残す */
    const val ORIGINAL_EPOCHS = 10

    /** 読み込んだ住宅データ */
    class Housing(val x: Array<DoubleArray>, val prices: DoubleArray) {
        val size: Int get() = prices.size
        val featureCount: Int get() = x[0].size
    }

    /** 平均と標準偏差。学習時と予測時で同じ値を使う必要がある */
    data class Scaler(val mean: Double, val std: Double) {
        fun scale(value: Double): Double = if (std == 0.0) 0.0 else (value - mean) / std
        fun unscale(value: Double): Double = value * std + mean
    }

    private fun scalerOf(values: DoubleArray): Scaler {
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        return Scaler(mean, sqrt(variance))
    }

    /**
     * 標準化した特徴量と価格、そして元に戻すための係数。
     *
     * **原著は標準化していない。** Keras の Adam は勾配の大きさを自動で調整するので、
     * 桁の違う特徴量（`Area` は 4 桁、`Resale` は 0 か 1）でも学習できる。
     * Smile の確率的勾配降下にはその仕組みが無く、標準化しないと発散する
     * （実測で RMSE が 10^29 に達した）。
     */
    class Scaled(
        val x: Array<DoubleArray>,
        val prices: DoubleArray,
        val priceScaler: Scaler,
    )

    /** 特徴量と価格をそれぞれ標準化する */
    fun standardize(data: Housing): Scaled {
        val featureScalers = (0 until data.featureCount).map { column ->
            scalerOf(DoubleArray(data.size) { data.x[it][column] })
        }
        val priceScaler = scalerOf(data.prices)

        return Scaled(
            Array(data.size) { row ->
                DoubleArray(data.featureCount) { column ->
                    featureScalers[column].scale(data.x[row][column])
                }
            },
            DoubleArray(data.size) { priceScaler.scale(data.prices[it]) },
            priceScaler,
        )
    }

    /**
     * ハイデラバードの住宅データを読み込む。
     *
     * `Location`（文字列）と `Price`（目的変数）を落として 38 列にする。
     */
    fun loadHousing(): Housing {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("Hyderabad.csv").toFile())
        val featureNames = frame.columnNames().filter { it != "Location" && it != "Price" }
        val columns = featureNames.map { name ->
            frame.getColumn(name).convertTo<Double>().toList()
        }
        val prices = frame.getColumn("Price").convertTo<Double>().toList()

        return Housing(
            Array(prices.size) { row -> DoubleArray(featureNames.size) { columns[it][row] } },
            prices.toDoubleArray(),
        )
    }

    /**
     * 原著と同じ形のネットワークを学習する。
     *
     * 回帰なので出力層は [OutputFunction.LINEAR]。値をそのまま出す。
     * 分類（[#13][Nb13NeuralNetworkBoundary]）ではシグモイドを使っていた。
     *
     * **出力層のビルダーも変える必要がある。** 分類で使った `Layer.mle`
     * （最尤法）に `LINEAR` を渡すと `Linear output function is not allowed
     * with likelihood cost function` で落ちる。回帰は `Layer.mse`
     * （平均二乗誤差）を使う。損失関数と出力関数の組み合わせに制約がある。
     *
     * **価格をそのまま学習させると発散する。** 1000 万の桁の値に対して
     * 勾配がそのまま掛かるためである。ここでは学習率を小さくして対処する。
     */
    fun fit(scaled: Scaled, epochs: Int = EPOCHS, seed: Int = 0): MLP {
        smile.math.MathEx.setSeed(seed.toLong())

        val network = MLP(
            Layer.input(scaled.x[0].size),
            Layer.rectifier(HIDDEN_UNITS[0]),
            Layer.rectifier(HIDDEN_UNITS[1]),
            Layer.rectifier(HIDDEN_UNITS[2]),
            Layer.mse(1, OutputFunction.LINEAR),
        )
        repeat(epochs) { network.update(scaled.x, scaled.prices) }
        return network
    }

    /** 全物件の価格を予測する。標準化を戻して円単位にする */
    fun predict(model: MLP, scaled: Scaled): DoubleArray =
        DoubleArray(scaled.prices.size) { scaled.priceScaler.unscale(model.predict(scaled.x[it])) }

    /** 学習データに対する RMSE。円単位で返す */
    fun rmse(model: MLP, data: Housing, scaled: Scaled): Double {
        val predictions = predict(model, scaled)
        val squared = data.prices.indices.sumOf { index ->
            val difference = data.prices[index] - predictions[index]
            difference * difference
        }
        return sqrt(squared / data.size)
    }

    /**
     * 常に平均価格を答えたときの RMSE。
     *
     * これを下回れなければ、ネットワークは何も学習できていない。
     */
    fun baselineRmse(data: Housing): Double {
        val mean = data.prices.average()
        return sqrt(data.prices.sumOf { (it - mean) * (it - mean) } / data.size)
    }
}
