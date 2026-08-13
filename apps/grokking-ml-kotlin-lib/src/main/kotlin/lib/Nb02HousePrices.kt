package lib

import kotlin.math.sqrt
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.OLS

/**
 * 原著ノートブック #02 `Chapter_03_Linear_Regression/House_price_predictions.ipynb`。
 *
 * ハイデラバードの住宅データ 2518 件から価格を予測する。面積 1 つだけを使う単回帰と、
 * 全 40 列を前処理してから使う重回帰の 2 本立てになっている。
 *
 * 前処理は原著の手順をそのまま踏襲する。
 * 1. 欠損（`9` で符号化されている）を含む末尾の行を落とす
 * 2. 数値列（`Area` と `No. of Bedrooms`）を標準化する
 * 3. カテゴリ列（`Location`）を one-hot 符号化する
 *
 * pandas の `get_dummies` にあたるものが Kotlin DataFrame には無いので、そこは自前で書く。
 */
object Nb02HousePrices {

    /** 原著が欠損ありとして切り落とす位置。これ以降の行は `9` で符号化された欠損を含む */
    const val VALID_ROWS = 2434

    /** 標準化に使った平均と標準偏差。新しい物件も同じ値で変換する必要がある */
    data class Standardizer(
        val areaMean: Double,
        val areaStd: Double,
        val bedroomsMean: Double,
        val bedroomsStd: Double,
    ) {
        fun transform(area: Double, bedrooms: Double): Pair<Double, Double> =
            (area - areaMean) / areaStd to (bedrooms - bedroomsMean) / bedroomsStd
    }

    /** 前処理を通したあとのデータ一式 */
    data class Preprocessed(
        /** 列名 -> 値。列の順序を保つため LinkedHashMap を使う */
        val features: Map<String, DoubleArray>,
        val labels: DoubleArray,
        val standardizer: Standardizer,
    ) {
        val columnNames: List<String> get() = features.keys.toList()
        val rowCount: Int get() = labels.size
    }

    /** ハイデラバードの住宅データを読み込む。2518 行 × 40 列 */
    fun loadData(): DataFrame<*> = DataFrame.readCSV(Datasets.path("Hyderabad.csv").toFile())

    private fun DataFrame<*>.doubleColumn(name: String): DoubleArray =
        getColumn(name).convertTo<Double>().toList().toDoubleArray()

    private fun DataFrame<*>.stringColumn(name: String): List<String> =
        getColumn(name).toList().map { it.toString() }

    /**
     * 面積 1 列だけで価格を予測する単回帰。
     *
     * scikit-learn は `data[['Area']]` と 2 次元にして渡すが、Smile は列に名前のある
     * データフレームを組み立てて [Formula.lhs] で目的変数を指すのが作法である。
     */
    fun fitAreaOnly(data: DataFrame<*>): Nb01LinearRegression.Line {
        val area = data.doubleColumn("Area")
        val price = data.doubleColumn("Price")
        val df = smile.data.DataFrame.of(
            DoubleVector.of("Area", area),
            DoubleVector.of("Price", price),
        )
        val model = OLS.fit(Formula.lhs("Price"), df)
        return Nb01LinearRegression.Line(model.coefficients()[0], model.intercept())
    }

    /** 不偏分散（ddof = 1）の標準偏差。pandas の `std()` の既定に合わせる */
    private fun DoubleArray.sampleStd(): Double {
        val mean = average()
        return sqrt(sumOf { (it - mean) * (it - mean) } / (size - 1))
    }

    /** 原著の前処理 3 手順をそのまま行う */
    fun preprocess(data: DataFrame<*>): Preprocessed {
        val allArea = data.doubleColumn("Area")
        val allBedrooms = data.doubleColumn("No. of Bedrooms")
        val allPrice = data.doubleColumn("Price")
        val allLocation = data.stringColumn("Location")

        val area = allArea.copyOf(VALID_ROWS)
        val bedrooms = allBedrooms.copyOf(VALID_ROWS)
        val price = allPrice.copyOf(VALID_ROWS)
        val location = allLocation.subList(0, VALID_ROWS)

        val standardizer = Standardizer(
            areaMean = area.average(),
            areaStd = area.sampleStd(),
            bedroomsMean = bedrooms.average(),
            bedroomsStd = bedrooms.sampleStd(),
        )

        // 列の順序は Python 版と合わせる。Area, No. of Bedrooms, 残りの数値列, 地域の one-hot
        val features = LinkedHashMap<String, DoubleArray>()
        features["Area"] = DoubleArray(VALID_ROWS) {
            (area[it] - standardizer.areaMean) / standardizer.areaStd
        }
        features["No. of Bedrooms"] = DoubleArray(VALID_ROWS) {
            (bedrooms[it] - standardizer.bedroomsMean) / standardizer.bedroomsStd
        }

        val passthrough = data.columnNames()
            .filter { it !in setOf("Price", "Area", "No. of Bedrooms", "Location") }
        for (name in passthrough) {
            features[name] = data.doubleColumn(name).copyOf(VALID_ROWS)
        }

        // pandas の get_dummies に相当する処理。地域名を昇順に並べて列を作る
        for (name in location.distinct().sorted()) {
            features["Location_$name"] =
                DoubleArray(VALID_ROWS) { if (location[it] == name) 1.0 else 0.0 }
        }

        return Preprocessed(features, price, standardizer)
    }

    /**
     * 前処理済みの全特徴量で重回帰を学習する。
     *
     * one-hot 符号化した地域の列は合計すると常に 1 になり、切片と線形従属になる。
     * 既定の QR 分解では正則性を仮定するので、擬似逆行列を求める SVD を明示的に選ぶ。
     */
    fun fitAllFeatures(prepared: Preprocessed): smile.regression.LinearModel {
        val vectors = buildList {
            add(DoubleVector.of("Price", prepared.labels))
            prepared.features.forEach { (name, values) -> add(DoubleVector.of(name, values)) }
        }
        val df = smile.data.DataFrame.of(*vectors.toTypedArray())
        val properties = java.util.Properties().apply {
            setProperty("smile.ols.method", "svd")
        }
        return OLS.fit(Formula.lhs("Price"), df, properties)
    }

    /** 二乗平均平方根誤差 */
    fun rmse(labels: DoubleArray, predictions: DoubleArray): Double {
        var squaredSum = 0.0
        for (i in labels.indices) {
            val difference = labels[i] - predictions[i]
            squaredSum += difference * difference
        }
        return sqrt(squaredSum / labels.size)
    }

    /** 学習データ全体に対する予測を返す */
    fun predictAll(
        model: smile.regression.LinearModel,
        prepared: Preprocessed,
    ): DoubleArray {
        val names = prepared.columnNames
        return DoubleArray(prepared.rowCount) { row ->
            model.predict(DoubleArray(names.size) { column ->
                prepared.features.getValue(names[column])[row]
            })
        }
    }

    /**
     * 新しい物件の価格を予測する。
     *
     * 学習時と同じ列・同じ順序の 1 行を組み立てるのが要点である。地域の列は
     * 1 つだけを 1 にして残りは 0 にする。
     */
    fun predictNewHouse(
        model: smile.regression.LinearModel,
        prepared: Preprocessed,
        area: Double,
        bedrooms: Int,
        location: String = "Gachibowli",
    ): Double {
        val names = prepared.columnNames
        val locationColumn = "Location_$location"
        require(locationColumn in names) { "学習データに無い地域です: $location" }

        val (scaledArea, scaledBedrooms) =
            prepared.standardizer.transform(area, bedrooms.toDouble())
        val row = DoubleArray(names.size)
        names.forEachIndexed { index, name ->
            row[index] = when (name) {
                "Area" -> scaledArea
                "No. of Bedrooms" -> scaledBedrooms
                locationColumn -> 1.0
                else -> 0.0
            }
        }
        return model.predict(row)
    }
}
