package lib

import kotlin.math.sqrt
import kotlin.random.Random
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.regression.OLS

/**
 * 原著ノートブック #01 `Chapter_03_Linear_Regression/Coding_linear_regression.ipynb`。
 *
 * 部屋数から住宅価格を予測する線形回帰を、3 つのトリック（simple / absolute / square）で
 * 学習したあと、同じ問題を Smile の [OLS] に解かせて突き合わせる。
 * scikit-learn の `LinearRegression` に相当するのが Smile の `OLS` である。
 *
 * 乱数は Python の Mersenne Twister と別物なので、トリックの学習結果は原著と
 * 一致しない。一方 [fitWithSmile] は閉じた式で解くため、原著と同じ数値になる。
 */
object Nb01LinearRegression {

    /** 原著が使うデータセット。部屋数と価格 */
    val FEATURES = intArrayOf(1, 2, 3, 5, 6, 7)
    val LABELS = intArrayOf(155, 197, 244, 356, 407, 448)

    /** 学習された直線。原著の `price_per_room` と `base_price` に対応する */
    data class Line(val pricePerRoom: Double, val basePrice: Double) {
        fun predict(numRooms: Double): Double = basePrice + pricePerRoom * numRooms
    }

    /** 学習の途中経過。原著が学習ループの中で描いていたものを記録する */
    data class TrainingLog(
        val line: Line,
        /** 各エポック開始時点の直線 */
        val history: List<Line>,
        /** 各エポック開始時点の RMSE */
        val errors: List<Double>,
    )

    /** どのトリックで学習するか。原著はコメントアウトで切り替えていた */
    enum class Trick { SIMPLE, ABSOLUTE, SQUARE }

    /**
     * シンプルトリック。予測が外れた向きに、小さな乱数だけ直線を動かす。
     *
     * 原著の 4 つの `if` をそのまま写している。3 番目の分岐だけ `basePrice` を
     * 減らす非対称な書き方も原著のままである。
     */
    fun simpleTrick(
        basePrice: Double,
        pricePerRoom: Double,
        numRooms: Double,
        price: Double,
        random: Random,
    ): Line {
        val smallRandom1 = random.nextDouble() * 0.1
        val smallRandom2 = random.nextDouble() * 0.1
        val predictedPrice = basePrice + pricePerRoom * numRooms
        var slope = pricePerRoom
        var intercept = basePrice
        if (price > predictedPrice && numRooms > 0) {
            slope += smallRandom1
            intercept += smallRandom2
        }
        if (price > predictedPrice && numRooms < 0) {
            slope -= smallRandom1
            intercept += smallRandom2
        }
        if (price < predictedPrice && numRooms > 0) {
            slope -= smallRandom1
            intercept -= smallRandom2
        }
        if (price < predictedPrice && numRooms < 0) {
            slope -= smallRandom1
            intercept += smallRandom2
        }
        return Line(slope, intercept)
    }

    /** 絶対トリック。外れた向きへ、学習率と部屋数に比例した幅で動かす */
    fun absoluteTrick(
        basePrice: Double,
        pricePerRoom: Double,
        numRooms: Double,
        price: Double,
        learningRate: Double,
    ): Line {
        val predictedPrice = basePrice + pricePerRoom * numRooms
        return if (price > predictedPrice) {
            Line(pricePerRoom + learningRate * numRooms, basePrice + learningRate)
        } else {
            Line(pricePerRoom - learningRate * numRooms, basePrice - learningRate)
        }
    }

    /** 二乗トリック。誤差の大きさにも比例して動かす。分岐が要らなくなる */
    fun squareTrick(
        basePrice: Double,
        pricePerRoom: Double,
        numRooms: Double,
        price: Double,
        learningRate: Double,
    ): Line {
        val predictedPrice = basePrice + pricePerRoom * numRooms
        val error = price - predictedPrice
        return Line(
            pricePerRoom + learningRate * numRooms * error,
            basePrice + learningRate * error,
        )
    }

    /** 二乗平均平方根誤差。原著は NumPy の内積で書いていた */
    fun rmse(labels: IntArray, prediction: Double): Double {
        val squaredSum = labels.sumOf { label ->
            val difference = label - prediction
            difference * difference
        }
        return sqrt(squaredSum / labels.size)
    }

    /**
     * トリックを繰り返して直線を学習する。
     *
     * 乱数の消費順序（重みの初期化 2 回 → 毎エポックの添字 1 回）は原著と同じに
     * してあるが、Kotlin の [Random] は Python と別のアルゴリズムなので、
     * 同じ種を渡しても同じ数列にはならない。数値の一致ではなく収束で検証する。
     */
    fun linearRegression(
        features: IntArray = FEATURES,
        labels: IntArray = LABELS,
        learningRate: Double = 0.01,
        epochs: Int = 1000,
        trick: Trick = Trick.SQUARE,
        seed: Int = 0,
    ): TrainingLog {
        val random = Random(seed)
        var line = Line(random.nextDouble(), random.nextDouble())
        val history = ArrayList<Line>(epochs)
        val errors = ArrayList<Double>(epochs)

        repeat(epochs) {
            history.add(line)
            // 原著は features[0] だけを使って誤差を測っている
            errors.add(rmse(labels, features[0] * line.pricePerRoom + line.basePrice))

            val i = random.nextInt(features.size)
            val numRooms = features[i].toDouble()
            val price = labels[i].toDouble()

            line = when (trick) {
                Trick.SQUARE ->
                    squareTrick(line.basePrice, line.pricePerRoom, numRooms, price, learningRate)
                Trick.ABSOLUTE ->
                    absoluteTrick(line.basePrice, line.pricePerRoom, numRooms, price, learningRate)
                Trick.SIMPLE ->
                    simpleTrick(line.basePrice, line.pricePerRoom, numRooms, price, random)
            }
        }

        return TrainingLog(line, history, errors)
    }

    /**
     * 同じ問題を Smile の [OLS] に解かせる。
     *
     * scikit-learn は特徴量を 2 次元配列に `reshape` させるが、Smile は列に名前の
     * ある [DataFrame] を要求する。式は [Formula.lhs] で「どの列が目的変数か」を指定する。
     */
    fun fitWithSmile(
        features: IntArray = FEATURES,
        labels: IntArray = LABELS,
    ): Line {
        val df = DataFrame.of(
            DoubleVector.of("rooms", DoubleArray(features.size) { features[it].toDouble() }),
            DoubleVector.of("price", DoubleArray(labels.size) { labels[it].toDouble() }),
        )
        val model = OLS.fit(Formula.lhs("price"), df)
        // Smile は切片を coefficients() に含めず intercept() で返す
        return Line(model.coefficients()[0], model.intercept())
    }
}
