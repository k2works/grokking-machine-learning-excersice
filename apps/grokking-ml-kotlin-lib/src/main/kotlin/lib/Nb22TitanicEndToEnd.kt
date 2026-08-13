package lib

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV

/**
 * 原著ノートブック #22 `Chapter_13_End_to_end_example/End_to_end_example.ipynb`。
 *
 * タイタニックの生存予測を、前処理からモデル選択まで通しでやる回である。
 * 本の最終章で、これまでのアルゴリズムが一堂に会する。
 *
 * **移植で難しいのは学習ではなく前処理と分割** である。
 * 欠損の埋め方・one-hot の列順・訓練とテストの切り分け方が合わなければ、
 * どんなモデルを使っても数字は合わない。ここではまず
 * **前処理と分割を原著と一致させる** ことに集中する。
 *
 * scikit-learn の `train_test_split(random_state=100)` は
 * **メルセンヌツイスタの並べ替え** を使うので、JVM では同じ分割を作れない。
 * その代わり、原著と同じ件数（534 / 178 / 179）になることと、
 * 分割が全体を覆うことを確かめる。
 */
object Nb22TitanicEndToEnd {

    /** 年齢の離散化の区切り。10 歳刻みで 8 区間 */
    val AGE_BINS = listOf(0, 10, 20, 30, 40, 50, 60, 70, 80)

    /** 学習に使わない列 */
    val DROPPED_COLUMNS = listOf("Name", "Ticket", "PassengerId")

    /** 1 人ぶんの生データ（必要な列だけ） */
    data class Passenger(
        val survived: Int,
        val pclass: Int,
        val sex: String,
        val age: Double?,
        val sibSp: Int,
        val parch: Int,
        val fare: Double,
        val embarked: String?,
    )

    /** 生のタイタニックデータ 891 行 */
    fun loadRaw(): List<Passenger> {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("titanic.csv").toFile())
        val survived = frame.getColumn("Survived").convertTo<Int>().toList()
        val pclass = frame.getColumn("Pclass").convertTo<Int>().toList()
        val sex = frame.getColumn("Sex").convertTo<String>().toList()
        val age = frame.getColumn("Age").toList().map { (it as? Number)?.toDouble() }
        val sibSp = frame.getColumn("SibSp").convertTo<Int>().toList()
        val parch = frame.getColumn("Parch").convertTo<Int>().toList()
        val fare = frame.getColumn("Fare").convertTo<Double>().toList()
        // Kotlin DataFrame は文字列列を Char 等に推論することがある。
        // 型で受けずに文字列化してから空を欠損とみなす
        val embarked = frame.getColumn("Embarked").toList().map { it?.toString()?.ifBlank { null } }

        return survived.indices.map {
            Passenger(
                survived[it], pclass[it], sex[it], age[it],
                sibSp[it], parch[it], fare[it], embarked[it],
            )
        }
    }

    /** `Cabin` は列ごと落とすので、そもそも読まない。欠損数だけ数える */
    fun cabinMissingCount(): Int {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("titanic.csv").toFile())
        return frame.getColumn("Cabin").toList().count { it == null }
    }

    /** 欠損のある列と、その数 */
    fun missingCounts(data: List<Passenger>): Map<String, Int> = mapOf(
        "Age" to data.count { it.age == null },
        "Cabin" to cabinMissingCount(),
        "Embarked" to data.count { it.embarked == null },
    )

    /** 年齢の中央値。原著は 28.0 */
    fun medianAge(data: List<Passenger>): Double {
        val ages = data.mapNotNull { it.age }.sorted()
        val middle = ages.size / 2
        return if (ages.size % 2 == 0) (ages[middle - 1] + ages[middle]) / 2.0 else ages[middle]
    }

    /**
     * 欠損を片付ける。
     *
     * - `Age` は中央値（28.0）で埋める
     * - `Embarked` は `U`（Unknown）という新しい区分にする
     */
    fun clean(raw: List<Passenger>): List<Passenger> {
        val median = medianAge(raw)
        return raw.map { it.copy(age = it.age ?: median, embarked = it.embarked ?: "U") }
    }

    /** 年齢がどの区間に入るか。`pandas.cut` と同じ **左を開き右を閉じる** 区間 */
    fun ageBin(age: Double): Int = AGE_BINS.indices.drop(1).first { age <= AGE_BINS[it] } - 1

    /** one-hot 符号化した特徴量の列名。原著の列順をそのまま保つ */
    val FEATURE_NAMES: List<String> = buildList {
        add("SibSp")
        add("Parch")
        add("Fare")
        addAll(listOf("Sex_female", "Sex_male"))
        addAll(listOf("Embarked_C", "Embarked_Q", "Embarked_S", "Embarked_U"))
        addAll(listOf("Pclass_1", "Pclass_2", "Pclass_3"))
        for (index in 0 until AGE_BINS.size - 1) {
            add("Categorized_age_(${AGE_BINS[index]}, ${AGE_BINS[index + 1]}]")
        }
    }

    /** 1 人ぶんの特徴量ベクトル */
    fun featuresOf(passenger: Passenger): DoubleArray {
        val values = DoubleArray(FEATURE_NAMES.size)
        values[0] = passenger.sibSp.toDouble()
        values[1] = passenger.parch.toDouble()
        values[2] = passenger.fare
        values[FEATURE_NAMES.indexOf("Sex_${passenger.sex}")] = 1.0
        values[FEATURE_NAMES.indexOf("Embarked_${passenger.embarked}")] = 1.0
        values[FEATURE_NAMES.indexOf("Pclass_${passenger.pclass}")] = 1.0
        val bin = ageBin(passenger.age!!)
        values[FEATURE_NAMES.indexOf("Categorized_age_(${AGE_BINS[bin]}, ${AGE_BINS[bin + 1]}]")] = 1.0
        return values
    }

    /** 前処理した特徴量とラベル */
    class Prepared(val x: Array<DoubleArray>, val y: IntArray) {
        val size: Int get() = y.size
        val featureCount: Int get() = if (x.isEmpty()) 0 else x[0].size
    }

    /** 生データを特徴量に直す */
    fun preprocess(data: List<Passenger>): Prepared =
        Prepared(data.map { featuresOf(it) }.toTypedArray(), data.map { it.survived }.toIntArray())

    /**
     * 訓練 / 検証 / テストの 3 分割。
     *
     * 原著と同じ件数（534 / 178 / 179）にするが、**同じ行が同じ組に入る
     * わけではない**。scikit-learn の並べ替えは JVM では再現できない。
     */
    class Split(private val prepared: Prepared, seed: Long = 100L) {
        private val order: List<Int> = prepared.y.indices.shuffled(kotlin.random.Random(seed))

        // 原著は 6:4 に切り、その 4 をさらに半分ずつにする。
        // scikit-learn は `test_size` の側を **切り上げる**（ceil）ので、
        // 891 -> 534 / 357、357 -> 178 / 179 になる
        private val restSize = Math.ceil(prepared.size * 0.4).toInt()
        private val trainSize = prepared.size - restSize
        private val testSize = Math.ceil(restSize * 0.5).toInt()
        private val validationSize = restSize - testSize

        private fun slice(from: Int, count: Int): Pair<Array<DoubleArray>, IntArray> {
            val indices = order.subList(from, from + count)
            return indices.map { prepared.x[it] }.toTypedArray() to indices.map { prepared.y[it] }.toIntArray()
        }

        val train = slice(0, trainSize)
        val validation = slice(trainSize, validationSize)
        val test = slice(trainSize + validationSize, testSize)
    }

    /** 「全員死亡」と答えたときの正解率 */
    fun majorityBaseline(labels: IntArray): Double =
        labels.count { it == 0 }.toDouble() / labels.size

    /** 正解率 */
    fun accuracy(predicted: IntArray, actual: IntArray): Double {
        val correct = predicted.indices.count { predicted[it] == actual[it] }
        return correct.toDouble() / actual.size
    }

    /**
     * F1 スコア。適合率と再現率の調和平均。
     *
     * 正解率だけでは不十分な理由を見せるための指標である。
     * 生存者が少数派なので、「全員死亡」と答えても正解率は 6 割を超える。
     */
    fun f1Score(predicted: IntArray, actual: IntArray): Double {
        val truePositive = predicted.indices.count { predicted[it] == 1 && actual[it] == 1 }
        val falsePositive = predicted.indices.count { predicted[it] == 1 && actual[it] == 0 }
        val falseNegative = predicted.indices.count { predicted[it] == 0 && actual[it] == 1 }

        if (truePositive == 0) return 0.0
        val precision = truePositive.toDouble() / (truePositive + falsePositive)
        val recall = truePositive.toDouble() / (truePositive + falseNegative)
        return 2 * precision * recall / (precision + recall)
    }

    // ---- モデル ----

    /** Smile のデータフレームに直す。木系のモデルは `Formula` で列名を要求する */
    fun frameOf(x: Array<DoubleArray>, y: IntArray): smile.data.DataFrame {
        val labelField = smile.data.type.StructField(
            "Survived",
            smile.data.type.DataTypes.IntegerType,
            smile.data.measure.NominalScale("died", "survived"),
        )
        val vectors = buildList<smile.data.vector.BaseVector<*, *, *>> {
            FEATURE_NAMES.forEachIndexed { column, name ->
                add(smile.data.vector.DoubleVector.of(name, DoubleArray(y.size) { x[it][column] }))
            }
            add(smile.data.vector.IntVector.of(labelField, y))
        }
        return smile.data.DataFrame.of(*vectors.toTypedArray())
    }

    private val formula = smile.data.formula.Formula.lhs("Survived")

    /**
     * 原著が試す 7 つのモデル。名前 -> 予測関数を返す。
     *
     * 原著は種を渡していない。木系は乱数を使うので、
     * **指定しないと毎回変わる**。ここでは再現のために 0 を固定する。
     */
    fun fitAll(split: Split): Map<String, (Array<DoubleArray>) -> IntArray> {
        val (trainX, trainY) = split.train
        val trainFrame = frameOf(trainX, trainY)
        smile.math.MathEx.setSeed(0L)

        fun fromFrame(model: smile.classification.Classifier<smile.data.Tuple>):
            (Array<DoubleArray>) -> IntArray = { rows ->
            val frame = frameOf(rows, IntArray(rows.size))
            IntArray(rows.size) { model.predict(frame[it]) }
        }

        fun fromVector(model: smile.classification.Classifier<DoubleArray>):
            (Array<DoubleArray>) -> IntArray = { rows -> IntArray(rows.size) { model.predict(rows[it]) } }

        val signed = IntArray(trainY.size) { if (trainY[it] == 1) 1 else -1 }

        return linkedMapOf(
            "Logistic regression" to
                fromVector(smile.classification.LogisticRegression.binomial(trainX, trainY)),
            "Decision tree" to fromFrame(
                smile.classification.DecisionTree.fit(
                    formula, trainFrame, smile.base.cart.SplitRule.GINI, 20, 10000, 1,
                ),
            ),
            // scikit-learn の GaussianNB に対応する。列ごとに正規分布を当てる
            "Naive Bayes" to naiveBayesPredictor(trainX, trainY),
            // scikit-learn の SVC の既定は RBF カーネル・gamma='scale'
            "SVM" to svmPredictor(trainX, signed),
            "Random forest" to fromFrame(
                smile.classification.RandomForest.fit(
                    formula, trainFrame, 100, 4, smile.base.cart.SplitRule.GINI, 20, 10000, 1, 1.0,
                ),
            ),
            "Gradient boosting" to fromFrame(
                smile.classification.GradientTreeBoost.fit(formula, trainFrame),
            ),
            "AdaBoost" to fromFrame(
                smile.classification.AdaBoost.fit(formula, trainFrame, 50, 10000, 2, 1),
            ),
        )
    }

    /** ガウシアン素朴ベイズの予測関数 */
    private fun naiveBayesPredictor(
        trainX: Array<DoubleArray>,
        trainY: IntArray,
    ): (Array<DoubleArray>) -> IntArray {
        val model = gaussianNaiveBayes(trainX, trainY)
        return { rows -> IntArray(rows.size) { model(rows[it]) } }
    }

    /** RBF カーネル SVM の予測関数。ラベルは Smile の要求どおり ±1 で渡す */
    private fun svmPredictor(
        trainX: Array<DoubleArray>,
        signed: IntArray,
    ): (Array<DoubleArray>) -> IntArray {
        val gamma = 1.0 / (FEATURE_NAMES.size * varianceOf(trainX))
        val kernel = smile.math.kernel.GaussianKernel(Math.sqrt(1.0 / (2 * gamma)))
        val model = smile.classification.SVM.fit(trainX, signed, kernel, 1.0, 1e-3)
        return { rows -> IntArray(rows.size) { if (model.predict(rows[it]) == 1) 1 else 0 } }
    }

    /** 全要素の分散。scikit-learn の `gamma='scale'` が使うもの */
    fun varianceOf(x: Array<DoubleArray>): Double {
        val values = x.flatMap { it.asIterable() }
        val mean = values.average()
        return values.sumOf { (it - mean) * (it - mean) } / values.size
    }

    /**
     * ガウシアン素朴ベイズ。
     *
     * Smile の `NaiveBayes` は多項分布・ベルヌーイ分布しか持たず、
     * **連続値を扱う `GaussianNB` に当たるものが無い**。
     * 列ごとに正規分布を当てる素直な実装を書いた。
     */
    fun gaussianNaiveBayes(x: Array<DoubleArray>, y: IntArray): (DoubleArray) -> Int {
        val classes = listOf(0, 1)
        // scikit-learn の `var_smoothing`。**全特徴量の最大分散 × 1e-9** を足す。
        // one-hot の列は分散がごく小さいので、固定値 1e-9 を足すだけでは足りない。
        // それをやると正解率が 0.38 まで落ちた（多数派を答えるより悪い）
        val smoothing = 1e-9 * (0 until FEATURE_NAMES.size).maxOf { column ->
            val values = x.map { it[column] }
            val mean = values.average()
            values.sumOf { (it - mean) * (it - mean) } / values.size
        }
        val stats = classes.associateWith { label ->
            val rows = x.indices.filter { y[it] == label }
            List(FEATURE_NAMES.size) { column ->
                val values = rows.map { x[it][column] }
                val mean = values.average()
                val variance = values.sumOf { (it - mean) * (it - mean) } / values.size + smoothing
                mean to variance
            }
        }
        val priors = classes.associateWith { label ->
            Math.log(y.count { it == label }.toDouble() / y.size)
        }

        return { row ->
            classes.maxByOrNull { label ->
                priors.getValue(label) +
                    stats.getValue(label).mapIndexed { column, (mean, variance) ->
                        -0.5 * Math.log(2 * Math.PI * variance) -
                            (row[column] - mean) * (row[column] - mean) / (2 * variance)
                    }.sum()
            }!!
        }
    }
}
