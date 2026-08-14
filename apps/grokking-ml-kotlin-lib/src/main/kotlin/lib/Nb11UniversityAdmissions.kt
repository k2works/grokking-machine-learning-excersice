package lib

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.io.readCSV
import smile.base.cart.SplitRule
import smile.classification.DecisionTree
import smile.data.formula.Formula
import smile.data.measure.NominalScale
import smile.data.type.DataTypes
import smile.data.type.StructField
import smile.data.vector.DoubleVector
import smile.data.vector.IntVector

/**
 * 原著ノートブック #11 `Chapter_09_Decision_Trees/University_Admissions.ipynb`。
 *
 * 大学院の入学審査データ 400 件から、合格するかどうかを決定木で当てる。
 * 原著は **同じデータで木の大きさを変えて、過学習の様子を見せる** 構成になっている。
 *
 * Smile の `DecisionTree` は葉の最小件数（`nodeSize`）を指定できるが、
 * scikit-learn の `min_samples_split` にあたるものは無い。
 * そこは制限が 1 つ少ない状態で学習することになる。
 */
object Nb11UniversityAdmissions {

    /** 原著が合格とみなす基準 */
    const val ADMISSION_THRESHOLD = 0.75

    /** 特徴量の列。原著の順序をそのまま保つ */
    val FEATURE_NAMES = listOf(
        "GRE Score", "TOEFL Score", "University Rating",
        "SOP", "LOR", "CGPA", "Research",
    )

    /** 2 特徴量だけで学習するときに使う列 */
    val EXAM_FEATURES = listOf("GRE Score", "TOEFL Score")

    /** 分割に使われた特徴量としきい値 */
    data class Split(val feature: String, val threshold: Double)

    /** 読み込んだデータ。列名 -> 値と、合否ラベル */
    class Admissions(val columns: Map<String, DoubleArray>, val admitted: IntArray) {
        val size: Int get() = admitted.size

        /** 指定した列だけで Smile のデータフレームを組み立てる */
        fun toFrame(featureNames: List<String>): smile.data.DataFrame {
            val labelField =
                StructField("Admitted", DataTypes.IntegerType, NominalScale("false", "true"))
            val vectors = buildList {
                featureNames.forEach { add(DoubleVector.of(it, columns.getValue(it))) }
                add(IntVector.of(labelField, admitted))
            }
            return smile.data.DataFrame.of(*vectors.toTypedArray())
        }
    }

    /**
     * 入学審査データを読み込み、合否のラベルを付ける。
     *
     * 原著は `Chance of Admit`（合格確率）を 0.75 で切って 2 値にし、元の列を落としている。
     */
    fun loadData(): Admissions {
        val frame: DataFrame<*> = DataFrame.readCSV(Datasets.path("Admission_Predict.csv").toFile())
        val columns = FEATURE_NAMES.associateWith { name ->
            frame.getColumn(name).convertTo<Double>().toList().toDoubleArray()
        }
        val chance = frame.getColumn("Chance of Admit").convertTo<Double>().toList()
        val admitted = IntArray(chance.size) { if (chance[it] >= ADMISSION_THRESHOLD) 1 else 0 }
        return Admissions(columns, admitted)
    }

    /**
     * 決定木を学習する。`maxDepth` は **Smile の数え方** をそのまま渡す。
     *
     * [#10][Nb10DecisionTreeBoundary] の 12 点データでは「Smile の深さ = scikit-learn の
     * 深さ + 1」で対応が付いた。しかしこの 400 件のデータでは対応が付かない。
     * 実測すると次のように **飛び飛びに変化** し、途中の深さでは木が育たない。
     *
     * | Smile の `maxDepth` | 節の数 | 正解率 |
     * | ---: | ---: | ---: |
     * | 1 | 1 | 0.550 |
     * | 2〜4 | 3 | 0.880 |
     * | 5 以上 | 15 | 0.890 |
     *
     * `maxNodes` を 2 から 10000 まで変えても結果は変わらなかったので、
     * 深さの制限とは別の停止条件が効いている。深さを scikit-learn に読み替えるのは
     * 諦め、**実測した値で設定する**。
     */
    fun fit(
        data: Admissions,
        featureNames: List<String> = FEATURE_NAMES,
        maxDepth: Int = 20,
        minLeafSize: Int = 1,
    ): DecisionTree =
        DecisionTree.fit(
            Formula.lhs("Admitted"),
            data.toFrame(featureNames),
            SplitRule.GINI,
            maxDepth,
            10000,
            minLeafSize,
        )

    /** 制限なしの木。訓練データを完全に覚えてしまう */
    fun fitFull(data: Admissions): DecisionTree = fit(data)

    /**
     * 原著が「過学習しない小さい木」として作る設定。
     *
     * 原著は `max_depth=3` / `min_samples_leaf=10` / `min_samples_split=10` の
     * 3 つを掛けて **15 節・正解率 0.885** の木を得た。Smile には
     * `min_samples_split` にあたるものが無く、深さの対応も付かない。
     *
     * ここでは **原著の 2 つの予測を再現できる設定** を選んだ（`maxDepth = 4`）。
     * 節は 3 つ（分割 1 つ）と原著より小さく、正解率は 0.88 になる。
     * 節の数を原著に合わせたい場合は [fitSameSizeAsOriginal] を使う。
     */
    fun fitSmaller(data: Admissions): DecisionTree =
        fit(data, FEATURE_NAMES, maxDepth = 4, minLeafSize = 10)

    /**
     * 原著と同じ **15 節** になる設定（`maxDepth = 5`・葉の最小件数 10）。
     *
     * 正解率は 0.89 で原著の 0.885 に近いが、**木の中身は違う**。
     * 原著が例に挙げた出願者（CGPA 8.9）を不合格と判定するので、
     * 「CGPA だけで判定が変わる」という原著の説明は再現できない。
     */
    fun fitSameSizeAsOriginal(data: Admissions): DecisionTree =
        fit(data, FEATURE_NAMES, maxDepth = 5, minLeafSize = 10)

    /** GRE と TOEFL の 2 特徴量だけで学習する */
    fun fitExams(data: Admissions, maxDepth: Int): DecisionTree =
        fit(data, EXAM_FEATURES, maxDepth = maxDepth)

    /** 学習データに対する正解率 */
    fun accuracy(
        model: DecisionTree,
        data: Admissions,
        featureNames: List<String> = FEATURE_NAMES,
    ): Double {
        val frame = data.toFrame(featureNames)
        val correct = (0 until data.size).count { model.predict(frame[it]) == data.admitted[it] }
        return correct.toDouble() / data.size
    }

    /** 1 人ぶんの出願情報から合否を予測する */
    fun predictApplicant(
        model: DecisionTree,
        values: List<Double>,
        featureNames: List<String> = FEATURE_NAMES,
    ): Boolean {
        val vectors = featureNames.mapIndexed { index, name ->
            DoubleVector.of(name, doubleArrayOf(values[index]))
        }
        val frame = smile.data.DataFrame.of(*vectors.toTypedArray())
        return model.predict(frame[0]) == 1
    }

    /** DOT 文字列から、分割に使われた条件を根から順に取り出す */
    fun splitConditions(model: DecisionTree): List<Split> =
        SPLIT_LABEL.findAll(model.dot())
            .map { Split(it.groupValues[1].trim(), it.groupValues[2].toDouble()) }
            .toList()

    // 列名に空白が入る（"GRE Score" など）ので、パターンにも空白を含める
    private val SPLIT_LABEL = Regex("""label=<([A-Za-z_0-9 ]+) &le; ([0-9.]+)<br/>""")

    /** 木に含まれる節と葉の数 */
    fun nodeCount(model: DecisionTree): Int =
        NODE_LABEL.findAll(model.dot()).count()

    private val NODE_LABEL = Regex("""^\s*\d+ \[label=""", RegexOption.MULTILINE)
}
