package lib

import smile.base.cart.SplitRule
import smile.classification.AdaBoost
import smile.classification.DecisionTree
import smile.classification.GradientTreeBoost
import smile.classification.RandomForest
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.measure.NominalScale
import smile.data.type.DataTypes
import smile.data.type.StructField
import smile.data.vector.IntVector

/**
 * 原著ノートブック #20 `Chapter_12_Ensemble_Methods/Random_forests_and_AdaBoost.ipynb`。
 *
 * 18 通のメールを 2 特徴量（`Lottery`・`Sale`）で振り分ける小さなデータに、
 * アンサンブル学習を次々と当てる回である。
 *
 * **「正解率が 1.0 になるのは良いことではない」** を見せるのが狙い。
 */
object Nb20EnsembleSpam {

    /** 原著が冒頭で設定する種 */
    const val SEED = 0L

    /** 原著の 18 通。[Lottery, Sale, Spam] */
    val EMAILS = arrayOf(
        intArrayOf(7, 8, 1), intArrayOf(3, 2, 0), intArrayOf(8, 4, 1),
        intArrayOf(2, 6, 0), intArrayOf(6, 5, 1), intArrayOf(9, 6, 1),
        intArrayOf(8, 5, 0), intArrayOf(7, 1, 0), intArrayOf(1, 9, 1),
        intArrayOf(4, 7, 0), intArrayOf(1, 3, 0), intArrayOf(3, 10, 1),
        intArrayOf(2, 2, 1), intArrayOf(9, 3, 0), intArrayOf(5, 3, 0),
        intArrayOf(10, 1, 0), intArrayOf(5, 9, 1), intArrayOf(10, 8, 1),
    )

    /** 原著が手作業で切り分ける 3 組。6 通ずつ **並び順のまま** 分ける */
    val BATCHES = listOf(
        listOf(0, 1, 2, 3, 4, 5),
        listOf(6, 7, 8, 9, 10, 11),
        listOf(12, 13, 14, 15, 16, 17),
    )

    /** 目的変数の列名 */
    const val LABEL = "Spam"

    private val labelField =
        StructField(LABEL, DataTypes.IntegerType, NominalScale("ham", "spam"))

    /** 指定した行だけを Smile のデータフレームにする */
    fun frameOf(indices: List<Int> = EMAILS.indices.toList()): DataFrame {
        val lottery = IntArray(indices.size) { EMAILS[indices[it]][0] }
        val sale = IntArray(indices.size) { EMAILS[indices[it]][1] }
        val spam = IntArray(indices.size) { EMAILS[indices[it]][2] }
        return DataFrame.of(
            IntVector.of("Lottery", lottery),
            IntVector.of("Sale", sale),
            IntVector.of(labelField, spam),
        )
    }

    /** 3 組のうち 1 つ */
    fun batch(index: Int): DataFrame = frameOf(BATCHES[index])

    private val formula = Formula.lhs(LABEL)

    /**
     * 決定木 1 本。
     *
     * `maxDepth` を大きく取れば、制限なしの `DecisionTreeClassifier` と同じく
     * 訓練データを丸暗記する。
     */
    fun fitDecisionTree(data: DataFrame, maxDepth: Int = 20, nodeSize: Int = 1): DecisionTree =
        DecisionTree.fit(formula, data, SplitRule.GINI, maxDepth, 10000, nodeSize)

    /**
     * 1 組ぶんのデータに深さ 1 の決定木（切り株）を当てる。
     *
     * [#10][Nb10DecisionTreeBoundary] で見たように **Smile の深さは
     * scikit-learn より 1 大きく数える**。scikit-learn の `max_depth=1`
     * （分割 1 つ）に対応するのは Smile の `maxDepth = 2` である。
     */
    fun fitWeakLearner(index: Int): DecisionTree = fitDecisionTree(batch(index), maxDepth = 2)

    /**
     * 原著と同じ設定（5 本・深さ 1）のランダムフォレスト。
     *
     * **`MathEx.setSeed` を呼んでも結果が固定できない。**
     * Smile の `RandomForest` は木をスレッド並列で育てるので、
     * どのスレッドがどの部分標本を取るかが実行ごとに変わる。
     * 40 回走らせた正解率の分布は次のとおりだった。
     *
     * | 正解率 | 回数 |
     * | ---: | ---: |
     * | 0.500 | 3 |
     * | 0.556 | 1 |
     * | 0.611 | 3 |
     * | 0.667 | 5 |
     * | 0.722 | 8 |
     * | 0.778 | 12 |
     * | 0.833 | 5 |
     * | 0.889 | 3 |
     *
     * 原著（scikit-learn）の 0.833 はこの分布の中に入るが、
     * **1 つの値として突き合わせることはできない**。
     * [#19][Nb19SvmKernels] の SVM は種で固定できたが、こちらはできない。
     */
    fun fitRandomForest(data: DataFrame = frameOf()): RandomForest {
        smile.math.MathEx.setSeed(SEED)
        return RandomForest.fit(formula, data, 5, 1, SplitRule.GINI, 2, 10000, 1, 1.0)
    }

    /** 原著と同じ設定（6 本）の AdaBoost */
    fun fitAdaBoost(data: DataFrame = frameOf()): AdaBoost {
        smile.math.MathEx.setSeed(SEED)
        return AdaBoost.fit(formula, data, 6, 10000, 2, 1)
    }

    /** 原著と同じ設定（5 本）の勾配ブースティング */
    fun fitGradientBoosting(data: DataFrame = frameOf()): GradientTreeBoost {
        smile.math.MathEx.setSeed(SEED)
        return GradientTreeBoost.fit(formula, data, 5, 6, 10000, 5, 0.05, 0.7)
    }

    /** 学習データに対する正解率 */
    fun accuracy(model: smile.classification.Classifier<smile.data.Tuple>, data: DataFrame): Double {
        val size = data.size()
        val correct = (0 until size).count { model.predict(data[it]) == data[it].getInt(LABEL) }
        return correct.toDouble() / size
    }

    /**
     * 深さ 1 の木が使った特徴量としきい値。
     *
     * Smile は木の構造を DOT 文字列でしか出さないので、そこから読み取る。
     */
    fun splitOf(tree: DecisionTree): Pair<String, Double>? =
        SPLIT_LABEL.find(tree.dot())?.let { it.groupValues[1] to it.groupValues[2].toDouble() }

    private val SPLIT_LABEL = Regex("""label=<([A-Za-z_0-9 ]+) &le; ([0-9.]+)<br/>""")
}
