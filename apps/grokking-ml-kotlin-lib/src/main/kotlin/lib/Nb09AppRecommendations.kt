package lib

import smile.base.cart.SplitRule
import smile.classification.DecisionTree
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.type.DataTypes
import smile.data.type.StructField
import smile.data.vector.IntVector

/**
 * 原著ノートブック #09 `Chapter_09_Decision_Trees/App_recommendations.ipynb`。
 *
 * 6 人のユーザーの「使っている端末」と「年齢」から、おすすめアプリを決定木で当てる。
 * 原著は同じデータを 2 通りの形で学習させ、木の形がどう変わるかを見せている。
 *
 * 1. 年齢を **カテゴリ**（若者 / 大人）に潰して one-hot 符号化する
 * 2. 年齢を **数値** のまま渡す
 *
 * scikit-learn と Smile では決定木の作り方が違う点が 2 つある。
 * - Smile は **多出力分類に対応していない**。カテゴリ版は 1 列のラベルに直す
 * - Smile の `DecisionTree` は既定で決定的に分割する。scikit-learn のように
 *   同点の候補から無作為に選ぶことはない
 */
object Nb09AppRecommendations {

    /** 原著の元データ。この 6 人ぶんの情報しかない */
    val PLATFORMS = listOf("iPhone", "iPhone", "Android", "iPhone", "Android", "Android")
    val AGES = intArrayOf(15, 25, 32, 35, 12, 14)
    val APPS = listOf(
        "Atom Count", "Check Mate Mate", "Beehive Finder",
        "Check Mate Mate", "Atom Count", "Atom Count",
    )

    /** アプリ名を 0 始まりの番号に直したもの。Smile はラベルを Int で扱う */
    val APP_CLASSES = APPS.distinct().sorted()

    private fun appIndices(): IntArray = IntArray(APPS.size) { APP_CLASSES.indexOf(APPS[it]) }

    private fun buildFrame(
        columns: Map<String, IntArray>,
        labels: IntArray,
        classCount: Int,
    ): DataFrame {
        // Smile は目的変数が「名義尺度」であることを型で示す必要がある。
        // ただの IntVector にすると回帰木として扱われてしまう
        val labelField = StructField(
            "App",
            DataTypes.IntegerType,
            smile.data.measure.NominalScale(*Array(classCount) { APP_CLASSES[it] }),
        )
        val vectors = buildList {
            columns.forEach { (name, values) -> add(IntVector.of(name, values)) }
            add(IntVector.of(labelField, labels))
        }
        return DataFrame.of(*vectors.toTypedArray())
    }

    /**
     * 年齢をカテゴリに潰した版を学習する。
     *
     * 原著は 3 列の目的変数を同時に予測する **多出力分類** にしているが、
     * Smile は対応していない。3 つのアプリを 1 列の名義尺度に直して学習する。
     * 結果として木の形は原著と変わりうるが、全問正解する点は同じである。
     */
    fun fitCategorical(): DecisionTree {
        val frame = buildFrame(
            linkedMapOf(
                "Platform_iPhone" to intArrayOf(1, 1, 0, 1, 0, 0),
                "Platform_Android" to intArrayOf(0, 0, 1, 0, 1, 1),
                "Age_Adult" to intArrayOf(0, 1, 1, 1, 0, 0),
                "Age_Young" to intArrayOf(1, 0, 0, 0, 1, 1),
            ),
            appIndices(),
            APP_CLASSES.size,
        )
        return fitTree(frame)
    }

    /**
     * 年齢を数値のまま渡した版を学習する。
     *
     * 決定木がしきい値を自分で決めるので、`Age <= 20` のような分割が現れる。
     */
    fun fitNumeric(): DecisionTree {
        val frame = buildFrame(
            linkedMapOf(
                "Age" to AGES,
                "Platform_iPhone" to intArrayOf(1, 1, 0, 1, 0, 0),
                "Platform_Android" to intArrayOf(0, 0, 1, 0, 1, 1),
            ),
            appIndices(),
            APP_CLASSES.size,
        )
        return fitTree(frame)
    }

    /**
     * 決定木を学習する。
     *
     * 引数 2 つの `DecisionTree.fit` は、葉の数の上限をデータ件数から自動で決める。
     * 6 件しかないと上限が 1 になり `Invalid maximum leaves: 1` で落ちるので、
     * ここでは上限を明示的に渡す。scikit-learn は既定で制限なしに育てる。
     */
    private fun fitTree(frame: DataFrame): DecisionTree =
        DecisionTree.fit(Formula.lhs("App"), frame, SplitRule.GINI, 20, 100, 1)

    /** 学習データに対する正解率 */
    fun accuracy(model: DecisionTree, frame: DataFrame): Double {
        val labels = appIndices()
        val correct = (0 until frame.nrow()).count { model.predict(frame[it]) == labels[it] }
        return correct.toDouble() / labels.size
    }

    /**
     * 木の構造を DOT 形式の文字列で得る。
     *
     * scikit-learn は `tree.plot_tree` で図を描くが、Smile は
     * **Graphviz の DOT 形式を文字列で返す** だけである。図にするには
     * 別途 Graphviz に通す必要がある。テキストなので差分は取りやすい。
     */
    fun toDot(model: DecisionTree): String = model.dot()

    /**
     * DOT 文字列から、分割に使われた特徴量としきい値を根から順に取り出す。
     *
     * Smile のラベルは `<Age &le; 20.0<br/>size = 6<br/>...>` という HTML 風の
     * 書式になっている。`&le;` は `<=` の実体参照である。
     */
    fun splits(model: DecisionTree): List<Split> =
        SPLIT_LABEL.findAll(model.dot())
            .map { Split(it.groupValues[1], it.groupValues[2].toDouble()) }
            .toList()

    /** 分割に使われた特徴量としきい値 */
    data class Split(val feature: String, val threshold: Double)

    /** DOT のノードラベルから特徴量名としきい値を取り出す正規表現 */
    private val SPLIT_LABEL = Regex("""label=<([A-Za-z_]+) &le; ([0-9.]+)<br/>""")

    /** 葉が予測するクラス名を、DOT に現れる順に取り出す */
    fun leafClasses(model: DecisionTree): List<String> =
        LEAF_LABEL.findAll(model.dot()).map { it.groupValues[1] }.toList()

    private val LEAF_LABEL = Regex("""label=<App = ([^<]+)<br/>""")
}
