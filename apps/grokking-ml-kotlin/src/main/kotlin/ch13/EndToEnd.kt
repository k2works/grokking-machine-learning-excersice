/**
 * 第 13 章: エンドツーエンドの実例。
 * 生データの前処理から、複数モデルの比較、評価までを 1 本のパイプラインに
 * つなぐ。これまでの章で作った部品を組み合わせるだけで実現できる。
 */
package ch13

import ch06.LogisticClassifier
import ch06.logisticRegression
import ch07.auc
import ch07.confusionMatrix
import ch07.f1Score
import ch07.precision
import ch07.recall
import ch09.Tree
import ch09.buildTree
import ch12.AdaBoost
import ch12.trainAdaBoost
import ch07.accuracy as matrixAccuracy

/** 生データの 1 行。 */
typealias Row = Map<String, String>

/** 特徴量ベクトル。 */
typealias Point = List<Double>

/** 前処理を終えた特徴量とラベル。 */
data class Dataset(
    val points: List<Point>,
    val labels: List<Int>,
    val featureNames: List<String>,
)

/** 数値に見えない値は既定値に落とす。欠損への最初の砦。 */
fun parseNumber(text: String, default: Double = 0.0): Double = text.toDoubleOrNull() ?: default

/** 中央値。欠損の穴埋めに使う。 */
fun median(values: List<Double>): Double {
    if (values.isEmpty()) return 0.0
    val ordered = values.sorted()
    val middle = ordered.size / 2
    return if (ordered.size % 2 == 1) ordered[middle] else (ordered[middle - 1] + ordered[middle]) / 2.0
}

/** 欠損を中央値で埋める。平均より外れ値に強い。 */
fun imputeMissing(column: List<Double?>): List<Double> {
    val filler = median(column.filterNotNull())
    return column.map { it ?: filler }
}

/** 最小 0・最大 1 に揃える。値の幅が違う特徴量を対等に扱うため。 */
fun normalize(column: List<Double>): List<Double> {
    val low = column.min()
    val high = column.max()
    if (high == low) return column.map { 0.0 }
    return column.map { (it - low) / (high - low) }
}

/** カテゴリ列を 0/1 の列に展開する。 */
fun oneHot(column: List<String>): Pair<List<List<Double>>, List<String>> {
    val categories = column.distinct().sorted()
    return column.map { value -> categories.map { if (value == it) 1.0 else 0.0 } } to categories
}

/** 生の行データを、数値の特徴量ベクトルとラベルに変換する。 */
fun buildDataset(rows: List<Row>, labelColumn: String): Dataset {
    val labels = rows.map { if (it[labelColumn] == "yes") 1 else 0 }
    val numericColumns = listOf("age", "income")
    val categoricalColumns = listOf("city")

    val columns = mutableListOf<List<Double>>()
    val names = mutableListOf<String>()
    numericColumns.forEach { name ->
        val raw = rows.map { row -> row[name]?.takeIf { it.isNotEmpty() }?.let { parseNumber(it) } }
        columns += normalize(imputeMissing(raw))
        names += name
    }
    categoricalColumns.forEach { name ->
        val (expanded, categories) = oneHot(rows.map { it[name].orEmpty() })
        categories.forEachIndexed { index, category ->
            columns += expanded.map { it[index] }
            names += "$name=$category"
        }
    }

    val points = rows.indices.map { i -> columns.map { it[i] } }
    return Dataset(points = points, labels = labels, featureNames = names)
}

/** 1 つのモデルの評価結果。 */
data class Evaluation(
    val name: String,
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val auc: Double,
)

/** 予測関数と確率関数から、第 7 章の指標をまとめて算出する。 */
fun evaluate(
    name: String,
    predict: (Point) -> Int,
    probability: (Point) -> Double,
    points: List<Point>,
    labels: List<Int>,
): Evaluation {
    val matrix = confusionMatrix(labels, points.map(predict))
    return Evaluation(
        name = name,
        accuracy = matrixAccuracy(matrix),
        precision = precision(matrix),
        recall = recall(matrix),
        f1 = f1Score(matrix),
        auc = auc(labels, points.map(probability)),
    )
}

/** 第 6 章のロジスティック回帰を評価する。 */
fun evaluateLogistic(model: LogisticClassifier, points: List<Point>, labels: List<Int>): Evaluation =
    evaluate("logistic", { model.predict(it) }, { model.predictProbability(it) }, points, labels)

/** 第 9 章の決定木を評価する。 */
fun evaluateTree(tree: Tree, points: List<Point>, labels: List<Int>): Evaluation =
    evaluate("tree", { tree.predict(it) }, { tree.predict(it).toDouble() }, points, labels)

/** 第 12 章の AdaBoost を評価する。ラベルは +1 / -1 なので 0 / 1 に戻す。 */
fun evaluateAdaBoost(model: AdaBoost, points: List<Point>, labels: List<Int>): Evaluation =
    evaluate(
        "adaboost",
        { if (model.predict(it) == 1) 1 else 0 },
        { model.score(it) },
        points,
        labels,
    )

/** 分割されたデータセット。第 4 章の Split を多次元の特徴量へ一般化したもの。 */
data class DataSplit(
    val trainPoints: List<Point>,
    val trainLabels: List<Int>,
    val testPoints: List<Point>,
    val testLabels: List<Int>,
)

/** 訓練用とテスト用に分割する。第 4 章と同じ手順を多次元の点に対して行う。 */
fun splitDataset(dataset: Dataset, testRatio: Double = 0.3, seed: Int = 0): DataSplit {
    val indices = dataset.points.indices.shuffled(kotlin.random.Random(seed))
    val testSize = (dataset.points.size * testRatio).toInt()
    val testIndices = indices.take(testSize)
    val trainIndices = indices.drop(testSize)
    return DataSplit(
        trainPoints = trainIndices.map { dataset.points[it] },
        trainLabels = trainIndices.map { dataset.labels[it] },
        testPoints = testIndices.map { dataset.points[it] },
        testLabels = testIndices.map { dataset.labels[it] },
    )
}

/** 前処理 → 分割 → 3 モデルの学習 → 評価までを一気に通す。 */
fun runPipeline(rows: List<Row>, labelColumn: String = "bought"): List<Evaluation> {
    val dataset = buildDataset(rows, labelColumn)
    val split = splitDataset(dataset, testRatio = 0.3, seed = 0)
    val trainX = split.trainPoints
    val trainY = split.trainLabels
    val testX = split.testPoints
    val testY = split.testLabels

    val (logistic, _) = logisticRegression(trainX, trainY, learningRate = 0.5, epochs = 2000, seed = 0)
    val tree = buildTree(trainX, trainY, maxDepth = 3)
    val boosted = trainAdaBoost(trainX, trainY.map { if (it == 1) 1 else -1 }, rounds = 5, maxDepth = 1)

    return listOf(
        evaluateLogistic(logistic, testX, testY),
        evaluateTree(tree, testX, testY),
        evaluateAdaBoost(boosted, testX, testY),
    )
}

/** F1 スコアがもっとも高いモデルを選ぶ。 */
fun bestByF1(evaluations: List<Evaluation>): Evaluation = evaluations.maxBy { it.f1 }
