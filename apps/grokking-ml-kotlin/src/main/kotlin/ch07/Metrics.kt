/**
 * 第 7 章: 分類モデルの評価指標。
 * 正解率だけでは分類モデルの良し悪しを測れない。混同行列を土台に、
 * 適合率・再現率・F1 スコア・ROC 曲線下面積（AUC）を実装する。
 */
package ch07

/** 混同行列。すべての指標の土台になる 4 つの数。 */
data class ConfusionMatrix(
    val truePositives: Int,
    val falsePositives: Int,
    val falseNegatives: Int,
    val trueNegatives: Int,
) {
    val total: Int get() = truePositives + falsePositives + falseNegatives + trueNegatives
}

/** 正解ラベルと予測から混同行列を作る。 */
fun confusionMatrix(labels: List<Int>, predictions: List<Int>): ConfusionMatrix {
    val counts = labels.zip(predictions).groupingBy { it }.eachCount()
    return ConfusionMatrix(
        truePositives = counts[1 to 1] ?: 0,
        falsePositives = counts[0 to 1] ?: 0,
        falseNegatives = counts[1 to 0] ?: 0,
        trueNegatives = counts[0 to 0] ?: 0,
    )
}

/** 0 除算を避ける割り算。分母が 0 なら 0 を返す。 */
private fun safeDivide(numerator: Int, denominator: Int): Double =
    if (denominator == 0) 0.0 else numerator.toDouble() / denominator

/** 正解率。全体のうち正しく当てた割合。 */
fun accuracy(matrix: ConfusionMatrix): Double =
    safeDivide(matrix.truePositives + matrix.trueNegatives, matrix.total)

/** 適合率。陽性と予測したもののうち、本当に陽性だった割合。 */
fun precision(matrix: ConfusionMatrix): Double =
    safeDivide(matrix.truePositives, matrix.truePositives + matrix.falsePositives)

/** 再現率。本当に陽性のもののうち、拾えた割合。 */
fun recall(matrix: ConfusionMatrix): Double =
    safeDivide(matrix.truePositives, matrix.truePositives + matrix.falseNegatives)

/** F ベータスコア。beta が大きいほど再現率を重視する。 */
fun fBetaScore(matrix: ConfusionMatrix, beta: Double = 1.0): Double {
    val p = precision(matrix)
    val r = recall(matrix)
    if (p == 0.0 && r == 0.0) return 0.0
    val betaSquared = beta * beta
    return (1 + betaSquared) * p * r / (betaSquared * p + r)
}

/** F1 スコア。適合率と再現率の調和平均。 */
fun f1Score(matrix: ConfusionMatrix): Double = fBetaScore(matrix, beta = 1.0)

/** 確率と閾値から 0 / 1 の予測を作る。 */
fun predictionsAtThreshold(probabilities: List<Double>, threshold: Double): List<Int> =
    probabilities.map { if (it >= threshold) 1 else 0 }

/** ROC 曲線の点列。閾値を動かしたときの (偽陽性率, 真陽性率) を返す。 */
fun rocPoints(labels: List<Int>, probabilities: List<Double>): List<Pair<Double, Double>> {
    val thresholds = (probabilities + listOf(0.0, 1.0 + 1e-9)).distinct().sortedDescending()
    return thresholds.map { threshold ->
        val matrix = confusionMatrix(labels, predictionsAtThreshold(probabilities, threshold))
        val falsePositiveRate =
            safeDivide(matrix.falsePositives, matrix.falsePositives + matrix.trueNegatives)
        falsePositiveRate to recall(matrix)
    }.sortedWith(compareBy({ it.first }, { it.second }))
}

/** ROC 曲線下の面積。台形則で積分する。 */
fun auc(labels: List<Int>, probabilities: List<Double>): Double =
    rocPoints(labels, probabilities)
        .zipWithNext { (x1, y1), (x2, y2) -> (x2 - x1) * (y1 + y2) / 2 }
        .sum()
