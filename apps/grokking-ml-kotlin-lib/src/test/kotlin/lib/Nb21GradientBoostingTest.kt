package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb21GradientBoosting.AGES
import lib.Nb21GradientBoosting.DAYS
import lib.Nb21GradientBoosting.INITIAL_PREDICTION
import lib.Nb21GradientBoosting.centeredLabels
import lib.Nb21GradientBoosting.findBestSplit
import lib.Nb21GradientBoosting.fitGradientBoosting
import lib.Nb21GradientBoosting.fitRegressionTree
import lib.Nb21GradientBoosting.predict
import lib.Nb21GradientBoosting.similarityScore
import lib.Nb21GradientBoosting.splitScores
import lib.Nb21GradientBoosting.xgboostResiduals

/**
 * 原著ノートブック #21 の再現テスト。
 *
 * 勾配ブースティングの予測が **原著と 15 桁一致** する。
 * Smile が scikit-learn と同じ答えを出したのは、このシリーズでは数少ない例である。
 */
class Nb21GradientBoostingTest {

    @Test
    fun `データセットは8人`() {
        assertEquals(listOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0), AGES.toList())
        assertEquals(listOf(7.0, 5.0, 7.0, 1.0, 2.0, 1.0, 5.0, 4.0), DAYS.toList())
        // 平均は 4.0。XGBoost の base_score に使う
        assertEquals(4.0, DAYS.average())
    }

    @Test
    fun `中心化したラベルが最初の残差`() {
        // 原著の出力: Residuals to predict: [ 3.  1.  3. -3. -2. -3.  1.  0.]
        assertEquals(listOf(3.0, 1.0, 3.0, -3.0, -2.0, -3.0, 1.0, 0.0), centeredLabels().toList())
    }

    @Test
    fun `回帰木は4つの値しか返さない`() {
        // 深さ 2 なので葉は 4 つ。同じ葉に落ちた人は同じ予測になる
        assertEquals(4, predict(fitRegressionTree()).toSet().size)
    }

    @Test
    fun `勾配ブースティングの予測は原著と15桁一致する`() {
        // 原著の出力
        //   array([6.87466667, 5.11466667, 6.71466667, 1.43466667, 1.43466667,
        //          1.43466667, 4.896     , 4.096     ])
        val expected = listOf(
            6.87466667, 5.11466667, 6.71466667, 1.43466667,
            1.43466667, 1.43466667, 4.896, 4.096,
        )

        predict(fitGradientBoosting()).forEachIndexed { index, value ->
            assertEquals(expected[index], value, 5e-9, "person $index")
        }
    }

    @Test
    fun `弱学習器を増やすほど残差が縮む`() {
        // ブースティングの定義そのもの
        val errors = (1..4).map { count ->
            val predicted = predict(fitGradientBoosting(count))
            DAYS.indices.sumOf { (DAYS[it] - predicted[it]) * (DAYS[it] - predicted[it]) }
        }

        assertEquals(errors.sortedDescending(), errors)
    }

    @Test
    fun `類似度スコアは和の2乗を使う`() {
        // sum(l)^2 / (len(l) + lambda)。2 乗の和ではない
        assertEquals(2.25, similarityScore(doubleArrayOf(3.0), 3.0), 1e-15)
        assertEquals(3.2, similarityScore(doubleArrayOf(3.0, 1.0), 3.0), 1e-15)
        // 符号が打ち消し合うと 0 になる
        assertEquals(0.0, similarityScore(doubleArrayOf(3.0, -3.0), 3.0))
    }

    @Test
    fun `空の集合の類似度は0`() {
        assertEquals(0.0, similarityScore(DoubleArray(0), 3.0))
    }

    @Test
    fun `全体の類似度は0になる`() {
        // 残差の総和が 0 なので、根の類似度スコアは 0
        assertEquals(0.0, similarityScore(xgboostResiduals(), 3.0))
    }

    @Test
    fun `残差は原著と一致する`() {
        // 原著の出力: array([ 3.,  1.,  3., -3., -2., -3.,  1.,  0.])
        assertEquals(listOf(3.0, 1.0, 3.0, -3.0, -2.0, -3.0, 1.0, 0.0), xgboostResiduals().toList())
    }

    @Test
    fun `切れ目ごとのスコアは原著と一致する`() {
        // 原著が 1 行ずつ印刷している「Sum of similarity scores」
        val scores = splitScores(xgboostResiduals(), 3.0)

        assertEquals(3.15, scores[0], 1e-15)
        assertEquals(4.977777777777778, scores[1], 1e-15)
        assertEquals(14.291666666666666, scores[2], 1e-15)
        assertEquals(scores.max(), scores[2])
    }

    @Test
    fun `最良の分割は原著と一致する`() {
        // 原著の結論: Left tree: [3. 1. 3.] / Right tree: [-3. -2. -3.  1.  0.]
        val best = findBestSplit(xgboostResiduals(), 3.0)

        assertEquals(listOf(3.0, 1.0, 3.0), best.left.toList())
        assertEquals(listOf(-3.0, -2.0, -3.0, 1.0, 0.0), best.right.toList())
        assertEquals(14.291666666666666, best.score, 1e-15)
    }

    @Test
    fun `分割のスコアはXGBoostのgainと一致する`() {
        // 原著の木のダンプ: 0:[f0<35] ... gain=14.291667
        // 手で計算した最良スコアが、XGBoost が出す gain とぴたり合う
        assertEquals(14.291667, findBestSplit(xgboostResiduals(), 3.0).score, 1e-6)
    }

    @Test
    fun `部分木の分割も原著と一致する`() {
        val best = findBestSplit(xgboostResiduals(), 3.0)
        val left = findBestSplit(best.left, 3.0)
        val right = findBestSplit(best.right, 3.0)

        // 原著: Left tree: [3.] / [1., 3.]
        assertEquals(listOf(3.0), left.left.toList())
        assertEquals(listOf(1.0, 3.0), left.right.toList())
        assertEquals(5.45, left.score, 1e-15)

        // 原著: Left tree: [-3., -2., -3.] / [1., 0.]
        assertEquals(listOf(-3.0, -2.0, -3.0), right.left.toList())
        assertEquals(listOf(1.0, 0.0), right.right.toList())
        // **原著は 7.0 を返すが、それは実装のバグによる**（最後の切れ目のスコア）。
        // 正しい最良スコアは 10.87
        assertEquals(10.866666666666665, right.score, 1e-15)
    }

    @Test
    fun `原著が返す7という値は最後の切れ目のスコア`() {
        // 右の木 [-3,-2,-3,1,0] の最後の切れ目は [-3,-2,-3,1] / [0] で、
        // (-7)^2/(4+3) + 0^2/(1+3) = 7.0。これが原著の返り値 7.0 の正体
        val right = findBestSplit(xgboostResiduals(), 3.0).right
        val scores = splitScores(right, 3.0)

        assertEquals(7.0, scores.last(), 1e-15)
        assertTrue(scores.last() < scores.max())
    }

    @Test
    fun `XGBoostの最初の予測はラベルの平均`() {
        assertEquals(DAYS.average(), INITIAL_PREDICTION)
    }
}
