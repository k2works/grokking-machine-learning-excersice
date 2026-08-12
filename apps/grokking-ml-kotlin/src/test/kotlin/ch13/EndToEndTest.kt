package ch13

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EndToEndTest {
    /** 「40 歳未満かつ収入 400 超なら購入」という規則に従う擬似データ。 */
    private fun makeRows(count: Int = 40, seed: Int = 1): List<Row> {
        val random = Random(seed)
        val cities = listOf("tokyo", "osaka", "kyoto")
        return List(count) { i ->
            val age = random.nextInt(18, 71)
            val income = random.nextInt(200, 901)
            val city = cities[random.nextInt(cities.size)]
            mapOf(
                // 9 行に 1 行は年齢が欠損している
                "age" to if (i % 9 == 0) "" else age.toString(),
                "income" to income.toString(),
                "city" to city,
                "bought" to if (age < 40 && income > 400) "yes" else "no",
            )
        }
    }

    private val rows = makeRows()

    @Test
    fun `parseNumber は数値に見えない値を既定値に落とす`() {
        assertEquals(42.0, parseNumber("42"), 1e-9)
        assertEquals(0.0, parseNumber("N/A"), 1e-9)
        assertEquals(-1.0, parseNumber("", default = -1.0), 1e-9)
    }

    @Test
    fun `median は奇数長と偶数長を扱う`() {
        assertEquals(2.0, median(listOf(3.0, 1.0, 2.0)), 1e-9)
        assertEquals(2.5, median(listOf(4.0, 1.0, 3.0, 2.0)), 1e-9)
    }

    @Test
    fun `median は空列で 0`() {
        assertEquals(0.0, median(emptyList()), 1e-9)
    }

    @Test
    fun `imputeMissing は中央値で埋める`() {
        assertEquals(listOf(1.0, 2.0, 3.0), imputeMissing(listOf(1.0, null, 3.0)))
    }

    @Test
    fun `imputeMissing は外れ値に強い`() {
        // 平均なら 1000 に引きずられるが、中央値なら影響を受けない
        val filled = imputeMissing(listOf(1.0, 2.0, 3.0, 1000.0, null))
        assertEquals(2.5, filled.last(), 1e-9)
    }

    @Test
    fun `normalize は 0 から 1 に写す`() {
        assertEquals(listOf(0.0, 0.5, 1.0), normalize(listOf(10.0, 20.0, 30.0)))
    }

    @Test
    fun `normalize は定数列で 0 除算しない`() {
        assertEquals(listOf(0.0, 0.0, 0.0), normalize(listOf(5.0, 5.0, 5.0)))
    }

    @Test
    fun `oneHot はカテゴリを辞書順に展開する`() {
        val (expanded, categories) = oneHot(listOf("b", "a", "b"))
        assertEquals(listOf("a", "b"), categories)
        assertEquals(listOf(listOf(0.0, 1.0), listOf(1.0, 0.0), listOf(0.0, 1.0)), expanded)
    }

    @Test
    fun `oneHot はちょうど 1 列だけ立てる`() {
        val (expanded, _) = oneHot(listOf("x", "y", "z"))
        expanded.forEach { assertEquals(1.0, it.sum(), 1e-9) }
    }

    @Test
    fun `buildDataset は数値の特徴量を作る`() {
        val dataset = buildDataset(rows, "bought")
        assertEquals(
            listOf("age", "income", "city=kyoto", "city=osaka", "city=tokyo"),
            dataset.featureNames,
        )
        assertEquals(rows.size, dataset.points.size)
        dataset.points.forEach { assertEquals(dataset.featureNames.size, it.size) }
    }

    @Test
    fun `buildDataset は数値列を正規化する`() {
        val dataset = buildDataset(rows, "bought")
        listOf(0, 1).forEach { index ->
            val column = dataset.points.map { it[index] }
            assertEquals(0.0, column.min(), 1e-9)
            assertEquals(1.0, column.max(), 1e-9)
        }
    }

    @Test
    fun `buildDataset はラベルを 0 と 1 に変換する`() {
        val dataset = buildDataset(rows, "bought")
        assertEquals(setOf(0, 1), dataset.labels.toSet())
        assertEquals(rows.count { it["bought"] == "yes" }, dataset.labels.sum())
    }

    @Test
    fun `buildDataset は欠損を埋める`() {
        val dataset = buildDataset(rows, "bought")
        // 欠損があっても、すべての点が有限の数値で埋まっている
        dataset.points.forEach { point -> point.forEach { assertTrue(it.isFinite()) } }
    }

    @Test
    fun `splitDataset はデータを失わずに分割する`() {
        val dataset = buildDataset(rows, "bought")
        val split = splitDataset(dataset)
        assertEquals(dataset.points.size, split.trainPoints.size + split.testPoints.size)
        assertEquals(dataset.labels.size, split.trainLabels.size + split.testLabels.size)
    }

    @Test
    fun `パイプラインは 3 つのモデルを評価する`() {
        assertEquals(listOf("logistic", "tree", "adaboost"), runPipeline(rows).map { it.name })
    }

    @Test
    fun `すべての指標は 0 から 1 の割合`() {
        runPipeline(rows).forEach { evaluation ->
            listOf(evaluation.accuracy, evaluation.precision, evaluation.recall, evaluation.f1, evaluation.auc)
                .forEach { assertTrue(it in 0.0..1.0, "$it") }
        }
    }

    @Test
    fun `どのモデルも当てずっぽうには勝つ`() {
        runPipeline(rows).forEach { assertTrue(it.auc > 0.5, "${it.name} auc=${it.auc}") }
    }

    @Test
    fun `bestByF1 は F1 が最大のモデルを選ぶ`() {
        val evaluations = listOf(
            Evaluation("a", accuracy = 0.9, precision = 0.5, recall = 0.5, f1 = 0.5, auc = 0.9),
            Evaluation("b", accuracy = 0.7, precision = 0.8, recall = 0.8, f1 = 0.8, auc = 0.7),
        )
        assertEquals("b", bestByF1(evaluations).name)
    }

    @Test
    fun `パイプラインは再現可能`() {
        assertEquals(runPipeline(rows), runPipeline(rows))
    }
}
