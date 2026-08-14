package lib

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb17BuildingDatasets.LINEAR
import lib.Nb17BuildingDatasets.ONE_CIRCLE
import lib.Nb17BuildingDatasets.SPECS
import lib.Nb17BuildingDatasets.TWO_CIRCLES
import lib.Nb17BuildingDatasets.generate
import lib.Nb17BuildingDatasets.linearRule
import lib.Nb17BuildingDatasets.load
import lib.Nb17BuildingDatasets.oneCircleRule
import lib.Nb17BuildingDatasets.ruleViolations
import lib.Nb17BuildingDatasets.twoCirclesRule

/**
 * 原著ノートブック #17 の再現テスト。
 *
 * 原著は種を固定せずに乱数でデータを作るので、生成は再現できない。
 * 代わりに **配布 CSV が原著の規則で作られたこと** を検証する。
 * 検証結果は Python 版・F# 版と完全に一致する。
 */
class Nb17BuildingDatasetsTest {

    @Test
    fun `配布CSVの行数は規則の点とノイズの合計`() {
        // 原著は「規則どおりの点」と「ノイズ」を続けて追加している
        assertEquals(60, LINEAR.total)
        assertEquals(110, ONE_CIRCLE.total)
        assertEquals(220, TWO_CIRCLES.total)
        SPECS.forEach { assertEquals(it.total, load(it).size, it.name) }
    }

    @Test
    fun `先頭の点は規則に1つも違反しない`() {
        // 配布 CSV の先頭 points 件は規則どおりに作られた点。
        // ここに違反が 1 つも無いことが、規則を正しく読めた証拠になる
        SPECS.forEach { spec ->
            val violations = ruleViolations(spec, load(spec))
            assertEquals(emptyList(), violations.filter { it < spec.points }, spec.name)
        }
    }

    @Test
    fun `違反はすべてノイズ部分にあり約半数`() {
        // ノイズはラベルを 0 か 1 で振り直すので、規則と食い違うのは約半分
        val expected = mapOf("linear" to 5, "one_circle" to 7, "two_circles" to 12)

        SPECS.forEach { spec ->
            val violations = ruleViolations(spec, load(spec))
            assertEquals(expected.getValue(spec.name), violations.size, spec.name)
            assertTrue(violations.all { it >= spec.points }, spec.name)
            assertTrue(violations.size.toDouble() / spec.noise in 0.2..0.8, spec.name)
        }
    }

    @Test
    fun `座標はマイナス3から3の範囲`() {
        // 原著の 6 * random() - 3
        SPECS.forEach { spec ->
            load(spec).forEach {
                assertTrue(it.x1 in -3.0..3.0 && it.x2 in -3.0..3.0, spec.name)
            }
        }
    }

    @Test
    fun `ラベルは0か1だけ`() {
        SPECS.forEach { spec ->
            assertEquals(listOf(0, 1), load(spec).map { it.y }.distinct().sorted(), spec.name)
        }
    }

    @Test
    fun `直線の規則は境界のちょうど上を0にする`() {
        // 原著は x + y > 0.5（等号を含まない）
        assertEquals(0, linearRule(0.25, 0.25))
        assertEquals(1, linearRule(0.3, 0.3))
    }

    @Test
    fun `円の規則は境界のちょうど上を0にする`() {
        // 原著は x^2 + y^2 < 2.8（等号を含まない）
        val radius = sqrt(2.8)
        assertEquals(0, oneCircleRule(radius, 0.0))
        assertEquals(1, oneCircleRule(radius - 1e-9, 0.0))
    }

    @Test
    fun `2つの円は重なりを持つ`() {
        // 中心 (1, 0) と (-1, 0)、半径 √2 ≒ 1.414。中心間の距離 2 より大きいので重なる
        assertEquals(1, twoCirclesRule(0.0, 0.0))
        assertEquals(0, twoCirclesRule(0.0, 2.0))
    }

    @Test
    fun `生成した点は規則どおりでノイズだけが外れる`() {
        SPECS.forEach { spec ->
            val generated = generate(spec, seed = 0)
            assertEquals(spec.total, generated.size, spec.name)
            assertEquals(
                emptyList(),
                ruleViolations(spec, generated).filter { it < spec.points },
                spec.name,
            )
        }
    }

    @Test
    fun `生成は種を固定すれば再現する`() {
        // 原著は種を固定していないので毎回違うものが出る
        assertEquals(generate(LINEAR, seed = 42), generate(LINEAR, seed = 42))
        assertTrue(generate(LINEAR, seed = 42) != generate(LINEAR, seed = 43))
    }

    @Test
    fun `生成したものは配布CSVとは一致しない`() {
        // 原著が種を固定していない以上、これは一致しようがない
        val generated = generate(LINEAR, seed = 0)
        val published = load(LINEAR)

        assertEquals(published.size, generated.size)
        assertTrue(generated[0].x1 != published[0].x1)
    }

    @Test
    fun `同じ種でもPythonとは違う点が出る`() {
        // Kotlin の Random と Python の random は別のアルゴリズム（Python は
        // メルセンヌツイスタ）。種を揃えても一致しない。
        // 原著が種を固定していれば、ここで手が止まっていた
        // Python 版の同じ種での 1 点目は 2.0665311091502883
        assertTrue(generate(LINEAR, seed = 0)[0].x1 != 2.0665311091502883)
    }
}
