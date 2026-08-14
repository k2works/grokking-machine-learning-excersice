package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import lib.Nb08GiniEntropy.ELEMENTS
import lib.Nb08GiniEntropy.bestSplit
import lib.Nb08GiniEntropy.counts
import lib.Nb08GiniEntropy.entropy
import lib.Nb08GiniEntropy.gini
import lib.Nb08GiniEntropy.splitImpurities

/**
 * 原著ノートブック #08 の再現テスト。
 *
 * NumPy しか使わない小さな回なので、原著の数値をすべて再現できる。
 */
class Nb08GiniEntropyTest {

    @Test
    fun `個数は初出順に並ぶ`() {
        // 原著の出力: [3, 2, 1]。A が 3、C が 2、B が 1 の順
        assertEquals(listOf(3, 2, 1), counts(ELEMENTS))
    }

    @Test
    fun `ジニ不純度は原著と同じ`() {
        // 原著の出力: 0.6111111111111112
        assertEquals(0.6111111111111112, gini(ELEMENTS), 1e-15)
    }

    @Test
    fun `エントロピーは原著と同じ`() {
        // 原著の出力: 1.4591479170272448
        assertEquals(1.4591479170272448, entropy(ELEMENTS), 1e-15)
    }

    @Test
    fun `同じ要素だけなら不純度は0`() {
        assertEquals(0.0, gini(listOf("A", "A", "A")), 1e-15)
        assertEquals(0.0, entropy(listOf("A", "A", "A")), 1e-15)
    }

    @Test
    fun `2クラスが半々ならジニは0_5でエントロピーは1`() {
        // 情報量 1 ビットぶん。コイン投げと同じ
        assertEquals(0.5, gini(listOf("A", "B")), 1e-15)
        assertEquals(1.0, entropy(listOf("A", "B")), 1e-15)
    }

    @Test
    fun `空のリストの扱いは2つで違う`() {
        // 原著はエントロピーだけ明示的に 0 を返す。ジニは 1 - 0 で 1 になる
        assertEquals(1.0, gini(emptyList()), 1e-15)
        assertEquals(0.0, entropy(emptyList()), 1e-15)
    }

    @Test
    fun `各分割の重み付き不純度は原著と同じ`() {
        // 原著のセル出力をそのまま期待値にしている
        val expected = listOf(
            Triple(0, 0.6111111111111112, 1.4591479170272446),
            Triple(1, 0.5333333333333333, 1.268273412406135),
            Triple(2, 0.41666666666666663, 1.0),
            Triple(3, 0.2222222222222222, 0.4591479170272448),
            Triple(4, 0.41666666666666663, 0.8741854163060886),
            Triple(5, 0.4666666666666667, 1.1424588287122237),
        )
        val splits = splitImpurities()

        for ((index, expectedGini, expectedEntropy) in expected) {
            assertEquals(expectedGini, splits[index].weightedGini, 1e-15, "gini at $index")
            assertEquals(expectedEntropy, splits[index].weightedEntropy, 1e-15, "entropy at $index")
        }
    }

    @Test
    fun `分割は6通り試される`() {
        // 0 から size - 1 まで。「左が全部・右が空」は試されない
        val splits = splitImpurities()

        assertEquals(6, splits.size)
        assertEquals(emptyList(), splits[0].left)
        assertEquals(listOf("C"), splits[5].right)
    }

    @Test
    fun `最良の分割はAのかたまりを切り離す`() {
        // ['A', 'A', 'A'] | ['C', 'B', 'C'] で両方の指標が最小になる
        val best = bestSplit()

        assertEquals(3, best.index)
        assertEquals(listOf("A", "A", "A"), best.left)
        assertEquals(listOf("C", "B", "C"), best.right)
        assertEquals(0.2222222222222222, best.weightedGini, 1e-15)
    }

    @Test
    fun `ジニとエントロピーは同じ分割を選ぶ`() {
        val splits = splitImpurities()

        assertEquals(
            splits.minBy { it.weightedGini }.index,
            splits.minBy { it.weightedEntropy }.index,
        )
    }

    @Test
    fun `分割しない場合の重み付き不純度は全体の不純度と一致する`() {
        // index 0 は左が空なので、右がそのまま全体になる。
        // ただしエントロピーは 1.4591479170272446 で、entropy(ELEMENTS) の
        // 1.4591479170272448 と最下位ビットだけ違う。重み付けの掛け算と割り算で
        // 丸めが 1 度多く入るため
        val split = splitImpurities()[0]

        assertEquals(gini(ELEMENTS), split.weightedGini, 1e-15)
        assertNotEquals(entropy(ELEMENTS), split.weightedEntropy)
        assertTrue(Math.abs(entropy(ELEMENTS) - split.weightedEntropy) < 1e-15)
    }
}
