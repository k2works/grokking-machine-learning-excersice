package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb04Perceptron.Boundary
import lib.Nb04Perceptron.FEATURES
import lib.Nb04Perceptron.LABELS
import lib.Nb04Perceptron.fitWithSmile
import lib.Nb04Perceptron.meanPerceptronError
import lib.Nb04Perceptron.perceptronAlgorithm
import lib.Nb04Perceptron.perceptronTrick
import lib.Nb04Perceptron.perceptronTrickExplicit
import lib.Nb04Perceptron.predictWithSmile
import lib.Nb04Perceptron.step

/**
 * 原著ノートブック #04 の再現テスト。
 *
 * トリックの単発の挙動は原著の数値と完全に一致する。学習ループは原著が乱数の種を
 * 与えていないため出力自体が実行のたびに変わるので、収束で検証する。
 */
class Nb04PerceptronTest {

    private val start = Boundary(doubleArrayOf(1.0, 2.0), -4.0)

    @Test
    fun `データセットは原著と同じ`() {
        assertEquals(
            listOf(listOf(1, 0), listOf(0, 2), listOf(1, 1), listOf(1, 2),
                listOf(1, 3), listOf(2, 2), listOf(2, 3), listOf(3, 2)),
            FEATURES.map { it.toList() },
        )
        assertEquals(listOf(0, 0, 0, 0, 1, 1, 1, 1), LABELS.toList())
    }

    @Test
    fun `ステップ関数は0で1を返す`() {
        // 境界をどちらに含めるかで結果が変わる。原著は 0 以上を 1 とする
        assertEquals(1, step(0.0))
        assertEquals(0, step(-1e-12))
        assertEquals(1, step(1.0))
    }

    @Test
    fun `スコアは重み付き和にバイアスを足す`() {
        // [2, 3] . [1, 2] - 4 = 2 + 6 - 4 = 4
        assertEquals(4.0, start.score(intArrayOf(2, 3)), 1e-12)
    }

    @Test
    fun `誤差は当たれば0外れればスコアの絶対値`() {
        assertEquals(0.0, start.error(intArrayOf(2, 3), 1), 1e-12)
        assertEquals(4.0, start.error(intArrayOf(2, 3), 0), 1e-12)
    }

    @Test
    fun `重み1と2バイアス-4の予測は原著と同じ`() {
        // 原著の出力
        //   0 0 / 1 0 / 0 0 / 1 1 / 1 0 / 1 0 / 1 0 / 1 0
        assertEquals(listOf(0, 1, 0, 1, 1, 1, 1, 1), FEATURES.map { start.predict(it) })
        assertEquals(
            listOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            FEATURES.indices.map { start.error(FEATURES[it], LABELS[it]) },
        )
    }

    @Test
    fun `平均パーセプトロン誤差は誤差の平均`() {
        assertEquals(1.0 / 8.0, meanPerceptronError(start), 1e-12)
    }

    @Test
    fun `短く書いた版のトリックは原著と同じ数値になる`() {
        // 原著の出力: ([0.9, 1.85], -4.1)
        val updated = perceptronTrick(start, FEATURES[6], 0)

        assertEquals(0.9, updated.weights[0], 1e-12)
        assertEquals(1.85, updated.weights[1], 1e-12)
        assertEquals(-4.1, updated.bias, 1e-12)
    }

    @Test
    fun `短く書いた版はバイアスを特徴量の数だけ動かす`() {
        // 原著の 2 つの実装は挙動が違う。短く書いた版はバイアスの更新が
        // 重みのループの内側にあり、特徴量が 2 つなので 2 回適用される
        val explicit = perceptronTrickExplicit(start, FEATURES[6], 0)
        val short = perceptronTrick(start, FEATURES[6], 0)

        // 重みの更新は一致する
        assertTrue(explicit.weights.contentEquals(short.weights))
        // バイアスだけ 2 倍動く
        assertEquals(-4.05, explicit.bias, 1e-12)
        assertEquals(-4.1, short.bias, 1e-12)
    }

    @Test
    fun `当たっているときは何も動かない`() {
        // FEATURES[4] = [1, 3] は重み [1, 2] バイアス -4 で予測 1、ラベルも 1
        val updated = perceptronTrick(start, FEATURES[4], 1)

        assertEquals(start, updated)
    }

    @Test
    fun `学習は誤差を下げる`() {
        val result = perceptronAlgorithm(epochs = 200)

        assertEquals(200, result.errors.size)
        assertTrue(result.errors.last() < result.errors.first())
    }

    @Test
    fun `学習後は全点を正しく分類できる`() {
        val result = perceptronAlgorithm(epochs = 500)

        assertEquals(LABELS.toList(), FEATURES.map { result.boundary.predict(it) })
    }

    @Test
    fun `Smile の線形 SVM も全点を正しく分類する`() {
        // 原著 scikit-learn の出力: Predictions: [0 0 0 0 1 1 1 1]
        // Smile にパーセプトロンは無いので線形 SVM で代替している
        assertEquals(LABELS.toList(), predictWithSmile(fitWithSmile()))
    }
}
