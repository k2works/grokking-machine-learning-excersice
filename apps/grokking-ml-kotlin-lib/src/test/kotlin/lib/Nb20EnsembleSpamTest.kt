package lib

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import lib.Nb20EnsembleSpam.BATCHES
import lib.Nb20EnsembleSpam.EMAILS
import lib.Nb20EnsembleSpam.accuracy
import lib.Nb20EnsembleSpam.batch
import lib.Nb20EnsembleSpam.fitAdaBoost
import lib.Nb20EnsembleSpam.fitDecisionTree
import lib.Nb20EnsembleSpam.fitGradientBoosting
import lib.Nb20EnsembleSpam.fitRandomForest
import lib.Nb20EnsembleSpam.fitWeakLearner
import lib.Nb20EnsembleSpam.frameOf
import lib.Nb20EnsembleSpam.splitOf

/**
 * 原著ノートブック #20 の再現テスト。
 *
 * 弱学習器 3 本は **選んだ特徴量としきい値まで** scikit-learn と一致する。
 * 一方、ランダムフォレストは **種を固定しても結果が変わる**。
 * ブースティング 2 種は決定的だが、実装の違いで原著とは揃わない。
 */
class Nb20EnsembleSpamTest {

    private val all = frameOf()

    @Test
    fun `データセットは18通`() {
        assertEquals(18, EMAILS.size)
        // スパムとそうでないものが 9 通ずつ
        assertEquals(9, EMAILS.count { it[2] == 1 })
    }

    @Test
    fun `制限なしの決定木は丸暗記する`() {
        // 原著の出力: 1.0
        // 18 点を完全に分けきる。良い結果に見えるが過学習そのもの
        assertEquals(1.0, accuracy(fitDecisionTree(all), all))
    }

    @Test
    fun `深さ1に制限すると丸暗記できない`() {
        // 分割 1 つでは 18 点を分けられない
        assertEquals(0.7777777777777778, accuracy(fitDecisionTree(all, maxDepth = 2), all), 1e-15)
    }

    @Test
    fun `3組は6通ずつ重複なく分かれる`() {
        assertEquals(listOf(6, 6, 6), BATCHES.map { it.size })
        assertEquals((0..17).toList(), BATCHES.flatten().sorted())
    }

    @Test
    fun `弱学習器の正解率は原著と一致する`() {
        // 原著の出力
        //   Weak learner 1 training accuracy: 1.0
        //   Weak learner 2 training accuracy: 1.0
        //   Weak learner 3 training accuracy: 0.8333333333333334
        val expected = listOf(1.0, 1.0, 0.8333333333333334)

        (0..2).forEach { index ->
            assertEquals(expected[index], accuracy(fitWeakLearner(index), batch(index)), 1e-15)
        }
    }

    @Test
    fun `弱学習器が選ぶ分割はscikit-learnと完全に一致する`() {
        // 特徴量もしきい値も同じところを選んだ。CART の分割規則が同じなら、
        // 6 点しかないデータでは解が一意に決まる
        assertEquals(listOf("Lottery" to 4.5, "Sale" to 8.0, "Sale" to 5.5), (0..2).map { splitOf(fitWeakLearner(it)) })
    }

    @Test
    fun `ランダムフォレストの正解率は固定できない`() {
        // 原著の出力は 0.8333333333333334。Smile は木をスレッド並列で育てるので、
        // MathEx.setSeed を呼んでも実行ごとに変わる。40 回の実測では
        // 0.500〜0.889 に散らばり、最頻値は 0.778 だった。
        // 原著の値もこの範囲に入るが、1 点として突き合わせることはできない
        val scores = (1..10).map { accuracy(fitRandomForest(), all) }

        assertTrue(scores.all { it in 0.4..1.0 }, "実測範囲を外れた: $scores")
    }

    @Test
    fun `AdaBoostは原著と一致しない`() {
        // 原著は 0.8888888888888888。Smile は 0.8333333333333334。
        // なお **現在の scikit-learn も原著と一致しない**（0.7777777777777778）。
        // 原著の既定 algorithm='SAMME.R' が 1.6 で削除されたため。
        // 3 つとも違う値になる、めずらしい回である
        assertEquals(0.8333333333333334, accuracy(fitAdaBoost(), all), 1e-15)
    }

    @Test
    fun `勾配ブースティングは原著と一致しない`() {
        // 原著は 0.8888888888888888。Smile は学習率も部分標本の割合も
        // 既定値が違うので揃わない
        assertEquals(0.7777777777777778, accuracy(fitGradientBoosting(), all), 1e-15)
    }

    @Test
    fun `アンサンブルは1本の木より正解率が低い`() {
        // 原著が「ブースティングは正確だが過学習からは遠い」と書いているところ。
        // 学習データの正解率だけを見れば、1 本の木（1.0）が最も高い
        val single = accuracy(fitDecisionTree(all), all)

        assertTrue(accuracy(fitAdaBoost(), all) < single)
        assertTrue(accuracy(fitGradientBoosting(), all) < single)
    }

    @Test
    fun `ブースティング2種は種を固定すれば再現する`() {
        // AdaBoost と勾配ブースティングは木を順番に育てるので決定的になる。
        // 並列に育てるランダムフォレストだけが固定できない
        assertEquals(accuracy(fitAdaBoost(), all), accuracy(fitAdaBoost(), all), 0.0)
        assertEquals(accuracy(fitGradientBoosting(), all), accuracy(fitGradientBoosting(), all), 0.0)
    }
}
