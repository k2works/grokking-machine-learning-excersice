package lib

import kotlin.math.exp

/**
 * 原著ノートブック #16 `Chapter_10_Neural_Networks/Plotting_Boundaries.ipynb`。
 *
 * **学習をしない回** である。重みを手で決めたネットワークを 2 つ作り、
 * その境界を描いて「1 層目は直線、2 層目は曲がる」ことを見せる。
 *
 * 題材は [#13][Nb13NeuralNetworkBoundary] と同じ「エイリアンが幸せかどうか」の 8 点。
 * ライブラリは要らない。**Kotlin の関数型が原著の高階関数にそのまま対応する** 回でもある。
 */
object Nb16PlottingBoundaries {

    /** 原著が図を描く範囲。境界の比較もこの範囲で行う */
    const val GRID_MIN = -0.5
    const val GRID_MAX = 3.0
    const val GRID_STEP = 0.005

    /** 出力を 1 と見なすしきい値。原著の `f(x, y) >= 0.5` */
    const val DECISION_THRESHOLD = 0.5

    /** エイリアン 1 匹ぶんの観測 */
    data class Alien(val aack: Int, val beep: Int, val happy: Int)

    /** 原著の 8 件 */
    val ALIEN_DATASET = listOf(
        Alien(1, 0, 0), Alien(2, 0, 0), Alien(0, 1, 0), Alien(0, 2, 0),
        Alien(1, 1, 1), Alien(1, 2, 1), Alien(2, 1, 1), Alien(2, 2, 1),
    )

    /** 階段関数。0 以上なら 1、そうでなければ 0 */
    fun step(x: Double): Int = if (x >= 0.0) 1 else 0

    /**
     * 原著の書き方 `exp(x) / (1 + exp(x))` をそのまま使う。
     *
     * 数学的には `1 / (1 + exp(-x))` と同じだが、**桁あふれの向きが逆** になる。
     * こちらは x が大きいときに `exp(x)` が無限大になり、
     * `Infinity / Infinity` で NaN を返す。原著が図を描く範囲では起きない。
     */
    fun sigmoid(x: Double): Double = exp(x) / (1.0 + exp(x))

    /** 1 層目の 1 つ目のニューロン。重み (6, 10)、バイアス -15 */
    fun line1(a: Double, b: Double): Double = step(6 * a + 10 * b - 15).toDouble()

    /** 1 層目の 2 つ目のニューロン。重み (10, 6)、バイアス -15 */
    fun line2(a: Double, b: Double): Double = step(10 * a + 6 * b - 15).toDouble()

    /** 常に 1 を返すニューロン。入力を一切見ない */
    @Suppress("UNUSED_PARAMETER")
    fun bias(a: Double, b: Double): Double = 1.0

    /**
     * 階段関数だけで組んだネットワーク。
     *
     * 2 層目は `line1 + line2 - 1.5 >= 0`。1 層目の出力は 0 か 1 なので、
     * 和が 1.5 以上になるのは **両方とも 1 のときだけ**。つまり AND である。
     */
    fun nnWithStep(a: Double, b: Double): Double =
        step(step(6 * a + 10 * b - 15) + step(10 * a + 6 * b - 15) - 1.5).toDouble()

    /** 同じ重みでシグモイドに置き換えたネットワーク。出力は連続値になる */
    fun nnWithSigmoid(a: Double, b: Double): Double =
        sigmoid(1.0 * sigmoid(6 * a + 10 * b - 15) + 1.0 * sigmoid(10 * a + 6 * b - 15) - 1.5)

    /** 原著の `h(x, y) = f(x, y) >= 0.5`。境界はこの判定で決まる */
    fun classify(f: (Double, Double) -> Double, a: Double, b: Double): Int =
        if (f(a, b) >= DECISION_THRESHOLD) 1 else 0

    /** 8 点それぞれの予測 */
    fun predictions(f: (Double, Double) -> Double): List<Int> =
        ALIEN_DATASET.map { classify(f, it.aack.toDouble(), it.beep.toDouble()) }

    /** 8 点に対する正解率 */
    fun accuracy(f: (Double, Double) -> Double): Double {
        val correct = predictions(f).zip(ALIEN_DATASET) { predicted, alien -> predicted == alien.happy }
        return correct.count { it }.toDouble() / ALIEN_DATASET.size
    }

    /** 原著の `np.arange(-0.5, 3, 0.005)` と同じ刻み */
    fun axis(): DoubleArray {
        val count = Math.ceil((GRID_MAX - GRID_MIN) / GRID_STEP).toInt()
        return DoubleArray(count) { GRID_MIN + it * GRID_STEP }
    }

    /**
     * 格子のうち、1 と判定される点の割合。
     *
     * 図を見なくても「境界がどこにあるか」を数値で比べられる。
     */
    fun regionRatio(f: (Double, Double) -> Double): Double {
        val values = axis()
        var ones = 0L
        for (x in values) for (y in values) ones += classify(f, x, y)
        return ones.toDouble() / (values.size.toLong() * values.size)
    }

    /** 2 つの関数の判定が食い違う格子点の割合 */
    fun disagreementRatio(f: (Double, Double) -> Double, g: (Double, Double) -> Double): Double {
        val values = axis()
        var different = 0L
        for (x in values) for (y in values) if (classify(f, x, y) != classify(g, x, y)) different++
        return different.toDouble() / (values.size.toLong() * values.size)
    }
}
