package lib

import kotlin.math.ln
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.sum

/**
 * 原著ノートブック #08 `Chapter_09_Decision_Trees/Gini_entropy_calculations.ipynb`。
 *
 * 決定木がどこで分割するかを決めるための 2 つの不純度、ジニ不純度とエントロピーを
 * 手で計算する。原著は NumPy しか使わない小さな回で、
 * **決定木の中身を理解するための下準備** にあたる。
 *
 * 分割位置を 1 つずつずらしながら重み付き不純度を見ると、
 * `['A', 'A', 'A'] | ['C', 'B', 'C']` でどちらの指標も最小になる。
 */
object Nb08GiniEntropy {

    /** 原著が使う 6 要素 */
    val ELEMENTS = listOf("A", "A", "A", "C", "B", "C")

    /** ある分割位置での、左右と重み付き不純度 */
    data class SplitImpurity(
        val index: Int,
        val left: List<String>,
        val right: List<String>,
        val weightedGini: Double,
        val weightedEntropy: Double,
    )

    /**
     * 要素ごとの個数を、**初めて現れた順** に返す。
     *
     * 原著は辞書に数えてから取り出しており、Python 3.7 以降の辞書は挿入順を保つ。
     * Kotlin では [LinkedHashMap] が同じ性質を持つ。`groupingBy` の結果も
     * `LinkedHashMap` なので、そのまま初出順になる。
     */
    fun counts(elements: List<String>): List<Int> =
        elements.groupingBy { it }.eachCount().values.toList()

    /**
     * ジニ不純度。1 から「同じクラスを 2 回続けて引く確率」を引いたもの。
     *
     * 空のリストに対しては 1 を返す。原著は特別扱いしておらず、
     * 空の合計が 0 になって `1 - 0` がそのまま 1 になる。
     * 重み付けのときは要素数 0 が掛かるので結果に影響しない。
     */
    fun gini(elements: List<String>): Double {
        val classCounts = counts(elements)
        val n = classCounts.sum()
        return 1.0 - classCounts.sumOf { count ->
            count.toDouble() * count / (n.toDouble() * n)
        }
    }

    /**
     * 情報エントロピー。原著はこちらだけ空のリストを明示的に 0 にしている。
     *
     * `log2(0)` が発散するので、空を通すと NaN になる。
     * ジニ不純度が特別扱い不要だったのと対照的である。
     *
     * NumPy の `np.log2` と `np.dot` にあたる部分を Multik で書いた。
     * Kotlin の標準ライブラリに `log2` はあるが、ここは原著の
     * 「ベクトルに対する一括演算」という形を保っている。
     */
    fun entropy(elements: List<String>): Double {
        if (elements.isEmpty()) return 0.0

        val classCounts = counts(elements)
        val n = classCounts.sum()
        val proportions = mk.ndarray(classCounts.map { 1.0 / n * it })
        val logs = proportions.map { log2(it) }
        return -(logs * proportions).sum()
    }

    /** 2 を底とする対数。Multik には要素ごとの log2 が無いので自分で書く */
    private fun log2(x: Double): Double = ln(x) / ln(2.0)

    /** 左右の不純度を、要素数で重み付けして平均する */
    private fun weighted(
        impurity: (List<String>) -> Double,
        left: List<String>,
        right: List<String>,
        total: Int,
    ): Double = 1.0 / total * (impurity(left) * left.size + impurity(right) * right.size)

    /**
     * 先頭から順に分割位置をずらし、それぞれの重み付き不純度を求める。
     *
     * 原著は 0 から `size - 1` まで回している。**右端では分割しない** ので、
     * 「左が全部・右が空」の場合は出てこない。
     */
    fun splitImpurities(elements: List<String> = ELEMENTS): List<SplitImpurity> =
        elements.indices.map { index ->
            val left = elements.subList(0, index)
            val right = elements.subList(index, elements.size)
            SplitImpurity(
                index = index,
                left = left,
                right = right,
                weightedGini = weighted(::gini, left, right, elements.size),
                weightedEntropy = weighted(::entropy, left, right, elements.size),
            )
        }

    /** 重み付きジニ不純度がもっとも小さい分割を返す */
    fun bestSplit(elements: List<String> = ELEMENTS): SplitImpurity =
        splitImpurities(elements).minBy { it.weightedGini }
}
