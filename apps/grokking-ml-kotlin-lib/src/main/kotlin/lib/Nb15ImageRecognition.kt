package lib

import java.util.zip.ZipFile
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.MLP

/**
 * 原著ノートブック #15 `Chapter_10_Neural_Networks/Image_recognition.ipynb`。
 *
 * MNIST の手書き数字 7 万枚を、[#13][Nb13NeuralNetworkBoundary] と同じ形の
 * ネットワークで分類する。違いは入力が 784 次元（28 × 28 画素）で、
 * 出力が 10 クラスになること。
 *
 * **Kotlin に MNIST のローダは無い。** 原著の
 * `keras.datasets.mnist.load_data()` は `mnist.npz` を取ってくるだけなので、
 * 同じファイルを共有データセットから読み、`.npy` の中身を自前で解析する。
 */
object Nb15ImageRecognition {

    /** 画像の大きさ */
    const val IMAGE_SIZE = 28
    const val INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE

    /** 原著のネットワークの形 */
    val HIDDEN_UNITS = intArrayOf(128, 64)
    const val CLASSES = 10

    /**
     * 原著の学習回数は 10 だが、Smile では時間がかかりすぎる。
     * 6 万枚 × 10 回は現実的でないので、既定は小さくしてある。
     */
    const val ORIGINAL_EPOCHS = 10

    /** MNIST の訓練セットとテストセット */
    class Mnist(
        val xTrain: Array<DoubleArray>,
        val yTrain: IntArray,
        val xTest: Array<DoubleArray>,
        val yTest: IntArray,
    )

    /**
     * NumPy の `.npy` 形式を読む。
     *
     * 形式は単純である。
     * - 6 バイトの magic `\x93NUMPY`
     * - メジャー・マイナー版が 1 バイトずつ
     * - ヘッダ長（版 1 なら 2 バイトのリトルエンディアン）
     * - Python の辞書リテラル（`{'descr': '|u1', 'fortran_order': False, 'shape': (60000, 28, 28), }`）
     * - 生のデータ
     *
     * MNIST は符号なし 8 ビット（`|u1`）なので、そのまま読める。
     */
    class NpyArray(val shape: IntArray, val data: ByteArray) {
        val size: Int get() = shape.fold(1) { acc, dimension -> acc * dimension }

        /** 符号なしとして解釈した値 */
        fun unsigned(index: Int): Int = data[index].toInt() and 0xFF
    }

    /** `.npy` の中身を解析する */
    fun parseNpy(bytes: ByteArray): NpyArray {
        require(bytes.size > 10) { "npy として短すぎます" }
        require(bytes[0] == 0x93.toByte() && String(bytes, 1, 5) == "NUMPY") {
            "npy の magic が合いません"
        }

        // 版 1 はヘッダ長が 2 バイト、版 2 以降は 4 バイト
        val major = bytes[6].toInt()
        val headerLengthBytes = if (major == 1) 2 else 4
        var headerLength = 0
        for (i in 0 until headerLengthBytes) {
            headerLength = headerLength or ((bytes[8 + i].toInt() and 0xFF) shl (8 * i))
        }
        val header = String(bytes, 8 + headerLengthBytes, headerLength)

        require("'descr': '|u1'" in header) { "符号なし 8 ビット以外は扱いません: $header" }

        val shape = SHAPE_PATTERN.find(header)!!.groupValues[1]
            .split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
            .toIntArray()

        val dataStart = 8 + headerLengthBytes + headerLength
        return NpyArray(shape, bytes.copyOfRange(dataStart, bytes.size))
    }

    private val SHAPE_PATTERN = Regex("""'shape':\s*\(([^)]*)\)""")

    /** `.npz`（zip）から指定した `.npy` を取り出して解析する */
    fun readNpz(name: String): NpyArray {
        ZipFile(Datasets.path("mnist.npz").toFile()).use { zip ->
            val entry = zip.getEntry(name) ?: error("$name が mnist.npz にありません")
            return parseNpy(zip.getInputStream(entry).readBytes())
        }
    }

    /**
     * MNIST を読み込む。
     *
     * 画像は 28 × 28 の 2 次元だが、[NpyArray] の時点で 1 次元に並んでいるので、
     * 784 ごとに切り出すだけで原著の `reshape(-1, 28*28)` と同じ形になる。
     */
    fun loadMnist(): Mnist {
        fun images(name: String): Array<DoubleArray> {
            val array = readNpz(name)
            val count = array.shape[0]
            return Array(count) { row ->
                DoubleArray(INPUT_DIM) { column -> array.unsigned(row * INPUT_DIM + column).toDouble() }
            }
        }

        fun labels(name: String): IntArray {
            val array = readNpz(name)
            return IntArray(array.shape[0]) { array.unsigned(it) }
        }

        return Mnist(images("x_train.npy"), labels("y_train.npy"), images("x_test.npy"), labels("y_test.npy"))
    }

    /** 原著の `batch_size` と同じ。**この値がそのまま学習の成否を決める**（[fit] 参照） */
    const val BATCH_SIZE = 10

    /** 学習率。Smile の既定（0.01）では 3 エポックでは足りないので上げてある */
    const val LEARNING_RATE = 0.1

    /**
     * 原著と同じ形のネットワークを学習する。
     *
     * 原著と違うところが 2 つある。どちらも **実測して必要だと分かった** ものである。
     *
     * 1. **画素値を 0〜1 に直してから渡す。** 原著は 0〜255 のまま渡しているが、
     *    [#14][Nb14HousePriceNetwork] と同じ理由で Smile では発散する。
     * 2. **[BATCH_SIZE] 件ずつに切って `update` を呼ぶ。**
     *    Smile の `MLP.update(x, y)` は渡した配列全体を 1 つのミニバッチとして扱う。
     *    訓練データをまるごと渡すと 1 エポックで重みが 1 回しか動かず、
     *    **正解率は当てずっぽうと同じ 0.09〜0.10 のまま** で、エラーも警告も出ない。
     *    10 件ずつに切ると、同じ 10000 件・3 エポックで 0.93 に届く。
     */
    fun fit(mnist: Mnist, epochs: Int, sampleSize: Int, seed: Int = 0): MLP {
        smile.math.MathEx.setSeed(seed.toLong())

        val x = Array(sampleSize) { row -> DoubleArray(INPUT_DIM) { mnist.xTrain[row][it] / 255.0 } }
        val y = IntArray(sampleSize) { mnist.yTrain[it] }

        val network = MLP(
            Layer.input(INPUT_DIM),
            Layer.rectifier(HIDDEN_UNITS[0]),
            Layer.rectifier(HIDDEN_UNITS[1]),
            Layer.mle(CLASSES, OutputFunction.SOFTMAX),
        )
        network.setLearningRate(smile.math.TimeFunction.constant(LEARNING_RATE))
        repeat(epochs) {
            var start = 0
            while (start + BATCH_SIZE <= sampleSize) {
                network.update(
                    Array(BATCH_SIZE) { x[start + it] },
                    IntArray(BATCH_SIZE) { y[start + it] },
                )
                start += BATCH_SIZE
            }
        }
        return network
    }

    /** 学習した重みの総数。原著の `model.summary()` が出す 109,386 と突き合わせる */
    fun parameterCount(): Int =
        (INPUT_DIM * HIDDEN_UNITS[0] + HIDDEN_UNITS[0]) +
            (HIDDEN_UNITS[0] * HIDDEN_UNITS[1] + HIDDEN_UNITS[1]) +
            (HIDDEN_UNITS[1] * CLASSES + CLASSES)

    /** テストセットに対する正解率 */
    fun testAccuracy(model: MLP, mnist: Mnist, sampleSize: Int = mnist.yTest.size): Double {
        val correct = (0 until sampleSize).count { row ->
            val scaled = DoubleArray(INPUT_DIM) { mnist.xTest[row][it] / 255.0 }
            model.predict(scaled) == mnist.yTest[row]
        }
        return correct.toDouble() / sampleSize
    }
}
