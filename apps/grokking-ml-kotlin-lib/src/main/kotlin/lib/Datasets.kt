package lib

import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/**
 * 共有データセット（`apps/grokking-ml-datasets`）へのアクセス。
 *
 * Python 版・F# 版と同じ CSV を読むことで、章ごとの数値を言語間で突き合わせられる。
 * サイズの大きい 2 本だけはリポジトリに入れず、初回参照時に原著リポジトリから取得する。
 */
object Datasets {
    /** 共有データセットディレクトリを差し替える環境変数。CI やノートブックで使う */
    private const val ENV_KEY = "GROKKING_ML_DATASETS"

    /** リポジトリに含めず初回にダウンロードするファイルと、その取得元 */
    private val remoteFiles = mapOf(
        "emails.csv" to
            "https://raw.githubusercontent.com/luisguiserrano/manning/master/" +
            "Chapter_08_Naive_Bayes/emails.csv",
        "IMDB_Dataset.csv" to
            "https://raw.githubusercontent.com/luisguiserrano/manning/master/" +
            "Chapter_06_Logistic_Regression/IMDB_Dataset.csv",
        // Keras が `keras.datasets.mnist.load_data()` で取りにいくのと同じファイル
        "mnist.npz" to "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    )

    /** 共有データセットディレクトリを返す */
    fun directory(): Path {
        System.getenv(ENV_KEY)?.let { return Paths.get(it) }
        // 実行時のカレントディレクトリは apps/grokking-ml-kotlin-lib になる
        return Paths.get("..", "grokking-ml-datasets").toAbsolutePath().normalize()
    }

    /** データセットの絶対パスを返す。未取得の大きいファイルはダウンロードする */
    fun path(name: String): Path {
        val path = directory().resolve(name)
        if (Files.exists(path)) return path

        val url = remoteFiles[name]
            ?: throw java.io.FileNotFoundException("データセットが見つかりません: $path")
        Files.createDirectories(path.parent)
        URI.create(url).toURL().openStream().use { input ->
            Files.copy(input, path)
        }
        return path
    }
}
