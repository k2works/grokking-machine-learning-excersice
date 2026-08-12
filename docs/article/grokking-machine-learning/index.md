# Grokking Machine Learning

「Grokking Machine Learning」（Luis G. Serrano 著）の学習用記事シリーズです。
**Python・Kotlin・F# の 3 言語** で、機械学習アルゴリズムを ML ライブラリに頼らず自前実装しながら解説します。

ライブラリの `fit()` を呼ぶのではなく、線形回帰の重み更新や決定木の分割基準を **自分の手で書く** ことで、アルゴリズムの中身を理解することを目的にしています。同じアルゴリズムを 3 言語で書き比べると、手続き型・オブジェクト指向・関数型それぞれの表現の違いも見えてきます。

## 3 言語統合比較

章ごとに 3 言語の実装を並べて比較する統合記事です。

- [統合記事 目次](all/index.md)

| 章 | タイトル | 統合記事 |
| :--- | :--- | :--- |
| 03 | 線形回帰 | [3 言語比較](all/ch03-linear-regression.md) |
| 04 | 過学習・未学習と正則化 | [3 言語比較](all/ch04-regularization.md) |
| 05 | パーセプトロン | [3 言語比較](all/ch05-perceptron.md) |
| 06 | ロジスティック回帰 | [3 言語比較](all/ch06-logistic-regression.md) |
| 07 | 分類モデルの評価指標 | [3 言語比較](all/ch07-metrics.md) |

## 言語別解説

### Python 版

Python 3.11+ を標準ライブラリのみで使い、`dataclass` と純関数でアルゴリズムを表現します。

- [Python 解説](python/index.md)

### Kotlin 版

Kotlin 2.0（JVM 21）の `data class` と不変更新で、型安全にアルゴリズムを表現します。

- [Kotlin 解説](kotlin/index.md)

### F# 版

F#（.NET 10）のレコード型・判別共用体・パイプライン演算子で、関数型スタイルにアルゴリズムを表現します。

- [F# 解説](fsharp/index.md)

## 章構成

| 章 | テーマ | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- | :--- |
| 01 | 機械学習とは何か | 執筆予定 | 執筆予定 | 執筆予定 |
| 02 | 機械学習の種類 | 執筆予定 | 執筆予定 | 執筆予定 |
| 03 | 線形回帰 | [ch03](python/ch03.md) | [ch03](kotlin/ch03.md) | [ch03](fsharp/ch03.md) |
| 04 | 過学習・未学習と正則化 | [ch04](python/ch04.md) | [ch04](kotlin/ch04.md) | [ch04](fsharp/ch04.md) |
| 05 | パーセプトロン | [ch05](python/ch05.md) | [ch05](kotlin/ch05.md) | [ch05](fsharp/ch05.md) |
| 06 | ロジスティック回帰 | [ch06](python/ch06.md) | [ch06](kotlin/ch06.md) | [ch06](fsharp/ch06.md) |
| 07 | 分類モデルの評価指標 | [ch07](python/ch07.md) | [ch07](kotlin/ch07.md) | [ch07](fsharp/ch07.md) |
| 08 | ナイーブベイズ | 執筆予定 | 執筆予定 | 執筆予定 |
| 09 | 決定木 | 執筆予定 | 執筆予定 | 執筆予定 |
| 10 | ニューラルネットワーク | 執筆予定 | 執筆予定 | 執筆予定 |
| 11 | サポートベクターマシンとカーネル法 | 執筆予定 | 執筆予定 | 執筆予定 |
| 12 | アンサンブル学習 | 執筆予定 | 執筆予定 | 執筆予定 |
| 13 | エンドツーエンドの実例 | 執筆予定 | 執筆予定 | 執筆予定 |

執筆計画の詳細は [アウトライン](outline.md) を参照してください。

## サンプルコード

| 言語 | ディレクトリ | テスト実行 |
| :--- | :--- | :--- |
| Python | `apps/grokking-ml-python` | `uv run pytest` |
| Kotlin | `apps/grokking-ml-kotlin` | `./gradlew test` |
| F# | `apps/grokking-ml-fsharp` | `dotnet test` |

## 参照

- [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) - 原著
- [原著のサンプルコード（GitHub）](https://github.com/luisguiserrano/manning)
- [Python 公式ドキュメント](https://docs.python.org/3/)
- [Kotlin 公式ドキュメント](https://kotlinlang.org/docs/home.html)
- [F# 公式ドキュメント](https://learn.microsoft.com/ja-jp/dotnet/fsharp/)
