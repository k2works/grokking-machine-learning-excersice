# Grokking Machine Learning

「Grokking Machine Learning」（Luis G. Serrano 著）の学習用記事シリーズです。
**Python・Kotlin・F# の 3 言語** で、機械学習アルゴリズムを ML ライブラリに頼らず自前実装しながら解説します。

ライブラリの `fit()` を呼ぶのではなく、線形回帰の重み更新や決定木の分割基準を **自分の手で書く** ことで、アルゴリズムの中身を理解することを目的にしています。同じアルゴリズムを 3 言語で書き比べると、手続き型・オブジェクト指向・関数型それぞれの表現の違いも見えてきます。

さらにもう 1 つの軸として、原著のノートブック 22 本を **主流の ML ライブラリで忠実に再現** する
[ライブラリ版](lib/index.md) があります。同じ問題を「自分で書いた場合」と「ライブラリに任せた場合」で
読み比べられます。

## 3 言語統合比較

章ごとに 3 言語の実装を並べて比較する統合記事です。

- [統合記事 目次](all/index.md)

| 章 | タイトル | 統合記事 |
| :--- | :--- | :--- |
| 01 | 機械学習とは何か | [概念解説](all/ch01-what-is-machine-learning.md) |
| 02 | 機械学習の種類 | [概念解説](all/ch02-types-of-machine-learning.md) |
| 03 | 線形回帰 | [3 言語比較](all/ch03-linear-regression.md) |
| 04 | 過学習・未学習と正則化 | [3 言語比較](all/ch04-regularization.md) |
| 05 | パーセプトロン | [3 言語比較](all/ch05-perceptron.md) |
| 06 | ロジスティック回帰 | [3 言語比較](all/ch06-logistic-regression.md) |
| 07 | 分類モデルの評価指標 | [3 言語比較](all/ch07-metrics.md) |
| 08 | ナイーブベイズ | [3 言語比較](all/ch08-naive-bayes.md) |
| 09 | 決定木 | [3 言語比較](all/ch09-decision-trees.md) |
| 10 | ニューラルネットワーク | [3 言語比較](all/ch10-neural-networks.md) |
| 11 | サポートベクターマシンとカーネル法 | [3 言語比較](all/ch11-svm.md) |
| 12 | アンサンブル学習 | [3 言語比較](all/ch12-ensembles.md) |
| 13 | エンドツーエンドの実例 | [3 言語比較](all/ch13-end-to-end.md) |

## 言語別解説

### Python 版

Python 3.11+ を標準ライブラリのみで使い、`dataclass` と純関数でアルゴリズムを表現します。

- [Python 解説](python/index.md)

### Kotlin 版

Kotlin 2.2（JVM 21）の `data class` と不変更新で、型安全にアルゴリズムを表現します。

- [Kotlin 解説](kotlin/index.md)

### F# 版

F#（.NET 10）のレコード型・判別共用体・パイプライン演算子で、関数型スタイルにアルゴリズムを表現します。

- [F# 解説](fsharp/index.md)

## 章構成

| 章 | テーマ | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- | :--- |
| 01 | 機械学習とは何か | [概念解説](all/ch01-what-is-machine-learning.md)（3 言語共通） | 同左 | 同左 |
| 02 | 機械学習の種類 | [概念解説](all/ch02-types-of-machine-learning.md)（3 言語共通） | 同左 | 同左 |
| 03 | 線形回帰 | [ch03](python/ch03.md) | [ch03](kotlin/ch03.md) | [ch03](fsharp/ch03.md) |
| 04 | 過学習・未学習と正則化 | [ch04](python/ch04.md) | [ch04](kotlin/ch04.md) | [ch04](fsharp/ch04.md) |
| 05 | パーセプトロン | [ch05](python/ch05.md) | [ch05](kotlin/ch05.md) | [ch05](fsharp/ch05.md) |
| 06 | ロジスティック回帰 | [ch06](python/ch06.md) | [ch06](kotlin/ch06.md) | [ch06](fsharp/ch06.md) |
| 07 | 分類モデルの評価指標 | [ch07](python/ch07.md) | [ch07](kotlin/ch07.md) | [ch07](fsharp/ch07.md) |
| 08 | ナイーブベイズ | [ch08](python/ch08.md) | [ch08](kotlin/ch08.md) | [ch08](fsharp/ch08.md) |
| 09 | 決定木 | [ch09](python/ch09.md) | [ch09](kotlin/ch09.md) | [ch09](fsharp/ch09.md) |
| 10 | ニューラルネットワーク | [ch10](python/ch10.md) | [ch10](kotlin/ch10.md) | [ch10](fsharp/ch10.md) |
| 11 | サポートベクターマシンとカーネル法 | [ch11](python/ch11.md) | [ch11](kotlin/ch11.md) | [ch11](fsharp/ch11.md) |
| 12 | アンサンブル学習 | [ch12](python/ch12.md) | [ch12](kotlin/ch12.md) | [ch12](fsharp/ch12.md) |
| 13 | エンドツーエンドの実例 | [ch13](python/ch13.md) | [ch13](kotlin/ch13.md) | [ch13](fsharp/ch13.md) |

執筆計画の詳細は [アウトライン](outline.md) を参照してください。

## ライブラリ版

原著のノートブック 22 本を、各言語の主流 ML ライブラリで忠実に再現する軸です。
自前実装版で書いた処理が、ライブラリではどの API に対応するのかを突き合わせられます。

- [ライブラリ版 索引](lib/index.md)

| 言語 | 主なライブラリ | 解説 |
| :--- | :--- | :--- |
| Python | NumPy / pandas / scikit-learn / Keras / XGBoost | [Python 版](lib/python/index.md) |
| Kotlin | Multik / Kotlin DataFrame / Smile / XGBoost4J / Kandy | [Kotlin 版](lib/kotlin/index.md) |
| F# | Math.NET / Deedle / ML.NET / Accord.Neuro / Plotly.NET | [F# 版](lib/fsharp/index.md) |

## サンプルコード

| 軸 | 言語 | ディレクトリ | テスト実行 | テスト数 |
| :--- | :--- | :--- | :--- | ---: |
| 自前実装 | Python | `apps/grokking-ml-python` | `uv run pytest` | 179 |
| 自前実装 | Kotlin | `apps/grokking-ml-kotlin` | `./gradlew test` | 178 |
| 自前実装 | F# | `apps/grokking-ml-fsharp` | `dotnet test` | 176 |
| ライブラリ | Python | `apps/grokking-ml-python-lib` | `uv run pytest` | 174 |
| ライブラリ | Kotlin | `apps/grokking-ml-kotlin-lib` | `./gradlew test` | 143 |
| ライブラリ | F# | `apps/grokking-ml-fsharp-lib` | `dotnet test` | 146 |

ライブラリ版は 22 本中 12 本を執筆済みです。

記事に載せたコードと数値は、すべてこの実装から転記しています。

## ノートブック

各章の実験は、3 種類のノートブックでも試せます。**実装本体を読み込んで動かす** ので、
記事・実装・ノートブックの 3 者が食い違いません。値を変えてその場で確かめられます。

| 言語 | 種類 | 一覧 | 実行環境 |
| :--- | :--- | :--- | :--- |
| Python | Jupyter Notebook | [Python 版の目次](python/index.md#jupyter-notebook) | `uv run jupyter lab` |
| Kotlin | Kotlin Notebook | [Kotlin 版の目次](kotlin/index.md#kotlin-notebook) | IntelliJ IDEA / Kotlin Jupyter カーネル |
| F# | Polyglot Notebook | [F# 版の目次](fsharp/index.md#polyglot-notebook) | VS Code Polyglot Notebooks / .NET Interactive |

ノートブックの正本は `apps/grokking-ml-*/notebooks/` にあります（実装を相対パスで読み込むため）。
ドキュメントサイトへはビルド時に取り込んでいます。

## 全 13 章を通して見えたこと

同じアルゴリズムを 3 言語で書くと、**言語機能とアルゴリズムの構造が噛み合う場面** がはっきりします。

| 章 | もっとも噛み合った言語と理由 |
| :--- | :--- |
| 3〜6 | 拮抗（状態の持ち方、種類を表す型、標準ライブラリの語彙） |
| 7 | **F#** — フィールド名が必須、整数除算が型エラーで止まる |
| 8 | 拮抗（Python の `Counter` は最短、F# は「無い」を型で表せる） |
| 9 | **F#** — 判別共用体が決定木そのもの |
| 10 | **F#** — 不変リストの `::` が逆伝播の向きに合致 |
| 11 | **F#** — カリー化された関数型に既存関数をそのまま代入 |
| 12 | 拮抗（`zip` が 2 本までという Kotlin の制約が現れる） |
| 13 | **Python** — 型に縛られず前章の部品を流用できた |

また、章をまたいで **同じ形が繰り返し現れます**。

- 第 6 章のシグモイドが、第 8 章のナイーブベイズの最終式と同型
- 第 6 章のロジスティックトリックが、第 10 章の出力層の誤差と同型
- 第 4 章の L2 正則化が、第 11 章ではマージンを広げる道具になる
- 対数を取る箇所には必ずクランプが要る（第 6・8・12 章）

## 参照

- [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) - 原著
- [原著のサンプルコード（GitHub）](https://github.com/luisguiserrano/manning)
- [Python 公式ドキュメント](https://docs.python.org/3/)
- [Kotlin 公式ドキュメント](https://kotlinlang.org/docs/home.html)
- [F# 公式ドキュメント](https://learn.microsoft.com/ja-jp/dotnet/fsharp/)
