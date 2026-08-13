# Grokking Machine Learning - Kotlin 版

Kotlin 2.2（JVM 21）を使い、機械学習アルゴリズムを **標準ライブラリのみ** で実装します。KotlinDL や Smile といった ML ライブラリは使いません。

`data class` による不変なモデル表現と、拡張関数・高階関数によるアルゴリズムの組み立てが Kotlin 版の焦点です。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | Kotlin 2.2（JVM ターゲット 21） |
| ビルド | Gradle（Kotlin DSL） |
| テスト | kotlin.test（JUnit 5 プラットフォーム） |
| 実行環境 | Nix devShell `kotlin` |
| サンプル実装 | `apps/grokking-ml-kotlin` |

### セットアップ

```bash
nix develop .#kotlin
```

Nix の `kotlin` devShell が JDK 21・Kotlin コンパイラ・Gradle を提供します。ビルドには Gradle Wrapper（`./gradlew`）を使うため、Gradle のバージョンは固定されます。

**JDK は Gradle Wrapper では固定できません。** Nix を使わない場合は、JDK 21 以上を用意してください。Kotlin プラグインが新しい JDK のバージョン文字列を解釈できないと、`Internal compiler error` という原因の分かりにくいエラーで止まります（本シリーズは JDK 21 と 25 で動作を確認しています）。

```bash
java -version   # 21 以上であること
```

### テストの実行

```bash
cd apps/grokking-ml-kotlin
./gradlew test
```

## 目次

| 章 | タイトル |
| :--- | :--- |
| 01 | [機械学習とは何か](../all/ch01-what-is-machine-learning.md)（3 言語共通） |
| 02 | [機械学習の種類](../all/ch02-types-of-machine-learning.md)（3 言語共通） |
| 03 | [線形回帰](ch03.md) |
| 04 | [過学習・未学習と正則化](ch04.md) |
| 05 | [パーセプトロン](ch05.md) |
| 06 | [ロジスティック回帰](ch06.md) |
| 07 | [分類モデルの評価指標](ch07.md) |
| 08 | [ナイーブベイズ](ch08.md) |
| 09 | [決定木](ch09.md) |
| 10 | [ニューラルネットワーク](ch10.md) |
| 11 | [サポートベクターマシンとカーネル法](ch11.md) |
| 12 | [アンサンブル学習](ch12.md) |
| 13 | [エンドツーエンドの実例](ch13.md) |

## Kotlin Notebook

各章の実験を、**実装本体を読み込んで** その場で動かせるノートブックです。`@file:DependsOn` でビルド済み JAR を読み込みます。IntelliJ IDEA の Kotlin Notebook プラグインでも開けます。
コードを複製していないので、記事・実装・ノートブックの 3 者が食い違いません。

```bash
cd apps/grokking-ml-kotlin
./gradlew jar   # 先に JAR を作る
jupyter lab notebooks/
```

| 章 | ノートブック |
| :--- | :--- |
| 第 3 章 線形回帰 | [ch03.ipynb](notebooks/ch03.ipynb) |
| 第 4 章 過学習・未学習と正則化 | [ch04.ipynb](notebooks/ch04.ipynb) |
| 第 5 章 パーセプトロン | [ch05.ipynb](notebooks/ch05.ipynb) |
| 第 6 章 ロジスティック回帰 | [ch06.ipynb](notebooks/ch06.ipynb) |
| 第 7 章 分類モデルの評価指標 | [ch07.ipynb](notebooks/ch07.ipynb) |
| 第 8 章 ナイーブベイズ | [ch08.ipynb](notebooks/ch08.ipynb) |
| 第 9 章 決定木 | [ch09.ipynb](notebooks/ch09.ipynb) |
| 第 10 章 ニューラルネットワーク | [ch10.ipynb](notebooks/ch10.ipynb) |
| 第 11 章 SVM とカーネル法 | [ch11.ipynb](notebooks/ch11.ipynb) |
| 第 12 章 アンサンブル学習 | [ch12.ipynb](notebooks/ch12.ipynb) |
| 第 13 章 エンドツーエンドの実例 | [ch13.ipynb](notebooks/ch13.ipynb) |

## 参照

- [シリーズ索引](../index.md)
- [3 言語統合比較](../all/index.md)
- [Kotlin 公式ドキュメント](https://kotlinlang.org/docs/home.html)
- [Gradle Kotlin DSL](https://docs.gradle.org/current/userguide/kotlin_dsl.html)
