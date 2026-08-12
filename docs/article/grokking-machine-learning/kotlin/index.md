# Grokking Machine Learning - Kotlin 版

Kotlin 2.0（JVM 21）を使い、機械学習アルゴリズムを **標準ライブラリのみ** で実装します。KotlinDL や Smile といった ML ライブラリは使いません。

`data class` による不変なモデル表現と、拡張関数・高階関数によるアルゴリズムの組み立てが Kotlin 版の焦点です。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | Kotlin 2.0（JVM ターゲット 21） |
| ビルド | Gradle（Kotlin DSL） |
| テスト | kotlin.test（JUnit 5 プラットフォーム） |
| 実行環境 | Nix devShell `kotlin` |
| サンプル実装 | `apps/grokking-ml-kotlin` |

### セットアップ

```bash
nix develop .#kotlin
```

Nix の `kotlin` devShell が JDK 21・Kotlin コンパイラ・Gradle を提供します。ビルドには Gradle Wrapper（`./gradlew`）を使うため、devShell の Gradle バージョンに関係なく同じ結果になります。

### テストの実行

```bash
cd apps/grokking-ml-kotlin
./gradlew test
```

## 目次

| 章 | タイトル |
| :--- | :--- |
| 03 | [線形回帰](ch03.md) |
| 04 | [過学習・未学習と正則化](ch04.md) |
| 05 | [パーセプトロン](ch05.md) |
| 06 | [ロジスティック回帰](ch06.md) |
| 07 | [分類モデルの評価指標](ch07.md) |
| 08 | [ナイーブベイズ](ch08.md) |
| 09 | [決定木](ch09.md) |

## 参照

- [シリーズ索引](../index.md)
- [3 言語統合比較](../all/index.md)
- [Kotlin 公式ドキュメント](https://kotlinlang.org/docs/home.html)
- [Gradle Kotlin DSL](https://docs.gradle.org/current/userguide/kotlin_dsl.html)
