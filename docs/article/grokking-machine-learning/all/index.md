# Grokking Machine Learning - 3 言語統合比較

Python・Kotlin・F# の 3 言語で同じ機械学習アルゴリズムを実装し、章ごとに読み比べる統合記事です。

同じアルゴリズムを 3 つのパラダイム（手続き型寄りの Python、オブジェクト指向と関数型を併せ持つ Kotlin、関数型の F#）で書くと、**どこが言語の違いで、どこがアルゴリズムの本質か** が分離して見えてきます。

## 言語の位置づけ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 主なパラダイム | マルチパラダイム（手続き型寄り） | オブジェクト指向 + 関数型 | 関数型ファースト |
| 型付け | 動的（型ヒントは任意） | 静的・null 安全 | 静的・型推論が強い |
| 不変データの表現 | `@dataclass(frozen=True)` | `data class` + `val` | レコード型（既定で不変） |
| 関数の扱い | 第一級（部分適用は `functools.partial`） | 第一級（ラムダ・関数参照） | 第一級（カリー化が既定） |
| ランタイム | CPython | JVM | .NET |

## 目次

| 章 | タイトル | 統合記事 |
| :--- | :--- | :--- |
| 03 | 線形回帰 | [3 言語比較](ch03-linear-regression.md) |
| 04 | 過学習・未学習と正則化 | [3 言語比較](ch04-regularization.md) |
| 05 | パーセプトロン | [3 言語比較](ch05-perceptron.md) |
| 06 | ロジスティック回帰 | [3 言語比較](ch06-logistic-regression.md) |

## 参照

- [シリーズ索引](../index.md)
- [Python 版](../python/index.md)
- [Kotlin 版](../kotlin/index.md)
- [F# 版](../fsharp/index.md)
