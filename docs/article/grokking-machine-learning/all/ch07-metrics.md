# 第 7 章 分類モデルの評価指標 - 3 言語比較

原著第 7 章の評価指標を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch07.md)・[Kotlin 版](../kotlin/ch07.md)・[F# 版](../fsharp/ch07.md) を参照してください。

## この章の性質

第 3〜6 章は学習アルゴリズムでしたが、本章に **学習は出てきません**。乱数も使いません。したがって **3 言語の実行結果は完全に一致します**。

| 章 | 3 言語の結果 | 理由 |
| :--- | :--- | :--- |
| 3・4 章 | 近いが一致しない | 乱数列が違う |
| 5 章 | 別々の解に到達 | 正しい境界線が無数にある |
| 6 章 | ほぼ一致 | 損失の最小点がひとつ |
| **7 章** | **完全に一致** | **乱数も学習もない純粋な計算** |

数値が一致するので、本章では言語ごとの **書き味と安全性** だけが比較の対象になります。

## 混同行列の構築

4 つの `int` フィールドが並ぶ型です。ここで言語ごとの安全性の差がはっきり出ます。

```python
@dataclass(frozen=True)
class ConfusionMatrix:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
```

```kotlin
data class ConfusionMatrix(
    val truePositives: Int,
    val falsePositives: Int,
    val falseNegatives: Int,
    val trueNegatives: Int,
)
```

```fsharp
type ConfusionMatrix =
    { TruePositives: int
      FalsePositives: int
      FalseNegatives: int
      TrueNegatives: int }
```

| 言語 | 位置引数での構築 | 取り違えの防ぎ方 |
| :--- | :--- | :--- |
| Python | できる（`ConfusionMatrix(3, 1, 2, 4)`） | 規律（キーワード引数を使う） |
| Kotlin | できる（`ConfusionMatrix(3, 1, 2, 4)`） | 規律（名前付き引数を使う） |
| F# | **できない** | **言語が強制** |

**F# のレコードは構築時に必ずフィールド名を書きます。** 4 つとも同じ型なので、Python と Kotlin では偽陽性と偽陰性を取り違えても型検査を通過します。この種の型は、実務でもっとも取り違えが起きやすい形です。

Python と Kotlin でこれを型で防ぐには、`NewType` や `value class` で 4 つを別の型にする方法があります。本シリーズでは規律（生成側で必ず名前付き引数）を選び、それをコード上で徹底しました。

## 数え上げ

```python
    counts = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    for label, prediction in zip(labels, predictions):
        counts[(label, prediction)] += 1
```

```kotlin
    val counts = labels.zip(predictions).groupingBy { it }.eachCount()
    // counts[1 to 1] ?: 0
```

```fsharp
    let counts =
        List.zip labels predictions
        |> List.countBy id
        |> Map.ofList
    // Map.tryFind key counts |> Option.defaultValue 0
```

| 言語 | 数え上げ | 不在キーの扱い |
| :--- | :--- | :--- |
| Python | 4 キーを 0 で初期化して手で加算 | 初期化済みなので不在なし |
| Kotlin | `groupingBy { it }.eachCount()` | `?: 0`（`Map` の添字は `V?`） |
| F# | `List.countBy id \|> Map.ofList` | `Option.defaultValue 0` |

Kotlin と F# は標準の集計関数 1 つで済みます。Python は手でループを書きましたが、代わりに **「起こりうる組は 4 つだけ」という事実がコードに現れます**（`defaultdict` を使うとこれが消えます）。

不在キーの扱いは、Kotlin と F# が型で明示させます。Kotlin は `V?`、F# は `Option` です。**「その組み合わせが 1 件もないかもしれない」という可能性が型に現れる** ので、忘れようがありません。

## 整数除算という落とし穴

指標はすべて「整数 ÷ 整数」です。ここが本章最大の罠でした。

```python
    return matrix.true_positives / predicted_positive
```

```kotlin
private fun safeDivide(numerator: Int, denominator: Int): Double =
    if (denominator == 0) 0.0 else numerator.toDouble() / denominator
```

```fsharp
let private safeDivide (numerator: int) (denominator: int) =
    if denominator = 0 then
        0.0
    else
        float numerator / float denominator
```

| 言語 | `3 / 4` の結果 | 変換を忘れたら |
| :--- | :--- | :--- |
| Python | `0.75`（`/` は常に浮動小数点除算） | 問題なし |
| Kotlin | `0`（整数除算） | **静かに 0 になる** |
| F# | `0`（整数除算） | 型エラーで止まる |

Kotlin がもっとも危険です。`numerator / denominator` は `Int` を返し、それが `Double` の戻り値型へ暗黙に昇格するため、**適合率 0.75 が 0.0 になってもコンパイルは通ります**。

F# は `float numerator / denominator` と片方だけ変換すると型エラーになります。両方の明示を強制されるぶん、書き忘れが実行時まで残りません。**第 4 章から「明示のコスト」として現れていた性質が、本章で利益に転じました。**

Python は `/` が常に浮動小数点除算なので、この問題自体が起きません（整数除算がほしいときは `//` を明示します）。

3 言語とも `safeDivide` 相当を 1 か所にまとめたことで、指標本体はそれぞれ 1 行の式になりました。分子と分母の定義だけがコードに残ります。

## 台形則

ROC 曲線の面積計算です。「隣り合う 2 点の台形を足す」という定義をどう書くか。

```python
    for (x1, y1), (x2, y2) in pairwise(points):
        area += (x2 - x1) * (y1 + y2) / 2
```

```kotlin
    rocPoints(labels, probabilities)
        .zipWithNext { (x1, y1), (x2, y2) -> (x2 - x1) * (y1 + y2) / 2 }
        .sum()
```

```fsharp
    rocPoints labels probabilities
    |> List.pairwise
    |> List.sumBy (fun ((x1, y1), (x2, y2)) -> (x2 - x1) * (y1 + y2) / 2.0)
```

3 言語とも「隣接ペア」を作る標準機能を持っていました。

| 言語 | 隣接ペア | 合計 |
| :--- | :--- | :--- |
| Python | `itertools.pairwise` | 手で加算（`for` ループ） |
| Kotlin | `zipWithNext { }`（変換も同時） | `.sum()` |
| F# | `List.pairwise` | `List.sumBy` |

Kotlin の `zipWithNext` はペア作成と変換を同時に行うため、もっとも短くなりました。F# は `pairwise` と `sumBy` の 2 段ですが、パイプラインで手順が素直に並びます。

Python は `pairwise` があるものの、合計は `for` ループになりました。`sum(... for ...)` と書くこともできますが、行が長くなるため素直なループを選んでいます。

なお最初は `zip(points, points[1:])` と書いていましたが、リンターが `pairwise` を勧めてきました。**リストの複製を作らない点でも、意図が明確な点でも `pairwise` が優ります。**

## タプルのソート

ROC の点列を辞書式に並べ替える箇所で差が出ました。

```python
    return sorted(points)
```

```kotlin
    }.sortedWith(compareBy({ it.first }, { it.second }))
```

```fsharp
    |> List.sort
```

Python のタプルと F# のタプルは **構造的比較を持つ** ため、そのままソートできます。Kotlin の `Pair` は `Comparable` を実装していないので、`compareBy` で比較器を組み立てる必要がありました。

**「タプルが値として振る舞うか」の差** です。Kotlin の `Pair` は汎用の入れ物であり、順序という概念を持ちません。データ指向の処理では、この一手間が繰り返し現れます。

## テストが思い込みを正した

AUC のテストで、当初こう書いて **失敗しました**。

```python
def test_auc_is_a_half_for_a_useless_ranking():
    # 陽性と陰性が交互に並ぶ、まったく情報のない並び
    labels = [1, 0, 1, 0]
    probabilities = [0.8, 0.6, 0.4, 0.2]
    assert auc(labels, probabilities) == pytest.approx(0.5)   # 実際は 0.75
```

「陽性と陰性が交互ならランダムと同じ」と考えたのですが、間違いでした。AUC は **陽性と陰性のすべての組のうち、陽性のほうが上位にある割合** です。

| 組 | 陽性の確率 | 陰性の確率 | 正しい順序か |
| :--- | ---: | ---: | :--- |
| 1 番目の陽性 vs 1 番目の陰性 | 0.8 | 0.6 | ○ |
| 1 番目の陽性 vs 2 番目の陰性 | 0.8 | 0.2 | ○ |
| 2 番目の陽性 vs 1 番目の陰性 | 0.4 | 0.6 | × |
| 2 番目の陽性 vs 2 番目の陰性 | 0.4 | 0.2 | ○ |

4 組中 3 組が正しい順序なので 0.75 です。0.5 になるのは陽性が最上位と最下位に 1 つずつある場合（`[1, 0, 0, 1]`）でした。

**実装が正しく、テストの期待値が間違っていた** ケースです。3 言語とも、修正後のテストとして両方（0.5 のケースと 0.75 のケース）を残しました。片方だけでは、次に読む人が同じ勘違いをします。

## 閾値と AUC の関係

第 6 章で確率を手に入れたことが、本章で 2 つの形で効きました。

**1. 閾値は運用上の選択である。** 同じモデルでも閾値を下げれば再現率が上がり適合率が下がります。医療診断なら見逃しを避けるため閾値を下げ、迷惑メール判定なら誤検知を避けるため上げます。モデルの再学習は不要です。

**2. AUC は閾値に依存しない。** 3 言語とも次を検証しています。

```python
    scaled = [0.99, 0.98, 0.02, 0.01]
    compressed = [0.55, 0.54, 0.46, 0.45]
    assert auc(labels, scaled) == pytest.approx(auc(labels, compressed))
```

確率が両極に分かれていても中央に固まっていても、**順位が同じなら AUC は同じ** です。第 6 章で「重みの大きさ＝確信の強さ」と述べましたが、その確信の強さは AUC には影響しません。**AUC はモデルの識別能力だけを取り出す指標** です。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | `/` が常に浮動小数点除算、タプルがそのままソート可 | `groupingBy`・`zipWithNext` など集計の語彙が豊富 | フィールド名が必須、整数除算が型エラーで止まる |
| 注意点 | 4 フィールドの取り違えを型が防がない | **整数除算が静かに 0 を返す**、`Pair` がソート不可 | 記述量がやや多い |

第 3〜6 章では言語ごとに強みと弱みが入れ替わってきましたが、本章のように **同じ型のフィールドが並び、整数の割り算が中心** の処理では、F# の「明示を強制する」設計がもっとも事故を防ぎました。逆に集計の書きやすさでは Kotlin の標準ライブラリが際立ちます。

## 参照

- [Python 版 第 7 章](../python/ch07.md)
- [Kotlin 版 第 7 章](../kotlin/ch07.md)
- [F# 版 第 7 章](../fsharp/ch07.md)
- [第 6 章 3 言語比較](ch06-logistic-regression.md)
- [統合記事 目次](index.md)
