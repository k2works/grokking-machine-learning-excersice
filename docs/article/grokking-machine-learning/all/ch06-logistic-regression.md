# 第 6 章 ロジスティック回帰 - 3 言語比較

原著第 6 章のロジスティック回帰を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch06.md)・[Kotlin 版](../kotlin/ch06.md)・[F# 版](../fsharp/ch06.md) を参照してください。

## 第 5 章からの差分

本章の実装は、第 5 章のパーセプトロンから **2 箇所しか変わりません**。

1. `predict`（0 か 1）と学習則の間に **シグモイド** を挟む
2. 誤差関数を **対数損失** に置き換える

それだけで、確率が手に入り、閾値を後から変えられ、正しく分類できた点からも学べるようになります。**小さな差分が挙動を大きく変える** ことが本章の要点です。

## シグモイドの数値的安定性

教科書の定義は `1 / (1 + exp(-x))` の 1 行ですが、3 言語ともそのままは書きませんでした。

```python
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)
```

```kotlin
fun sigmoid(x: Double): Double =
    if (x >= 0) {
        1.0 / (1.0 + exp(-x))
    } else {
        val exponential = exp(x)
        exponential / (1.0 + exponential)
    }
```

```fsharp
let sigmoid (x: float) =
    if x >= 0.0 then
        1.0 / (1.0 + exp (-x))
    else
        let exponential = exp x
        exponential / (1.0 + exponential)
```

| 言語 | 素朴な実装で `x = -1000` のとき | 影響 |
| :--- | :--- | :--- |
| Python | `OverflowError` を送出 | **例外で落ちる** |
| Kotlin | `exp(1000.0)` が `Infinity` | 落ちないが `Infinity` を経由 |
| F# | `exp 1000.0` が `infinity` | 落ちないが `infinity` を経由 |

**Python だけが例外で落ちます。** JVM と .NET は IEEE 754 に従って `Infinity` を返し、`1.0 / Infinity` が 0.0 になるため結果自体は正しくなります。

とはいえ 3 言語とも同じ対策を入れました。片方だけ「たまたま動く」実装に依存するのは、移植時にも将来の変更時にも危ういからです。**数学的に同じ式でも、浮動小数点の上では安全な形と危険な形があります。**

対策は 3 言語とも「負の入力では分子分母に `exp(x)` を掛けた等価な式に切り替える」で共通です。`exp(-1000)` は 0 に潰れるだけなので安全です。

なお F# は `exp` / `log` を `Math.Exp` などのメソッド経由ではなく **そのまま関数として** 書けます。数式との距離がもっとも近い書き方になります。

## 閾値をどう与えるか

確率が手に入ると、判定の基準（閾値）を後から変えられます。ここで言語機能の差が出ました。

```python
    def predict(self, point: Point, threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(point) >= threshold else 0
```

```kotlin
    fun predict(point: Point, threshold: Double = 0.5): Int =
        if (predictProbability(point) >= threshold) 1 else 0
```

```fsharp
let predictWith (threshold: float) (model: LogisticClassifier) (point: Point) =
    if predictProbability model point >= threshold then 1 else 0

let predict (model: LogisticClassifier) (point: Point) = predictWith 0.5 model point
```

| 言語 | 手段 | 関数の数 |
| :--- | :--- | ---: |
| Python | デフォルト引数 | 1 |
| Kotlin | デフォルト引数 | 1 |
| F# | 一般形 + 部分適用で既定値を固定 | 2 |

Python と Kotlin はデフォルト引数 1 つで済みます。**既存の呼び出しを一切壊さずに調整点を増やせました。**

F# にはデフォルト引数がないため、一般形と特殊形の 2 つの関数になります。手数は増えますが、「既定はこれ」という決定が関数定義として明示される利点があります。第 4 章の `train`、第 5 章の引数順の設計と同じ、**F# における「設定は部分適用で固定する」という一貫した解き方** です。

## 対数損失

第 5 章のパーセプトロン誤差には、全点が境界線上にある退化状態を 0 と評価する欠陥がありました。対数損失にその問題はありません。

| 正解 | 予測確率 | 損失 |
| ---: | ---: | ---: |
| 1 | 0.99 | 0.01 |
| 1 | 0.51 | 0.67 |
| 1 | 0.50 | 0.69 |
| 1 | 0.01 | 4.61 |

**当たったかどうかではなく、どれくらいの確信で当たったかを測ります。**

実装で共通して必要になったのが、`log(0)` を避けるクランプです。

```python
    epsilon = 1e-15
    probability = min(max(probability, epsilon), 1.0 - epsilon)
```

```kotlin
    val probability = min(max(model.predictProbability(point), epsilon), 1.0 - epsilon)
```

```fsharp
    let probability =
        predictProbability model point
        |> max epsilon
        |> min (1.0 - epsilon)
```

Python と Kotlin は `min(max(x, lo), hi)` という入れ子で、内側から読む必要があります。F# はパイプラインで「下限で切って上限で切る」という手順がそのまま並びます。**`max` / `min` を部分適用して繋げられるかどうかの差** です。

Kotlin には `coerceIn(lo, hi)` という専用の関数があり、実務ではそちらが読みやすくなります（本シリーズでは 3 言語で式を揃えるため `min`/`max` を使いました）。

クランプが必要な理由は 3 言語で同じです。シグモイドは数学的には 0 にも 1 にもなりませんが、浮動小数点では `sigmoid(-1000) == 0.0` になります。**数学的に起きないことが実装では起きます。**

## トリックの差分

第 5 章と本章のトリックを並べます。

```python
# 第 5 章
    error = label - model.predict(point)
    if error == 0:
        return model

# 第 6 章
    error = label - model.predict_probability(point)
```

| | パーセプトロン | ロジスティック |
| :--- | :--- | :--- |
| error の値 | -1, 0, +1 の 3 通り | -1 から +1 の連続値 |
| error の型（静的型付け言語） | `Int` | `Double` / `float` |
| 正しい点 | 動かない | **わずかに動く** |
| 分岐 | あり | **なし** |

3 言語とも **分岐が消えました**。これは単純化であると同時に、実装上も健全です。浮動小数点になった `error` を 0 と厳密比較する分岐は書くべきではなく、書く必要もありませんでした。

型変換の扱いは第 5 章と同じ構図です。

| 言語 | 記述 |
| :--- | :--- |
| Python | `label - model.predict_probability(point)`（動的） |
| Kotlin | `label - model.predictProbability(point)`（`Int - Double` を自動昇格） |
| F# | `float label - predictProbability model point`（**明示が必須**） |

「動く／動かない」の検証も、3 言語で対になっています。

```python
# 第 5 章: assert perceptron_trick(...) == model
# 第 6 章: assert moved != model
```

```kotlin
// 第 5 章: assertEquals(model, perceptronTrick(...))
// 第 6 章: assertNotEquals(model, moved)
```

```fsharp
// 第 5 章: Assert.Equal(model, perceptronTrick ...)
// 第 6 章: Assert.NotEqual(model, moved)
```

**同じ形のテストで逆の性質を固定しているのが 2 つの章の対比そのものです。** 3 言語とも値の等価性が自動で手に入っている（`@dataclass`・`data class`・レコード型）から 1 行で書けます。

## 実験結果

第 5 章と同じデータ、学習率 0.1、1000 エポック、シード 0 の結果です。

| 言語 | aack の重み | beep の重み | バイアス | 最終の平均対数損失 | 正解率 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Python | 2.2506 | 1.5327 | -5.6115 | 0.1638 | 1.0 |
| Kotlin | 2.0201 | 1.5903 | -5.5970 | 0.1582 | 1.0 |
| F# | 2.1482 | 1.5443 | -5.5948 | 0.1600 | 1.0 |

**第 5 章と違い、3 言語がほぼ同じ場所に収束しました。**

| | 第 5 章（パーセプトロン） | 第 6 章（ロジスティック） |
| :--- | :--- | :--- |
| 到達した解 | 言語ごとにバラバラ | ほぼ一致 |
| 理由 | 正しい境界線が無数にある | 損失に最小点がひとつしかない |
| 重みの絶対値 | 意味を持たない（比率だけ） | **確信の強さを表す** |
| 重みの大きさ | 0.01〜0.04 | 2.0〜2.3 |

第 5 章では乱数の違いで別々の直線に落ち着きましたが、本章では損失を下げる方向が一意に決まるため、どの経路を通っても同じ谷底へ向かいます。**乱数列の違いは到達点ではなく到達までの経路にしか影響しません。**

そして重みが 100 倍近く大きくなりました。パーセプトロンでは重みを定数倍しても同じ分類器でしたが、ロジスティックでは **重みを大きくすると確率が 0 と 1 に寄る**（確信が強くなる）ため、絶対値そのものに意味があります。

### 確率としての出力

| 言語 | `(3, 2)` の確率 | `(1, 0)` の確率 |
| :--- | ---: | ---: |
| Python | 0.9853 | 0.0335 |
| Kotlin | 0.9745 | 0.0272 |
| F# | 0.9809 | 0.0309 |

3 言語とも「明確に楽しい文」に 0.97 以上、「明確に悲しい文」に 0.04 未満を与えます。単に分類できているだけでなく、**どちらがより確からしいかまで答えられる** ようになりました。

## 損失の初期値が語ること

3 言語のテストは共通して次を検証しています。

```python
    # パーセプトロン誤差と違い、対数損失は初期状態でも 0 にならない
    assert losses[0] == pytest.approx(-math.log(0.5))
    assert losses[-1] < losses[0]
```

初期の損失は `-log(0.5) = 0.6931` です。全パラメータが 0 のとき、すべての予測は確率 0.5 になります。

第 5 章のパーセプトロン誤差は、同じ初期状態を **0（最良）** と評価しました。正解率が 0.5 しかないのに、です。対数損失は同じ状態を「五分五分で何も分かっていない」と正しく評価します。

その結果、第 5 章では書けなかった `losses[-1] < losses[0]`（損失が減った）が本章では素直に書けます。**損失関数を替えたことで、第 3 章と同じ形のテストが戻ってきました。** テストの書きやすさが、損失関数の良し悪しを映しています。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | 記述量が最小 | デフォルト引数 + `coerceIn` など語彙が豊富 | `exp`/`log` が関数、クランプがパイプラインで並ぶ |
| 注意点 | **シグモイドが例外で落ちうる唯一の言語** | 特になし | デフォルト引数がなく関数が 2 つに分かれる |

第 3 章は「状態の持ち方」、第 4 章は「種類を表す型の安全性」、第 5 章は「標準ライブラリの語彙」が差でした。本章で見えたのは **数値計算の落とし穴に対する言語の姿勢** です。同じ式を書いたとき、例外で落ちるか、`Infinity` を返して進むか。どちらにせよ対策は必要ですが、**問題に気づくきっかけの現れ方が違います**。

## 参照

- [Python 版 第 6 章](../python/ch06.md)
- [Kotlin 版 第 6 章](../kotlin/ch06.md)
- [F# 版 第 6 章](../fsharp/ch06.md)
- [第 5 章 3 言語比較](ch05-perceptron.md)
- [統合記事 目次](index.md)
