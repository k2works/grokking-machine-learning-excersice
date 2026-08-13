# 第 5 章 パーセプトロン - 3 言語比較

原著第 5 章のパーセプトロンを、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch05.md)・[Kotlin 版](../kotlin/ch05.md)・[F# 版](../fsharp/ch05.md) を参照してください。

## アルゴリズムの共通部分

1. 線形分類器 = 重みのリスト + バイアス（不変）
2. `score`（連続値）と `predict`（0/1）の分離
3. 誤分類した点だけを動かすパーセプトロントリック
4. パーセプトロン誤差と正解率
5. 第 3 章と同じ骨格の学習ループ

**第 4 章までとの最大の違いは、モデルの構造ではなく「いつ更新するか」です。** 回帰は全点で更新し、分類は誤分類した点だけで更新します。

## 特徴量ベクトルに名前を付ける

本章から入力が多次元になります。3 言語とも型に名前を付けました。

```python
Point = Sequence[float]
```

```kotlin
typealias Point = List<Double>
```

```fsharp
type Point = float list
```

| 言語 | 機構 | 新しい型か | 効果 |
| :--- | :--- | :--- | :--- |
| Python | 型エイリアス | いいえ | 型ヒントの可読性 |
| Kotlin | `typealias` | いいえ | シグネチャの可読性 |
| F# | 型略称 | いいえ | シグネチャの可読性 |

3 言語とも **別名であって新しい型ではありません。** 取り違えを型で防ぎたいなら、Kotlin なら `value class`、F# なら単一ケースの判別共用体、Python なら `NewType` を使うことになります。ここでは特徴量ベクトルと取り違える相手がいないため、別名で十分と判断しました。

## 内積の書き方

```python
    def score(self, point: Point) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, point))
```

```kotlin
    fun score(point: Point): Double =
        bias + weights.zip(point) { w, x -> w * x }.sum()
```

```fsharp
let score (model: Perceptron) (point: Point) =
    List.map2 (*) model.Weights point
    |> List.sum
    |> (+) model.Bias
```

F# の `List.map2 (*)` が際立ちます。**演算子をそのまま関数値として渡せる** ため、掛け算のラムダを書く必要がありません。Python と Kotlin はラムダ（あるいはジェネレータ式）で `w * x` を明示します。

`|> (+) model.Bias` も同様で、加算演算子の部分適用です。読み慣れるまでは記号的ですが、「合計してバイアスを足す」というパイプラインが素直に並びます。

## 「動かさない」をどう表すか

トリックの核心は、正しく分類できた点でモデルを変えないことです。

```python
    error = label - model.predict(point)
    if error == 0:
        return model
```

```kotlin
    val error = label - model.predict(point)
    if (error == 0) return model
```

```fsharp
    let error = float (label - predict model point)

    if error = 0.0 then
        model
    else
        { ... }
```

Python と Kotlin は **早期リターン**、F# は **`if` 式の一方の腕** として表現します。F# には早期リターンがありませんが、`if` が式なので「2 つの結果のどちらかを返す」と読めます。どちらの表現でも、規則がコードの目立つ位置に現れます。

この「動かない」性質は 3 言語とも等価比較でテストしています。

```python
assert perceptron_trick(model, (1.0, 2.0), label=1) == model
```

```kotlin
assertEquals(model, perceptronTrick(model, listOf(1.0, 2.0), label = 1))
```

```fsharp
Assert.Equal(model, perceptronTrick 0.01 model [ 1.0; 2.0 ] 1)
```

3 言語とも **値としての等価性** が自動で手に入っているからこの 1 行で済みます（`@dataclass`・`data class`・レコード型）。もし通常のクラスなら参照比較になり、同じインスタンスを返す実装ではテストが意図せず通ってしまいます。

## 型変換の扱い

`error` は整数、学習率は浮動小数点数です。ここで 3 言語の姿勢が分かれます。

| 言語 | 記述 | 変換 |
| :--- | :--- | :--- |
| Python | `learning_rate * error * x` | 暗黙（動的） |
| Kotlin | `learningRate * error * x` | 暗黙（演算子オーバーロード） |
| F# | `float (label - predict model point)` | **明示が必須** |

F# だけが変換を書かせます。第 4 章の `float (List.length labels)` と同じ設計思想で、手数は増えますが型の混在が事故になりません。

同じ話は集計にも現れます。

```kotlin
points.zip(labels).count { ... }.toDouble() / points.size
```

```fsharp
List.map2 (fun point label -> if predict model point = label then 1.0 else 0.0) points labels
|> List.average
```

Kotlin の `count` は `Int` を返すため `toDouble()` が要ります。F# は真偽値を 1.0 / 0.0 に写して `List.average` を使うことで、**「正解率 = 正解フラグの平均」という定義そのもの** をコードにしています。標準ライブラリに `average` があるかどうかが書き味を変えた例です。

## 学習ループ

第 3 章とまったく同じ骨格です。

| 言語 | ループ | 状態 |
| :--- | :--- | :--- |
| Python | `for _ in range(epochs)` | 再代入 + `list.append` |
| Kotlin | `repeat(epochs) { }` | `var` + `MutableList` |
| F# | `List.fold` | タプルを畳み込み、`::` で積んで `List.rev` |

第 3 章の線形回帰と本章のパーセプトロンで違うのは、呼び出すトリックだけです。**同じ骨格に別の学習則を差し込むと別のアルゴリズムになります。** この構造は第 6 章のロジスティック回帰でも変わりません。

## 実験結果

学習率 0.01、1000 エポック、シード 0 の結果です。

| 言語 | aack の重み | beep の重み | バイアス | 境界線 | 正解率 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| Python | 0.02 | 0.01 | -0.04 | `2·aack + beep = 4` | 1.0 |
| Kotlin | 0.01 | 0.01 | -0.03 | `aack + beep = 3` | 1.0 |
| F# | 0.04 | 0.02 | -0.08 | `2·aack + beep = 4` | 1.0 |

3 言語で **異なる境界線に到達しました。** これは実装の差ではなく、線形分離可能なデータには **正しい境界線が無数に存在する** ためです。Python 版と F# 版はたまたま同じ直線（重みは 2 倍違うが同じ線）に、Kotlin 版は別の直線に落ち着きました。

分類では重みの絶対値に意味がなく、**比率と符号だけが境界線を決めます**。すべての重みとバイアスを定数倍しても同じ分類器です。第 3 章の回帰では重みの大きさが予測値そのものを決めていたのと対照的で、ここが回帰と分類の本質的な違いのひとつです。

したがって 3 言語で検証すべきは **正解率が 1.0 になること** です。重みの値を比較してはいけません。

## パーセプトロン誤差の落とし穴

3 言語のテストは、共通して次の事実を固定しています。

```python
    # 初期モデル（重みもバイアスもすべて 0）はすべての点が境界線上にあるため誤差 0
    assert errors[0] == pytest.approx(0.0)
    # 学習の途中では誤差が生じ、最終的に 0 へ戻る
    assert max(errors) > 0.0
    assert errors[-1] == pytest.approx(0.0)
```

**初期状態（全パラメータ 0）の誤差が 0 になります。** すべての点のスコアが 0 なので、誤分類していても `|スコア| = 0` だからです。正解率は 0.5 しかないのに、誤差関数は最良と報告します。

| エポック | 平均誤差 | 正解率 |
| :--- | ---: | ---: |
| 最初 | 0.0 | 0.5 |
| 途中（最大） | 0.0275〜0.0325（言語による） | — |
| 最後 | 0.0 | 1.0 |

第 3 章では 3 言語とも `errors[-1] < errors[0]`（誤差が減った）を検証しました。本章で同じ検証を書くと `0.0 < 0.0` となり、3 言語すべてで落ちます。

**これは実装のバグではなく、誤差関数の性質です。** テストを通すために閾値を緩めるのではなく、実際の挙動（途中で誤差が生じ、最後に 0 へ戻る）を検証する形に書き換えました。前章までの型どおりのアサーションをそのまま持ち込まなかったことで、誤差関数の限界が浮かび上がりました。

学習の進み具合を見たいなら、誤差ではなく正解率を見ます。3 言語とも `accuracy` を別関数として用意したのはこのためです。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | 記述量が最小、ジェネレータ式で内積が 1 行 | 早期リターンと `data class` の等価性 | 演算子を関数として渡せる、`List.average` が定義を表す |
| 注意点 | 型エイリアスは実行時に何も守らない | `count` と `sumOf` で戻り値型が違う | 数値変換を毎回明示する |

第 3 章は「状態の持ち方」、第 4 章は「種類を表す型の安全性」が差でした。本章で見えたのは **標準ライブラリの語彙** です。`average` があるか、演算子を関数として渡せるか、といった差が、アルゴリズムの定義をそのまま書けるかどうかを分けました。

## 参照

- [Python 版 第 5 章](../python/ch05.md)
- [Kotlin 版 第 5 章](../kotlin/ch05.md)
- [F# 版 第 5 章](../fsharp/ch05.md)
- [第 4 章 3 言語比較](ch04-regularization.md)
- [統合記事 目次](index.md)
