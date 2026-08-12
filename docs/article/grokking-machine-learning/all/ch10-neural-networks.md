# 第 10 章 ニューラルネットワーク - 3 言語比較

原著第 10 章のニューラルネットワークを、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch10.md)・[Kotlin 版](../kotlin/ch10.md)・[F# 版](../fsharp/ch10.md) を参照してください。

## 本章で新しく現れる難しさ

第 3〜9 章のデータ構造は、せいぜい「数値のリスト」でした。本章は **入れ子のリスト（行列）** を、しかも **前後に往復しながら** 扱います。

| 章 | データ構造 | 走査の向き |
| :--- | :--- | :--- |
| 3〜6 章 | 重みのリスト | 前から後ろへ 1 回 |
| 8 章 | 単語 → 確率の写像 | 集合を 1 回 |
| 9 章 | 木（再帰構造） | 根から葉へ |
| **10 章** | **行列のリスト** | **前へ進み、後ろへ戻る** |

この「往復」が逆伝播であり、本章の実装の中心です。

## XOR が示すこと

| 手法 | 隠れニューロン数 | Python | Kotlin | F# |
| :--- | ---: | ---: | ---: | ---: |
| ロジスティック回帰（第 6 章） | なし | 0.75 | 0.75 | — |
| ニューラルネット | 1 | 0.75 | 0.75 | 0.75 |
| **ニューラルネット** | **4** | **1.00** | **1.00** | **1.00** |

**3 言語とも同じ結論に達しました。** 隠れ層を足しただけでは足りず、十分な幅が要ります。隠れ層 1 ニューロンのネットワークは「直線を 1 本引いてから変換する」だけなので、表現力はロジスティック回帰と変わりません。

Python 版と Kotlin 版のテストでは、**第 6 章の実装をそのまま呼んで** この比較を書いています。

```python
from grokking_ml.ch06_logistic_regression import accuracy as logistic_accuracy
from grokking_ml.ch06_logistic_regression import logistic_regression
```

```kotlin
import ch06.logisticRegression
import ch06.accuracy as logisticAccuracy
```

両言語ともインポート別名で名前の衝突を避けています。**章ごとにモジュール／パッケージを分けているからこそ、前章の手法との比較が実行可能なテストになります。**

F# 版では同じテストを書いていません。F# のモジュールは開いた瞬間に名前が入るため、`open` の順序で挙動が変わる書き方になりがちです。**モジュールシステムの性質に合わせて、テストの構成も変えました。**

## 行列をどう表すか

```python
Vector = list[float]
Matrix = list[list[float]]
```

```kotlin
data class Layer(val weights: List<List<Double>>, val biases: List<Double>)
```

```fsharp
type Layer =
    { Weights: float list list
      Biases: float list }
```

**3 言語とも、行列の形（行数・列数）を型で表せません。** 2×3 の行列に長さ 4 のベクトルを掛けようとしても、コンパイルは通り実行時に落ちます。

これは第 7 章で見た「同じ型のフィールドが 4 つ並ぶ混同行列」と似た状況ですが、より深刻です。**形の不一致は、静かに間違った答えを出すこともあります**（`zip` は短いほうに合わせるため、例外すら出ません）。

そこで 3 言語とも同じ対策を取りました。

1. `input_size` / `output_size` を **派生プロパティ** として持たせる
2. 形をテストで固定する
3. 逆伝播が形を保つことをテストする

```python
def test_backpropagate_keeps_the_shape():
    model = initial_network([2, 4, 1])
    updated = backpropagate(model, (1.0, 0.0), 1, learning_rate=0.5)
    assert len(updated.layers) == len(model.layers)
    for original, layer in zip(model.layers, updated.layers):
        assert layer.input_size == original.input_size
        assert layer.output_size == original.output_size
```

**型が守ってくれない部分は、テストで守ります。** この判断は第 7 章（混同行列の 4 フィールド）と同じです。

## 内積の書き方

```python
            sigmoid(bias + sum(w * x for w, x in zip(row, inputs)))
```

```kotlin
        weights.zip(biases) { row, bias ->
            sigmoid(bias + row.zip(inputs) { w, x -> w * x }.sum())
        }
```

```fsharp
    List.map2
        (fun row bias -> sigmoid (bias + (List.map2 (*) row inputs |> List.sum)))
        layer.Weights
        layer.Biases
```

F# の `List.map2 (*)` が際立ちます。**演算子をそのまま関数値として渡せる** ため、掛け算のラムダが要りません。第 5 章のパーセプトロンで見た書き方が、行列演算でも同じように効いています。

Kotlin は変換関数付きの `zip` を入れ子にすることで、中間リストを作らずに 2 段の走査を書けました。Python はジェネレータ式で同じことをしています。

**3 言語とも行列演算ライブラリを使わずに 3 行以内で書けました。** 教材としては、この「中身が見える」ことが重要です。

## 誤差を「先頭に積む」

逆伝播は出力層から入力側へ計算するため、結果を **逆順に組み立てる** ことになります。

```python
    deltas: list[Vector] = [[activations[-1][0] - label]]
    for index in range(len(model.layers) - 1, 0, -1):
        ...
        deltas.insert(0, [...])
```

```kotlin
    val deltas = ArrayDeque(listOf(listOf(activations.last().first() - label)))
    for (index in model.layers.size - 1 downTo 1) {
        ...
        deltas.addFirst(List(layer.inputSize) { i -> ... })
    }
```

```fsharp
    let deltas =
        [ layerCount - 1 .. -1 .. 1 ]
        |> List.fold
            (fun accumulated index -> ... :: accumulated)
            [ outputDelta ]
```

| 言語 | 先頭追加 | 計算量 | 選んだ理由 |
| :--- | :--- | :--- | :--- |
| Python | `list.insert(0, x)` | O(n) | 層数は小さく、素直さを優先 |
| Kotlin | `ArrayDeque.addFirst` | **O(1)** | 要求に合う容器を選んだ |
| F# | `x :: accumulated` | **O(1)** | 不変リストの標準操作 |

**F# は言語の自然な操作がそのまま最適でした。** 不変リストは先頭追加が O(1) であり、第 3 章で誤差の履歴を積んだときと同じ書き方です。

Kotlin は「先頭に積みたい」という要求から `ArrayDeque` を選びました。`List` のままでは O(n) の挿入になります。**データ構造の選択が、アルゴリズムの向きから決まった例です。**

Python は `list.insert(0, ...)` にしました。`deque` を使えば O(1) にできますが、層の数は多くても数十なので素直さを優先しています。**計算量の最適化が意味を持つ規模かどうかで判断が変わります。**

## 不変性という保険

3 言語とも、逆伝播は **新しいネットワークを返します**。元のモデルは変更しません。

```python
def test_network_is_immutable_across_training():
    original = initial_network([2, 4, 1], seed=3)
    snapshot = NeuralNetwork(layers=list(original.layers))
    backpropagate(original, (1.0, 0.0), 1, learning_rate=0.5)
    assert original == snapshot
```

```kotlin
@Test
fun `学習はモデルを破壊的に変更しない`() {
    val original = initialNetwork(listOf(2, 4, 1), seed = 3)
    val snapshot = NeuralNetwork(original.layers.toList())
    backpropagate(original, listOf(1.0, 0.0), 1, learningRate = 0.5)
    assertEquals(snapshot, original)
}
```

**F# 版にはこのテストがありません。** レコードとリストが言語レベルで不変なので、破壊的変更が起こりえないためです。**言語が保証する性質はテストしない** という判断です。

Python は `@dataclass(frozen=True)` でも中の `list` は可変なので、規律で守る必要があります。Kotlin は `List`（読み取り専用）と `data class` の組み合わせで、通常のコードでは不変です。

この不変性は、多次元配列を扱うときにとくに効きます。**行列を破壊的に更新する実装では、順伝播で記録した値がいつの間にか書き換わる、という追いにくいバグが起きます。** Kotlin 版で `Array` ではなく `List` を選んだのはこのためです。

## 出力層の誤差が 1 行になる理由

3 言語とも、逆伝播の最初の行はこれだけです。

```python
    deltas: list[Vector] = [[activations[-1][0] - label]]
```

`予測確率 - 正解ラベル`。それだけです。

これは偶然ではありません。**対数損失とシグモイドを組み合わせると、微分が約分されて `p - label` に簡約されます。**

| 要素 | 微分 |
| :--- | :--- |
| 対数損失を p で微分 | `-(label/p - (1-label)/(1-p))` |
| シグモイドを入力で微分 | `p(1-p)` |
| **連鎖律で掛け合わせる** | **`p - label`** |

第 6 章では「シグモイドを使う」「対数損失を使う」とだけ書きましたが、**この組み合わせを選んだ本当の理由が本章で現れました。** 別の損失関数（たとえば二乗誤差）を使うと、この簡約が起きず、勾配消失も悪化します。

そして `p - label` は **第 6 章のロジスティックトリックの式そのもの** です（符号は「誤差を足す」か「勾配を引く」かの違い）。第 8 章では最終式がシグモイドと同型になり、本章では更新式が第 6 章と同型になりました。**章が進むほど、同じ形が繰り返し現れます。**

## 勾配消失

```python
def test_sigmoid_derivative_vanishes_at_the_extremes():
    # 出力が 0 や 1 に近いと勾配がほぼ消える（勾配消失）
    assert sigmoid_derivative(0.999) < 0.002
    assert sigmoid_derivative(0.001) < 0.002
```

3 言語とも同じテストを書きました。シグモイドの微分は最大 0.25、両端では 0 に近づきます。

層を深く積むと、この 0.25 以下の値が層の数だけ掛け合わされます。10 層なら `0.25¹⁰ ≒ 1e-6` です。**入力側の層はほとんど学習しません。**

本章のネットワークは 2 層なので問題になりませんが、**「なぜ現代のネットワークが ReLU を使うのか」という問いの答えがこのテストにあります。**

## 実験結果

| 言語 | 初期損失 | 最終損失 | 正解率 |
| :--- | ---: | ---: | ---: |
| Python | 0.7518 | 0.0009 | 1.0 |
| Kotlin | 0.9238 | 0.0008 | 1.0 |
| F# | 0.7560 | 0.0008 | 1.0 |

**第 7〜9 章では 3 言語の数値が完全一致していましたが、本章では再び分かれました。** 重みを乱数で初期化するためです。

初期損失の差（0.75 対 0.92）は、初期の重みがどれだけ「たまたま良かったか」の差です。しかし **最終損失は 3 言語とも 0.001 未満に収束しました。** 出発点が違っても同じ谷底に着いています。

各点の予測確率も 3 言語で揃っています。

| 入力 | Python | Kotlin | F# | 正解 |
| :--- | ---: | ---: | ---: | ---: |
| (0, 0) | 0.0007 | 0.0002 | 0.0009 | 0 |
| (0, 1) | 0.9992 | 0.9995 | 0.9993 | 1 |
| (1, 0) | 0.9992 | 0.9991 | 0.9993 | 1 |
| (1, 1) | 0.0015 | 0.0016 | 0.0011 | 0 |

第 5 章では「正しい境界線が無数にある」ため 3 言語がバラバラの解に到達しました。本章では **どの言語も同じ振る舞いのモデル** に到達しています（内部の重みは違いますが、出力はほぼ同じです）。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | 内包表記で行列操作が短い | 変換付き `zip` の入れ子、`ArrayDeque` の選択 | `List.map2 (*)` で内積が 1 行、`::` が最適、不変性が言語保証 |
| 注意点 | `frozen=True` でも中のリストは可変 | `Array` を選ぶと不変性が壊れる | 降順範囲式 `[n .. -1 .. 1]` に慣れが要る |

第 9 章では F# の判別共用体がアルゴリズムの構造と噛み合いました。本章で噛み合ったのは **「逆順に積む」という操作** です。不変リストの `::` は、逆伝播の向きにそのまま合致しました。

一方で行列の形の安全性は、3 言語とも型システムの外にあります。**型で守れない部分をどうテストで守るかが、本章の共通の課題でした。**

## 参照

- [Python 版 第 10 章](../python/ch10.md)
- [Kotlin 版 第 10 章](../kotlin/ch10.md)
- [F# 版 第 10 章](../fsharp/ch10.md)
- [第 9 章 3 言語比較](ch09-decision-trees.md)
- [統合記事 目次](index.md)
