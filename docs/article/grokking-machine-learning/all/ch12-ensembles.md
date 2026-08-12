# 第 12 章 アンサンブル学習 - 3 言語比較

原著第 12 章のアンサンブル学習を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch12.md)・[Kotlin 版](../kotlin/ch12.md)・[F# 版](../fsharp/ch12.md) を参照してください。

## 本章は「前章の再利用」の章

第 10・11 章では前章の実装を **比較対象** として呼びました。本章では **構成部品** として使います。

```python
from grokking_ml.ch09_decision_trees import (
    Impurity, Leaf, Point, Tree, build_tree, gini_impurity, majority_label,
)
```

```kotlin
import ch09.Impurity
import ch09.Leaf
import ch09.Point
import ch09.Tree
import ch09.buildTree
import ch09.giniImpurity
import ch09.majorityLabel
```

```fsharp
open GrokkingMl.Ch09DecisionTrees
```

**第 9 章の決定木を 1 行も書き換えずに再利用できました。** 第 9 章で `Impurity` を差し替え可能な関数型として設計しておいたことが効いています。アンサンブルの関数もすべて `impurity` を引数で受け取り、そのまま木の構築へ渡すだけです。

**設計の良し悪しは、後の章で再利用しようとしたときに分かります。**

### 名前の衝突をどう避けるか

| 言語 | 機構 | 本章での対処 |
| :--- | :--- | :--- |
| Python | モジュール | 必要な名前だけ `from ... import` |
| Kotlin | パッケージ | `ch09.buildTree` と修飾可能、`accuracy` は多重定義 |
| F# | モジュール（`open` で平坦化） | **`forestPredict` のように接頭辞を付ける** |

F# だけが名前の工夫を要しました。`open` すると名前がそのまま入るため、第 9 章の `predict`（木の予測）と本章の予測関数が衝突します。**モジュールシステムの性質が、関数の命名に影響しました。**

Kotlin は `Forest` 用と `AdaBoost` 用に `accuracy` を **多重定義** できました。Python では `Forest | AdaBoost` という和型で 1 つの関数にしています。

## バギングが効かなかった

3 言語とも同じ結果になりました。

| モデル | Python | Kotlin | F# |
| :--- | ---: | ---: | ---: |
| 切り株 1 本 | 0.75 | 0.75 | 0.75 |
| バギング（切り株 10 本） | **0.75** | **0.75** | **0.75** |
| AdaBoost（切り株 10 本） | **1.00** | **1.00** | **1.00** |

**バギングは 1 本の切り株を 1 ミリも上回りませんでした。**

復元抽出でデータを揺らしても、**どの標本でも「x = 2.5 で切る」のが最良** なので、10 本ともほぼ同じ木になります。同じ意見を 10 回聞いても結論は変わりません。

```python
def test_forest_of_stumps_does_not_beat_a_single_stump_here():
    stump = build_tree(POINTS, LABELS, max_depth=1)
    forest = train_forest(POINTS, LABELS, tree_count=10, max_depth=1)
    # バギングは似た木ばかり作るので、この問題では改善しない
    assert accuracy(forest, POINTS, LABELS) <= tree_accuracy(stump, POINTS, LABELS) + 1e-9
```

**「効かないこと」をテストで固定したのは本シリーズで初めてです。** 手法を並べて「どれも良い」と書くのは簡単ですが、実装して測ると得手不得手が出ます。

バギングが効くのは、**個々の学習器が「違う間違い方」をするとき** です。深い木のように分散が大きい学習器なら平均化でばらつきが減りますが、切り株のように偏りが大きい学習器では何本集めても偏りは消えません。

## 復元抽出と分割の違い

第 4 章の `train_test_split` と本章の `bootstrap_sample` は、どちらも「データから標本を作る」操作ですが、目的が正反対です。

| | 第 4 章（分割） | 第 12 章（ブートストラップ） |
| :--- | :--- | :--- |
| 重複 | **なし** | **あり** |
| 大きさ | 元より小さい（訓練とテストに分ける） | **元と同じ** |
| 目的 | 未知データでの評価 | 学習器の多様性 |
| 実装 | シャッフルして切る | ランダムなインデックス列 |

```python
    indices = list(range(len(features)))
    rng.shuffle(indices)          # 第 4 章: 並べ替えて分ける
```

```python
    indices = [rng.randrange(len(points)) for _ in range(len(points))]   # 第 12 章: 引き直す
```

3 言語とも **インデックス列を作ってから両方のリストを引く** という同じ書き方です。特徴量とラベルを別々に抽出すると対応が壊れる、という第 4 章の教訓がそのまま活きています。

## 3 本のリストを同時に走査する

AdaBoost では「点・ラベル・重み」の 3 本を同時に扱います。ここで言語差が出ました。

```python
    wrong = sum(
        weight
        for point, label, weight in zip(points, labels, weights)
        if tree.predict(point) != label
    )
```

```kotlin
    val wrong = points.indices
        .filter { tree.predict(points[it]) != labels[it] }
        .sumOf { weights[it] }
```

```fsharp
    List.zip3 points labels weights
    |> List.sumBy (fun (point, label, weight) -> if predict tree point <> label then weight else 0.0)
```

| 言語 | 3 本の走査 | 書き味 |
| :--- | :--- | :--- |
| Python | `zip(a, b, c)`（任意個） | 自然 |
| Kotlin | `zip` は **2 本まで** → インデックス経由 | 一手間 |
| F# | `List.zip3`（3 本まで） | 自然 |

**Kotlin の `zip` が 2 本までという制約が、ここで初めて問題になりました。** `zip` を重ねると `Pair<Pair<A, B>, C>` という入れ子ができ、分解が煩雑になります。インデックス経由のほうが素直と判断しました。

F# の `List.zip3` は 3 本まで対応します（4 本以上は同じ問題に当たります）。Python の `zip` は任意個を受け取れるので制約がありません。

## 発言権の式

```python
    return 0.5 * math.log((1.0 - clamped) / clamped)
```

3 言語とも同じ式です。この 1 行に 3 つの性質が詰まっており、それぞれテストで固定しました。

| 誤り率 | 発言権 | 意味 |
| ---: | ---: | :--- |
| 0.05 | 1.47 | よく当たるので強く聞く |
| 0.5 | **0.0** | 当てずっぽうなので無視 |
| 0.7 | **-0.42** | 逆張りすれば役立つ |

**誤り率 0.5 で発言権がちょうど 0 になります。** 第 7 章で AUC が 0.5 のときランダムを意味したのと同じ構図です。「情報がない」状態が数式上できれいに 0 になる指標は、それだけで信頼できます。

クランプ（`epsilon`）は 3 言語とも同じ理由で必要でした。誤り率 0 のとき `log(1/0)` が無限大になるためです。

| 章 | クランプが必要だった場所 |
| :--- | :--- |
| 第 6 章 | 対数損失（`log(0)` を避ける） |
| 第 8 章 | ナイーブベイズの対数確率 |
| **第 12 章** | **発言権の対数** |

**対数を取るところには必ずクランプが要る**、というパターンが 3 度繰り返されました。F# はいずれも `|> max lo |> min hi` のパイプライン、Python と Kotlin は `min(max(x, lo), hi)` の入れ子で書いています。

## 状態の畳み込み

AdaBoost の学習ループは「学習器の列」と「重み」を同時に更新します。

```python
    weights = [1.0 / len(points)] * len(points)
    learners: list[WeightedTree] = []
    for _ in range(rounds):
        ...
        learners.append(WeightedTree(tree=tree, weight=alpha))
        weights = [...]
```

```kotlin
    var weights = List(points.size) { 1.0 / points.size }
    val learners = mutableListOf<WeightedTree>()
    repeat(rounds) {
        ...
        learners += WeightedTree(tree, alpha)
        weights = updated.map { it / total }
    }
```

```fsharp
    let step (learners, weights) _ =
        ...
        (learners @ [ { Tree = tree; Weight = alpha } ], updated |> List.map (fun w -> w / total))

    let learners, _ = List.fold step ([], initialWeights) [ 1..rounds ]
```

| 言語 | 手法 | 可変変数 |
| :--- | :--- | :--- |
| Python | ループ + 再代入 + `append` | 2 個 |
| Kotlin | `repeat` + `var` + `MutableList` | 2 個 |
| F# | **タプルを `fold`** | なし |

**F# の `fold` は本シリーズで 4 度目です**（第 3 章の学習ループ、第 8 章の対数確率、第 10 章の順伝播、本章）。「複数の状態を同時に更新する」という要求に、F# は一貫して同じ道具で応えます。

Kotlin の `var weights` は再代入していますが、代入するのは常に **新しい不変リスト** です。要素を書き換えているわけではありません。

## AdaBoost が作った役割分担

3 言語とも、まったく同じ学習器の列を作りました（**AdaBoost に乱数がない** ためです）。

| ラウンド | 発言権 | 分割 |
| ---: | ---: | :--- |
| 1 | 0.5493 | x < 2.5 なら +1 |
| 2 | 0.8047 | x < 6.5 なら -1 |
| 3 | 0.6931 | （全部 +1 の葉） |
| 4 | 0.7332 | x < 2.5 なら +1 |

**1 本目は左端、2 本目は右端に注目しました。** 1 本目が右端の `x = 7, 8` を外したので、その重みが上がり、2 本目が拾いに行ったのです。

重み付き投票の結果、`x = 7, 8` では「1 本目は -1、2 本目は +1」となり、発言権の大きい 2 本目（0.8047）が勝ちます。**誰も指示していないのに役割分担が生まれました。** これが「弱い学習器を集めて強くする」ということです。

## ラベルの表現が効いた

第 11 章で SVM のためにラベルを **+1 / -1** に変えました。本章でもその表現が効きます。

```python
        weights = [
            weight * math.exp(-alpha * label * tree.predict(point))
            for point, label, weight in zip(points, labels, weights)
        ]
```

`label * tree.predict(point)` は、当たれば +1、外れれば -1 です。**0 / 1 のラベルではこの積が意味を持ちません。**

| 章 | ラベルの表現 | 理由 |
| :--- | :--- | :--- |
| 5・6・8・9 章 | 0 / 1 | 確率や多数決に自然 |
| 11 章 | **+1 / -1** | `label × score` が「正しい側への距離」になる |
| 12 章 | **+1 / -1** | `label × predict` が「当たり外れ」になる |

**表現の選択が、式の簡潔さを決めます。** 第 5 章で「0/1 で十分」と判断した箇所が、第 11 章以降では別の表現に変わりました。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | `zip` が任意個、和型で 1 つの `accuracy` | 型ごとの `accuracy` 多重定義で意図が明確 | `List.zip3`、`fold` によるタプル畳み込み |
| 注意点 | 型エイリアスは実行時に守られない | **`zip` が 2 本まで** | `open` で名前が衝突するため接頭辞が要る |

本章でもっとも重要な発見は、言語差ではなく **手法の得手不得手** でした。バギングと AdaBoost はどちらも「弱学習器を集める」手法ですが、この問題ではバギングがまったく効かず、AdaBoost が完全に解きました。

**「実装して測る」ことでしか分からないことがあります。** 3 言語すべてで同じ結論に達したことが、それが実装の癖ではなく手法の性質であることの裏付けになりました。

## 参照

- [Python 版 第 12 章](../python/ch12.md)
- [Kotlin 版 第 12 章](../kotlin/ch12.md)
- [F# 版 第 12 章](../fsharp/ch12.md)
- [第 11 章 3 言語比較](ch11-svm.md)
- [統合記事 目次](index.md)
