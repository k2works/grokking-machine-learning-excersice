# 第 9 章 決定木 - 3 言語比較

原著第 9 章の決定木を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch09.md)・[Kotlin 版](../kotlin/ch09.md)・[F# 版](../fsharp/ch09.md) を参照してください。

## この章の主題は「木をどう型で表すか」

第 3〜8 章のモデルは、すべて **数値のリスト**（重み、確率表）でした。本章のモデルは初めて **再帰的な構造** になります。

```text
Tree = Leaf(ラベル) または Node(質問, Tree, Tree)
```

この「A または B で、B が A を含む」という形をどう表現するかが、3 言語の設計の分かれ目です。

本章も乱数を使わないため、**3 言語の結果は完全に一致します**（第 7・8 章と同じ）。

## 木の型定義

```python
@dataclass(frozen=True)
class Leaf:
    label: int

    def predict(self, point: Point) -> int:
        return self.label


@dataclass(frozen=True)
class Node:
    split: Split
    left: Tree
    right: Tree

    def predict(self, point: Point) -> int:
        branch = self.left if self.split.matches(point) else self.right
        return branch.predict(point)


Tree = Leaf | Node
```

```kotlin
sealed interface Tree {
    fun predict(point: Point): Int
}

data class Leaf(val label: Int) : Tree {
    override fun predict(point: Point): Int = label
}

data class Node(val split: Split, val left: Tree, val right: Tree) : Tree {
    override fun predict(point: Point): Int =
        if (split.matches(point)) left.predict(point) else right.predict(point)
}
```

```fsharp
type Tree =
    | Leaf of label: int
    | Node of split: Split * left: Tree * right: Tree

let rec predict (tree: Tree) (point: Point) =
    match tree with
    | Leaf label -> label
    | Node(split, left, right) -> predict (if matches split point then left else right) point
```

| 言語 | 機構 | 行数（型定義） | ケースの網羅性 |
| :--- | :--- | ---: | :--- |
| Python | 2 クラス + 型エイリアスの和 | 約 15 | 検査されない |
| Kotlin | `sealed interface` + 2 つの `data class` | 約 12 | `when` で検査される |
| F# | 判別共用体 | **3** | `match` で検査される |

**F# は 3 行で終わります。** 「木は葉か内部ノードのどちらか」という定義がそのまま型宣言です。クラスもインターフェースも要りません。

Python と Kotlin は、`predict` をメソッドとして各クラスに持たせる **オブジェクト指向の解き方** になりました。分岐が多態で解決されるので、`predict` の実装自体は簡潔です。

## 木を走査する操作

`predict` は多態で書けますが、`depth`（深さ）や `leafCount`（葉の数）のような **構造を調べる操作** は事情が違います。

```python
def depth(tree: Tree) -> int:
    if isinstance(tree, Leaf):
        return 0
    return 1 + max(depth(tree.left), depth(tree.right))
```

```kotlin
fun depth(tree: Tree): Int = when (tree) {
    is Leaf -> 0
    is Node -> 1 + maxOf(depth(tree.left), depth(tree.right))
}
```

```fsharp
let rec depth (tree: Tree) =
    match tree with
    | Leaf _ -> 0
    | Node(_, left, right) -> 1 + max (depth left) (depth right)
```

| 言語 | 分岐 | 網羅性 | フィールドの取り出し |
| :--- | :--- | :--- | :--- |
| Python | `isinstance` | 検査されない | `tree.left` |
| Kotlin | `when` + スマートキャスト | **検査される**（`else` 不要） | `tree.left`（自動キャスト） |
| F# | `match` + パターン | **検査される** | **パターンで同時に束縛** |

Kotlin の `when` に `else` がないことに注目してください。`sealed interface` なので、コンパイラは 2 ケースで尽きると知っています。**将来 `Tree` に第 3 のケースを足したら、この `when` はコンパイルエラーになります。**

F# は分解も同時に行います。`Node(_, left, right)` の 1 行で、使わないフィールドを `_` で捨てつつ、必要な 2 つに名前が付きます。**「葉のラベルは深さの計算に関係ない」ことがコードに現れます。**

Python は `isinstance` の後に暗黙の前提（残りは `Node`）が残ります。型チェッカー（mypy 等）を使えば網羅性を検査できますが、実行時には守られません。

### 木の形をテストする

構造の検査で差がもっとも出ます。

```python
def test_tree_structure_is_inspectable():
    tree = build_tree(POINTS, LABELS, max_depth=1)
    assert isinstance(tree, Node)
    assert isinstance(tree.left, Leaf)
    assert isinstance(tree.right, Leaf)
```

```kotlin
@Test
fun `木の構造を検査できる`() {
    val tree = buildTree(points, labels, maxDepth = 1)
    assertTrue(tree is Node)
    assertTrue((tree as Node).left is Leaf)
    assertTrue(tree.right is Leaf)
}
```

```fsharp
[<Fact>]
let ``木の構造をパターンマッチで検査できる`` () =
    match buildTreeWith giniImpurity 1 1 points labels with
    | Node(_, Leaf _, Leaf _) -> ()
    | _ -> failwith "深さ 1 の木は葉 2 枚を持つはずだった"
```

**F# は `Node(_, Leaf _, Leaf _)` という 1 つのパターンが構造の主張そのもの** です。Python と Kotlin は 3 つのアサーションに分解する必要があります。入れ子パターンが書けるかどうかが、そのまま表現力の差になりました。

## 不純度を差し替え可能にする

3 言語とも「不純度の測り方」を引数で受け取れる形にしました。

```python
Impurity = Callable[[Sequence[int]], float]
```

```kotlin
typealias Impurity = (List<Int>) -> Double
val giniImpurity: Impurity = { labels -> ... }
```

```fsharp
type Impurity = int list -> float
let giniImpurity: Impurity = fun labels -> ...
```

3 言語とも関数型に名前を付けられます。Kotlin と F# では、**関数を `fun`/`let f x =` ではなく「関数型の値」として定義した** のが判断点です。`val giniImpurity: Impurity = { ... }` と書くことで、「これは差し替え可能な戦略である」という意図が型注釈に現れます。

結果として、ジニとエントロピーの比較テストが 3 言語とも 1 行で書けました。

```python
assert build_tree(POINTS, LABELS, impurity=gini_impurity) == build_tree(POINTS, LABELS, impurity=entropy)
```

**木全体の等価比較が 1 行で済むのは、3 言語とも値の等価性が自動生成されるから** です（`@dataclass`・`data class`・判別共用体）。手で書けば再帰関数が必要になります。

## 「良い分割がない」をどう返すか

```python
def best_split(...) -> tuple[Split, float] | None:
    best: tuple[Split, float] | None = None
    for split in candidate_splits(points):
        ...
        if best is None or gain > best[1]:
            best = (split, gain)
    if best is None or best[1] <= 0.0:
        return None
    return best
```

```kotlin
fun bestSplit(...): Pair<Split, Double>? =
    candidateSplits(points)
        .mapNotNull { split -> ... }
        .maxByOrNull { it.second }
        ?.takeIf { it.second > 0.0 }
```

```fsharp
let bestSplit (impurity: Impurity) (points: Point list) (labels: int list) =
    candidateSplits points
    |> List.choose (fun split -> ...)
    |> function
        | [] -> None
        | candidates ->
            let best = candidates |> List.maxBy snd
            if snd best > 0.0 then Some best else None
```

| 言語 | 不在の表現 | 書き方 |
| :--- | :--- | :--- |
| Python | `X \| None` | ループ + 手動の最大値追跡 |
| Kotlin | `X?` | `mapNotNull` → `maxByOrNull` → `takeIf` |
| F# | `X option` | `List.choose` → `function` で場合分け |

**Kotlin がもっとも宣言的になりました。** 3 つの標準関数を繋ぐだけで「候補を絞り、最大を取り、条件を満たさなければ捨てる」が表現できます。null 安全な `?.` によって、途中で `null` になっても後続がスキップされます。

F# の `|> function | [] -> ...` は、パイプラインの途中に場合分けを差し込む書き方です。`List.maxBy` が空リストで例外を投げるため、この分岐が必要でした。**「空の可能性を型で扱わない標準関数」は F# にも存在します。**

Python は素朴なループになりました。`max(..., key=..., default=None)` を使えば短くできますが、「片側が空になる分割を除外する」条件も同時に扱う必要があり、素直なループのほうが読みやすいと判断しました。

## 候補の列挙

「各特徴量について、値の中点を集める」という二重ループです。

```python
    for feature in range(len(points[0])):
        values = sorted({point[feature] for point in points})
        for low, high in pairwise(values):
            splits.append(Split(feature=feature, threshold=(low + high) / 2))
```

```kotlin
    points.first().indices.flatMap { feature ->
        points.map { it[feature] }.distinct().sorted()
            .zipWithNext { low, high -> Split(feature, (low + high) / 2) }
    }
```

```fsharp
    [ for feature in 0 .. List.length (List.head points) - 1 do
          let values = points |> List.map (fun point -> point[feature]) |> List.distinct |> List.sort

          for low, high in List.pairwise values do
              { Feature = feature; Threshold = (low + high) / 2.0 } ]
```

Kotlin は `flatMap` + `zipWithNext` で式 1 つ。F# はリスト内包表記の中に `let` と二重 `for` を書けるため、**手順の入れ子がそのまま入れ子** になります。Python は素朴な二重ループです。

3 言語とも第 7 章と同じ「隣接ペア」の機能（`pairwise` / `zipWithNext` / `List.pairwise`）を使っています。

## 実験結果

3 言語とも同じ木に到達しました。

```text
Node(split=Split(feature=1, threshold=20.0), left=Leaf(label=1), right=Leaf(label=0))
```

| 項目 | 値 |
| :--- | ---: |
| 深さ | 1 |
| 葉の数 | 2 |
| 正解率 | 1.0 |

### 各候補の情報利得（根の不純度 0.4688）

| 特徴量 | 閾値 | 情報利得 |
| :--- | ---: | ---: |
| 性別 | 0.5 | 0.0312 |
| 年齢 | 13.0 | 0.1116 |
| 年齢 | 14.5 | 0.2604 |
| **年齢** | **20.0** | **0.4688** |
| 年齢 | 28.5 | 0.2812 |
| 年齢 | 33.5 | 0.1688 |
| 年齢 | 37.5 | 0.0938 |
| 年齢 | 47.5 | 0.0402 |

「年齢 < 20」の利得が根の不純度と一致します。**この 1 回の質問で不純度が完全に 0 になる** ということです。性別の利得は 0.0312 しかありません。

**決定木は自力で「年齢が効き、性別は関係ない」を見つけました。** 第 3〜6 章では人間が特徴量を選んでいましたが、木は特徴量の選択自体を学習に含みます。

## 説明可能性という利点

第 6 章のロジスティック回帰は、同じ種類の問題に対して「重み 2.02 と 1.59、バイアス -5.60」というモデルを返しました。本章の決定木は「20 歳未満なら推薦」を返します。

| モデル | 表現 | 人間への説明 |
| :--- | :--- | :--- |
| ロジスティック回帰 | 重みベクトル | 「特徴量の重み付き和が閾値を超えた」 |
| ナイーブベイズ | 単語ごとの確率 | 「lottery が 0.8、winning が 0.75 だから」 |
| **決定木** | **質問の連鎖** | **「20 歳未満だから」** |

3 言語とも、木の構造が `print` / `println` / `printfn` するだけで読めます（`@dataclass` の `__repr__`、`data class` の `toString`、判別共用体の既定の `ToString`）。**モデルを目視で検証できるのは決定木ならではです。**

## 過学習との関係

停止条件のうち 2 つ（最大深さ、最小サンプル数）は、**第 4 章の正則化と同じ役割** です。

| 章 | モデルの複雑さ | 抑える手段 |
| :--- | :--- | :--- |
| 第 4 章 | 多項式の次数 | L1 / L2 正則化、次数の制限 |
| 第 9 章 | 木の深さ | 最大深さ、最小サンプル数 |

木は放っておくと、訓練データの 1 点ごとに葉を作るまで育ちます。それは訓練誤差 0・テスト誤差最悪という第 4 章で見た過学習そのものです。

本章のデータでは深さ 1 で完全に分かれたため、この制限は働きませんでした。3 言語とも `max_depth = 1` を明示したときだけ動作を確認しています。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | 記述量が少なく、`Tree = Leaf \| Node` で和型を表現できる | `sealed interface` の網羅性検査、`mapNotNull`→`maxByOrNull`→`takeIf` の宣言的な連鎖 | **判別共用体が木そのもの。3 行で型定義、入れ子パターンで構造をテスト** |
| 注意点 | `isinstance` の網羅性が実行時に守られない | 構造検査でキャストが必要（`(tree as Node)`） | `List.maxBy` が空リストで例外を投げる |

第 3〜8 章では、章ごとに 3 言語の得手不得手が入れ替わってきました。本章は **F# が明確に有利** です。決定木は代数的データ型そのものであり、判別共用体とパターンマッチはまさにそれを扱うために設計された機能だからです。

言語の機能とアルゴリズムの構造が噛み合うと、コードは短くなるだけでなく **定義そのものになります**。

## 参照

- [Python 版 第 9 章](../python/ch09.md)
- [Kotlin 版 第 9 章](../kotlin/ch09.md)
- [F# 版 第 9 章](../fsharp/ch09.md)
- [第 8 章 3 言語比較](ch08-naive-bayes.md)
- [統合記事 目次](index.md)
