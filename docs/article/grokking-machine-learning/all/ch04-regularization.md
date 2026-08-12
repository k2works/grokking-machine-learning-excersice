# 第 4 章 過学習・未学習と正則化 - 3 言語比較

原著第 4 章の多項式回帰と正則化を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch04.md)・[Kotlin 版](../kotlin/ch04.md)・[F# 版](../fsharp/ch04.md) を参照してください。

## アルゴリズムの共通部分

1. 多項式モデル = 重みのリスト + バイアス（不変）
2. 訓練データとテストデータの分割
3. 正則化の種類 = 3 通りの列挙（なし / L1 / L2）
4. 二乗トリック + 正則化項の 1 点更新
5. 次数を変えて訓練誤差とテスト誤差を比較

第 3 章との違いは、**重みが 1 個からリストになった** ことと、**種類を表す型（列挙）が登場した** ことです。この 2 点で 3 言語の表現力の差がはっきり出ます。

## 重みのリストをどう不変に保つか

| 言語 | 表現 | 不変性の担保 |
| :--- | :--- | :--- |
| Python | `weights: tuple[float, ...]` | `tuple` にしないと `frozen=True` でも中身を変更できる |
| Kotlin | `val weights: List<Double>` | `List` は読み取り専用インターフェース |
| F# | `Weights: float list` | リストは既定で不変 |

```python
@dataclass(frozen=True)
class PolynomialModel:
    weights: tuple[float, ...]
    bias: float
```

```kotlin
data class PolynomialModel(val weights: List<Double>, val bias: Double)
```

```fsharp
type PolynomialModel =
    { Weights: float list
      Bias: float }
```

Python だけが注意を要します。`@dataclass(frozen=True)` が凍結するのは **属性の再代入だけ** で、`list` を持たせれば `model.weights[0] = 999` は通ってしまいます。`tuple` を選ぶ判断が必要です。

Kotlin の `List` は読み取り専用インターフェースであり、`MutableList` へのキャストという抜け道は残りますが、通常のコードでは不変として扱えます。F# は何も考えなくても不変です。

### 次数を派生させる

3 言語とも「重みの個数 = 次数」を **派生値** として持たせ、食い違いが起きない設計にしています。

```python
    @property
    def degree(self) -> int:
        return len(self.weights)
```

```kotlin
    val degree: Int get() = weights.size
```

```fsharp
    member this.Degree = List.length this.Weights
```

同じ意図が、プロパティ・ゲッター付き `val`・メンバーという 3 つの構文で表現されます。

## 正則化の種類を表す型

ここが本章でもっとも差の出る箇所です。

```python
class Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"


def regularization_gradient(weight: float, kind: Regularization, strength: float) -> float:
    if kind is Regularization.L1:
        return strength * (1.0 if weight > 0 else -1.0 if weight < 0 else 0.0)
    if kind is Regularization.L2:
        return strength * 2.0 * weight
    return 0.0
```

```kotlin
enum class Regularization { NONE, L1, L2 }

fun regularizationGradient(weight: Double, kind: Regularization, strength: Double): Double =
    when (kind) {
        Regularization.NONE -> 0.0
        Regularization.L1 -> strength * when {
            weight > 0 -> 1.0
            weight < 0 -> -1.0
            else -> 0.0
        }
        Regularization.L2 -> strength * 2.0 * weight
    }
```

```fsharp
type Regularization =
    | NoRegularization
    | L1
    | L2

let regularizationGradient (weight: float) (kind: Regularization) (strength: float) =
    match kind with
    | NoRegularization -> 0.0
    | L1 -> strength * (if weight > 0.0 then 1.0 elif weight < 0.0 then -1.0 else 0.0)
    | L2 -> strength * 2.0 * weight
```

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 型 | `Enum` | `enum class` | 判別共用体 |
| 分岐 | `if` の連鎖 | 式としての `when` | `match` |
| 網羅性検査 | なし | あり（式として使えば） | あり（警告） |
| ケース追加時 | 気づけない（最後の `return` に落ちる） | コンパイルエラー | 警告 |

**この差は将来の変更で効きます。** `ElasticNet` を追加したとき、Kotlin はコンパイルが通らず、F# は警告が出ます。Python は静かに「正則化なし」として扱われ、バグが実行時まで残ります。

Python でこれを防ぐには、最後を `raise ValueError(f"unknown: {kind}")` にするか、`match` 文（3.10 以降）を使ったうえで型チェッカーに頼ることになります。**言語が守ってくれない部分は、規律で埋めるしかありません。**

F# には固有の注意点があります。ケース名を `None` にすると標準の `Option.None` と衝突するため、`NoRegularization` という名前にしています。判別共用体のケース名はモジュールの名前空間を共有するので、汎用的な単語は避ける必要があります。

## 4 つの値を返す

分割関数は 4 本のリストを返します。ここで 3 言語の設計判断が分かれました。

```python
def train_test_split(...) -> tuple[list[float], list[float], list[float], list[float]]:
    ...
    return (train_x, train_y, test_x, test_y)
```

```kotlin
data class Split(
    val trainFeatures: List<Double>,
    val trainLabels: List<Double>,
    val testFeatures: List<Double>,
    val testLabels: List<Double>,
)
```

```fsharp
type Split =
    { TrainFeatures: float list
      TrainLabels: float list
      TestFeatures: float list
      TestLabels: float list }
```

Python は 4 要素タプルのままにしました。呼び出し側が `train_x, train_y, test_x, test_y = ...` と **必ず分解代入する** 慣習があり、その場で名前が付くためです。

Kotlin と F# は名前付きの型を定義しました。Kotlin の `Triple` は 3 要素までしかなく、`Quadruple` は存在しません。F# の 4 要素タプルは `fst` / `snd` すら使えません。**要素が 3 つを超えたら名前付きの型** というのが両言語での実務的な線引きです。

型を定義した副作用として、Kotlin と F# は `split.testFeatures` のように **必要なフィールドだけ** 参照できるようになりました。Python 版は使わない値にも変数名を付ける（あるいは `_` で捨てる）必要があります。

## 学習ループ

第 3 章では誤差の履歴も畳み込んでいましたが、本章では最終モデルだけが必要です。

```python
    model = PolynomialModel(tuple(0.0 for _ in range(degree)), 0.0)
    for _ in range(epochs):
        i = rng.randrange(len(features))
        model = square_trick(model, features[i], labels[i], learning_rate, kind, strength)
    return model
```

```kotlin
    var model = PolynomialModel(List(degree) { 0.0 }, 0.0)
    repeat(epochs) {
        val i = random.nextInt(features.size)
        model = squareTrick(model, features[i], labels[i], learningRate, kind, strength)
    }
    return model
```

```fsharp
    let step model _ =
        let i = rng.Next(List.length features)
        squareTrick learningRate kind strength model features[i] labels[i]

    List.fold step initial [ 1..epochs ]
```

畳み込む状態が 1 つに減ったことで、F# 版の `step` が 2 行になりました。第 3 章ではタプル `(model, errors)` を畳み込んでいたぶん複雑でしたが、本章では `List.fold` がもっとも簡潔です。

3 言語とも初期値を **すべて 0** にしています。第 3 章はランダムな初期値でしたが、多項式では次数が上がるほど `x^n` が大きくなり、初期値によっては発散するためです。

## ハイパーパラメータの渡し方

本章では引数が 8 個まで増えました。ここで言語の機能差が実際の書き味に直結します。

| 言語 | 手段 | 呼び出し例 |
| :--- | :--- | :--- |
| Python | デフォルト引数 + キーワード引数 | `polynomial_regression(x, y, degree=5, kind=Regularization.L2, strength=0.01)` |
| Kotlin | デフォルト引数 + 名前付き引数 | `polynomialRegression(x, y, degree = 5, kind = Regularization.L2, strength = 0.01)` |
| F# | 部分適用で設定済み関数を作る | `let train degree kind strength = polynomialRegression degree 0.01 20000 kind strength 0` |

Python と Kotlin はほぼ同じ解き方です。F# にはデフォルト引数がないため、代わりに **設定済みの関数を作る** という関数型らしい解き方になります。

```fsharp
let train (degree: int) (kind: Regularization) (strength: float) =
    polynomialRegression degree 0.01 20000 kind strength 0

// 呼び出し
let model = train 5 L2 0.01 trainFeatures trainLabels
```

引数順を「ハイパーパラメータ → データ」にしてあるからこそ、この部分適用が成立します。第 3 章で決めた引数順の方針が、章をまたいで効いています。

## 実験結果

`y = 2x + 3` に小さなノイズを乗せた 10 点を、7 点の訓練データと 3 点のテストデータに分けた結果です。

### 次数と汎化ギャップ

汎化ギャップ = テスト RMSE − 訓練 RMSE です。

| 次数 | Python | Kotlin | F# |
| ---: | ---: | ---: | ---: |
| 1 | 0.046 | 0.195 | 0.142 |
| 3 | 0.244 | 0.238 | 0.241 |
| 5 | 0.235 | 0.233 | 0.283 |

絶対値は言語ごとに違いますが、**1 次から高次へ移るとギャップが広がる** という傾向は 3 言語で共通しています。

### 正則化（5 次モデル、λ = 0.01）のテスト RMSE

| 正則化 | Python | Kotlin | F# |
| :--- | ---: | ---: | ---: |
| なし | 0.2947 | 0.2963 | 0.3370 |
| L1 | 0.2231 | 0.2220 | 0.2088 |
| L2 | **0.1239** | **0.1807** | **0.1498** |

3 言語とも L2 がもっともテスト誤差を下げました。

### L1 が 0 にした重み

| 言語 | 3 次の重み | 4 次の重み |
| :--- | ---: | ---: |
| Python | -0.0003 | -0.0005 |
| Kotlin | 0.0001 | 0.0002 |
| F# | -0.0011 | -0.0001 |

いずれも「ちょうど 0」にはなっていません。劣勾配による確率的更新では厳密な 0 に張り付かないためです。3 言語のテストは共通して `< 5e-3` で検証しています。**もっとも 0 から遠かった F# 版の値（-0.0011）に合わせて閾値を決めました。** 数学的な理想ではなく、実装が実際に返す値を基準にするための判断です。

## なぜ数値が一致しないのか

第 3 章と同じく、乱数アルゴリズムが言語ごとに違うためです。本章ではさらに **データ分割そのものが変わる** ため、差が大きくなります。

| 言語 | テストデータに選ばれた x |
| :--- | :--- |
| Python | 0.6, 0.9, -1.2 |
| Kotlin | 1.2, 0.6, -1.2 |
| F# | -0.3, 1.2, 0.6 |

3 点だけを取り分けるので、どの点が選ばれるかでテスト誤差は大きく動きます。**したがって「3 言語で数値が一致すること」を検証してはいけません。** 3 言語のテストはすべて、次の 2 種類だけを検証しています。

| 対象 | 検証方法 |
| :--- | :--- |
| 1 点分の更新・勾配の計算 | 手計算した期待値と厳密比較 |
| 学習の結果 | 「ギャップが広がる」「重みが縮む」といった **関係性** |

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | デフォルト引数とキーワード引数で設定が読みやすい | 網羅性検査 + デフォルト引数の両立 | 部分適用で設定済み関数を作れる、既定で不変 |
| 注意点 | `Enum` 分岐の漏れを言語が検出しない、`tuple` を選ばないと不変にならない | `var` が学習ループに残る | デフォルト引数がなく、長い式はオフサイドに引っかかる |

第 3 章では「状態の持ち方」と「設定値の渡し方」が差でした。本章ではそこに **「種類を表す型の安全性」** が加わりました。分岐の網羅性をコンパイラが見てくれるかどうかは、章を重ねてケースが増えるほど効いてきます。

## 参照

- [Python 版 第 4 章](../python/ch04.md)
- [Kotlin 版 第 4 章](../kotlin/ch04.md)
- [F# 版 第 4 章](../fsharp/ch04.md)
- [第 3 章 3 言語比較](ch03-linear-regression.md)
- [統合記事 目次](index.md)
