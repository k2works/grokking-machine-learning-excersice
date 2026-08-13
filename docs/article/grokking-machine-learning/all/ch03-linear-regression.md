# 第 3 章 線形回帰 - 3 言語比較

原著第 3 章の線形回帰を、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch03.md)・[Kotlin 版](../kotlin/ch03.md)・[F# 版](../fsharp/ch03.md) を参照してください。

## アルゴリズムの共通部分

3 言語とも次の構成で実装しています。言語が違っても、アルゴリズムの骨格はまったく同じです。

1. モデル = 傾きと切片の 2 値（不変）
2. トリック = `(モデル, 1 点) -> 新しいモデル` の純関数
3. RMSE = モデル全体の誤差
4. 学習ループ = ランダムに 1 点選び、二乗トリックを繰り返す

## モデル表現

| 言語 | 表現 | 不変性 | 構造的等価性 |
| :--- | :--- | :--- | :--- |
| Python | `@dataclass(frozen=True)` | オプトイン（`frozen=True`） | 自動生成 |
| Kotlin | `data class` + `val` | `val` で宣言 | 自動生成 |
| F# | レコード型 | 既定で不変 | 既定であり |

```python
@dataclass(frozen=True)
class Model:
    slope: float
    intercept: float

    def predict(self, rooms: float) -> float:
        return self.slope * rooms + self.intercept
```

```kotlin
data class Model(val slope: Double, val intercept: Double) {
    fun predict(rooms: Double): Double = slope * rooms + intercept
}
```

```fsharp
type Model =
    { Slope: float
      Intercept: float }

let predict (model: Model) (rooms: float) = model.Slope * rooms + model.Intercept
```

Python と Kotlin は `predict` をモデルのメソッドとして持たせています。F# はモジュールの関数として外に置いており、これがそのまま `List.map (predict model)` の部分適用につながります。

**不変性の既定値** が 3 言語で異なる点に注目してください。F# は何も書かなければ不変、Kotlin は `val` を選ぶ、Python は `frozen=True` を明示する。この差は、以降の章でモデルが複雑になるほど効いてきます。

## 二乗トリック

アルゴリズムの中心です。3 言語とも同じ 2 本の式に落ち着きます。

```python
def square_trick(model: Model, rooms: float, price: float, learning_rate: float) -> Model:
    error = price - model.predict(rooms)
    return Model(
        model.slope + learning_rate * rooms * error,
        model.intercept + learning_rate * error,
    )
```

```kotlin
fun squareTrick(model: Model, rooms: Double, price: Double, learningRate: Double): Model {
    val error = price - model.predict(rooms)
    return Model(
        slope = model.slope + learningRate * rooms * error,
        intercept = model.intercept + learningRate * error,
    )
}
```

```fsharp
let squareTrick (learningRate: float) (model: Model) (rooms: float) (price: float) =
    let error = price - predict model rooms

    { Slope = model.Slope + learningRate * rooms * error
      Intercept = model.Intercept + learningRate * error }
```

違いは **引数の順序** です。Python と Kotlin はモデルを先頭、学習率を末尾（キーワード引数・デフォルト引数で扱いやすい位置）に置いています。F# は逆に学習率を先頭に置き、`squareTrick 0.01` という部分適用で「学習率を固定したトリック」を作れるようにしています。

同じ関数でも、**言語が提供する呼び出し方の慣習に合わせて引数順を決める** ことになります。

## RMSE

```python
def rmse(labels: Sequence[float], predictions: Sequence[float]) -> float:
    n = len(labels)
    total = sum((label - prediction) ** 2 for label, prediction in zip(labels, predictions))
    return math.sqrt(total / n)
```

```kotlin
fun rmse(labels: List<Double>, predictions: List<Double>): Double {
    val total = labels.zip(predictions).sumOf { (label, prediction) ->
        val difference = label - prediction
        difference * difference
    }
    return sqrt(total / labels.size)
}
```

```fsharp
let rmse (labels: float list) (predictions: float list) =
    let n = float (List.length labels)

    List.zip labels predictions
    |> List.sumBy (fun (label, prediction) -> (label - prediction) ** 2.0)
    |> fun total -> sqrt (total / n)
```

3 言語とも `zip` → 合計 → 平方根という同じ流れです。表現の差は次のとおりです。

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 合計 | ジェネレータ式 + `sum` | `sumOf` | `List.sumBy` |
| 分解 | `for a, b in ...` | ラムダの分解宣言 | タプルパターン |
| 数値変換 | 暗黙（`int / int` は float） | 暗黙（演算子オーバーロード） | 明示（`float (...)` が必要） |
| 流れの表現 | 内側から外側へ | メソッドチェーン | パイプライン演算子 |

F# の明示的な数値変換は書く量が増えますが、意図しない整数除算という定番のバグを型検査で防ぎます。

## 学習ループ

ここが 3 言語でもっとも表現が分かれる箇所です。

```python
    rng = random.Random(seed)
    model = Model(rng.random(), rng.random())
    errors: list[float] = []
    for _ in range(epochs):
        errors.append(model_rmse(model, features, labels))
        i = rng.randrange(len(features))
        model = square_trick(model, features[i], labels[i], learning_rate)
    return model, errors
```

```kotlin
    val random = Random(seed)
    var model = Model(random.nextDouble(), random.nextDouble())
    val errors = mutableListOf<Double>()
    repeat(epochs) {
        errors += modelRmse(model, features, labels)
        val i = random.nextInt(features.size)
        model = squareTrick(model, features[i], labels[i], learningRate)
    }
    return model to errors
```

```fsharp
    let step (model, errors) _ =
        let errors = modelRmse model features labels :: errors
        let i = rng.Next(List.length features)
        let model = squareTrick learningRate model features[i] labels[i]
        (model, errors)

    let model, errors = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev errors)
```

| 言語 | ループの表現 | 状態の扱い |
| :--- | :--- | :--- |
| Python | `for _ in range(epochs)` | 変数の再代入 + リストへの追加 |
| Kotlin | `repeat(epochs) { }` | `var` + `MutableList`（関数内に閉じ込め） |
| F# | `List.fold` | 状態をタプルで畳み込み（再代入なし） |

Python と Kotlin は **可変状態を関数の内部に閉じ込め、外へは不変の値を返す** という方針です。F# は畳み込みで可変状態そのものを避けています。

どれが正しいということはありません。ただし共通しているのは、**関数の外から見える振る舞いは 3 言語とも純粋** だという点です。同じ入力（同じシード）に対して常に同じ出力を返します。

## 学習結果の比較

学習率 0.01、1000 エポック、シード 0 での結果です。

| 言語 | 傾き | 切片 | 学習後の RMSE | 初期の RMSE |
| :--- | ---: | ---: | ---: | ---: |
| Python | 51.0443 | 91.5945 | 7.4501 | 315.77 |
| Kotlin | 52.4794 | 88.3304 | 7.2976 | 317.01 |
| F# | 52.7060 | 90.5543 | 7.0309 | 316.24 |

真の関係は傾き 50・切片 100 です。3 言語とも同じ程度に近い直線へ収束していますが、値は一致しません。理由は 2 つあります。

1. **乱数アルゴリズムが言語ごとに違う** — 同じシードでも選ばれる点の並びが異なります。
2. **SGD は 1 点ずつしか見ない** — 最後に選ばれた点の影響が残るため、厳密な最小値には停まりません。

つまり「3 言語で数値が一致すること」を目指すのは誤りです。**同じくらいの誤差に収束すること** が正しい検証対象になります。

## テストの書き方

この性質から、3 言語ともテストは 2 段構えにしています。

| 対象 | 検証方法 |
| :--- | :--- |
| トリック 1 回分 | 手計算した期待値と厳密比較（浮動小数点の許容誤差付き） |
| 学習全体 | 収束範囲・誤差が減少したことの検証 |

```python
assert model.slope == pytest.approx(50.0, abs=5.0)
assert errors[-1] < errors[0]
```

```kotlin
assertTrue(model.slope in 45.0..55.0, "slope=${model.slope}")
assertTrue(errors.last() < errors.first())
```

```fsharp
Assert.InRange(model.Slope, 45.0, 55.0)
Assert.True(List.last errors < List.head errors)
```

| 言語 | 範囲検証 | 浮動小数点比較 | テスト名 |
| :--- | :--- | :--- | :--- |
| Python | `pytest.approx(値, abs=許容)` | `pytest.approx` | 関数名（英語） |
| Kotlin | 範囲リテラル `in 45.0..55.0` | `assertEquals(期待, 実測, 許容)` | バッククォートで日本語可 |
| F# | `Assert.InRange` | `Assert.Equal(期待, 実測, 桁数)` | バッククォート 2 つで日本語可 |

Kotlin の範囲リテラルは意図がもっとも読み取りやすく、Python の `pytest.approx` は「中心値 ± 許容」という書き方で真の値との関係を表せます。

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | 記述量が最小、実験の反復が速い | 型安全と簡潔さの両立、名前付き引数で意図が明確 | 部分適用で設定済み関数を作れる、既定で不変 |
| 注意点 | 不変性を明示しないと守られない | 学習ループで `var` が必要になる | 数値変換を明示する手間 |

アルゴリズムの本質（2 本の更新式）は 3 言語でまったく同じでした。違いが出たのは **状態の持ち方** と **設定値の渡し方** です。以降の章でモデルが複雑になるほど、この 2 点の差が設計に効いてきます。

## 参照

- [Python 版 第 3 章](../python/ch03.md)
- [Kotlin 版 第 3 章](../kotlin/ch03.md)
- [F# 版 第 3 章](../fsharp/ch03.md)
- [統合記事 目次](index.md)
