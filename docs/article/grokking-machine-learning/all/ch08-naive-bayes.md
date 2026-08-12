# 第 8 章 ナイーブベイズ - 3 言語比較

原著第 8 章のナイーブベイズを、Python・Kotlin・F# で実装した結果を比較します。各言語の詳細は [Python 版](../python/ch08.md)・[Kotlin 版](../kotlin/ch08.md)・[F# 版](../fsharp/ch08.md) を参照してください。

## 数値が完全に一致する 2 つ目の章

第 7 章に続き、本章も **3 言語で数値が完全に一致します**。

| 文書 | Python | Kotlin | F# |
| :--- | ---: | ---: | ---: |
| （空） | 0.3750 | 0.3750 | 0.3750 |
| project deadline | 0.0909 | 0.0909 | 0.0909 |
| sale today | 0.2308 | 0.2308 | 0.2308 |
| lottery | 0.7059 | 0.7059 | 0.7059 |
| lottery winning | 0.8780 | 0.8780 | 0.8780 |
| lottery winning sale | 0.9153 | 0.9153 | 0.9153 |

第 7 章は「学習がない」から一致しました。本章は **学習はあるが乱数がない** から一致します。数え上げは決定的な操作なので、どの言語で実行しても同じ答えになります。

**ナイーブベイズにはエポックも学習率もシードもありません。** 唯一のハイパーパラメータは平滑化の強さだけです。

## クラス別の集計

学習の実体は「ラベルごとに単語を数える」だけです。ここで 3 言語の集計語彙の差が出ました。

```python
    for document, label in zip(documents, labels):
        words = set(tokenize(document))
        if label == 1:
            spam_word_counts.update(words)
            spam_documents += 1
        else:
            ham_word_counts.update(words)
            ham_documents += 1
```

```kotlin
    val byLabel = documents.zip(labels).groupBy({ it.second }, { tokenize(it.first).toSet() })
    fun countWords(label: Int): Map<String, Int> =
        byLabel[label].orEmpty().flatten().groupingBy { it }.eachCount()
```

```fsharp
    let wordSetsFor label =
        List.zip documents labels
        |> List.filter (fun (_, l) -> l = label)
        |> List.map (fst >> tokenize >> Set.ofList)

    let countWords label =
        wordSetsFor label
        |> List.collect Set.toList
        |> List.countBy id
        |> Map.ofList
```

| 言語 | 手法 | 可変変数 | 走査回数 |
| :--- | :--- | :--- | ---: |
| Python | ループ + `Counter.update` | 4 個 | 1 |
| Kotlin | `groupBy` 2 引数版 → `groupingBy().eachCount()` | なし | 2 |
| F# | `filter` → `map` → `countBy` | なし | 2（ラベルごと） |

Python は `Counter.update(集合)` が「集合の各要素を 1 ずつ増やす」を一手でやってくれるため、素直なループが最短になりました。ただし可変変数が 4 つ出てきます。

Kotlin と F# は宣言的に書けますが、ラベルで分けてから数えるので走査が 2 段になります。**読みやすさと走査回数のトレードオフ** です。データ量が小さい本章では読みやすさを取りました。

3 言語に共通するのは、**同じ処理をスパム用とハム用で 2 度書かない** ための工夫です。Kotlin と F# はローカル関数（`countWords`）に切り出し、外側の変数をクロージャで捕捉しています。トップレベル関数にすると引数が増え、どこで使うのかも曖昧になります。

F# の `fst >> tokenize >> Set.ofList` は **関数合成** です。3 段の変換がラムダなしで 1 つの関数になり、処理の流れが左から右に読めます。

## 「キーが無い」をどう表すか

数え上げの結果を引くとき、その単語が一度も現れていない場合があります。

```python
    spam = model.spam_word_counts[word] + smoothing   # Counter は 0 を返す
```

```kotlin
    val spam = (model.spamWordCounts[word] ?: 0) + smoothing
```

```fsharp
    let countIn counts =
        Map.tryFind word counts |> Option.defaultValue 0 |> float
```

| 言語 | 型 | 不在時 | 「無い」と「0 件」の区別 |
| :--- | :--- | :--- | :--- |
| Python | `Counter[str]` | **0 を返す** | つかない |
| Kotlin | `Map<String, Int>` | `null` を返す | つく（`null` vs `0`） |
| F# | `Map<string, int>` | `None` を返す | つく（`None` vs `Some 0`） |

Python の `Counter` は「存在しないキーは 0」という規約を持つため、`+ smoothing` がそのまま書けます。もっとも短いコードになりますが、**「一度も現れていない」と「0 回と記録されている」の区別が消えます**。

Kotlin と F# は不在を型に持ち上げます。テストでもその違いが出ました。

```kotlin
assertEquals(null, model.hamWordCounts["lottery"])
```

```fsharp
Assert.Equal(None, Map.tryFind "lottery" model.HamWordCounts)
```

```python
assert model.ham_word_counts["lottery"] == 0
```

**Python では「0 件だった」としか書けません。** 3 言語とも同じ事実（lottery は通常文書に現れない）を検証していますが、表現の精度が違います。

## 平滑化が型の問題も解いた

第 7 章では、`Int / Int` が整数除算になる罠が Kotlin と F# にありました。本章では起きません。

```kotlin
    val spam = (model.spamWordCounts[word] ?: 0) + smoothing   // Int + Double = Double
    return spam / (spam + ham)                                  // Double / Double
```

平滑化（`Double`）を足した時点で `Double` に昇格するため、割り算の型を気にする必要がありません。**確率を両端から引き離すための処理が、たまたま型の落とし穴も塞いでいます。**

F# も同様ですが、こちらは `|> float` で明示的に変換してから足しています。暗黙の昇格がないためです。

## 2 つの累積値をどう更新するか

対数空間での合成は、`logSpam` と `logHam` を同時に更新します。

```python
    for word in words:
        if word not in model.vocabulary:
            continue
        spam_given_word = word_spam_probability(model, word, smoothing)
        log_spam += math.log(_safe(spam_given_word))
        log_ham += math.log(_safe(1.0 - spam_given_word))
```

```kotlin
    tokenize(document).toSet().filter { it in model.vocabulary }.forEach { word ->
        val spamGivenWord = wordSpamProbability(model, word, smoothing)
        logSpam += ln(safe(spamGivenWord))
        logHam += ln(safe(1.0 - spamGivenWord))
    }
```

```fsharp
    let accumulate (logSpam, logHam) word =
        let spamGivenWord = wordSpamProbabilityWith smoothing model word
        (logSpam + log (safe spamGivenWord), logHam + log (safe (1.0 - spamGivenWord)))

    let logSpam, logHam =
        Set.fold accumulate (log (safe prior), log (safe (1.0 - prior))) words
```

| 言語 | 手法 | 可変変数 |
| :--- | :--- | :--- |
| Python | ループ + 再代入 | 2 個 |
| Kotlin | `forEach` + `var` の再代入 | 2 個 |
| F# | **タプルを畳み込む** | なし |

F# は第 3 章の学習ループ（`(model, errors)` の畳み込み）と同じ形です。**「複数の値を同時に更新したい」という場面で、F# は一貫してタプルの `fold` を使えます。**

Python と Kotlin も `reduce` / `fold` で書けますが、`Pair` の分解と再構築が入るぶん読みにくくなります。**言語の標準的な書き方に従うほうが、読み手にとって素直です。**

## すべてがシグモイドに帰着する

3 言語とも、最後の 1 行はこの形です。

```python
    return 1.0 / (1.0 + math.exp(min(max(log_ham - log_spam, -700.0), 700.0)))
```

```kotlin
    return 1.0 / (1.0 + exp(min(max(logHam - logSpam, -700.0), 700.0)))
```

```fsharp
    1.0 / (1.0 + exp (logHam - logSpam |> max -700.0 |> min 700.0))
```

`1 / (1 + exp(x))` は **第 6 章のシグモイドとまったく同じ関数** です。

| 章 | シグモイドに入れるもの | 由来 |
| :--- | :--- | :--- |
| 第 6 章 | 重み付き和 `w・x + b` | 勾配降下法で学習した重み |
| 第 8 章 | 対数確率の差 `log(P(ham)) - log(P(spam))` | ベイズの定理と数え上げ |

**まったく違う道筋から、同じ関数に辿り着きます。** これは偶然ではなく、「2 つの対立仮説の対数尤度比を確率に戻す」という操作がシグモイドそのものだからです。第 6 章のロジスティック回帰と第 8 章のナイーブベイズは、しばしば「生成モデルと識別モデルの対」として並べて論じられます。実装がそれを見せてくれました。

クランプの書き方だけが 3 言語で分かれます。F# のパイプライン（`|> max -700.0 |> min 700.0`）は第 6 章の `logLoss` と同じ形で、「下限で切って上限で切る」が手順どおりに並びます。

## 平滑化がないとどうなるか

3 言語とも同じテストで平滑化の必要性を固定しています。

```python
def test_smoothing_keeps_probabilities_away_from_the_extremes():
    model = train(DOCUMENTS, LABELS)
    for word in model.vocabulary:
        probability = word_spam_probability(model, word)
        assert 0.0 < probability < 1.0
```

`lottery` はスパム文書に 3 回、通常文書に 0 回現れます。平滑化なしなら確率 1.0 です。**確率 1.0 は「絶対にスパム」を意味し、他のどんな単語が来ても覆せません。** 掛け算の中に 0 が 1 つ混じればすべてが 0 になるのと同じで、たった 1 語が判定を支配します。

すべてのカウントに 1 を足すと 0.8 になり、他の単語の証拠と競り合えるようになります。

未知語はスパム 0 件・通常 0 件なので、平滑化後はちょうど 0.5 です。**「何も知らない単語は判定に寄与しない」** という性質が自然に出ます。3 言語とも次を検証しました。

```python
def test_unknown_words_do_not_change_the_prediction():
    known = predict_probability(model, "lottery")
    with_unknown = predict_probability(model, "lottery zzzz qqqq")
    assert with_unknown == pytest.approx(known)
```

## この章のまとめ

| 観点 | Python | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| 強み | `Counter` により集計と既定値 0 が最短で書ける | `groupBy` 2 引数版で分類と変換が同時にできる | 関数合成とタプル畳み込みで可変変数がゼロ |
| 注意点 | 「キーが無い」と「0 件」を区別できない | `?: 0` と `orEmpty()` が繰り返し必要 | 記述量がやや多い |

第 7 章では F# の「明示を強制する」設計が事故を防ぎました。本章で目立ったのは **「無い」を型で表せるかどうか** です。Python の `Counter` は最短のコードを与えますが、その代償として情報がひとつ失われます。どちらを取るかは、扱うデータの性質によります。

## 参照

- [Python 版 第 8 章](../python/ch08.md)
- [Kotlin 版 第 8 章](../kotlin/ch08.md)
- [F# 版 第 8 章](../fsharp/ch08.md)
- [第 7 章 3 言語比較](ch07-metrics.md)
- [統合記事 目次](index.md)
