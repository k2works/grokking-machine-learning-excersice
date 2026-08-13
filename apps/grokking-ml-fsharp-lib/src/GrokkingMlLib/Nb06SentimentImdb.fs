/// 原著ノートブック #06 `Chapter_06_Logistic_Regression/Sentiment_analysis_IMDB.ipynb`。
///
/// IMDB の映画レビュー 50000 件を、単語の出現回数だけからロジスティック回帰で
/// 肯定・否定に分類する。学習した係数がそのまま「単語の感情スコア」になる。
///
/// **.NET には `CountVectorizer` に相当するものが無い**（ML.NET の
/// `FeaturizeText` は n-gram やハッシュ化まで含む別物）。3 言語で同じ語彙を
/// 作るため、ベクトル化は自前で書く。
module GrokkingMlLib.Nb06SentimentImdb

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open Deedle

/// 原著が使う語彙の上限
[<Literal>]
let MaxFeatures = 2000

/// scikit-learn の既定と同じトークン正規表現（2 文字以上の単語）
let private tokenPattern = Regex(@"\b\w\w+\b", RegexOptions.Compiled)

/// scikit-learn 内蔵の英語ストップワード 318 語。共有データセットから読む
let englishStopWords () =
    Datasets.path "sklearn_english_stop_words.txt"
    |> File.ReadAllLines
    |> Array.filter (String.IsNullOrWhiteSpace >> not)
    |> Set.ofArray

/// 小文字化して 2 文字以上の単語を取り出し、ストップワードを除く
let tokenize (stopWords: Set<string>) (document: string) =
    tokenPattern.Matches(document.ToLowerInvariant())
    |> Seq.map (fun m -> m.Value)
    |> Seq.filter (fun token -> not (stopWords.Contains token))
    |> List.ofSeq

/// 出現回数ベクトル。疎な表現で持つ。
///
/// **密な行列にしてはいけない。** IMDB は 50000 件 × 2000 語なので、
/// 倍精度の密行列にすると 800 MB になる。実際の非ゼロ要素は数パーセントである。
type SparseCounts = (int * float) list

/// 学習した語彙。添字がそのまま特徴量の添字になる
type Vocabulary =
    { Words: string[]
      IndexOf: IReadOnlyDictionary<string, int> }

    member this.Size = this.Words.Length

/// scikit-learn の `CountVectorizer` と同じ規則で語彙を学習する。
///
/// 揃えているのは次の 4 点である。
/// 1. 小文字化する
/// 2. トークンは正規表現 `\b\w\w+\b` で切り出す
/// 3. ストップワードを除く
/// 4. 残った語をコーパス全体の出現回数で並べ、上位 `maxFeatures` 語を採る
///
/// 語彙は最後に **辞書順** に並べ替える。scikit-learn の
/// `get_feature_names_out()` と同じで、その順序が特徴量の添字になる。
let fitVocabulary (stopWords: Set<string>) (maxFeatures: int) (documents: string seq) =
    let counts = Dictionary<string, int>()

    for document in documents do
        for token in tokenize stopWords document do
            counts.[token] <- (match counts.TryGetValue token with
                               | true, value -> value
                               | _ -> 0)
                              + 1

    // 出現回数の多い順。同数なら辞書順にして結果を決定的にする
    let byCountThenWord (left: KeyValuePair<string, int>) (right: KeyValuePair<string, int>) =
        match compare right.Value left.Value with
        | 0 -> String.CompareOrdinal(left.Key, right.Key)
        | difference -> difference

    let words =
        counts
        |> Seq.sortWith byCountThenWord
        |> Seq.truncate maxFeatures
        |> Seq.map (fun kv -> kv.Key)
        |> Seq.sortWith (fun a b -> String.CompareOrdinal(a, b))
        |> Array.ofSeq

    { Words = words
      IndexOf = words |> Array.mapi (fun i w -> w, i) |> dict |> Dictionary }

/// 学習済みの語彙で 1 件を出現回数ベクトルにする
let transformOne (stopWords: Set<string>) (vocabulary: Vocabulary) (document: string) : SparseCounts =
    let counts = Dictionary<int, float>()

    for token in tokenize stopWords document do
        match vocabulary.IndexOf.TryGetValue token with
        | true, index ->
            counts.[index] <- (match counts.TryGetValue index with
                               | true, value -> value
                               | _ -> 0.0)
                              + 1.0
        | _ -> ()

    counts |> Seq.map (fun kv -> kv.Key, kv.Value) |> Seq.sortBy fst |> List.ofSeq

/// レビューと 0 / 1 のラベル
type Reviews = { Texts: string[]; Sentiments: int[] }

/// 学習済みモデルと、それを作るのに使った語彙
type SentimentModel =
    { Vocabulary: Vocabulary
      /// 語彙と同じ順に並んだ係数
      Coefficients: float[]
      Intercept: float }

    /// 単語と、その係数（感情スコア）の対応表
    member this.WordSentiments() =
        Array.map2 (fun word weight -> word, weight) this.Vocabulary.Words this.Coefficients

    /// 係数が大きい順に単語を返す
    member this.MostPositiveWords(count: int) =
        this.WordSentiments() |> Array.sortByDescending snd |> Array.truncate count |> Array.map fst

    /// 係数が小さい順に単語を返す
    member this.MostNegativeWords(count: int) =
        this.WordSentiments() |> Array.sortBy snd |> Array.truncate count |> Array.map fst

/// IMDB のレビューを読み込み、`sentiment` を 0 / 1 に置き換える。
///
/// このファイルは約 63 MB あるためリポジトリには含めていない。
let loadReviews () =
    let frame = Datasets.loadFrame "IMDB_Dataset.csv"
    let texts = frame.GetColumn<string>("review") |> Series.values |> Array.ofSeq
    let sentiments =
        frame.GetColumn<string>("sentiment")
        |> Series.values
        |> Seq.map (fun s -> if s = "positive" then 1 else 0)
        |> Array.ofSeq

    { Texts = texts; Sentiments = sentiments }

/// 疎な出現回数ベクトルに対するロジスティック回帰を、勾配降下で学習する。
///
/// ML.NET は特徴量を固定長の `float32[]` で要求するため、50000 × 2000 の
/// 密配列（約 400 MB）を作ることになる。ここは疎なまま解くほうが素直なので、
/// 自前の勾配降下で書いた。L2 正則化を掛けて scikit-learn の既定に寄せる。
let fitSparse
    (features: SparseCounts[])
    (labels: int[])
    (vocabulary: Vocabulary)
    (learningRate: float)
    (lambda: float)
    (epochs: int)
    =
    let weights = Array.zeroCreate<float> vocabulary.Size
    let mutable intercept = 0.0
    let n = float features.Length

    for _ in 1..epochs do
        let gradient = Array.zeroCreate<float> vocabulary.Size
        let mutable interceptGradient = 0.0

        // 疎ベクトルなので、非ゼロの要素だけを見れば済む
        for i in 0 .. features.Length - 1 do
            let score =
                features.[i] |> List.sumBy (fun (index, value) -> weights.[index] * value)

            let predicted = 1.0 / (1.0 + exp (-(score + intercept)))
            let error = float labels.[i] - predicted

            for (index, value) in features.[i] do
                gradient.[index] <- gradient.[index] + error * value

            interceptGradient <- interceptGradient + error

        for j in 0 .. vocabulary.Size - 1 do
            // L2 正則化。切片は罰則の対象にしない
            weights.[j] <- weights.[j] + learningRate * (gradient.[j] / n - lambda * weights.[j])

        intercept <- intercept + learningRate * interceptGradient / n

    { Vocabulary = vocabulary
      Coefficients = weights
      Intercept = intercept }

/// レビューをベクトル化してロジスティック回帰を学習する
let fit (reviews: Reviews) (maxFeatures: int) (epochs: int) =
    let stopWords = englishStopWords ()
    let vocabulary = fitVocabulary stopWords maxFeatures reviews.Texts
    let features = reviews.Texts |> Array.map (transformOne stopWords vocabulary)
    fitSparse features reviews.Sentiments vocabulary 0.5 1e-4 epochs
