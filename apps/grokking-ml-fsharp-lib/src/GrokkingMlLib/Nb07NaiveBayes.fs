/// 原著ノートブック #07 `Chapter_08_Naive_Bayes/Coding_naive_Bayes.ipynb`。
///
/// メール 5728 通からスパム判定器をナイーブベイズで作る。
/// **この回だけ原著が scikit-learn を使わず、pandas と辞書だけで書いている。**
/// だからライブラリの差が出ず、F# でも原著の数値をそのまま再現できる。
///
/// 原著は最後の確率計算で `np.compat.long` を使っており、これが浮動小数点数を
/// **整数に切り捨てる**。切り捨てないと原著の出力と 8 桁目から食い違うので、
/// その挙動もそのまま写してある。
module GrokkingMlLib.Nb07NaiveBayes

open System
open System.Collections.Generic
open System.Numerics
open System.Text.RegularExpressions
open Deedle

/// ある単語が、スパムとハムそれぞれ何通に現れたか。
///
/// 原著は 1 から数え始める。ラプラス平滑化にあたり、
/// 「一度も見ていない側」の確率が 0 になるのを防ぐ。
type WordCounts = { Spam: int; Ham: int }

/// 数え始めの値
let initialCounts = { Spam = 1; Ham = 1 }

/// 学習に使ったメール全体の統計
type Corpus =
    { Total: int; Spam: int }

    member this.Ham = this.Total - this.Spam

    /// 事前確率。何も情報が無いときにスパムと判断する確率
    member this.SpamProbability = float this.Spam / float this.Total

/// 単語ごとの出現数と、コーパス全体の統計
type NaiveBayesModel =
    { Words: IReadOnlyDictionary<string, WordCounts>
      Corpus: Corpus }

/// メールと、スパムかどうかのラベル
type Emails = { Texts: string[]; Spam: int[] }

let private whitespace = Regex(@"\s+", RegexOptions.Compiled)

/// メール本文を、重複を除いた小文字の単語集合にする。
///
/// 原著は `list(set(text.lower().split()))`。**同じ単語が何度出ても 1 回** と
/// 数えるのがこの実装の前提である。
///
/// Python の `split()` は空白の連続をまとめて区切る。`Regex.Split` は
/// 空文字列を残すので、そこを落とす必要がある。
let processEmail (text: string) =
    whitespace.Split(text.ToLowerInvariant())
    |> Array.filter (fun token -> token <> "")
    |> Set.ofArray

/// メールのデータセットを読み込む。5728 通、うち 1368 通がスパム。
///
/// このファイルは約 8.5 MB あるためリポジトリには含めていない。
let loadEmails () =
    let frame = Datasets.loadFrame "emails.csv"

    { Texts = frame.GetColumn<string>("text") |> Series.values |> Array.ofSeq
      Spam = frame.GetColumn<int>("spam") |> Series.values |> Array.ofSeq }

/// 単語ごとに、スパム・ハムそれぞれの出現通数を数える
let train (emails: Emails) =
    let words = Dictionary<string, WordCounts>()

    for i in 0 .. emails.Texts.Length - 1 do
        for word in processEmail emails.Texts.[i] do
            let current =
                match words.TryGetValue word with
                | true, counts -> counts
                | _ -> initialCounts

            words.[word] <-
                if emails.Spam.[i] = 1 then
                    { current with Spam = current.Spam + 1 }
                else
                    { current with Ham = current.Ham + 1 }

    { Words = words
      Corpus =
        { Total = emails.Texts.Length
          Spam = emails.Spam |> Array.sumBy (fun s -> if s = 1 then 1 else 0) } }

/// 単語 1 つだけを見たときの、スパムである確率。
///
/// ベイズの定理そのものではなく、単純に「その単語を含むメールのうち
/// スパムの割合」を返す。原著もそう書いている。
let predictBayes (model: NaiveBayesModel) (word: string) =
    let counts = model.Words.[word.ToLowerInvariant()]
    float counts.Spam / float (counts.Spam + counts.Ham)

/// メール全体を見たときの、スパムである確率。
///
/// 「単語の出現が互いに独立」と仮定して、単語ごとの尤度比を掛け合わせる。
/// 語彙にない単語は無視するので、知らない単語ばかりのメールは事前確率に落ちる。
///
/// 最後に整数へ切り捨てているのは、原著の `np.compat.long` に合わせるため。
/// 切り捨てないと 8 桁目から数値が変わる。
let predictNaiveBayes (model: NaiveBayesModel) (email: string) =
    let corpus = model.Corpus

    let productSpams, productHams =
        processEmail email
        |> Set.fold
            (fun (spams, hams) word ->
                match model.Words.TryGetValue word with
                | true, counts ->
                    spams * (float counts.Spam / float corpus.Spam * float corpus.Total),
                    hams * (float counts.Ham / float corpus.Ham * float corpus.Total)
                | _ -> spams, hams)
            (float corpus.Spam, float corpus.Ham)

    // 原著の np.compat.long と同じ切り捨て。値は 10^20 を超えることがあるので
    // int64 では溢れる。Python の int は多倍長なので BigInteger で受ける
    let truncatedSpams = BigInteger productSpams
    let truncatedHams = BigInteger productHams

    // Python は int / int を多倍長のまま計算してから float にする。
    // decimal では桁が足りないので、比を BigInteger の割り算で作ってから float にする
    let scale = BigInteger.Pow(BigInteger 10, 20)
    float (truncatedSpams * scale / (truncatedSpams + truncatedHams)) / 1e20
