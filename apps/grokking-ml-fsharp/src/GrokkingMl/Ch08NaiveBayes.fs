/// 第 8 章: ナイーブベイズ。
/// 「単語が独立に出現する」という（実際には正しくない）仮定を置くことで、
/// ベイズの定理による分類を単なる掛け算に単純化する。
module GrokkingMl.Ch08NaiveBayes

/// 文章を小文字の単語列に分解する。
let tokenize (text: string) =
    text.ToLowerInvariant().Split(' ')
    |> Array.filter (fun word -> word <> "")
    |> Array.toList

/// ナイーブベイズ分類器。単語ごとの出現回数からスパム確率を求める。
type NaiveBayesClassifier =
    { SpamWordCounts: Map<string, int>
      HamWordCounts: Map<string, int>
      SpamDocuments: int
      HamDocuments: int }

    member this.TotalDocuments = this.SpamDocuments + this.HamDocuments

    member this.Vocabulary =
        Set.union (Set.ofSeq (Map.keys this.SpamWordCounts)) (Set.ofSeq (Map.keys this.HamWordCounts))

/// 文書とラベル（1 がスパム）から分類器を学習する。
let train (documents: string list) (labels: int list) =
    // 同じ単語が何度出ても 1 文書につき 1 回だけ数える（ベルヌーイ型）
    let wordSetsFor label =
        List.zip documents labels
        |> List.filter (fun (_, l) -> l = label)
        |> List.map (fst >> tokenize >> Set.ofList)

    let countWords label =
        wordSetsFor label
        |> List.collect Set.toList
        |> List.countBy id
        |> Map.ofList

    { SpamWordCounts = countWords 1
      HamWordCounts = countWords 0
      SpamDocuments = List.length (wordSetsFor 1)
      HamDocuments = List.length (wordSetsFor 0) }

/// その単語を含む文書がスパムである確率。ラプラス平滑化つき。
let wordSpamProbabilityWith (smoothing: float) (model: NaiveBayesClassifier) (word: string) =
    let countIn counts =
        Map.tryFind word counts |> Option.defaultValue 0 |> float

    let spam = countIn model.SpamWordCounts + smoothing
    let ham = countIn model.HamWordCounts + smoothing
    spam / (spam + ham)

/// 平滑化 1.0 での単語のスパム確率。
let wordSpamProbability (model: NaiveBayesClassifier) (word: string) =
    wordSpamProbabilityWith 1.0 model word

/// 事前確率。何も見ないときのスパム率。
let priorSpamProbability (model: NaiveBayesClassifier) =
    if model.TotalDocuments = 0 then
        0.0
    else
        float model.SpamDocuments / float model.TotalDocuments

/// log(0) を避けるためにごくわずかに内側へ丸める。
let private safe (probability: float) =
    let epsilon = 1e-15
    probability |> max epsilon |> min (1.0 - epsilon)

/// 文書がスパムである確率。対数空間で計算する。
let predictProbabilityWith (smoothing: float) (model: NaiveBayesClassifier) (document: string) =
    if model.TotalDocuments = 0 then
        0.0
    else
        let prior = priorSpamProbability model

        // 学習時に見ていない単語は何も語らないので無視する
        let words =
            tokenize document
            |> Set.ofList
            |> Set.filter (fun word -> Set.contains word model.Vocabulary)

        let accumulate (logSpam, logHam) word =
            let spamGivenWord = wordSpamProbabilityWith smoothing model word
            (logSpam + log (safe spamGivenWord), logHam + log (safe (1.0 - spamGivenWord)))

        let logSpam, logHam =
            Set.fold accumulate (log (safe prior), log (safe (1.0 - prior))) words

        // log の差から確率へ戻す（シグモイドと同じ形）
        1.0 / (1.0 + exp (logHam - logSpam |> max -700.0 |> min 700.0))

/// 平滑化 1.0 での文書のスパム確率。
let predictProbability (model: NaiveBayesClassifier) (document: string) =
    predictProbabilityWith 1.0 model document

/// 閾値による 0 / 1 の分類。
let predictWith (threshold: float) (model: NaiveBayesClassifier) (document: string) =
    if predictProbability model document >= threshold then 1 else 0

/// 閾値 0.5 による分類。
let predict (model: NaiveBayesClassifier) (document: string) = predictWith 0.5 model document

/// 正解率。
let accuracy (model: NaiveBayesClassifier) (documents: string list) (labels: int list) =
    List.map2 (fun document label -> if predict model document = label then 1.0 else 0.0) documents labels
    |> List.average
