module GrokkingMl.Tests.Ch08NaiveBayesTests

open Xunit
open GrokkingMl.Ch08NaiveBayes

// 原著と同じ「lottery / sale / winning」を含むスパム判定
let documents =
    [ "lottery sale"
      "lottery winning"
      "winning lottery sale"
      "sale today"
      "meeting tomorrow"
      "project meeting"
      "lunch meeting today"
      "project deadline" ]

let labels = [ 1; 1; 1; 0; 0; 0; 0; 0 ]

let model = train documents labels

[<Fact>]
let ``tokenize は小文字化して分割する`` () =
    Assert.Equal<string list>([ "lottery"; "winning"; "today" ], tokenize "Lottery WINNING today")

[<Fact>]
let ``train はクラスごとの文書数を数える`` () =
    Assert.Equal(3, model.SpamDocuments)
    Assert.Equal(5, model.HamDocuments)
    Assert.Equal(8, model.TotalDocuments)

[<Fact>]
let ``train は同じ単語を 1 文書につき 1 回だけ数える`` () =
    let repeated = train [ "spam spam spam" ] [ 1 ]
    Assert.Equal(1, Map.find "spam" repeated.SpamWordCounts)

[<Fact>]
let ``事前確率はスパム文書の割合`` () =
    Assert.Equal(3.0 / 8.0, priorSpamProbability model, 6)

[<Fact>]
let ``wordSpamProbability はラプラス平滑化を使う`` () =
    // lottery はスパム 3 件、ハム 0 件。平滑化なしなら 1.0 になってしまう
    Assert.Equal(3, Map.find "lottery" model.SpamWordCounts)
    Assert.Equal(None, Map.tryFind "lottery" model.HamWordCounts)
    // (3+1) / ((3+1) + (0+1)) = 0.8
    Assert.Equal(0.8, wordSpamProbability model "lottery", 6)

[<Fact>]
let ``平滑化により確率が両端に張り付かない`` () =
    for word in model.Vocabulary do
        Assert.InRange(wordSpamProbability model word, 1e-9, 1.0 - 1e-9)

[<Fact>]
let ``未知語は五分五分になる`` () =
    Assert.Equal(0.5, wordSpamProbability model "unseen", 6)

[<Fact>]
let ``未知語は予測を変えない`` () =
    Assert.Equal(predictProbability model "lottery", predictProbability model "lottery zzzz qqqq", 6)

[<Fact>]
let ``スパム語は確率を上げる`` () =
    Assert.True(predictProbability model "lottery winning" > 0.5)
    Assert.True(predictProbability model "project deadline" < 0.5)

[<Fact>]
let ``スパム語が重なるほど確信が強まる`` () =
    Assert.True(predictProbability model "lottery winning" > predictProbability model "lottery")

[<Fact>]
let ``確率は 0 から 1 の範囲に収まる`` () =
    for document in documents @ [ ""; "lottery lottery lottery winning sale" ] do
        Assert.InRange(predictProbability model document, 0.0, 1.0)

[<Fact>]
let ``空の文書は事前確率を返す`` () =
    Assert.Equal(priorSpamProbability model, predictProbability model "", 6)

[<Fact>]
let ``predictWith は閾値を使う`` () =
    Assert.Equal(1, predict model "lottery winning")
    Assert.Equal(0, predictWith 0.99 model "lottery winning")

[<Fact>]
let ``分類器は訓練データを分離する`` () =
    Assert.Equal(1.0, accuracy model documents labels, 6)

[<Fact>]
let ``学習していないモデルは 0 を返す`` () =
    let untrained = train [] []
    Assert.Equal(0.0, priorSpamProbability untrained, 6)
    Assert.Equal(0.0, predictProbability untrained "lottery", 6)
