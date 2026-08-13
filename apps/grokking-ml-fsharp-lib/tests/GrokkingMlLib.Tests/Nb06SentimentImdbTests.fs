/// 原著ノートブック #06 の再現テスト。
///
/// IMDB のデータセットは約 63 MB あり、リポジトリに含めていない。
/// **テストからは自動ダウンロードしない**。未取得なら該当テストを飛ばす。
///
/// 語彙の作り方そのものは、小さなコーパスで常に検証している。
module GrokkingMlLib.Tests.Nb06SentimentImdbTests

open System
open System.IO
open Xunit
open GrokkingMlLib
open GrokkingMlLib.Nb06SentimentImdb

/// 語彙の作り方を確かめるための小さなコーパス
let private tinyReviews =
    { Texts =
        [| "This movie was wonderful and the acting was superb"
           "A dreadful waste of time, truly awful acting"
           "Wonderful direction and a superb cast"
           "Awful script and a dreadful waste" |]
      Sentiments = [| 1; 0; 1; 0 |] }

let private stopWords = englishStopWords ()

let private imdbAvailable () =
    File.Exists(Path.Combine(Datasets.directory (), "IMDB_Dataset.csv"))

[<Fact>]
let ``ストップワードは318語ある`` () =
    // scikit-learn 1.9.0 の ENGLISH_STOP_WORDS から書き出したもの
    Assert.Equal(318, stopWords.Count)
    Assert.Contains("the", stopWords)
    Assert.Contains("and", stopWords)
    Assert.DoesNotContain("movie", stopWords)

[<Fact>]
let ``ベクトル化は2文字以上の単語だけを拾う`` () =
    // トークン正規表現は \b\w\w+\b なので 1 文字の語は落ちる
    let tokens = tokenize stopWords "a wonderful movie"

    Assert.Contains("wonderful", tokens)
    Assert.DoesNotContain("a", tokens)

[<Fact>]
let ``ベクトル化は小文字に揃える`` () =
    Assert.Equal<string list>(
        [ "wonderful"; "wonderful"; "wonderful" ],
        tokenize stopWords "Wonderful WONDERFUL wonderful"
    )

[<Fact>]
let ``ストップワードはトークンから除かれる`` () =
    let tokens = tokenize stopWords "the movie and the acting"

    Assert.Contains("movie", tokens)
    Assert.Contains("acting", tokens)
    Assert.DoesNotContain("the", tokens)
    Assert.DoesNotContain("and", tokens)

[<Fact>]
let ``maxFeatures は出現回数の上位を採る`` () =
    // rare は 1 回、common は 3 回、middle は 2 回
    let vocabulary =
        fitVocabulary stopWords 2 [ "common middle rare"; "common middle"; "common" ]

    Assert.Equal<string[]>([| "common"; "middle" |], vocabulary.Words)

[<Fact>]
let ``語彙は辞書順に並ぶ`` () =
    // scikit-learn の get_feature_names_out() と同じ。添字が特徴量の添字になる
    let vocabulary =
        fitVocabulary stopWords 50 [ "zebra apple mango apple zebra apple" ]

    Assert.Equal<string[]>(Array.sortWith (fun a b -> String.CompareOrdinal(a, b)) vocabulary.Words, vocabulary.Words)

[<Fact>]
let ``出現回数ベクトルは非ゼロ要素だけを持つ`` () =
    let vocabulary = fitVocabulary stopWords 50 [ "apple apple zebra"; "zebra" ]
    let first = transformOne stopWords vocabulary "apple apple zebra"
    let second = transformOne stopWords vocabulary "zebra"
    let appleIndex = Array.findIndex ((=) "apple") vocabulary.Words

    Assert.Equal(2.0, first |> List.find (fst >> (=) appleIndex) |> snd, 12)
    // 疎表現なので 0 の要素は保持されない
    Assert.Equal(1, second.Length)

[<Fact>]
let ``小さなコーパスでも単語の重みが感情を反映する`` () =
    let model = fit tinyReviews 20 500
    let weights = model.WordSentiments() |> Map.ofArray

    // 肯定的なレビューにだけ出る語は正、否定的なレビューにだけ出る語は負
    Assert.True(weights.["wonderful"] > 0.0)
    Assert.True(weights.["superb"] > 0.0)
    Assert.True(weights.["dreadful"] < 0.0)
    Assert.True(weights.["waste"] < 0.0)
    // 両方に出る語は 0 に近い
    Assert.True(abs weights.["acting"] < abs weights.["wonderful"])

[<Fact>]
let ``係数の数は語彙の数と一致する`` () =
    let model = fit tinyReviews 20 100

    Assert.Equal(model.Vocabulary.Size, model.Coefficients.Length)
    Assert.Equal(model.Vocabulary.Size, (model.WordSentiments()).Length)

[<Fact>]
let ``IMDB の語彙は原著と同じ添字になる`` () =
    if imdbAvailable () then
        let reviews = loadReviews ()
        let vocabulary = fitVocabulary stopWords MaxFeatures reviews.Texts

        Assert.Equal(50000, reviews.Texts.Length)
        Assert.Equal(MaxFeatures, vocabulary.Size)
        // 自前のベクトル化が scikit-learn と同じ語彙を作れている証拠。
        // 原著の出力に現れる添字（1964 wonderfully / 1921 waste）と一致する
        Assert.Equal(1964, Array.findIndex ((=) "wonderfully") vocabulary.Words)
        Assert.Equal(1921, Array.findIndex ((=) "waste") vocabulary.Words)

[<Fact>]
let ``IMDB では感情語の符号が正しく分かれる`` () =
    if imdbAvailable () then
        // 自前の勾配降下は lbfgs ほど速く収束しない。300 エポックでは
        // 上位に出るのが原著の wonderfully / funniest ではなく excellent /
        // perfect のような頻出語になる。符号の向きは正しく学習できている
        let model = fit (loadReviews ()) MaxFeatures 300
        let weights = model.WordSentiments() |> Map.ofArray

        for word in [ "excellent"; "wonderful"; "superb"; "brilliant" ] do
            Assert.True(weights.[word] > 0.0, $"{word} は正のはず")

        for word in [ "waste"; "worst"; "awful"; "boring" ] do
            Assert.True(weights.[word] < 0.0, $"{word} は負のはず")
