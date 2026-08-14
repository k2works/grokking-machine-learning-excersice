/// 原著ノートブック #07 の再現テスト。
///
/// この回は原著が scikit-learn を使わないので、**すべての数値を再現できる**。
/// `np.compat.long` による切り捨てまで写しているので 16 桁まで一致する。
///
/// `emails.csv` は約 8.5 MB あり、リポジトリに含めていない。
/// テストからは自動ダウンロードせず、未取得ならスキップする。
module GrokkingMlLib.Tests.Nb07NaiveBayesTests

open System.IO
open Xunit
open GrokkingMlLib
open GrokkingMlLib.Nb07NaiveBayes

/// 数え方を確かめるための小さなコーパス
let private tinyEmails =
    { Texts = [| "win lottery now"; "lottery lottery lottery"; "meeting at noon"; "lunch meeting" |]
      Spam = [| 1; 1; 0; 0 |] }

let private emailsAvailable () =
    File.Exists(Path.Combine(Datasets.directory (), "emails.csv"))

[<Fact>]
let ``メールは小文字の単語集合になる`` () =
    // 原著は list(set(text.lower().split()))。同じ単語は 1 回しか数えない
    Assert.Equal<Set<string>>(set [ "lottery"; "the"; "win" ], processEmail "Win WIN the lottery")

[<Fact>]
let ``空白の連続はまとめて区切る`` () =
    // Python の split() は空白の連続をまとめる。Regex.Split は空文字列を残すので落とす
    Assert.Equal<Set<string>>(set [ "win"; "lottery" ], processEmail "win   lottery")

[<Fact>]
let ``出現数は1から数え始める`` () =
    // ラプラス平滑化。一度も見ていない側の確率が 0 にならないようにする
    Assert.Equal({ Spam = 1; Ham = 1 }, initialCounts)

[<Fact>]
let ``小さなコーパスで出現通数を数える`` () =
    let trained = train tinyEmails

    // lottery はスパム 2 通に出る。1 から数え始めるので Spam は 3
    Assert.Equal({ Spam = 3; Ham = 1 }, trained.Words.["lottery"])
    Assert.Equal({ Spam = 1; Ham = 3 }, trained.Words.["meeting"])

[<Fact>]
let ``同じ単語が何度出ても1通と数える`` () =
    // 2 通目は "lottery lottery lottery" だが、1 通ぶんしか数えない
    Assert.Equal(3, (train tinyEmails).Words.["lottery"].Spam)

[<Fact>]
let ``語彙にない単語は事前確率を返す`` () =
    // スパム 2 通 / 全 4 通
    Assert.Equal(0.5, predictNaiveBayes (train tinyEmails) "zzzz", 12)

[<Fact>]
let ``データセットは原著と同じ規模`` () =
    if emailsAvailable () then
        let model = train (loadEmails ())

        // 原著の出力
        //   Number of emails: 5728 / Number of spam emails: 1368
        //   Probability of spam: 0.2388268156424581
        Assert.Equal(5728, model.Corpus.Total)
        Assert.Equal(1368, model.Corpus.Spam)
        Assert.Equal(0.2388268156424581, model.Corpus.SpamProbability, 15)

[<Fact>]
let ``単語ごとの出現数と予測は原著と同じ`` () =
    if emailsAvailable () then
        let model = train (loadEmails ())

        // 原著の出力
        //   model['lottery'] -> {'spam': 9, 'ham': 1}
        //   model['sale']    -> {'spam': 39, 'ham': 42}
        //   predict_bayes('lottery') -> 0.9 / ('sale') -> 0.48148148148148145
        Assert.Equal({ Spam = 9; Ham = 1 }, model.Words.["lottery"])
        Assert.Equal({ Spam = 39; Ham = 42 }, model.Words.["sale"])
        Assert.Equal(0.9, predictBayes model "lottery", 15)
        Assert.Equal(0.48148148148148145, predictBayes model "sale", 15)

[<Fact>]
let ``メール全体の予測は原著と同じ`` () =
    if emailsAvailable () then
        let model = train (loadEmails ())

        // 原著のセル出力をそのまま期待値にしている
        let expected =
            [ "lottery sale", 0.9638144992048691
              "Hi mom how are you", 0.12554358867164464
              "meet me at the lobby of the hotel at nine am", 6.964603508395961e-05
              "enter the lottery to win three million dollars", 0.9995234218677428
              "buy cheap lottery easy money now", 0.999973472265966
              "Grokking Machine Learning by Luis Serrano", 0.4197107645488719
              "asdfgh", 0.2388268156424581 ]

        for (email, value) in expected do
            let actual = predictNaiveBayes model email
            Assert.True(abs (actual - value) <= abs value * 1e-14, $"{email}: {actual} <> {value}")

[<Fact>]
let ``知らない単語を足しても結果は変わらない`` () =
    if emailsAvailable () then
        let model = train (loadEmails ())

        Assert.Equal(
            predictNaiveBayes model "Hi mom how are you",
            predictNaiveBayes model "Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd",
            15
        )

[<Fact>]
let ``切り捨てを外すと原著と食い違う`` () =
    if emailsAvailable () then
        let model = train (loadEmails ())
        let corpus = model.Corpus

        // 切り捨てなしで同じ計算をすると 0.9638144470140118 になり、
        // 原著の 0.9638144992048691 と 8 桁目から分かれる
        let spams, hams =
            [ "lottery"; "sale" ]
            |> List.fold
                (fun (s, h) word ->
                    let counts = model.Words.[word]
                    s * (float counts.Spam / float corpus.Spam * float corpus.Total),
                    h * (float counts.Ham / float corpus.Ham * float corpus.Total))
                (float corpus.Spam, float corpus.Ham)

        let withoutTruncation = spams / (spams + hams)

        Assert.Equal(0.9638144470140118, withoutTruncation, 14)
        Assert.True(abs (withoutTruncation - 0.9638144992048691) > 1e-9)
