/// 原著ノートブック #22 の再現テスト。
///
/// **前処理と分割の件数は原著と完全に一致** する。
/// モデルの成績は、分割の中身が違うので一致しない（`split` の説明を参照）。
module GrokkingMlLib.Tests.Nb22TitanicEndToEndTests

open Xunit
open GrokkingMlLib.Nb22TitanicEndToEnd

let private raw = loadRaw ()
let private prepared = preprocess (clean raw)
let private data = split prepared 100

[<Fact>]
let ``データセットは891行`` () =
    // 原著の出力: The dataset has 891 rows
    Assert.Equal(891, raw.Length)

[<Fact>]
let ``生存者は342人`` () =
    // 原著の出力: 342 passengers survived out of 891
    Assert.Equal(342, raw |> Array.filter (fun p -> p.Survived = 1) |> Array.length)

[<Fact>]
let ``欠損の数は原著と一致する`` () =
    // 原著の isna().sum() の出力
    let counts = missingCounts raw

    Assert.Equal(177, counts.["Age"])
    Assert.Equal(687, counts.["Cabin"])
    Assert.Equal(2, counts.["Embarked"])

[<Fact>]
let ``年齢の中央値は28`` () =
    // 原著の出力: 28.0
    Assert.Equal(28.0, medianAge raw, 15)

[<Fact>]
let ``前処理後は欠損がなくなる`` () =
    let cleaned = clean raw

    Assert.All(cleaned, fun p -> Assert.True(p.Age.IsSome && p.Embarked.IsSome))
    // Embarked の欠損は U という新しい区分になる
    Assert.Equal(2, cleaned |> Array.filter (fun p -> p.Embarked = Some "U") |> Array.length)

[<Fact>]
let ``年齢の区切りは10歳刻みで8区間`` () =
    Assert.Equal<int[]>([| 0; 10; 20; 30; 40; 50; 60; 70; 80 |], ageBins)
    Assert.Equal(8, ageBins.Length - 1)

[<Fact>]
let ``年齢の区間は左を開き右を閉じる`` () =
    // pandas.cut の既定。10 歳ちょうどは (0, 10] に入る
    Assert.Equal(0, ageBin 10.0)
    Assert.Equal(1, ageBin 10.1)
    Assert.Equal(2, ageBin 28.0)

[<Fact>]
let ``特徴量は20列になる`` () =
    // SibSp + Parch + Fare + Sex 2 + Embarked 4 + Pclass 3 + 年齢 8。
    // Python 版はここに Survived を含めて 21 列と数えている
    Assert.Equal(20, featureNames.Length)
    Assert.Equal(20, prepared.FeatureCount)

[<Fact>]
let ``one_hotは各群でちょうど1つだけ立つ`` () =
    let row = featuresOf (clean raw).[0]

    let sumOf (prefix: string) =
        featureNames
        |> Array.mapi (fun i name -> if name.StartsWith prefix then row.[i] else 0.0)
        |> Array.sum

    Assert.Equal(1.0, sumOf "Sex_", 15)
    Assert.Equal(1.0, sumOf "Embarked_", 15)
    Assert.Equal(1.0, sumOf "Pclass_", 15)
    Assert.Equal(1.0, sumOf "Categorized_age_", 15)

[<Fact>]
let ``分割の件数は原著と一致する`` () =
    // 原著の出力: 534 / 178 / 179。
    // scikit-learn は test_size の側を切り上げる
    Assert.Equal(534, data.Train.Size)
    Assert.Equal(178, data.Validation.Size)
    Assert.Equal(179, data.Test.Size)

[<Fact>]
let ``分割は全体を覆う`` () =
    Assert.Equal(891, data.Train.Size + data.Validation.Size + data.Test.Size)

[<Fact>]
let ``学習できるのは3モデル`` () =
    // 原著は 7 つだが、木系 4 つは自前実装を 20 列に広げていないので外した
    Assert.Equal<string list>(
        [ "Logistic regression"; "Naive Bayes"; "SVM" ],
        fitAll data |> List.map fst
    )

[<Fact>]
let ``ロジスティック回帰と素朴ベイズは多数派より良い`` () =
    // 「全員死亡」と答えると 0.551
    let baseline = majorityBaseline data.Validation.Y
    let scores = fitAll data |> List.map (fun (name, predict) -> name, accuracy (predict data.Validation.X) data.Validation.Y)

    for name in [ "Logistic regression"; "Naive Bayes" ] do
        Assert.True(scores |> List.exists (fun (n, s) -> n = name && s > baseline))

[<Fact>]
let ``SVMは正解率とF1の差が大きい`` () =
    // 原著も同じ傾向（正解率 0.680 に対し F1 は 0.400）。
    // 多数派に寄せた予測をしているので、正解率だけでは見抜けない
    let _, predict = fitAll data |> List.find (fun (name, _) -> name = "SVM")
    let predicted = predict data.Validation.X

    Assert.True(
        accuracy predicted data.Validation.Y - f1Score predicted data.Validation.Y > 0.25
    )

[<Fact>]
let ``F1スコアの定義を確かめる`` () =
    // 適合率 1.0・再現率 0.5 なら F1 は 2/3
    Assert.Equal(2.0 / 3.0, f1Score [| 1; 0; 0; 0 |] [| 1; 1; 0; 0 |], 15)
    // 1 つも当てられなければ 0
    Assert.Equal(0.0, f1Score [| 0; 0 |] [| 1; 1 |], 15)

[<Fact>]
let ``同じ種なら分割は再現する`` () =
    // Accord と違い、System.Random は種を渡せば決定的
    let first = split prepared 100
    let second = split prepared 100

    Assert.Equal<int[]>(first.Validation.Y, second.Validation.Y)
