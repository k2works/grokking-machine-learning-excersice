/// 原著ノートブック #09 の再現テスト。
///
/// 自前の CART は同点の分割候補から **列の順に最初のもの** を選ぶ。
/// その規則で、数値版は原著とまったく同じ木（`Age <= 20.0` → `Platform_iPhone <= 0.5`）になった。
module GrokkingMlLib.Tests.Nb09AppRecommendationsTests

open Xunit
open GrokkingMlLib.Nb09AppRecommendations

[<Fact>]
let ``元データは6人ぶん`` () =
    Assert.Equal(6, platforms.Length)
    Assert.Equal<int[]>([| 15; 25; 32; 35; 12; 14 |], ages)
    Assert.Equal(3, (apps |> Array.distinct).Length)

[<Fact>]
let ``候補のしきい値は隣り合う値の中点`` () =
    // 15 と 25 の間なら 20。scikit-learn も同じ規則
    Assert.Equal<float[]>([| 13.0; 14.5; 20.0; 28.5; 33.5 |], candidateThresholds (Array.map float ages))

[<Fact>]
let ``ジニ不純度は8章と同じ定義`` () =
    Assert.Equal(0.0, gini [| "A"; "A" |], 12)
    Assert.Equal(0.5, gini [| "A"; "B" |], 12)
    Assert.Equal(0.0, gini [||], 12)

[<Fact>]
let ``数値版は原著とまったく同じ木になる`` () =
    // 原著の出力
    //   X[0] <= 20.0 （X[0] は Age）
    //   X[1] <= 0.5  （X[1] は Platform_iPhone）
    let tree = fit numericDataset

    Assert.Equal<(string * float) list>(
        [ "Age", 20.0; "Platform_iPhone", 0.5 ],
        splits tree
    )

[<Fact>]
let ``カテゴリ版の根も年齢で分割する`` () =
    // 原著は Age_Young、こちらは裏返しの Age_Adult を選ぶ。分割としては同じ
    let rootSplit = splits (fit categoricalDataset) |> List.head

    Assert.Contains(fst rootSplit, [ "Age_Young"; "Age_Adult" ])
    Assert.Equal(0.5, snd rootSplit, 12)

[<Fact>]
let ``木は分割2つと葉3つになる`` () =
    // 原著の木も 5 節（内部 2 + 葉 3）
    for dataset in [ numericDataset; categoricalDataset ] do
        let tree = fit dataset

        Assert.Equal(2, (splits tree).Length)
        Assert.Equal(3, (leafPredictions tree).Length)

[<Fact>]
let ``葉は3つのアプリをそれぞれ予測する`` () =
    // 全問正解するので、3 つの葉が 3 つのアプリに 1 対 1 で対応する
    Assert.Equal<string list>(
        [ "Atom Count"; "Beehive Finder"; "Check Mate Mate" ],
        leafPredictions (fit numericDataset) |> List.sort
    )

[<Fact>]
let ``両方の版が全問正解する`` () =
    // 原著の出力: どちらも score 1.0
    Assert.Equal(1.0, accuracy (fit numericDataset) numericDataset, 12)
    Assert.Equal(1.0, accuracy (fit categoricalDataset) categoricalDataset, 12)

[<Fact>]
let ``学習は決定的で毎回同じ木になる`` () =
    // scikit-learn は同点の候補から無作為に選ぶため木の形が変わるが、
    // 列の順に最初のものを選ぶ規則にしたので毎回同じになる
    Assert.Equal(fit numericDataset, fit numericDataset)

[<Fact>]
let ``判別共用体が木の形をそのまま表す`` () =
    // Node と Leaf のパターンマッチで構造を直接確かめられる
    match fit numericDataset with
    | Node("Age", 20.0, Leaf "Atom Count", Node("Platform_iPhone", 0.5, right, left)) ->
        Assert.Equal(Leaf "Beehive Finder", right)
        Assert.Equal(Leaf "Check Mate Mate", left)
    | other -> failwith $"想定と違う木になった: {other}"

[<Fact>]
let ``未知の組み合わせも予測できる`` () =
    // 学習データに無い「Android の 20 歳」を投げてみる
    let tree = fit numericDataset
    let prediction = predict tree numericDataset [| 20.0; 0.0; 1.0 |]

    Assert.Contains(prediction, apps)
