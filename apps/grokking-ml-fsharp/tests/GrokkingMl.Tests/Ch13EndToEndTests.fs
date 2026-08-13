module GrokkingMl.Tests.Ch13EndToEndTests

open System
open Xunit
open GrokkingMl.Ch13EndToEnd

/// 「40 歳未満かつ収入 400 超なら購入」という規則に従う擬似データ。
let makeRows (count: int) (seed: int) : Row list =
    let rng = Random(seed)
    let cities = [ "tokyo"; "osaka"; "kyoto" ]

    List.init count (fun i ->
        let age = rng.Next(18, 71)
        let income = rng.Next(200, 901)
        let city = cities[rng.Next(List.length cities)]

        Map.ofList
            [ // 9 行に 1 行は年齢が欠損している
              "age", (if i % 9 = 0 then "" else string age)
              "income", string income
              "city", city
              "bought", (if age < 40 && income > 400 then "yes" else "no") ])

let rows = makeRows 40 1

[<Fact>]
let ``parseNumber は数値に見えない値を既定値に落とす`` () =
    Assert.Equal(42.0, parseNumber 0.0 "42", 6)
    Assert.Equal(0.0, parseNumber 0.0 "N/A", 6)
    Assert.Equal(-1.0, parseNumber -1.0 "", 6)

[<Fact>]
let ``median は奇数長と偶数長を扱う`` () =
    Assert.Equal(2.0, median [ 3.0; 1.0; 2.0 ], 6)
    Assert.Equal(2.5, median [ 4.0; 1.0; 3.0; 2.0 ], 6)

[<Fact>]
let ``median は空列で 0`` () = Assert.Equal(0.0, median [], 6)

[<Fact>]
let ``imputeMissing は中央値で埋める`` () =
    Assert.Equal<float list>([ 1.0; 2.0; 3.0 ], imputeMissing [ Some 1.0; None; Some 3.0 ])

[<Fact>]
let ``imputeMissing は外れ値に強い`` () =
    // 平均なら 1000 に引きずられるが、中央値なら影響を受けない
    let filled = imputeMissing [ Some 1.0; Some 2.0; Some 3.0; Some 1000.0; None ]
    Assert.Equal(2.5, List.last filled, 6)

[<Fact>]
let ``normalize は 0 から 1 に写す`` () =
    Assert.Equal<float list>([ 0.0; 0.5; 1.0 ], normalize [ 10.0; 20.0; 30.0 ])

[<Fact>]
let ``normalize は定数列で 0 除算しない`` () =
    Assert.Equal<float list>([ 0.0; 0.0; 0.0 ], normalize [ 5.0; 5.0; 5.0 ])

[<Fact>]
let ``oneHot はカテゴリを辞書順に展開する`` () =
    let expanded, categories = oneHot [ "b"; "a"; "b" ]
    Assert.Equal<string list>([ "a"; "b" ], categories)
    Assert.Equal<float list list>([ [ 0.0; 1.0 ]; [ 1.0; 0.0 ]; [ 0.0; 1.0 ] ], expanded)

[<Fact>]
let ``oneHot はちょうど 1 列だけ立てる`` () =
    let expanded, _ = oneHot [ "x"; "y"; "z" ]
    Assert.All(expanded, fun row -> Assert.Equal(1.0, List.sum row, 6))

[<Fact>]
let ``buildDataset は数値の特徴量を作る`` () =
    let dataset = buildDataset "bought" rows

    Assert.Equal<string list>(
        [ "age"; "income"; "city=kyoto"; "city=osaka"; "city=tokyo" ],
        dataset.FeatureNames
    )

    Assert.Equal(List.length rows, List.length dataset.Points)
    Assert.All(dataset.Points, fun point -> Assert.Equal(List.length dataset.FeatureNames, List.length point))

[<Fact>]
let ``buildDataset は数値列を正規化する`` () =
    let dataset = buildDataset "bought" rows

    for index in [ 0; 1 ] do
        let column = dataset.Points |> List.map (fun point -> point[index])
        Assert.Equal(0.0, List.min column, 6)
        Assert.Equal(1.0, List.max column, 6)

[<Fact>]
let ``buildDataset はラベルを 0 と 1 に変換する`` () =
    let dataset = buildDataset "bought" rows
    Assert.Equal<int list>([ 0; 1 ], dataset.Labels |> List.distinct |> List.sort)

    Assert.Equal(
        rows |> List.filter (fun row -> Map.find "bought" row = "yes") |> List.length,
        List.sum dataset.Labels
    )

[<Fact>]
let ``buildDataset は欠損を埋める`` () =
    let dataset = buildDataset "bought" rows
    // 欠損があっても、すべての点が有限の数値で埋まっている
    Assert.All(dataset.Points, fun point -> Assert.All(point, fun value -> Assert.True(Double.IsFinite value)))

[<Fact>]
let ``splitDataset はデータを失わずに分割する`` () =
    let dataset = buildDataset "bought" rows
    let split = splitDataset 0.3 0 dataset

    Assert.Equal(
        List.length dataset.Points,
        List.length split.TrainPoints + List.length split.TestPoints
    )

[<Fact>]
let ``パイプラインは 3 つのモデルを評価する`` () =
    let names = runPipeline "bought" rows |> List.map (fun evaluation -> evaluation.Name)
    Assert.Equal<string list>([ "logistic"; "tree"; "adaboost" ], names)

[<Fact>]
let ``すべての指標は 0 から 1 の割合`` () =
    for evaluation in runPipeline "bought" rows do
        for value in [ evaluation.Accuracy; evaluation.Precision; evaluation.Recall; evaluation.F1; evaluation.Auc ] do
            Assert.InRange(value, 0.0, 1.0)

[<Fact>]
let ``どのモデルも当てずっぽうには勝つ`` () =
    for evaluation in runPipeline "bought" rows do
        Assert.True(evaluation.Auc > 0.5, $"{evaluation.Name} auc={evaluation.Auc}")

[<Fact>]
let ``bestByF1 は F1 が最大のモデルを選ぶ`` () =
    let evaluations =
        [ { Name = "a"; Accuracy = 0.9; Precision = 0.5; Recall = 0.5; F1 = 0.5; Auc = 0.9 }
          { Name = "b"; Accuracy = 0.7; Precision = 0.8; Recall = 0.8; F1 = 0.8; Auc = 0.7 } ]

    Assert.Equal("b", (bestByF1 evaluations).Name)

[<Fact>]
let ``パイプラインは再現可能`` () =
    Assert.Equal<Evaluation list>(runPipeline "bought" rows, runPipeline "bought" rows)
