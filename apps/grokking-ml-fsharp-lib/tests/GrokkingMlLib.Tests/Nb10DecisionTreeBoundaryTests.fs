/// 原著ノートブック #10 の再現テスト。
///
/// 自前の CART が scikit-learn とまったく同じ 3 つの分割を選び、
/// 深さ 1 の正解率（10 / 12）と決定境界（x = 5.2 で切り替わる）まで一致した。
module GrokkingMlLib.Tests.Nb10DecisionTreeBoundaryTests

open Xunit
open GrokkingMlLib.Nb10DecisionTreeBoundary

/// 実質無制限の深さ
let private unlimited = 20

[<Fact>]
let ``データセットは12点で半々に分かれる`` () =
    Assert.Equal(12, x0.Length)
    Assert.Equal<string[]>(Array.append (Array.create 6 "0") (Array.create 6 "1"), y)

[<Fact>]
let ``ジニの木は全問正解する`` () =
    // 原著の出力: decision_tree.score(features, labels) -> 1.0
    Assert.Equal(1.0, accuracy (fit Gini unlimited), 12)

[<Fact>]
let ``エントロピーの木も全問正解する`` () =
    // 原著の出力: decision_tree_entropy.score(features, labels) -> 1.0
    Assert.Equal(1.0, accuracy (fit Entropy unlimited), 12)

[<Fact>]
let ``ジニとエントロピーは同じ木になる`` () =
    // 分割の候補に同点が無いので、どちらの指標でも同じ順序で選ばれる。
    // #09 では同点があって木の形が変わったのと対照的
    Assert.Equal(fit Gini unlimited, fit Entropy unlimited)

[<Fact>]
let ``木は scikit-learn と同じ3つの条件で分割する`` () =
    // scikit-learn も x_0 <= 5.0 / x_1 <= 8.0 / x_1 <= 2.5 を選ぶ
    Assert.Equal<(string * float) list>(
        [ "x_0", 5.0; "x_1", 8.0; "x_1", 2.5 ],
        splitConditions (fit Gini unlimited)
    )

[<Fact>]
let ``深さ1の木は1本の直線になる`` () =
    // 原著の「1 本の縦線または横線」。根を深さ 0 と数えるので分割が 1 つ入る
    Assert.Equal<(string * float) list>([ "x_0", 5.0 ], splitConditions (fit Gini 1))

[<Fact>]
let ``深さ1の木は12点中10点しか当てられない`` () =
    // 原著と同じ 0.8333。直線 1 本では 2 点を取り違える
    Assert.Equal(10.0 / 12.0, accuracy (fit Gini 1), 12)

[<Fact>]
let ``深さ0の木は根だけになる`` () =
    // 分割が 1 つも入らない。12 点の半分しか当たらない
    Assert.Empty(splitConditions (fit Gini 0))
    Assert.Equal(0.5, accuracy (fit Gini 0), 12)

[<Fact>]
let ``arange は終端を含まない`` () =
    // NumPy の np.arange と同じ挙動にする
    Assert.Equal<float[]>([| 0.0; 0.5; 1.0; 1.5 |], arange 0.0 2.0 0.5)

[<Fact>]
let ``格子は原著と同じ大きさになる`` () =
    // x は 0 から 10 まで、y は 0 から 11 まで、刻みは 0.2。
    // 終端を含まないので 50 × 55 になる
    let grid = decisionGrid (fit Gini unlimited) PlotStep

    Assert.Equal(0.2, PlotStep, 12)
    Assert.Equal(0.0, grid.XValues.[0], 12)
    Assert.Equal(0.0, grid.YValues.[0], 12)
    Assert.Equal(55, grid.RowCount)
    Assert.Equal(50, grid.ColumnCount)

[<Fact>]
let ``格子の予測は0か1しかない`` () =
    let grid = decisionGrid (fit Gini unlimited) PlotStep
    let distinct = grid.Predictions |> Array.collect id |> Array.distinct |> Array.sort

    Assert.Equal<string[]>([| "0"; "1" |], distinct)

[<Fact>]
let ``境界は軸に平行になる`` () =
    // 決定木の境界は長方形の集まりなので、予測が変わる x 座標は
    // 分割しきい値の直後だけに限られる
    let changes = boundaryColumns (decisionGrid (fit Gini unlimited) PlotStep)

    Assert.Equal(1, changes.Length)
    Assert.Equal(5.2, changes.[0], 9)

[<Fact>]
let ``深さ1の境界は縦線なので全行で同じ`` () =
    let grid = decisionGrid (fit Gini 1) PlotStep

    Assert.All(grid.Predictions, fun row -> Assert.Equal<string[]>(grid.Predictions.[0], row))

[<Fact>]
let ``深い木の境界は行によって変わる`` () =
    // x_1 での分割が入るので、行ごとにパターンが違う
    let grid = decisionGrid (fit Gini unlimited) PlotStep
    let patterns = grid.Predictions |> Array.map List.ofArray |> Array.distinct

    Assert.True(patterns.Length > 1)

[<Fact>]
let ``学習は決定的で毎回同じ格子になる`` () =
    let first = decisionGrid (fit Gini unlimited) PlotStep
    let second = decisionGrid (fit Gini unlimited) PlotStep

    Assert.Equal<string[][]>(first.Predictions, second.Predictions)
