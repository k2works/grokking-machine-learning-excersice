/// 原著ノートブック #13 の再現テスト。
///
/// Accord.Neuro には ReLU も softmax も Dropout も無い。
/// さらに **重みの初期化を自分で呼ばないと学習が始まらない**。
/// 初期化さえすれば、原著より良い正解率（1.0）に届く — つまり過学習する。
module GrokkingMlLib.Tests.Nb13NeuralNetworkBoundaryTests

open Xunit
open GrokkingMlLib.Nb13NeuralNetworkBoundary

let private data = loadCircle ()

[<Fact>]
let ``データセットは110点`` () =
    Assert.Equal(110, data.Size)
    Assert.Equal(2, data.X.[0].Length)

[<Fact>]
let ``ラベルは偏っている`` () =
    // 84 対 26。円の内側が少数派になる
    Assert.Equal(26, Array.sum data.Y)
    Assert.Equal(84, data.Size - Array.sum data.Y)

[<Fact>]
let ``ネットワークの隠れ層は原著と同じ`` () =
    // Dense(128) と Dense(64)
    Assert.Equal<int[]>([| 128; 64 |], hiddenUnits)

[<Fact>]
let ``重みを初期化しないと学習が始まらない`` () =
    // NguyenWidrow を呼ばないと、何エポック回しても常に同じクラスを答える。
    // Keras は層を作った時点で自動的に初期化するので、移植のときに見落としやすい
    let network = fitWithoutInitialization data 1000 0

    Assert.Equal(26.0 / 110.0, accuracy network data, 12)

[<Fact>]
let ``重みを初期化すれば完全に分類できる`` () =
    // 原著（Keras・Dropout あり）は 0.88 前後。Dropout が無いぶん過学習して 1.0 になる
    Assert.Equal(1.0, accuracy (fit data Epochs 0) data, 12)

[<Fact>]
let ``原著の100エポックでも多数派より良くなる`` () =
    // 初期化さえしてあれば、少ない回数でも学習は進む
    Assert.True(accuracy (fit data OriginalEpochs 0) data > 84.0 / 110.0)

[<Fact>]
let ``arange は終端を含まない`` () =
    Assert.Equal<float[]>([| 0.0; 0.5; 1.0; 1.5 |], arange 0.0 2.0 0.5)

[<Fact>]
let ``格子の予測は0か1しかない`` () =
    let grid = decisionGrid (fit data Epochs 0) data PlotStep
    let distinct = grid |> Array.collect id |> Array.distinct |> Array.sort

    Assert.Equal<int[]>([| 0; 1 |], distinct)

[<Fact>]
let ``境界は行ごとに変わる`` () =
    // 決定木（#10）の境界は軸に平行で、切り替わる位置が行によらなかった。
    // ニューラルネットワークは曲線を引けるので、行ごとに変わる
    let changes = boundaryChangesPerRow (decisionGrid (fit data Epochs 0) data PlotStep)

    Assert.True((changes |> List.distinct |> List.length) > 1, $"changes={changes}")

[<Fact>]
let ``境界は閉じた形になる`` () =
    // 円形のデータなので、内側を囲む境界ができる。
    // 少なくとも 1 行は「外・内・外」と 2 回切り替わる
    let changes = boundaryChangesPerRow (decisionGrid (fit data Epochs 0) data PlotStep)

    Assert.True(List.max changes >= 2, $"changes={changes}")

[<Fact>]
let ``種を固定すれば同じ結果になる`` () =
    // Accord は重みの初期化に乱数を使う。Generator.Seed で固定できる
    Assert.Equal(accuracy (fit data 200 7) data, accuracy (fit data 200 7) data, 12)
