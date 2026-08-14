/// ライブラリ版の実行環境が揃っていることを確認するスモークテスト。
///
/// 章の実装を書く前に、原著の NumPy / pandas / scikit-learn / Keras に対応する
/// ライブラリが実際に学習まで走ることをここで担保する。
module GrokkingMlLib.Tests.EnvironmentTests

open System.IO
open Xunit
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.ML
open Microsoft.ML.Data
open GrokkingMlLib

[<CLIMutable>]
type Sample = { X: float32; [<ColumnName("Label")>] IsLarge: bool }

[<Fact>]
let ``共有データセットを読み込める`` () =
    let frame = Datasets.loadFrame "Hyderabad.csv"
    Assert.Contains("Price", frame.ColumnKeys)
    Assert.True(frame.RowCount > 1000)

[<Fact>]
let ``未登録のデータセットはエラーになる`` () =
    Assert.Throws<FileNotFoundException>(fun () -> Datasets.path "does_not_exist.csv" |> ignore)
    |> ignore

[<Fact>]
let ``MathNet で行列演算ができる`` () =
    let m: Matrix<float> = DenseMatrix.ofRowList [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ]
    let identity = m * m.Inverse()
    Assert.Equal(1.0, identity.[0, 0], 6)
    Assert.Equal(0.0, identity.[0, 1], 6)

[<Fact>]
let ``MathNet で最小二乗の直線を当てはめられる`` () =
    // ML.NET の Ols トレーナーは MKL ネイティブライブラリを要求し、macOS x64 では
    // 動かない。厳密な最小二乗は Math.NET 側で行う方針とする。
    let xs = [| 1.0; 2.0; 3.0; 4.0 |]
    let ys = [| 3.0; 5.0; 7.0; 9.0 |]
    // Fit.Line は構造体タプルを返すので struct パターンで受ける
    let struct (intercept, slope) = MathNet.Numerics.Fit.Line(xs, ys)

    Assert.Equal(2.0, slope, 6)
    Assert.Equal(1.0, intercept, 6)

[<Fact>]
let ``ML_NET のパイプラインで二値分類を学習できる`` () =
    // x > 100 かどうかを当てる、線形分離可能な問題
    let ctx = MLContext(seed = 0)
    let rows = [ for i in 1..200 -> { X = float32 i; IsLarge = i > 100 } ]
    let data = ctx.Data.LoadFromEnumerable rows

    let pipeline =
        EstimatorChain()
            .Append(ctx.Transforms.Concatenate("Features", [| "X" |]))
            .Append(ctx.Transforms.NormalizeMinMax "Features")
            .Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression())

    let model = pipeline.Fit data
    let metrics = ctx.BinaryClassification.Evaluate(model.Transform data)

    Assert.True(metrics.Accuracy > 0.95)

[<Fact>]
let ``Accord で小さなニューラルネットワークが学習できる`` () =
    let inputs = [| [| 0.0; 0.0 |]; [| 1.0; 1.0 |]; [| 0.0; 1.0 |]; [| 1.0; 0.0 |] |]
    let outputs = [| [| 0.0 |]; [| 1.0 |]; [| 0.0 |]; [| 1.0 |] |]
    let network = Accord.Neuro.ActivationNetwork(Accord.Neuro.SigmoidFunction(), 2, 4, 1)
    let teacher = Accord.Neuro.Learning.BackPropagationLearning(network)

    for _ in 1..2000 do
        teacher.RunEpoch(inputs, outputs) |> ignore

    // x1 = 1 のとき 1、x1 = 0 のとき 0 に寄る
    Assert.True(network.Compute([| 1.0; 0.0 |]).[0] > 0.5)
    Assert.True(network.Compute([| 0.0; 1.0 |]).[0] < 0.5)
