module GrokkingMl.Tests.Ch04RegularizationTests

open Xunit
open GrokkingMl.Ch04Regularization

// y = 2x + 3 に小さなノイズを乗せたデータ
let features = [ -1.5; -1.2; -0.9; -0.6; -0.3; 0.0; 0.3; 0.6; 0.9; 1.2 ]
let labels = [ 0.08; 0.32; 1.07; 1.63; 2.54; 3.11; 3.84; 3.95; 4.75; 5.12 ]

let split () = trainTestSplit 0.3 0 features labels

[<Fact>]
let ``polynomialFeatures は x の累乗を並べる`` () =
    Assert.Equal<float list>([ 2.0; 4.0; 8.0 ], polynomialFeatures 2.0 3)

[<Fact>]
let ``predict はすべての次数を使う`` () =
    // 1 + 2*2 + 3*4 = 17
    Assert.Equal(17.0, predict { Weights = [ 2.0; 3.0 ]; Bias = 1.0 } 2.0, 6)

[<Fact>]
let ``L1 の勾配は重みの符号を使う`` () =
    Assert.Equal(0.1, regularizationGradient 5.0 L1 0.1, 6)
    Assert.Equal(-0.1, regularizationGradient -5.0 L1 0.1, 6)
    Assert.Equal(0.0, regularizationGradient 0.0 L1 0.1, 6)

[<Fact>]
let ``L2 の勾配は重みに比例する`` () =
    Assert.Equal(1.0, regularizationGradient 5.0 L2 0.1, 6)
    Assert.Equal(-1.0, regularizationGradient -5.0 L2 0.1, 6)

[<Fact>]
let ``正則化なしの勾配は 0`` () =
    Assert.Equal(0.0, regularizationGradient 5.0 NoRegularization 0.1, 6)

[<Fact>]
let ``squareTrick は正則化なしで誤差に比例して動く`` () =
    // 予測 = 1 + 2*3 = 7、誤差 = 10 - 7 = 3
    let moved = squareTrick 0.01 NoRegularization 0.0 { Weights = [ 2.0 ]; Bias = 1.0 } 3.0 10.0
    Assert.Equal(2.09, moved.Weights[0], 6) // 2 + 0.01 * 3 * 3
    Assert.Equal(1.03, moved.Bias, 6) // 1 + 0.01 * 3

[<Fact>]
let ``squareTrick は L2 で重みを 0 に引き戻す`` () =
    let model = { Weights = [ 2.0 ]; Bias = 1.0 }
    let plain = squareTrick 0.01 NoRegularization 0.0 model 3.0 10.0
    let regularized = squareTrick 0.01 L2 0.1 model 3.0 10.0
    Assert.True(regularized.Weights[0] < plain.Weights[0])
    // 2 + 0.01 * (3*3 - 0.1*2*2)
    Assert.Equal(2.086, regularized.Weights[0], 6)

[<Fact>]
let ``squareTrick はバイアスを正則化しない`` () =
    let model = { Weights = [ 2.0 ]; Bias = 1.0 }
    let plain = squareTrick 0.01 NoRegularization 0.0 model 3.0 10.0
    let regularized = squareTrick 0.01 L2 0.1 model 3.0 10.0
    Assert.Equal(plain.Bias, regularized.Bias, 6)

[<Fact>]
let ``trainTestSplit はデータを失わずに分割する`` () =
    let s = split ()
    Assert.Equal(3, List.length s.TestFeatures)
    Assert.Equal(7, List.length s.TrainFeatures)
    Assert.Equal<float list>(List.sort features, List.sort (s.TrainFeatures @ s.TestFeatures))
    Assert.Equal<float list>(List.sort labels, List.sort (s.TrainLabels @ s.TestLabels))

[<Fact>]
let ``trainTestSplit は特徴量とラベルの対応を保つ`` () =
    let s = split ()

    List.zip (s.TrainFeatures @ s.TestFeatures) (s.TrainLabels @ s.TestLabels)
    |> List.iter (fun (x, y) -> Assert.Equal(labels[List.findIndex ((=) x) features], y, 6))

[<Fact>]
let ``1 次モデルは直線的なデータに適合する`` () =
    let model = train 1 NoRegularization 0.0 features labels
    Assert.InRange(model.Weights[0], 1.7, 2.3)
    Assert.InRange(model.Bias, 2.7, 3.3)
    Assert.True(modelRmse model features labels < 0.3)

[<Fact>]
let ``高次モデルは過学習する`` () =
    let s = split ()
    let simple = train 1 NoRegularization 0.0 s.TrainFeatures s.TrainLabels
    let complex = train 5 NoRegularization 0.0 s.TrainFeatures s.TrainLabels

    let gap (model: PolynomialModel) =
        modelRmse model s.TestFeatures s.TestLabels
        - modelRmse model s.TrainFeatures s.TrainLabels

    // 次数を上げると訓練誤差は下がる
    let simpleTrainError = modelRmse simple s.TrainFeatures s.TrainLabels
    let complexTrainError = modelRmse complex s.TrainFeatures s.TrainLabels
    Assert.True(complexTrainError < simpleTrainError)
    // しかし汎化ギャップ（テスト誤差 - 訓練誤差）は広がる
    Assert.True(gap complex > gap simple, $"simple={gap simple} complex={gap complex}")

[<Fact>]
let ``L2 正則化はテスト誤差を改善する`` () =
    let s = split ()
    let plain = train 5 NoRegularization 0.0 s.TrainFeatures s.TrainLabels
    let regularized = train 5 L2 0.01 s.TrainFeatures s.TrainLabels

    let plainTestError = modelRmse plain s.TestFeatures s.TestLabels
    let regularizedTestError = modelRmse regularized s.TestFeatures s.TestLabels
    Assert.True(regularizedTestError < plainTestError)

[<Fact>]
let ``L1 正則化は不要な重みをほぼ 0 にする`` () =
    let s = split ()
    let plain = train 5 NoRegularization 0.0 s.TrainFeatures s.TrainLabels
    let regularized = train 5 L1 0.01 s.TrainFeatures s.TrainLabels

    Assert.True(weightMagnitude regularized < weightMagnitude plain)
    Assert.True(regularized.Weights |> List.filter (fun w -> abs w < 5e-3) |> List.length >= 2)
    Assert.Empty(plain.Weights |> List.filter (fun w -> abs w < 5e-3))
