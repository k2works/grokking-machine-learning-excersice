/// 原著ノートブック #03 の再現テスト。
///
/// データ生成に乱数を使うので、原著の数値（0.1528 / 0.1037）とは一致しない。
/// 検証するのは **正則化が持つ性質** である。正則化なしでは係数が爆発し、
/// L1・L2 はそれを抑えてテスト誤差を下げる。
///
/// F# は Lasso を座標降下で自前実装しているので、scikit-learn と同じく
/// 係数が厳密なゼロになる。Smile を使う Kotlin 版とはここが違う。
module GrokkingMlLib.Tests.Nb03PolynomialRegularizationTests

open Xunit
open GrokkingMlLib.Nb03PolynomialRegularization

let private dataset = generateDataset SampleSize 0 0.2

[<Fact>]
let ``多項式は係数の添字が次数に対応する`` () =
    // -x^2 + 2 なので x = 0 で 2、x = 1 で 1、x = 2 で -2
    Assert.Equal(2.0, polynomial polynomialCoefficients 0.0, 12)
    Assert.Equal(1.0, polynomial polynomialCoefficients 1.0, 12)
    Assert.Equal(-2.0, polynomial polynomialCoefficients 2.0, 12)

[<Fact>]
let ``データセットは40点で訓練32テスト8に分かれる`` () =
    // 原著の出力
    //   Shape of X_train: (32,)  Shape of X_test: (8,)
    Assert.Equal(40, dataset.X.Length)
    Assert.Equal(32, dataset.XTrain.Length)
    Assert.Equal(8, dataset.XTest.Length)
    Assert.Equal(32, dataset.YTrain.Length)
    Assert.Equal(8, dataset.YTest.Length)

[<Fact>]
let ``生成した点は元の多項式の近くにある`` () =
    // ノイズの標準偏差は 0.1 なので、離れても 0.5 には収まる
    Array.iter2
        (fun x y -> Assert.InRange(y, polynomial polynomialCoefficients x - 0.5, polynomial polynomialCoefficients x + 0.5))
        dataset.X
        dataset.Y

[<Fact>]
let ``多項式特徴量は次数の数だけ列を作る`` () =
    let features = polynomialFeatures dataset.XTrain Degree

    // 定数列は入らない
    Assert.Equal(32, features.RowCount)
    Assert.Equal(Degree, features.ColumnCount)
    // 1 列目は x そのもの、2 列目は x の 2 乗
    Assert.Equal(dataset.XTrain.[0], features.[0, 0], 12)
    Assert.Equal(dataset.XTrain.[0] ** 2.0, features.[0, 1], 12)

[<Fact>]
let ``自前の Lasso は係数を厳密なゼロにする`` () =
    let model = trainPolynomialRegression dataset.XTrain dataset.YTrain Degree L1 0.01

    // 軟判定しきい値関数を通すので、多くの係数がちょうど 0 になる
    let zeros = model.Coefficients |> Array.filter (fun c -> c = 0.0) |> Array.length
    Assert.True(zeros >= Degree / 2, $"zeros={zeros}")

[<Fact>]
let ``L2 正則化は係数をゼロにしないが小さく保つ`` () =
    let model = trainPolynomialRegression dataset.XTrain dataset.YTrain Degree L2 0.01

    Assert.Equal(0, model.Coefficients |> Array.filter (fun c -> c = 0.0) |> Array.length)
    Assert.True(model.Coefficients |> Array.map abs |> Array.max < 10.0)

[<Fact>]
let ``正則化なしは係数が爆発する`` () =
    let model = trainPolynomialRegression dataset.XTrain dataset.YTrain Degree NoRegularization 0.0
    let maxCoefficient = model.Coefficients |> Array.map abs |> Array.max

    // 元の多項式の係数は -1 と 2 だけなのに、桁違いの係数が現れる
    Assert.True(maxCoefficient > 100.0, $"max={maxCoefficient}")

[<Fact>]
let ``正則化はテスト誤差を下げる`` () =
    let evaluate regularization alpha =
        let model = trainPolynomialRegression dataset.XTrain dataset.YTrain Degree regularization alpha
        evaluateModel model dataset.XTest dataset.YTest

    let noReg = evaluate NoRegularization 0.0
    let l1 = evaluate L1 0.01
    let l2 = evaluate L2 0.01

    Assert.True(l1 < noReg, $"l1={l1} noReg={noReg}")
    Assert.True(l2 < noReg, $"l2={l2} noReg={noReg}")

[<Fact>]
let ``正則化ありのモデルは元の多項式に近い予測をする`` () =
    let model = trainPolynomialRegression dataset.XTrain dataset.YTrain Degree L2 0.01

    // 元の多項式 -x^2 + 2 は x = 0 で 2 を返す
    Assert.InRange(model.Predict 0.0, 1.7, 2.3)
