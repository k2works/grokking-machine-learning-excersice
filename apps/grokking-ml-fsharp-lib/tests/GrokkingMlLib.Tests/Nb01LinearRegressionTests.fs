/// 原著ノートブック #01 の再現テスト。
///
/// 閉じた式で解く `fitWithMathNet` は原著の数値と完全に一致する。トリックによる
/// 学習は乱数列が Python と違うので、収束先で検証する。
module GrokkingMlLib.Tests.Nb01LinearRegressionTests

open System
open Xunit
open GrokkingMlLib.Nb01LinearRegression

[<Fact>]
let ``データセットは原著と同じ`` () =
    Assert.Equal<int[]>([| 1; 2; 3; 5; 6; 7 |], features)
    Assert.Equal<int[]>([| 155; 197; 244; 356; 407; 448 |], labels)

[<Fact>]
let ``二乗トリックは誤差に比例して動く`` () =
    // 予測 0 + 1 * 2 = 2、実測 10 なので誤差は 8
    let line = squareTrick 0.01 { PricePerRoom = 1.0; BasePrice = 0.0 } 2.0 10.0

    Assert.Equal(1.0 + 0.01 * 2.0 * 8.0, line.PricePerRoom, 12)
    Assert.Equal(0.0 + 0.01 * 8.0, line.BasePrice, 12)

[<Fact>]
let ``絶対トリックは誤差の大きさに依存しない`` () =
    let start = { PricePerRoom = 1.0; BasePrice = 0.0 }
    let small = absoluteTrick 0.01 start 2.0 10.0
    let large = absoluteTrick 0.01 start 2.0 802.0

    // レコードは構造的等価性を持つのでそのまま比べられる
    Assert.Equal(small, large)

[<Fact>]
let ``シンプルトリックは予測を実測へ近づける`` () =
    let line = simpleTrick (Random 0) { PricePerRoom = 1.0; BasePrice = 0.0 } 2.0 10.0

    Assert.True(line.PricePerRoom > 1.0)
    Assert.True(line.BasePrice > 0.0)

[<Fact>]
let ``rmse は誤差の二乗平均平方根`` () =
    // ラベル 1, 2, 3 に対して予測は一律 2。差は -1, 0, 1
    Assert.Equal(sqrt (2.0 / 3.0), rmse [| 1; 2; 3 |] 2.0, 12)

[<Fact>]
let ``学習の途中経過をエポック数だけ記録する`` () =
    let result = linearRegression features labels 0.01 50 Square 0

    Assert.Equal(50, result.History.Length)
    Assert.Equal(50, result.Errors.Length)
    Assert.True(List.last result.Errors < List.head result.Errors)

[<Fact>]
let ``二乗トリックは最小二乗解へ収束する`` () =
    // 乱数列が Python と違うので原著の 51.044 / 91.594 には一致しない。
    // 閉じた式の解に十分近づくことで検証する
    let exact = fitWithMathNet features labels
    let result = linearRegression features labels 0.01 10000 Square 0

    Assert.Equal(exact.PricePerRoom, result.Line.PricePerRoom, 1.0)
    Assert.Equal(exact.BasePrice, result.Line.BasePrice, 1.0)

[<Fact>]
let ``MathNet の解は原著と同じ数値になる`` () =
    // 原著 scikit-learn の出力
    //   Coefficient: [50.39285714]
    //   Intercept: 99.59523809523819
    let line = fitWithMathNet features labels

    Assert.Equal(50.39285714, line.PricePerRoom, 8)
    Assert.Equal(99.59523809523819, line.BasePrice, 8)

[<Fact>]
let ``MathNet の4部屋の予測は原著と同じ数値になる`` () =
    // 原著の出力: Predicted label for feature 4: [301.16666667]
    let line = fitWithMathNet features labels

    Assert.Equal(301.16666667, line.Predict 4.0, 7)

[<Fact>]
let ``既定の設定でも学習が進む`` () =
    let result = linearRegressionDefault ()

    Assert.Equal(1000, result.History.Length)
    Assert.True(List.last result.Errors < List.head result.Errors)
