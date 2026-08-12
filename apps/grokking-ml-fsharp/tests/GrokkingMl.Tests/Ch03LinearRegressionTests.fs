module GrokkingMl.Tests.Ch03LinearRegressionTests

open System
open Xunit
open GrokkingMl.Ch03LinearRegression

let features = [ 1.0; 2.0; 3.0; 5.0; 6.0; 7.0 ]
let labels = [ 155.0; 197.0; 244.0; 356.0; 407.0; 448.0 ]

[<Fact>]
let ``predict は slope * rooms + intercept を返す`` () =
    Assert.Equal(250.0, predict { Slope = 50.0; Intercept = 100.0 } 3.0, 6)

[<Fact>]
let ``absoluteTrick は予測が低いとき点に近づく`` () =
    let moved = absoluteTrick 0.01 { Slope = 50.0; Intercept = 100.0 } 3.0 300.0
    Assert.Equal(50.03, moved.Slope, 6)
    Assert.Equal(100.01, moved.Intercept, 6)

[<Fact>]
let ``absoluteTrick は予測が高いとき点から離れる向きに下げる`` () =
    let moved = absoluteTrick 0.01 { Slope = 50.0; Intercept = 100.0 } 3.0 200.0
    Assert.Equal(49.97, moved.Slope, 6)
    Assert.Equal(99.99, moved.Intercept, 6)

[<Fact>]
let ``squareTrick の移動量は誤差に比例する`` () =
    let moved = squareTrick 0.01 { Slope = 50.0; Intercept = 100.0 } 3.0 300.0
    Assert.Equal(51.5, moved.Slope, 6)
    Assert.Equal(100.5, moved.Intercept, 6)

[<Fact>]
let ``simpleTrick は absoluteTrick と同じ向きに動く`` () =
    let model = { Slope = 50.0; Intercept = 100.0 }
    let moved = simpleTrick (Random(0)) model 3.0 300.0
    Assert.True(moved.Slope > model.Slope)
    Assert.True(moved.Intercept > model.Intercept)

[<Fact>]
let ``rmse は誤差ゼロのとき 0 を返す`` () =
    Assert.Equal(0.0, rmse [ 1.0; 2.0 ] [ 1.0; 2.0 ], 6)

[<Fact>]
let ``rmse は二乗平均平方根を返す`` () =
    Assert.Equal(sqrt 2.5, rmse [ 1.0; 2.0 ] [ 2.0; 4.0 ], 6)

[<Fact>]
let ``linearRegression は住宅データセットに適合する`` () =
    let model, errors = linearRegression 0.01 1000 0 features labels
    Assert.InRange(model.Slope, 45.0, 55.0)
    Assert.InRange(model.Intercept, 80.0, 120.0)
    Assert.True(modelRmse model features labels < 15.0)
    Assert.Equal(1000, List.length errors)
    Assert.True(List.last errors < List.head errors)
