module GrokkingMl.Tests.Ch05PerceptronTests

open Xunit
open GrokkingMl.Ch05Perceptron

// 原著と同じ「悲しい／楽しい」文の分類データ
// 特徴量 = (aack の出現数, beep の出現数)、ラベル = 1 が楽しい
let points: Point list =
    [ [ 1.0; 0.0 ]
      [ 0.0; 2.0 ]
      [ 1.0; 1.0 ]
      [ 1.0; 2.0 ]
      [ 1.0; 3.0 ]
      [ 2.0; 2.0 ]
      [ 2.0; 3.0 ]
      [ 3.0; 2.0 ] ]

let labels = [ 0; 0; 0; 0; 1; 1; 1; 1 ]

let model = { Weights = [ 1.0; 2.0 ]; Bias = -4.0 }

[<Fact>]
let ``score は重み付き和`` () =
    // -4 + 1*1 + 2*2 = 1
    Assert.Equal(1.0, score model [ 1.0; 2.0 ], 6)

[<Fact>]
let ``predict はスコアの符号を使う`` () =
    Assert.Equal(1, predict model [ 1.0; 2.0 ])
    Assert.Equal(0, predict model [ 1.0; 1.0 ])

[<Fact>]
let ``predict は境界線上を正のクラスとして扱う`` () =
    let onLine = { Weights = [ 1.0; 1.0 ]; Bias = -2.0 }
    Assert.Equal(0.0, score onLine [ 1.0; 1.0 ], 6)
    Assert.Equal(1, predict onLine [ 1.0; 1.0 ])

[<Fact>]
let ``perceptronTrick は正しく分類できた点では動かない`` () =
    Assert.Equal(model, perceptronTrick 0.01 model [ 1.0; 2.0 ] 1)

[<Fact>]
let ``perceptronTrick は誤分類した正の点に近づく`` () =
    // 予測 0、ラベル 1 なので誤差 +1
    let moved = perceptronTrick 0.1 model [ 1.0; 1.0 ] 1
    Assert.Equal(1.1, moved.Weights[0], 6) // 1 + 0.1 * 1 * 1
    Assert.Equal(2.1, moved.Weights[1], 6) // 2 + 0.1 * 1 * 1
    Assert.Equal(-3.9, moved.Bias, 6) // -4 + 0.1

[<Fact>]
let ``perceptronTrick は誤分類した負の点から離れる`` () =
    // 予測 1、ラベル 0 なので誤差 -1
    let moved = perceptronTrick 0.1 model [ 1.0; 2.0 ] 0
    Assert.Equal(0.9, moved.Weights[0], 6) // 1 - 0.1 * 1
    Assert.Equal(1.8, moved.Weights[1], 6) // 2 - 0.1 * 2
    Assert.Equal(-4.1, moved.Bias, 6) // -4 - 0.1

[<Fact>]
let ``perceptronError は正解のとき 0`` () =
    Assert.Equal(0.0, perceptronError model [ 1.0; 2.0 ] 1, 6)

[<Fact>]
let ``perceptronError は誤りのときスコアの絶対値`` () =
    // スコア -1 で予測 0、ラベル 1 なので誤差は |-1| = 1
    Assert.Equal(1.0, perceptronError model [ 1.0; 1.0 ] 1, 6)

[<Fact>]
let ``meanPerceptronError は全点の平均`` () =
    let sample: Point list = [ [ 1.0; 2.0 ]; [ 1.0; 1.0 ] ]
    Assert.Equal(0.5, meanPerceptronError model sample [ 1; 1 ], 6)

[<Fact>]
let ``accuracy は正解した割合`` () =
    let sample: Point list = [ [ 1.0; 2.0 ]; [ 1.0; 1.0 ] ]
    Assert.Equal(0.5, accuracy model sample [ 1; 1 ], 6)

[<Fact>]
let ``perceptronAlgorithm はデータを分離する`` () =
    let trained, errors = perceptronAlgorithm 0.01 1000 0 points labels
    Assert.Equal(1.0, accuracy trained points labels, 6)
    Assert.Equal(1000, List.length errors)
    // 初期モデル（重みもバイアスもすべて 0）はすべての点が境界線上にあるため誤差 0
    Assert.Equal(0.0, List.head errors, 6)
    // 学習の途中では誤差が生じ、最終的に 0 へ戻る
    Assert.True(List.max errors > 0.0)
    Assert.Equal(0.0, List.last errors, 6)

[<Fact>]
let ``perceptronAlgorithm は分離可能なデータで誤差 0 に到達する`` () =
    let trained, _ = perceptronAlgorithm 0.01 1000 0 points labels
    Assert.Equal(0.0, meanPerceptronError trained points labels, 6)
