module GrokkingMl.Tests.Ch06LogisticRegressionTests

open Xunit
open GrokkingMl.Ch06LogisticRegression

// 第 5 章と同じ「悲しい／楽しい」文の分類データ
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
let ``sigmoid は 0 で 0.5 を返す`` () = Assert.Equal(0.5, sigmoid 0.0, 6)

[<Fact>]
let ``sigmoid は単調増加`` () =
    Assert.True(sigmoid -1.0 < sigmoid 0.0)
    Assert.True(sigmoid 0.0 < sigmoid 1.0)

[<Fact>]
let ``sigmoid の値域は 0 から 1`` () =
    for x in [ -1000.0; -10.0; 0.0; 10.0; 1000.0 ] do
        Assert.InRange(sigmoid x, 0.0, 1.0)

[<Fact>]
let ``sigmoid は大きな負の入力でも溢れない`` () =
    // 素朴な 1/(1+exp(-x)) 実装は exp のオーバーフローで壊れる
    Assert.Equal(0.0, sigmoid -1000.0, 6)
    Assert.Equal(1.0, sigmoid 1000.0, 6)

[<Fact>]
let ``sigmoid は 0 を中心に対称`` () =
    Assert.Equal(1.0, sigmoid 2.0 + sigmoid -2.0, 6)

[<Fact>]
let ``predictProbability はスコアにシグモイドを適用する`` () =
    // スコア = -4 + 1*1 + 2*2 = 1
    Assert.Equal(1.0, score model [ 1.0; 2.0 ], 6)
    Assert.Equal(sigmoid 1.0, predictProbability model [ 1.0; 2.0 ], 6)

[<Fact>]
let ``predictWith は閾値を使う`` () =
    Assert.Equal(1, predict model [ 1.0; 2.0 ])
    Assert.Equal(0, predictWith 0.8 model [ 1.0; 2.0 ])

[<Fact>]
let ``logLoss は自信を持って正解したとき小さい`` () =
    let confident = { Weights = [ 10.0; 20.0 ]; Bias = -40.0 }
    Assert.True(logLoss confident [ 1.0; 2.0 ] 1 < logLoss model [ 1.0; 2.0 ] 1)

[<Fact>]
let ``logLoss は自信を持って間違えたとき大きい`` () =
    let confident = { Weights = [ 10.0; 20.0 ]; Bias = -40.0 }
    Assert.True(logLoss confident [ 1.0; 2.0 ] 0 > 5.0)

[<Fact>]
let ``logLoss は定義どおり`` () =
    let probability = sigmoid 1.0
    Assert.Equal(-log probability, logLoss model [ 1.0; 2.0 ] 1, 6)
    Assert.Equal(-log (1.0 - probability), logLoss model [ 1.0; 2.0 ] 0, 6)

[<Fact>]
let ``meanLogLoss は全点の平均`` () =
    let zero = { Weights = [ 0.0; 0.0 ]; Bias = 0.0 }
    // すべての予測が 0.5 なので、どのラベルでも損失は -log(0.5)
    Assert.Equal(-log 0.5, meanLogLoss zero points labels, 6)

[<Fact>]
let ``logisticTrick は正しく分類できた点でも動く`` () =
    // 予測は 1（正解）だが、確率は 0.73 なのでまだ動く
    let moved = logisticTrick 0.1 model [ 1.0; 2.0 ] 1
    Assert.NotEqual(model, moved)
    Assert.True(moved.Weights[0] > model.Weights[0])

[<Fact>]
let ``logisticTrick は勾配どおりに動く`` () =
    let error = 1.0 - sigmoid 1.0
    let moved = logisticTrick 0.1 model [ 1.0; 2.0 ] 1
    Assert.Equal(1.0 + 0.1 * error * 1.0, moved.Weights[0], 6)
    Assert.Equal(2.0 + 0.1 * error * 2.0, moved.Weights[1], 6)
    Assert.Equal(-4.0 + 0.1 * error, moved.Bias, 6)

[<Fact>]
let ``logisticTrick は誤った正の予測から離れる`` () =
    // 予測確率 0.73 に対しラベル 0 なので誤差は負
    let moved = logisticTrick 0.1 model [ 1.0; 2.0 ] 0
    Assert.True(moved.Weights[0] < model.Weights[0])
    Assert.True(moved.Bias < model.Bias)

[<Fact>]
let ``logisticRegression はデータを分離する`` () =
    let trained, losses = logisticRegression 0.1 1000 0 points labels
    Assert.Equal(1.0, accuracy trained points labels, 6)
    Assert.Equal(1000, List.length losses)
    // パーセプトロン誤差と違い、対数損失は初期状態でも 0 にならない
    Assert.Equal(-log 0.5, List.head losses, 6)
    Assert.True(List.last losses < List.head losses)

[<Fact>]
let ``logisticRegression は確率として使える出力を返す`` () =
    let trained, _ = logisticRegression 0.1 1000 0 points labels
    Assert.True(predictProbability trained [ 3.0; 2.0 ] > 0.5)
    Assert.True(predictProbability trained [ 1.0; 0.0 ] < 0.5)
