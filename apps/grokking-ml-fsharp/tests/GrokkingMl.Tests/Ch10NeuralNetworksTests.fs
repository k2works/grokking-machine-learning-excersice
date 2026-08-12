module GrokkingMl.Tests.Ch10NeuralNetworksTests

open Xunit
open GrokkingMl.Ch10NeuralNetworks

// XOR。直線では分けられない代表例
let xorPoints: Point list = [ [ 0.0; 0.0 ]; [ 0.0; 1.0 ]; [ 1.0; 0.0 ]; [ 1.0; 1.0 ] ]
let xorLabels = [ 0; 1; 1; 0 ]

[<Fact>]
let ``sigmoidDerivative は出力から計算できる`` () =
    // s'(x) = s(x) * (1 - s(x))。出力が 0.5 のとき最大の 0.25
    Assert.Equal(0.25, sigmoidDerivative 0.5, 6)
    Assert.Equal(0.25, sigmoidDerivative (sigmoid 0.0), 6)

[<Fact>]
let ``sigmoidDerivative は両端で消える`` () =
    // 出力が 0 や 1 に近いと勾配がほぼ消える（勾配消失）
    Assert.True(sigmoidDerivative 0.999 < 0.002)
    Assert.True(sigmoidDerivative 0.001 < 0.002)

[<Fact>]
let ``forward は重み付き和にシグモイドを適用する`` () =
    let layer = { Weights = [ [ 1.0; 2.0 ] ]; Biases = [ -4.0 ] }
    // -4 + 1*1 + 2*2 = 1
    Assert.Equal(sigmoid 1.0, forward layer [ 1.0; 2.0 ] |> List.head, 6)

[<Fact>]
let ``layer は自身の形を報告する`` () =
    let layer =
        { Weights = [ [ 1.0; 2.0; 3.0 ]; [ 4.0; 5.0; 6.0 ] ]
          Biases = [ 0.0; 0.0 ] }

    Assert.Equal(3, layer.InputSize)
    Assert.Equal(2, layer.OutputSize)

[<Fact>]
let ``forwardAll はすべての層の出力を記録する`` () =
    let activations = forwardAll (initialNetwork [ 2; 3; 1 ] 0) [ 0.5; 0.5 ]
    Assert.Equal(3, List.length activations) // 入力 + 隠れ層 + 出力層
    Assert.Equal(2, List.length activations[0])
    Assert.Equal(3, List.length activations[1])
    Assert.Equal(1, List.length activations[2])

[<Fact>]
let ``initialNetwork は指定した形を持つ`` () =
    let model = initialNetwork [ 2; 4; 1 ] 0
    Assert.Equal(2, List.length model.Layers)
    Assert.Equal(2, model.Layers[0].InputSize)
    Assert.Equal(4, model.Layers[0].OutputSize)
    Assert.Equal(4, model.Layers[1].InputSize)
    Assert.Equal(1, model.Layers[1].OutputSize)

[<Fact>]
let ``initialNetwork は再現可能`` () =
    Assert.Equal(initialNetwork [ 2; 3; 1 ] 7, initialNetwork [ 2; 3; 1 ] 7)
    Assert.NotEqual(initialNetwork [ 2; 3; 1 ] 7, initialNetwork [ 2; 3; 1 ] 8)

[<Fact>]
let ``predictProbability は 0 から 1 の範囲に収まる`` () =
    let model = initialNetwork [ 2; 4; 1 ] 0

    for point in xorPoints do
        Assert.InRange(predictProbability model point, 0.0, 1.0)

[<Fact>]
let ``backpropagate はその点の損失を下げる`` () =
    let model = initialNetwork [ 2; 4; 1 ] 1
    let point = [ 1.0; 0.0 ]
    let before = logLoss model point 1
    let after = logLoss (backpropagate 0.5 model point 1) point 1
    Assert.True(after < before)

[<Fact>]
let ``backpropagate は形を保つ`` () =
    let model = initialNetwork [ 2; 4; 1 ] 0
    let updated = backpropagate 0.5 model [ 1.0; 0.0 ] 1
    Assert.Equal(List.length model.Layers, List.length updated.Layers)

    List.zip model.Layers updated.Layers
    |> List.iter (fun (original, layer) ->
        Assert.Equal(original.InputSize, layer.InputSize)
        Assert.Equal(original.OutputSize, layer.OutputSize))

[<Fact>]
let ``backpropagate は新しいネットワークを返す`` () =
    let model = initialNetwork [ 2; 4; 1 ] 0
    Assert.NotEqual(model, backpropagate 0.5 model [ 1.0; 0.0 ] 1)

[<Fact>]
let ``隠れニューロンが 1 つでは XOR を解けない`` () =
    let model, _ = trainWith 1 0.5 20000 0 xorPoints xorLabels
    Assert.True(accuracy model xorPoints xorLabels < 1.0)

[<Fact>]
let ``隠れ層があれば XOR を解ける`` () =
    let model, losses = trainWith 4 0.5 20000 0 xorPoints xorLabels
    Assert.Equal(1.0, accuracy model xorPoints xorLabels, 6)
    Assert.True(List.last losses < List.head losses)

[<Fact>]
let ``学習後は自信のある予測になる`` () =
    let model, _ = trainWith 4 0.5 20000 0 xorPoints xorLabels
    Assert.True(predictProbability model [ 0.0; 0.0 ] < 0.1)
    Assert.True(predictProbability model [ 0.0; 1.0 ] > 0.9)
    Assert.True(predictProbability model [ 1.0; 0.0 ] > 0.9)
    Assert.True(predictProbability model [ 1.0; 1.0 ] < 0.1)

[<Fact>]
let ``学習前の損失は五分五分あたりから始まる`` () =
    let loss = meanLogLoss (initialNetwork [ 2; 4; 1 ] 0) xorPoints xorLabels
    Assert.InRange(loss, 0.5, 1.2)
