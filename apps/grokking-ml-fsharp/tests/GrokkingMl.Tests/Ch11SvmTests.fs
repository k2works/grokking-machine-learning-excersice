module GrokkingMl.Tests.Ch11SvmTests

open Xunit
open GrokkingMl.Ch11Svm

// 第 5 章と同じデータ。ただし SVM ではラベルを +1 / -1 で表す
let points: Point list =
    [ [ 1.0; 0.0 ]
      [ 0.0; 2.0 ]
      [ 1.0; 1.0 ]
      [ 1.0; 2.0 ]
      [ 1.0; 3.0 ]
      [ 2.0; 2.0 ]
      [ 2.0; 3.0 ]
      [ 3.0; 2.0 ] ]

let labels = [ -1; -1; -1; -1; 1; 1; 1; 1 ]
let perceptronLabels = [ 0; 0; 0; 0; 1; 1; 1; 1 ]

// XOR。線形カーネルでは分けられない
let xorPoints: Point list = [ [ 0.0; 0.0 ]; [ 0.0; 1.0 ]; [ 1.0; 0.0 ]; [ 1.0; 1.0 ] ]
let xorLabels = [ -1; 1; 1; -1 ]

let sample = { Weights = [ 1.0; 1.0 ]; Bias = -3.0 }

[<Fact>]
let ``linearKernel は内積`` () =
    Assert.Equal(11.0, linearKernel [ 1.0; 2.0 ] [ 3.0; 4.0 ], 6)

[<Fact>]
let ``polynomialKernel は定義どおり`` () =
    // (1*3 + 2*4 + 1)^2 = 144
    Assert.Equal(144.0, polynomialKernel 2 1.0 [ 1.0; 2.0 ] [ 3.0; 4.0 ], 6)

[<Fact>]
let ``rbfKernel は同じ点で 1`` () =
    Assert.Equal(1.0, rbfKernel 1.0 [ 1.0; 2.0 ] [ 1.0; 2.0 ], 6)

[<Fact>]
let ``rbfKernel は距離とともに減衰する`` () =
    let kernel = rbfKernel 1.0
    let near = kernel [ 0.0; 0.0 ] [ 0.5; 0.0 ]
    let far = kernel [ 0.0; 0.0 ] [ 3.0; 0.0 ]
    Assert.True(0.0 < far)
    Assert.True(far < near)
    Assert.True(near < 1.0)

[<Fact>]
let ``predict は +1 と -1 を返す`` () =
    Assert.Equal(1, predict sample [ 2.0; 2.0 ])
    Assert.Equal(-1, predict sample [ 1.0; 1.0 ])

[<Fact>]
let ``hingeLoss はマージンの外で 0`` () =
    Assert.Equal(0.0, hingeLoss sample [ 3.0; 2.0 ] 1, 6)

[<Fact>]
let ``hingeLoss はマージンの内側を罰する`` () =
    // スコア 0.5、ラベル +1。正解だがマージンの内側なので 1 - 0.5 = 0.5
    Assert.Equal(0.5, hingeLoss sample [ 1.5; 2.0 ] 1, 6)

[<Fact>]
let ``hingeLoss は誤差とともに増える`` () =
    // スコア -1、ラベル +1。1 - (-1) = 2
    Assert.Equal(2.0, hingeLoss sample [ 1.0; 1.0 ] 1, 6)

[<Fact>]
let ``svmStep は自信のある点をほとんど動かさない`` () =
    let moved = svmStep 0.01 0.1 sample [ 3.0; 2.0 ] 1
    // マージンの外なので、重みを縮める力しか働かない
    Assert.Equal(sample.Bias, moved.Bias, 6)
    Assert.True(moved.Weights[0] < sample.Weights[0])

[<Fact>]
let ``svmStep はマージンの内側の点を押し返す`` () =
    let moved = svmStep 0.01 0.0 sample [ 1.5; 2.0 ] 1
    Assert.True(moved.Weights[0] > sample.Weights[0])
    Assert.True(moved.Bias > sample.Bias)

[<Fact>]
let ``SVM はデータを分離する`` () =
    let model, errors = trainSvmWith 0.01 20000 0.1 0 points labels
    Assert.Equal(1.0, accuracy model points labels, 6)
    Assert.True(List.last errors < List.head errors)

[<Fact>]
let ``SVM はパーセプトロンより広いマージンを残す`` () =
    let perceptron, _ = GrokkingMl.Ch05Perceptron.perceptronAlgorithm 0.01 1000 0 points perceptronLabels

    let asSvm =
        { Weights = perceptron.Weights
          Bias = perceptron.Bias }

    let svm, _ = trainSvmWith 0.01 20000 0.01 0 points labels

    // どちらも完全に分離できている
    Assert.Equal(1.0, accuracy asSvm points labels, 6)
    Assert.Equal(1.0, accuracy svm points labels, 6)
    // しかしパーセプトロンは分離できた時点で止まるため、余白がない
    Assert.Equal(0.0, margin asSvm points, 6)
    Assert.True(margin svm points > 0.5)

[<Fact>]
let ``正則化が弱いほどマージンは広がる`` () =
    let loose, _ = trainSvmWith 0.01 20000 0.01 0 points labels
    let tight, _ = trainSvmWith 0.01 20000 0.1 0 points labels
    Assert.True(margin loose points > margin tight points)

[<Fact>]
let ``svmError は損失と罰則の和`` () =
    let meanLoss = List.map2 (hingeLoss sample) points labels |> List.average
    Assert.Equal(meanLoss, svmError 0.0 sample points labels, 6)
    Assert.Equal(meanLoss + 0.2, svmError 0.1 sample points labels, 6)

[<Fact>]
let ``線形カーネルでは XOR を分けられない`` () =
    let model = trainKernelClassifier linearKernel xorPoints xorLabels
    Assert.True(kernelAccuracy model xorPoints xorLabels < 1.0)

[<Fact>]
let ``多項式カーネルは XOR を解く`` () =
    let model = trainKernelClassifier (polynomialKernel 2 1.0) xorPoints xorLabels
    Assert.Equal(1.0, kernelAccuracy model xorPoints xorLabels, 6)

[<Fact>]
let ``RBF カーネルは XOR を解く`` () =
    let model = trainKernelClassifier (rbfKernel 1.0) xorPoints xorLabels
    Assert.Equal(1.0, kernelAccuracy model xorPoints xorLabels, 6)

[<Fact>]
let ``カーネル分類器は訓練点を保持する`` () =
    let model = trainKernelClassifier (rbfKernel 1.0) xorPoints xorLabels
    Assert.Equal(List.length xorPoints, List.length model.Points)
    Assert.Equal(List.length xorPoints, List.length model.Weights)

[<Fact>]
let ``重みが増えるのは誤分類した点だけ`` () =
    let model = trainKernelClassifierWith (rbfKernel 1.0) 0.1 200 0 xorPoints xorLabels
    Assert.Contains(model.Weights, fun w -> w > 0.0)
    Assert.All(model.Weights, fun w -> Assert.True(w >= 0.0))
