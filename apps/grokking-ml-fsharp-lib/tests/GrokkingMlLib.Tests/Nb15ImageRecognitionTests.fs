/// 原著ノートブック #15 の再現テスト。
///
/// `mnist.npz` は約 11 MB あり、リポジトリに含めていない。
/// 未取得のときはダウンロードせず、その場で終わる（テストをネットワークに依存させない）。
module GrokkingMlLib.Tests.Nb15ImageRecognitionTests

open System.IO
open Xunit
open GrokkingMlLib
open GrokkingMlLib.Nb15ImageRecognition

let private mnistAvailable () =
    File.Exists(Path.Combine(Datasets.directory (), "mnist.npz"))

/// 読み込みは重いので 1 回だけ行う
let private mnist = lazy (loadMnist ())

[<Fact>]
let ``画像は28かける28の784次元`` () =
    Assert.Equal(28, ImageSize)
    Assert.Equal(784, InputDim)

[<Fact>]
let ``重みの総数は原著と同じ109386`` () =
    // 原著の model.summary() が出す Total params
    Assert.Equal(109386, parameterCount)

[<Fact>]
let ``層の構成は原著と同じ`` () =
    Assert.Equal<int[]>([| 128; 64 |], hiddenUnits)
    Assert.Equal(10, Classes)

[<Fact>]
let ``データ件数は訓練6万テスト1万`` () =
    if mnistAvailable () then
        let data = mnist.Value
        Assert.Equal(60000, data.YTrain.Length)
        Assert.Equal(10000, data.YTest.Length)

[<Fact>]
let ``原著が例に挙げるラベルが一致する`` () =
    if mnistAvailable () then
        let data = mnist.Value
        Assert.Equal(2, data.YTrain.[5])
        Assert.Equal(4, data.YTest.[4])
        // 原著が「モデルが間違える例」として挙げる 18 番目の正解
        Assert.Equal(3, data.YTest.[18])

[<Fact>]
let ``画素値は0から255`` () =
    if mnistAvailable () then
        let data = mnist.Value
        Assert.Equal(255.0, Array.max data.XTrain.[0], 1.0)
        Assert.Equal(0.0, Array.min data.XTrain.[0], 1.0)

[<Fact>]
let ``小さく学習しても当てずっぽうより十分に良い`` () =
    if mnistAvailable () then
        let data = mnist.Value
        let network = fit data 5 2000 0

        // 10 クラスなので当てずっぽうは 0.1
        Assert.True(testAccuracy network data 500 > 0.7)
