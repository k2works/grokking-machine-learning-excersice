/// 原著ノートブック #05 の再現テスト。
///
/// 手書きの学習ループは #04 と同じく種が与えられていないため再現できない。
/// あわせて、原著が示す「対数損失の別の書き方」が実際には対数損失と一致しないことを
/// テストで固定してある。
module GrokkingMlLib.Tests.Nb05LogisticRegressionTests

open Xunit
open GrokkingMlLib.Nb05LogisticRegression

let private start = { Weights = [| 1.0; 1.0 |]; Bias = 0.0 }

[<Fact>]
let ``データセットは原著と同じ`` () =
    // #04 と似ているが、最後の 2 点が入れ替わっている
    Assert.Equal<int[][]>(
        [| [| 1; 0 |]; [| 0; 2 |]; [| 1; 1 |]; [| 1; 2 |]
           [| 1; 3 |]; [| 2; 2 |]; [| 3; 2 |]; [| 2; 3 |] |],
        features
    )
    Assert.Equal<int[]>([| 0; 0; 0; 0; 1; 1; 1; 1 |], labels)

[<Fact>]
let ``シグモイドは0で0_5を返す`` () =
    Assert.Equal(0.5, sigmoid 0.0, 12)
    Assert.Equal(1.0, sigmoid 100.0, 12)
    Assert.Equal(0.0, sigmoid -100.0, 12)

[<Fact>]
let ``シグモイドの2つの書き方は一致する`` () =
    // 原著が使う exp(x)/(1+exp(x)) と、教科書の 1/(1+exp(-x))
    for x in [ -3.0; -0.5; 0.0; 1.7; 4.2 ] do
        Assert.Equal(1.0 / (1.0 + exp -x), sigmoid x, 12)

[<Fact>]
let ``ソフト relu は relu をなめらかにしたもの`` () =
    Assert.Equal(20.0, softRelu 20.0, 8)
    Assert.Equal(0.0, softRelu -20.0, 8)
    Assert.Equal(log 2.0, softRelu 0.0, 12)

[<Fact>]
let ``対数損失は当たっていても0にはならない`` () =
    // features.[4] = [1, 3] はラベル 1。スコア 4 なのでよく当たっている
    let confident = start.LogLoss(features.[4], 1)
    // features.[0] = [1, 0] はラベル 0。スコア 1 なので外している
    let wrong = start.LogLoss(features.[0], 0)

    Assert.InRange(confident, 0.0, 0.1)
    Assert.True(wrong > 1.0)

[<Fact>]
let ``原著の別の書き方は対数損失と一致しない`` () =
    // pred は 0〜1 の確率なので (pred - label) は ±1 にならない
    Array.iteri2
        (fun i f l ->
            if abs (start.Score f) > 1e-9 then
                Assert.True(
                    abs (alternateLogLossOriginal start f l - start.LogLoss(f, l)) > 1e-6,
                    $"index {i} で一致してしまった"
                ))
        features
        labels

[<Fact>]
let ``正しい別の書き方は対数損失と厳密に一致する`` () =
    // ラベルが 0 なら +1、1 なら -1 を掛ける。つまり (1 - 2 * label)
    for f in features do
        for label in [ 0; 1 ] do
            Assert.Equal(start.LogLoss(f, label), alternateLogLoss start f label, 12)

[<Fact>]
let ``ロジスティックトリックはバイアスを1回だけ動かす`` () =
    // #04 の「短く書いた版」はバイアスをループの内側で更新していたが、
    // こちらは外側にあるので 1 回だけ適用される
    let updated = logisticTrick 0.05 start features.[0] 0

    Assert.Equal((0.0 - start.Prediction features.[0]) * 0.05, updated.Bias, 12)

[<Fact>]
let ``確信を持って間違えた点ほど大きく動く`` () =
    let close = logisticTrick 0.05 { Weights = [| 1.0; 1.0 |]; Bias = -3.0 } [| 1; 2 |] 1
    let far = logisticTrick 0.05 { Weights = [| 1.0; 1.0 |]; Bias = -10.0 } [| 1; 2 |] 1

    Assert.True(far.Weights.[0] > close.Weights.[0])

[<Fact>]
let ``合計対数損失は各点の合計`` () =
    let expected = Array.map2 (fun f l -> start.LogLoss(f, l)) features labels |> Array.sum

    Assert.Equal(expected, totalLogLoss start features labels, 12)

[<Fact>]
let ``学習は誤差を下げる`` () =
    let result = logisticRegressionAlgorithm features labels 0.01 500 0

    Assert.Equal(500, result.Errors.Length)
    Assert.True(List.last result.Errors < List.head result.Errors)

[<Fact>]
let ``学習後は全点を正しく分類できる`` () =
    // 対数損失は当たっていても 0 にならないので、パーセプトロンより収束に時間がかかる
    let result = logisticRegressionAlgorithm features labels 0.01 20000 0

    Assert.Equal<int[]>(
        labels,
        features |> Array.map (fun f -> if result.Boundary.Prediction f >= 0.5 then 1 else 0)
    )

[<Fact>]
let ``ML_NET のロジスティック回帰も全点を正しく分類する`` () =
    // 原著 scikit-learn の出力: Logistic Regression Predictions: [0 0 0 0 1 1 1 1]
    let predict = fitWithMlNet features labels

    Assert.Equal<int[]>(labels, features |> Array.map predict)
