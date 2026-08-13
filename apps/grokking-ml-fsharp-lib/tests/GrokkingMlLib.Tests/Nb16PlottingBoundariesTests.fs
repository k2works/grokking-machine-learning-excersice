/// 原著ノートブック #16 の再現テスト。
///
/// 原著は図しか出さないので、突き合わせる数値が無い。
/// そこで **境界を数値に直して** 検証する。8 点の判定は 3 言語で完全に一致する。
module GrokkingMlLib.Tests.Nb16PlottingBoundariesTests

open Xunit
open GrokkingMlLib.Nb16PlottingBoundaries

[<Fact>]
let ``データセットは8点`` () =
    Assert.Equal(8, alienDataset.Length)
    // 幸せなエイリアンと不幸せなエイリアンが半分ずつ
    Assert.Equal<int list>([ 0; 0; 0; 0; 1; 1; 1; 1 ], alienDataset |> List.map (fun a -> a.Happy))

[<Fact>]
let ``階段関数は0で1になる`` () =
    // 原著は x >= 0。0 ちょうどは 1 の側に入る
    Assert.Equal(1.0, step 0.0, 15)
    Assert.Equal(0.0, step -1e-15, 15)

[<Fact>]
let ``シグモイドは0で半分`` () =
    Assert.Equal(0.5, sigmoid 0.0, 15)
    // 原著の書き方 exp(x)/(1+exp(x))。1/(1+exp(-x)) と同じ値になる
    Assert.Equal(0.8807970779778823, sigmoid 2.0, 15)

[<Fact>]
let ``1層目の2つの直線は対称`` () =
    // 重みが (6, 10) と (10, 6) なので、aack と beep を入れ替えた関係
    Assert.Equal<int list>([ 0; 0; 0; 1; 1; 1; 1; 1 ], predictions line1)
    Assert.Equal<int list>([ 0; 1; 0; 0; 1; 1; 1; 1 ], predictions line2)
    // 領域の広さは同じ
    Assert.Equal(regionRatio line1, regionRatio line2, 15)

[<Fact>]
let ``1層目だけでは8点を分けられない`` () =
    // 直線 1 本では 1 点ずつ間違える
    Assert.Equal(0.875, accuracy line1, 15)
    Assert.Equal(0.875, accuracy line2, 15)

[<Fact>]
let ``バイアスは入力を見ない`` () =
    Assert.Equal<int list>(List.replicate 8 1, predictions bias)
    Assert.Equal(1.0, regionRatio bias, 15)

[<Fact>]
let ``2層目はANDになっている`` () =
    // 1 層目の出力は 0 か 1。その和が 1.5 以上になるのは両方 1 のときだけ
    Assert.Equal(1.0, nnWithStep 1.0 1.0, 15)
    Assert.Equal(0.0, nnWithStep 2.0 0.0, 15) // line2 だけが 1
    Assert.Equal(0.0, nnWithStep 0.0 2.0, 15) // line1 だけが 1

[<Fact>]
let ``階段関数のネットワークは8点すべて正解`` () =
    Assert.Equal<int list>([ 0; 0; 0; 0; 1; 1; 1; 1 ], predictions nnWithStep)
    Assert.Equal(1.0, accuracy nnWithStep, 15)

[<Fact>]
let ``シグモイド版は1点だけ外す`` () =
    // aack=1 beep=1 の点だけ 0 と答える。正解は 1
    Assert.Equal<int list>([ 0; 0; 0; 0; 0; 1; 1; 1 ], predictions nnWithSigmoid)
    Assert.Equal(0.875, accuracy nnWithSigmoid, 15)

[<Fact>]
let ``外した点は判定の境目にある`` () =
    // 0.4905 で、しきい値 0.5 をわずかに下回る
    Assert.Equal(0.4905304218, nnWithSigmoid 1.0 1.0, 9)
    Assert.Equal(0, classify nnWithSigmoid 1.0 1.0)

[<Fact>]
let ``シグモイドは出力が飽和する`` () =
    // 内側のシグモイドが 1 に近づくので、外側は sigmoid(0.5) で頭打ちになる
    Assert.Equal(0.6224593117, nnWithSigmoid 2.0 2.0, 9)
    Assert.Equal(0.6224593312, nnWithSigmoid 3.0 3.0, 9)
    Assert.True(nnWithSigmoid 3.0 3.0 - nnWithSigmoid 2.0 2.0 < 1e-7)

[<Fact>]
let ``格子は原著と同じ700刻み`` () =
    Assert.Equal(700, axis.Length)
    Assert.Equal(-0.5, axis.[0], 15)
    // こちらは `min + i * step` なのでちょうど 0.0 になる。
    // NumPy の arange は 4.44e-16 になる
    Assert.Equal(0.0, axis.[100], 15)

[<Fact>]
let ``2つのネットワークの境界はほぼ重なる`` () =
    // 図では見分けが付かないが、格子の 0.48% だけ判定が食い違う
    Assert.Equal(0.0048, disagreementRatio nnWithStep nnWithSigmoid, 4)

[<Fact>]
let ``ANDの領域は各直線より狭い`` () =
    // Python 版は 0.5544327、F# 版・Kotlin 版は 0.5543918。**格子の作り方が違う**。
    // NumPy の arange は誤差を累積するので、-0.5 + 100 * 0.005 が
    // 0.0 ではなく 4.44e-16 になる。境界ちょうどの点の判定がそこで分かれる
    Assert.Equal(0.5544, regionRatio nnWithStep, 4)
    Assert.True(regionRatio nnWithStep < regionRatio line1)
