/// 原著ノートブック #19 の再現テスト。
///
/// 原著は 9 つの正解率を印刷する。Accord で完全に一致したのは **3 つ** だった。
/// どこがどれだけ違うかを表にして記録する。
module GrokkingMlLib.Tests.Nb19SvmKernelsTests

open Xunit
open GrokkingMlLib.Nb19SvmKernels

let private linearData = load "linear"
let private oneCircle = load "one_circle"
let private twoCircles = load "two_circles"

[<Fact>]
let ``データセットは17で作ったもの`` () =
    Assert.Equal(60, linearData.Size)
    Assert.Equal(110, oneCircle.Size)
    Assert.Equal(220, twoCircles.Size)

[<Fact>]
let ``線形カーネルは原著より少し低い`` () =
    // 原著は C=1 で 0.933、C=0.01 で 0.867、C=100 で 0.917。
    // Accord の LinearDualCoordinateDescent は座標降下法で、
    // libsvm の SMO とは別の解に落ち着く
    let score c =
        let model = fitLinear linearData c
        accuracy (fun row -> model.Decide row) linearData

    Assert.Equal(0.8833333333333333, score 1.0, 15)
    Assert.Equal(0.8833333333333333, score 0.01, 15)
    Assert.Equal(0.85, score 100.0, 15)

[<Fact>]
let ``多項式カーネルは4次で原著と完全に一致する`` () =
    // 原著は degree=2 で 0.891、degree=4 で 0.900
    let scaled = scaledForPolynomial oneCircle

    let score degree =
        let model = fitKernel scaled (polynomialKernel degree) 1.0
        accuracy (fun row -> model.Decide row) scaled

    Assert.Equal(0.9, score 2, 15)
    Assert.Equal(0.9, score 4, 15) // 原著と完全に一致

[<Fact>]
let ``gammaを特徴量に畳み込まないと4次で破綻する`` () =
    // 座標をそのまま渡すと内積が最大 18 になり、4 乗で 10 万を超える。
    // カーネル行列の値が大きすぎて SMO が解けず、正解率 0.318 まで落ちる
    let model = fitKernel oneCircle (polynomialKernel 4) 1.0

    Assert.Equal(0.3181818181818182, accuracy (fun row -> model.Decide row) oneCircle, 15)

[<Fact>]
let ``RBFのgammaを変えると原著の4つ中2つが完全に一致する`` () =
    let score gamma =
        let model = fitKernel twoCircles (rbfKernel gamma) 1.0
        accuracy (fun row -> model.Decide row) twoCircles

    // 原著: 0.8772727272727273 / 0.9045454545454545 / 0.9636363636363636 / 0.990909090909091
    Assert.Equal(0.8818181818181818, score 0.1, 15)
    Assert.Equal(0.9636363636363636, score 10.0, 15) // 完全一致
    Assert.Equal(0.990909090909091, score 100.0, 15) // 完全一致

[<Fact>]
let ``gamma1の正解率は実行環境で変わる`` () =
    // **同じ種を渡しても OS で結果が違う。** 手元の macOS では 0.9091、
    // CI の Linux では 0.9045 になった。後者は原著と完全に一致する。
    //
    // Accord の SMO は作業集合の選び方を浮動小数点の比較で決めるので、
    // 丸めがわずかに違うだけで反復の道筋が分かれ、別の解に落ち着く。
    // [#19](Nb19SvmKernels.fs) で種を固定して同一マシンでの再現性は
    // 確保できたが、**マシンをまたいだ再現性は別の話** だった。
    //
    // 1 点に固定できないので、観測した 2 つの値のどちらかであることを確かめる
    let model = fitKernel twoCircles (rbfKernel 1.0) 1.0
    let score = accuracy (fun row -> model.Decide row) twoCircles

    Assert.True(
        abs (score - 0.9090909090909091) < 1e-12 || abs (score - 0.9045454545454545) < 1e-12,
        $"想定外の正解率: {score}"
    )

[<Fact>]
let ``gammaを上げるほど正解率が単調に上がる`` () =
    // 原著と同じ傾向。値がずれても、章が示したい性質は再現できている
    let scores =
        [ 0.1; 1.0; 10.0; 100.0 ]
        |> List.map (fun gamma ->
            let model = fitKernel twoCircles (rbfKernel gamma) 1.0
            accuracy (fun row -> model.Decide row) twoCircles)

    Assert.Equal<float list>(List.sort scores, scores)

[<Fact>]
let ``種を固定すれば結果は再現する`` () =
    // fitLinear / fitKernel は毎回 Generator.Seed を設定する。
    // 設定しないと C = 100 で 0.917 と 0.683 が交互に出た
    let score () =
        let model = fitLinear linearData 100.0
        accuracy (fun row -> model.Decide row) linearData

    Assert.Equal(score (), score (), 15)

[<Fact>]
let ``scaleはscikit-learnのgamma_scaleと同じ式`` () =
    // 1 / (特徴量数 * 全体の分散)。特徴量は 2 列なので 1 / (2 * var)
    let values = oneCircle.X |> Array.collect id
    let mean = Array.average values
    let variance = values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float values.Length

    Assert.Equal(1.0 / (2.0 * variance), scaleOf oneCircle, 15)
    Assert.True(scaleOf oneCircle > 0.0)
