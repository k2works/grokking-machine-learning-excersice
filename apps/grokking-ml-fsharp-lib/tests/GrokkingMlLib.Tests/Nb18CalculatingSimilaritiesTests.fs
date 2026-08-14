/// 原著ノートブック #18 の再現テスト。
///
/// 類似度も手書きの予測式も、**原著の出力と 16 桁まで一致する**。
/// 一方、SVM の係数は解き方が違うので比べられない（後述）。
module GrokkingMlLib.Tests.Nb18CalculatingSimilaritiesTests

open Xunit
open GrokkingMlLib.Nb18CalculatingSimilarities

[<Fact>]
let ``データセットは7点`` () =
    // 原点と、その上下左右と斜め 2 点
    Assert.Equal(7, Size)
    Assert.Equal<int[]>([| 0; 0; 0; 1; 1; 1; 1 |], y)

[<Fact>]
let ``同じ点の類似度は1`` () =
    // exp(0) = 1
    Assert.Equal(1.0, similarity [| 0.0; 0.0 |] [| 0.0; 0.0 |], 15)
    Assert.Equal(1.0, similarity [| 3.0; -2.0 |] [| 3.0; -2.0 |], 15)

[<Fact>]
let ``類似度は距離の2乗で決まる`` () =
    let origin = [| 0.0; 0.0 |]
    // exp(-||x-y||^2)。距離 1 なら exp(-1)
    Assert.Equal(exp -1.0, similarity origin [| 1.0; 0.0 |], 15)
    // 距離 √2 なら exp(-2)
    Assert.Equal(exp -2.0, similarity origin [| 1.0; 1.0 |], 15)
    // 原著のセルに残っている出力 1.522997974471263e-08 は (0,0) と (3,3) の類似度
    // 桁が小さいので相対誤差で比べる（Assert.Equal の桁指定は 15 が上限）
    Assert.True(abs (similarity origin [| 3.0; 3.0 |] / 1.522997974471263e-08 - 1.0) < 1e-15)

[<Fact>]
let ``類似度は対称`` () =
    let matrix = similarityMatrix ()

    for i in 0 .. Size - 1 do
        for j in 0 .. Size - 1 do
            Assert.Equal(matrix.[i].[j], matrix.[j].[i], 15)

[<Fact>]
let ``対角成分はすべて1`` () =
    let matrix = similarityMatrix ()

    for i in 0 .. Size - 1 do
        Assert.Equal(1.0, matrix.[i].[i], 15)

[<Fact>]
let ``類似度行列は原著の表と一致する`` () =
    let matrix = similarityMatrix ()

    // 原著の表の 1 行目: 1.000000 0.367879 0.367879 0.367879 0.367879 0.135335 ...
    Assert.Equal(0.36787944117144233, matrix.[0].[1], 15)
    Assert.Equal(0.1353352832366127, matrix.[0].[5], 15)
    // 2 行目の Sim4: 0.018316（距離の 2 乗が 4）
    Assert.Equal(0.01831563888873418, matrix.[1].[4], 15)
    // 3 行目の Sim5: 0.006738（距離の 2 乗が 5）
    Assert.Equal(0.006737946999085467, matrix.[2].[5], 15)

[<Fact>]
let ``特徴量は9列になる`` () =
    // 元の x1, x2 に類似度 7 列を足す
    Assert.Equal(9, FeatureCount)

    for row in features () do
        Assert.Equal(9, row.Length)

[<Fact>]
let ``SVMは7点すべてを正しく分類する`` () =
    // ML.NET の LinearSvm は確率的勾配法で解くので、libsvm の SMO を使う
    // scikit-learn とは **係数が一致しない**。原著の svm.coef_ とは
    // 突き合わせられないので、代わりに分類結果で確かめる
    let predict = fit ()
    let predicted = features () |> Array.map predict

    Assert.Equal<int[]>(y, predicted)

[<Fact>]
let ``手書きの予測式は原著と桁まで一致する`` () =
    // 原著の出力をそのまま並べたもの
    let expected =
        [| -0.7293294335267746
           -0.9749464141121803
           -0.9749464141121804
           0.9884223081103513
           0.9884223081103514
           0.8650001793912898
           0.8650001793912898 |]

    Assert.Equal<float[]>(expected, trainingPredictions ())

[<Fact>]
let ``予測の符号は正解ラベルと合う`` () =
    // 原著のコメント「ラベルが 1 なら正、0 なら負になるはず」
    trainingPredictions ()
    |> Array.iteri (fun index value -> Assert.Equal(y.[index] = 1, value > 0.0))

[<Fact>]
let ``対称な2点は同じ値になる`` () =
    // (0,1) と (1,0) は x1 と x2 を入れ替えた関係。データ全体もその入れ替えで
    // 不変なので、予測値も一致する
    Assert.Equal(svmRbfPrediction [| 0.0; 1.0 |], svmRbfPrediction [| 1.0; 0.0 |], 15)
    Assert.Equal(svmRbfPrediction [| -1.0; 1.0 |], svmRbfPrediction [| 1.0; -1.0 |], 15)

[<Fact>]
let ``遠く離れた点の予測は0に近づく`` () =
    // RBF は距離とともに指数的に減る。データから離れると判断できなくなる
    Assert.True(abs (svmRbfPrediction [| 10.0; 10.0 |]) < 1e-30)

[<Fact>]
let ``点の並びは原著と同じ`` () =
    Assert.Equal<float[]>([| 0.0; 0.0 |], points.[0])
    Assert.Equal<float[]>([| 1.0; -1.0 |], points.[6])
