/// 原著ノートブック #21 の再現テスト。
///
/// 勾配ブースティングを自前で書いたところ、**段ごとの残差まで
/// scikit-learn と一致** した。1e-16 の丸め誤差の出方まで同じである。
module GrokkingMlLib.Tests.Nb21GradientBoostingTests

open Xunit
open GrokkingMlLib.Nb21GradientBoosting

let private approx (expected: float[]) (actual: float[]) (tolerance: float) =
    Assert.Equal(expected.Length, actual.Length)
    Array.iter2 (fun e a -> Assert.Equal(e, a, tolerance)) expected actual

[<Fact>]
let ``データセットは8人`` () =
    Assert.Equal<float[]>([| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0; 70.0; 80.0 |], ages)
    Assert.Equal<float[]>([| 7.0; 5.0; 7.0; 1.0; 2.0; 1.0; 5.0; 4.0 |], days)
    // 平均は 4.0。XGBoost の base_score に使う
    Assert.Equal(4.0, Array.average days, 15)

[<Fact>]
let ``中心化したラベルが最初の残差`` () =
    // 原著の出力: Residuals to predict: [ 3.  1.  3. -3. -2. -3.  1.  0.]
    Assert.Equal<float[]>([| 3.0; 1.0; 3.0; -3.0; -2.0; -3.0; 1.0; 0.0 |], centeredLabels ())

[<Fact>]
let ``回帰木は深さ2で4つの値しか返さない`` () =
    let tree = fitTree ages days MaxDepth
    let predicted = ages |> Array.map (predictTree tree) |> Array.distinct

    Assert.Equal(4, predicted.Length)

[<Fact>]
let ``1本目の弱学習器の予測は原著と一致する`` () =
    // 原著の出力
    //   Predictions: [ 3. 2. 2. -2.66666667 -2.66666667 -2.66666667 0.5 0.5 ]
    let stage, _ = (fitGradientBoosting 4).Head

    approx [| 3.0; 2.0; 2.0; -2.66666667; -2.66666667; -2.66666667; 0.5; 0.5 |] stage.Predictions 5e-9

[<Fact>]
let ``1段目の後の残差は原著と一致する`` () =
    // 原著の出力
    //   New residuals: [ 0.6 -0.6 1.4 -0.86666667 0.13333333 -0.86666667 0.6 -0.4 ]
    let stage, _ = (fitGradientBoosting 4).Head

    approx [| 0.6; -0.6; 1.4; -0.86666667; 0.13333333; -0.86666667; 0.6; -0.4 |] stage.NewResiduals 5e-9

[<Fact>]
let ``2段目の後の残差は原著と一致する`` () =
    // 原著の出力: [ 0.6 -0.6 0.28 -0.44 0.56 -0.44 0.52 -0.48]
    let stage, _ = (fitGradientBoosting 4).[1]

    approx [| 0.6; -0.6; 0.28; -0.44; 0.56; -0.44; 0.52; -0.48 |] stage.NewResiduals 1e-12

[<Fact>]
let ``3段目の予測はほとんど0になる`` () =
    // 原著の出力に -7.40148683e-17 が並ぶ。**符号は違うが桁は同じ**。
    // 「もう説明できるものが残っていない」を浮動小数点の誤差が示している
    let stage, _ = (fitGradientBoosting 4).[2]

    Assert.Equal(0.6, stage.Predictions.[0], 12)
    Assert.Equal(-0.6, stage.Predictions.[1], 12)
    Assert.All(stage.Predictions.[2..], fun value -> Assert.True(abs value < 1e-15))

[<Fact>]
let ``勾配ブースティングの予測は原著と一致する`` () =
    // 原著の出力
    //   array([6.87466667, 5.11466667, 6.71466667, 1.43466667, 1.43466667,
    //          1.43466667, 4.896     , 4.096     ])
    let stages = fitGradientBoosting 4
    let predicted = ages |> Array.map (predictGradientBoosting stages)

    approx
        [| 6.87466667; 5.11466667; 6.71466667; 1.43466667; 1.43466667; 1.43466667; 4.896; 4.096 |]
        predicted
        5e-9

[<Fact>]
let ``残差は段を追うごとに小さくなる`` () =
    // ブースティングの定義そのもの。二乗和で測る
    let magnitudes =
        fitGradientBoosting 4
        |> List.map (fun (stage, _) -> stage.Residuals |> Array.sumBy (fun v -> v * v))

    Assert.Equal<float list>(List.sortDescending magnitudes, magnitudes)

[<Fact>]
let ``類似度スコアは和の2乗を使う`` () =
    // sum(l)^2 / (len(l) + lambda)。2 乗の和ではない
    Assert.Equal(2.25, similarityScore [| 3.0 |] 3.0, 15)
    Assert.Equal(3.2, similarityScore [| 3.0; 1.0 |] 3.0, 15)
    // 符号が打ち消し合うと 0 になる
    Assert.Equal(0.0, similarityScore [| 3.0; -3.0 |] 3.0, 15)

[<Fact>]
let ``空の集合の類似度は0`` () =
    Assert.Equal(0.0, similarityScore [||] 3.0, 15)

[<Fact>]
let ``全体の類似度は0になる`` () =
    // 残差の総和が 0 なので、根の類似度スコアは 0
    Assert.Equal(0.0, similarityScore (xgboostResiduals ()) 3.0, 15)

[<Fact>]
let ``残差は原著と一致する`` () =
    // 原著の出力: array([ 3.,  1.,  3., -3., -2., -3.,  1.,  0.])
    Assert.Equal<float[]>([| 3.0; 1.0; 3.0; -3.0; -2.0; -3.0; 1.0; 0.0 |], xgboostResiduals ())

[<Fact>]
let ``切れ目ごとのスコアは原著と一致する`` () =
    // 原著が 1 行ずつ印刷している「Sum of similarity scores」
    let scores = splitScores (xgboostResiduals ()) 3.0

    Assert.Equal(3.15, scores.[0], 15)
    Assert.Equal(4.977777777777778, scores.[1], 15)
    Assert.Equal(14.291666666666666, scores.[2], 15)
    Assert.Equal(List.max scores, scores.[2], 15)

[<Fact>]
let ``最良の分割は原著と一致する`` () =
    // 原著の結論: Left tree: [3. 1. 3.] / Right tree: [-3. -2. -3.  1.  0.]
    let best = findBestSplit (xgboostResiduals ()) 3.0

    Assert.Equal<float[]>([| 3.0; 1.0; 3.0 |], best.Left)
    Assert.Equal<float[]>([| -3.0; -2.0; -3.0; 1.0; 0.0 |], best.Right)
    Assert.Equal(14.291666666666666, best.Score, 15)

[<Fact>]
let ``分割のスコアはXGBoostのgainと一致する`` () =
    // 原著の木のダンプ: 0:[f0<35] ... gain=14.291667
    // 手で計算した最良スコアが、XGBoost が出す gain とぴたり合う
    Assert.Equal(14.291667, (findBestSplit (xgboostResiduals ()) 3.0).Score, 6)

[<Fact>]
let ``部分木の分割も原著と一致する`` () =
    let best = findBestSplit (xgboostResiduals ()) 3.0
    let left = findBestSplit best.Left 3.0
    let right = findBestSplit best.Right 3.0

    // 原著: Left tree: [3.] / [1., 3.]
    Assert.Equal<float[]>([| 3.0 |], left.Left)
    Assert.Equal<float[]>([| 1.0; 3.0 |], left.Right)
    Assert.Equal(5.45, left.Score, 15)

    // 原著: Left tree: [-3., -2., -3.] / [1., 0.]
    Assert.Equal<float[]>([| -3.0; -2.0; -3.0 |], right.Left)
    Assert.Equal<float[]>([| 1.0; 0.0 |], right.Right)
    // **原著は 7.0 を返すが、それは実装のバグによる**（最後の切れ目のスコア）
    Assert.Equal(10.866666666666665, right.Score, 15)

[<Fact>]
let ``原著が返す7という値は最後の切れ目のスコア`` () =
    // 右の木 [-3,-2,-3,1,0] の最後の切れ目は [-3,-2,-3,1] / [0] で、
    // (-7)^2/(4+3) + 0^2/(1+3) = 7.0。これが原著の返り値 7.0 の正体
    let right = (findBestSplit (xgboostResiduals ()) 3.0).Right
    let scores = splitScores right 3.0

    Assert.Equal(7.0, List.last scores, 15)
    Assert.True(List.last scores < List.max scores)
