module GrokkingMl.Tests.Ch07MetricsTests

open Xunit
open GrokkingMl.Ch07Metrics

// 1000 人に 10 人が罹る病気を、全員「陰性」と判定するモデル
let sickLabels = List.replicate 10 1 @ List.replicate 990 0
let alwaysHealthy = List.replicate 1000 0

let sample =
    { TruePositives = 3
      FalsePositives = 1
      FalseNegatives = 2
      TrueNegatives = 4 }

[<Fact>]
let ``confusionMatrix は 4 つの場合を数える`` () =
    let matrix = confusionMatrix [ 1; 1; 0; 0 ] [ 1; 0; 1; 0 ]
    Assert.Equal(1, matrix.TruePositives)
    Assert.Equal(1, matrix.FalseNegatives)
    Assert.Equal(1, matrix.FalsePositives)
    Assert.Equal(1, matrix.TrueNegatives)
    Assert.Equal(4, matrix.Total)

[<Fact>]
let ``正解率が高くてもモデルが役立たずなことがある`` () =
    let matrix = confusionMatrix sickLabels alwaysHealthy
    // 990/1000 を当てているので正解率は 99%
    Assert.Equal(0.99, accuracy matrix, 6)
    // しかし病人を 1 人も見つけられていない
    Assert.Equal(0.0, recall matrix, 6)

[<Fact>]
let ``precision は陽性予測の信頼度`` () =
    // 陽性と予測した 4 件のうち 3 件が当たり
    Assert.Equal(0.75, precision sample, 6)

[<Fact>]
let ``recall は陽性をどれだけ拾えたか`` () =
    // 実際の陽性 5 件のうち 3 件を拾えた
    Assert.Equal(0.6, recall sample, 6)

[<Fact>]
let ``precision と recall はトレードオフ`` () =
    // 全員を陽性と予測すると再現率は 1.0 だが適合率は下がる
    let aggressive = confusionMatrix sickLabels (List.replicate 1000 1)
    Assert.Equal(1.0, recall aggressive, 6)
    Assert.Equal(0.01, precision aggressive, 6)

[<Fact>]
let ``f1Score は調和平均`` () =
    Assert.Equal(2.0 * 0.75 * 0.6 / (0.75 + 0.6), f1Score sample, 6)

[<Fact>]
let ``f1Score は偏りを算術平均より強く罰する`` () =
    // 適合率 1.0、再現率 0.1 のモデル
    let unbalanced =
        { TruePositives = 1
          FalsePositives = 0
          FalseNegatives = 9
          TrueNegatives = 90 }

    let arithmeticMean = (precision unbalanced + recall unbalanced) / 2.0
    Assert.True(f1Score unbalanced < arithmeticMean)

[<Fact>]
let ``fBetaScore は beta が大きいほど再現率を重視する`` () =
    // 適合率 0.75、再現率 0.6
    Assert.True(fBetaScore 2.0 sample < f1Score sample)
    Assert.True(fBetaScore 0.5 sample > f1Score sample)

[<Fact>]
let ``定義できない指標は 0 を返す`` () =
    let empty =
        { TruePositives = 0
          FalsePositives = 0
          FalseNegatives = 0
          TrueNegatives = 0 }

    Assert.Equal(0.0, accuracy empty, 6)
    Assert.Equal(0.0, precision empty, 6)
    Assert.Equal(0.0, recall empty, 6)
    Assert.Equal(0.0, f1Score empty, 6)

[<Fact>]
let ``predictionsAtThreshold は閾値で 0 と 1 に分ける`` () =
    let probabilities = [ 0.9; 0.6; 0.4; 0.1 ]
    Assert.Equal<int list>([ 1; 1; 0; 0 ], predictionsAtThreshold 0.5 probabilities)
    Assert.Equal<int list>([ 0; 0; 0; 0 ], predictionsAtThreshold 0.95 probabilities)
    Assert.Equal<int list>([ 1; 1; 1; 1 ], predictionsAtThreshold 0.05 probabilities)

[<Fact>]
let ``閾値を下げると再現率が上がり適合率が下がる`` () =
    let labels = [ 1; 1; 0; 0 ]
    let probabilities = [ 0.9; 0.4; 0.6; 0.1 ]
    let strict = confusionMatrix labels (predictionsAtThreshold 0.8 probabilities)
    let loose = confusionMatrix labels (predictionsAtThreshold 0.3 probabilities)
    Assert.True(recall loose > recall strict)
    Assert.True(precision loose < precision strict)

[<Fact>]
let ``rocPoints は両端の角から始まり角で終わる`` () =
    let points = rocPoints [ 1; 1; 0; 0 ] [ 0.9; 0.6; 0.4; 0.1 ]
    Assert.Equal((0.0, 0.0), List.head points)
    Assert.Equal((1.0, 1.0), List.last points)

[<Fact>]
let ``auc は完全な順位付けで 1`` () =
    Assert.Equal(1.0, auc [ 1; 1; 0; 0 ] [ 0.9; 0.8; 0.2; 0.1 ], 6)

[<Fact>]
let ``auc は情報のない順位付けで 0.5`` () =
    // 陽性が最上位と最下位に 1 つずつ
    Assert.Equal(0.5, auc [ 1; 0; 0; 1 ] [ 0.8; 0.6; 0.4; 0.2 ], 6)

[<Fact>]
let ``auc は正しく並んだ組の割合`` () =
    // 陽性 2 件と陰性 2 件の組 4 通りのうち、3 通りで陽性が上位
    Assert.Equal(0.75, auc [ 1; 0; 1; 0 ] [ 0.8; 0.6; 0.4; 0.2 ], 6)

[<Fact>]
let ``auc は完全に逆転した順位付けで 0`` () =
    Assert.Equal(0.0, auc [ 0; 0; 1; 1 ] [ 0.9; 0.8; 0.2; 0.1 ], 6)

[<Fact>]
let ``auc は閾値に依存しない`` () =
    let labels = [ 1; 1; 0; 0 ]
    let scaled = [ 0.99; 0.98; 0.02; 0.01 ]
    let compressed = [ 0.55; 0.54; 0.46; 0.45 ]
    Assert.Equal(auc labels scaled, auc labels compressed, 6)
