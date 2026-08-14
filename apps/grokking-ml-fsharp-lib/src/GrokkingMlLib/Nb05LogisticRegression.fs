/// 原著ノートブック #05 `Chapter_06_Logistic_Regression/Coding_logistic_regression.ipynb`。
///
/// #04 と同じ形の 8 点を、今度はロジスティック回帰で分ける。
/// ステップ関数がシグモイドに、パーセプトロン誤差が対数損失に置き換わる。
///
/// 原著は対数損失の「別の書き方」も示しているが、**その式は対数損失と一致しない**。
/// 正しい形と並べて、両方を実装してある。
module GrokkingMlLib.Nb05LogisticRegression

open System
open Microsoft.ML
open Microsoft.ML.Data

/// 原著が使う 8 点。#04 と似ているが最後の 2 点が入れ替わっている
let features =
    [| [| 1; 0 |]; [| 0; 2 |]; [| 1; 1 |]; [| 1; 2 |]
       [| 1; 3 |]; [| 2; 2 |]; [| 3; 2 |]; [| 2; 3 |] |]

let labels = [| 0; 0; 0; 0; 1; 1; 1; 1 |]

/// シグモイド関数。
///
/// 原著のコメントどおり `exp(x) / (1 + exp(x))` で書く。教科書によくある
/// `1 / (1 + exp(-x))` と数学的には同じだが、x が大きな負の数のときに
/// `exp(-x)` が溢れない形になっている。
let sigmoid (x: float) = exp x / (1.0 + exp x)

/// ソフト ReLU。`log(1 + exp(x))` で、ReLU をなめらかにしたもの
let softRelu (x: float) = log (1.0 + exp x)

/// 分離直線。重みとバイアスの組
type Boundary =
    { Weights: float[]
      Bias: float }

    /// 重み付き和にバイアスを足したもの
    member this.Score(features: int[]) =
        this.Bias
        + (Array.map2 (fun w (f: int) -> w * float f) this.Weights features |> Array.sum)

    /// 予測確率。0 / 1 ではなく 0〜1 の連続値を返す
    member this.Prediction(features: int[]) = sigmoid (this.Score features)

    /// 対数損失。当たっていても 0 にはならず、確信の度合いで連続的に変わる
    member this.LogLoss(features: int[], label: int) =
        let pred = this.Prediction features
        -float label * log pred - float (1 - label) * log (1.0 - pred)

/// 学習の結果と途中経過
type TrainingLog =
    { Boundary: Boundary
      /// 各エポック開始時点の対数損失の合計
      Errors: float list }

/// 全点の対数損失の合計。原著は平均ではなく合計を取っている
let totalLogLoss (boundary: Boundary) (features: int[][]) (labels: int[]) =
    Array.map2 (fun f l -> boundary.LogLoss(f, l)) features labels |> Array.sum

/// 原著が「対数損失の別の書き方」として示す式。
///
/// **実際には対数損失と一致しない。** `pred` は 0〜1 の確率なので、
/// `(pred - label)` は -1 か +1 ではなく中間の値になる。
/// スコアが 0 のときだけ両者が一致する。詳しくは記事を参照。
let alternateLogLossOriginal (boundary: Boundary) (features: int[]) (label: int) =
    softRelu ((boundary.Prediction features - float label) * boundary.Score features)

/// 対数損失と厳密に等しい「別の書き方」。
///
/// ラベルが 0 なら +1、1 なら -1 を掛ける。つまり `(1 - 2 * label)`。
let alternateLogLoss (boundary: Boundary) (features: int[]) (label: int) =
    softRelu (float (1 - 2 * label) * boundary.Score features)

/// ロジスティックトリック。パーセプトロンのトリックと同じ形をしている。
///
/// 違いは `pred` が 0 / 1 ではなく 0〜1 の連続値であること。
/// #04 の「短く書いた版」と違い、**バイアスの更新はループの外側にある**。
let logisticTrick (learningRate: float) (boundary: Boundary) (features: int[]) (label: int) =
    let pred = boundary.Prediction features
    let delta = float label - pred

    { Weights =
        Array.map2 (fun w (f: int) -> w + delta * float f * learningRate) boundary.Weights features
      Bias = boundary.Bias + delta * learningRate }

/// トリックを繰り返して分離直線を学習する。
///
/// #04 と同じく、原著は点の選択に種を与えていない標準ライブラリの `random` を
/// 使っている。原著の出力 `([1.2019, 0.7009], -2.7884)` は再現できない。
let logisticRegressionAlgorithm
    (features: int[][])
    (labels: int[])
    (learningRate: float)
    (epochs: int)
    (seed: int)
    =
    let random = Random(seed)
    let mutable boundary = { Weights = Array.create features.[0].Length 1.0; Bias = 0.0 }
    let mutable errors = []

    for _ in 1..epochs do
        errors <- totalLogLoss boundary features labels :: errors
        let j = random.Next features.Length
        boundary <- logisticTrick learningRate boundary features.[j] labels.[j]

    { Boundary = boundary
      Errors = List.rev errors }

[<CLIMutable>]
type Sample =
    { [<VectorType(2)>]
      Features: single[]
      Label: bool }

[<CLIMutable>]
type Prediction =
    { PredictedLabel: bool
      Probability: single }

/// 同じ問題を ML.NET の `LbfgsLogisticRegression` に解かせる。
///
/// scikit-learn の `LogisticRegression` は既定で lbfgs ソルバを使う。
/// ML.NET も同名のトレーナーを持つので、いちばん近い対応になる。
let fitWithMlNet (features: int[][]) (labels: int[]) =
    let ctx = MLContext(seed = 0)

    let rows =
        Array.map2
            (fun (f: int[]) (l: int) ->
                { Features = f |> Array.map single
                  Label = (l = 1) })
            features
            labels

    let data = ctx.Data.LoadFromEnumerable rows
    let model = ctx.BinaryClassification.Trainers.LbfgsLogisticRegression().Fit data
    let engine = ctx.Model.CreatePredictionEngine<Sample, Prediction> model

    fun (row: int[]) ->
        let prediction =
            engine.Predict { Features = row |> Array.map single; Label = false }

        if prediction.PredictedLabel then 1 else 0
