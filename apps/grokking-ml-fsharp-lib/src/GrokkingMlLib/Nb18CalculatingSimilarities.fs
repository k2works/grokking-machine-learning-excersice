/// 原著ノートブック #18 `Chapter_11_Support_Vector_Machines/Calculating_similarities.ipynb`。
///
/// RBF カーネルの正体を **手で計算して見せる** 回である。
///
/// 7 点の小さなデータに対して全対の類似度を計算し、7 列の特徴量を足してから
/// **線形** SVM を当てる。カーネルトリックを使わず、
/// 「カーネルとは特徴量を増やすことだ」を目に見える形にしている。
module GrokkingMlLib.Nb18CalculatingSimilarities

open Microsoft.ML
open Microsoft.ML.Data

/// 原著が予測に使う符号。ラベル 0 を -1 に読み替えたもの
let predictionSigns = [| -1.0; -1.0; -1.0; 1.0; 1.0; 1.0; 1.0 |]

/// 原著の 7 点。原点とその周りに 6 点が並ぶ
let x1 = [| 0.0; -1.0; 0.0; 0.0; 1.0; -1.0; 1.0 |]
let x2 = [| 0.0; 0.0; -1.0; 1.0; 0.0; 1.0; -1.0 |]
let y = [| 0; 0; 0; 1; 1; 1; 1 |]

/// 点の数
[<Literal>]
let Size = 7

/// 元の 2 列に類似度 7 列を足した特徴量の数
[<Literal>]
let FeatureCount = 9

/// 座標の並び。原著の `data[['x1','x2']]`
let points = Array.init Size (fun i -> [| x1.[i]; x2.[i] |])

/// 原著の類似度。RBF（ガウス）カーネルそのもの。
///
/// `exp(-(x1-y1)^2 - (x2-y2)^2)` は `exp(-||x-y||^2)`、
/// つまり γ = 1 の RBF カーネルである。
/// 同じ点なら 1、離れるほど急速に 0 に近づく。
let similarity (a: float[]) (b: float[]) =
    let dx = a.[0] - b.[0]
    let dy = a.[1] - b.[1]
    exp (-dx * dx - dy * dy)

/// 全対の類似度を並べた 7 × 7 の行列
let similarityMatrix () =
    Array.init Size (fun i -> Array.init Size (fun j -> similarity points.[i] points.[j]))

/// SVM に渡す特徴量。x1・x2 に類似度 7 列を足した 9 列
let features () =
    let matrix = similarityMatrix ()

    Array.init Size (fun row ->
        Array.init FeatureCount (fun column ->
            match column with
            | 0 -> x1.[row]
            | 1 -> x2.[row]
            | _ -> matrix.[column - 2].[row]))

[<CLIMutable>]
type Sample =
    { [<VectorType(9)>]
      Features: single[]
      Label: bool }

[<CLIMutable>]
type Prediction = { PredictedLabel: bool; Score: single }

/// **線形** SVM を、類似度を足した特徴量に当てる。
///
/// `kernel='rbf'` を使わないのが要点である。カーネルは既に特徴量として
/// 展開済みなので、線形で足りる。
///
/// ML.NET の `LinearSvm` が scikit-learn の `SVC(kernel='linear')` に
/// いちばん近い。ただし **解き方が違う**（ML.NET は確率的勾配法、
/// scikit-learn は libsvm の SMO）ので、係数は一致しない。
///
/// **`numberOfIterations` を明示しないと原点を誤分類する。** 既定のままだと
/// 7 点のうち原点（ラベル 0）だけを 1 と答えた。原点は 6 点のちょうど真ん中で
/// 最も判定が難しい。1000 回に増やすと 7 点すべて正しくなる。
let fit () =
    let ctx = MLContext(seed = 0)

    let rows =
        Array.init Size (fun i ->
            { Features = features().[i] |> Array.map single
              Label = (y.[i] = 1) })

    let data = ctx.Data.LoadFromEnumerable rows
    let model = ctx.BinaryClassification.Trainers.LinearSvm(numberOfIterations = 1000).Fit data
    let engine = ctx.Model.CreatePredictionEngine<Sample, Prediction> model

    fun (row: float[]) ->
        let prediction =
            engine.Predict
                { Features = row |> Array.map single
                  Label = false }

        if prediction.PredictedLabel then 1 else 0

/// 原著が手で書いた予測式。
///
/// 学習した SVM の係数ではなく、**ラベルの符号をそのまま重みにする**。
/// `similarity` の重み付き和が正なら 1、負なら 0 と読む。
let svmRbfPrediction (newPoint: float[]) =
    Array.init Size (fun i -> similarity newPoint points.[i] * predictionSigns.[i])
    |> Array.sum

/// 7 点それぞれに対する [svmRbfPrediction] の値
let trainingPredictions () = points |> Array.map svmRbfPrediction
