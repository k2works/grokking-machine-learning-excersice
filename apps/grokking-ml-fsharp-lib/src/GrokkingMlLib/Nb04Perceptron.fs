/// 原著ノートブック #04 `Chapter_05_Perceptron_Algorithm/Coding_perceptron_algorithm.ipynb`。
///
/// 2 次元の 8 点を直線で 2 クラスに分ける。パーセプトロンのトリックを手で書いてから、
/// 同じ問題を ML.NET の `AveragedPerceptron` に解かせる。
///
/// 原著はトリックを 2 通り書いており、**2 つ目は 1 つ目と挙動が違う**。
/// バイアスの更新が重みのループの内側にあり、特徴量の数だけ繰り返し適用されるためである。
/// 原著のセル出力もその挙動を前提にしているので、両方を実装して差が見えるようにした。
module GrokkingMlLib.Nb04Perceptron

open System
open Microsoft.ML
open Microsoft.ML.Data

/// 原著が使う 8 点。aack と beep の出現回数を模した 2 次元の特徴量
let features =
    [| [| 1; 0 |]; [| 0; 2 |]; [| 1; 1 |]; [| 1; 2 |]
       [| 1; 3 |]; [| 2; 2 |]; [| 2; 3 |]; [| 3; 2 |] |]

let labels = [| 0; 0; 0; 0; 1; 1; 1; 1 |]

/// ステップ関数。0 以上なら 1、そうでなければ 0
let step (x: float) = if x >= 0.0 then 1 else 0

/// 分離直線。重みとバイアスの組
type Boundary =
    { Weights: float[]
      Bias: float }

    /// 重み付き和にバイアスを足したもの。直線からの符号つき距離に比例する
    member this.Score(features: int[]) =
        this.Bias + (Array.map2 (fun w (f: int) -> w * float f) this.Weights features |> Array.sum)

    /// スコアをステップ関数に通した予測ラベル
    member this.Predict(features: int[]) = step (this.Score features)

    /// パーセプトロン誤差。当たっていれば 0、外れていればスコアの絶対値
    member this.Error(features: int[], label: int) =
        if this.Predict features = label then 0.0 else abs (this.Score features)

/// 学習の結果と途中経過
type TrainingLog =
    { Boundary: Boundary
      /// 各エポック開始時点の平均パーセプトロン誤差
      Errors: float list }

/// 全点のパーセプトロン誤差の平均
let meanPerceptronError (boundary: Boundary) (features: int[][]) (labels: int[]) =
    Array.map2 (fun f l -> boundary.Error(f, l)) features labels
    |> Array.sum
    |> fun total -> total / float features.Length

/// 原著が最初に示すトリック。当たっていれば何もせず、外れたら向きを見て動かす。
///
/// バイアスの更新はループの **外側** にあり、1 回だけ適用される。
let perceptronTrickExplicit
    (learningRate: float)
    (boundary: Boundary)
    (features: int[])
    (label: int)
    =
    let pred = boundary.Predict features

    if pred = label then
        boundary
    elif label = 1 && pred = 0 then
        { Weights = Array.map2 (fun w (f: int) -> w + float f * learningRate) boundary.Weights features
          Bias = boundary.Bias + learningRate }
    else
        { Weights = Array.map2 (fun w (f: int) -> w - float f * learningRate) boundary.Weights features
          Bias = boundary.Bias - learningRate }

/// 原著が「短く書いた版」として示すトリック。以降の学習ループはこちらを使う。
///
/// `label - pred` が符号を持つので分岐が要らなくなる。ただし原著のコードでは
/// **バイアスの更新が重みのループの内側にある**。特徴量が 2 つなら学習率が
/// 2 回足され、`perceptronTrickExplicit` の 2 倍動く。
///
/// 原著のセル出力（`[0.9, 1.85], -4.1`）はこの挙動を前提にしているので、そのまま写している。
let perceptronTrick (learningRate: float) (boundary: Boundary) (features: int[]) (label: int) =
    let pred = boundary.Predict features
    let delta = float (label - pred)

    { Weights =
        Array.map2 (fun w (f: int) -> w + delta * float f * learningRate) boundary.Weights features
      // 原著はここをループの内側に置いているので、特徴量の数だけ足される
      Bias = boundary.Bias + delta * learningRate * float features.Length }

/// トリックを繰り返して分離直線を学習する。
///
/// 原著は `np.random.seed(42)` を呼んでいるが、点の選択に使っているのは
/// **標準ライブラリの `random.randint`** で、こちらに種を与えていない。
/// 原著の出力 `([0.55, 0.25], -1.1)` は実行のたびに変わる値であり再現できない。
let perceptronAlgorithm
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
        errors <- meanPerceptronError boundary features labels :: errors
        let i = random.Next features.Length
        boundary <- perceptronTrick learningRate boundary features.[i] labels.[i]

    { Boundary = boundary
      Errors = List.rev errors }

[<CLIMutable>]
type Sample =
    { [<VectorType(2)>]
      Features: single[]
      Label: bool }

[<CLIMutable>]
type Prediction = { PredictedLabel: bool }

/// 同じ問題を ML.NET の `AveragedPerceptron` に解かせる。
///
/// scikit-learn の `Perceptron` にいちばん近いトレーナーである。ML.NET は
/// 二値分類のラベルを **bool** で扱うので、0 / 1 から変換する。
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
    let trainer = ctx.BinaryClassification.Trainers.AveragedPerceptron(numberOfIterations = 100)
    let model = trainer.Fit data
    let engine = ctx.Model.CreatePredictionEngine<Sample, Prediction> model

    fun (row: int[]) ->
        let prediction =
            engine.Predict { Features = row |> Array.map single; Label = false }

        if prediction.PredictedLabel then 1 else 0
