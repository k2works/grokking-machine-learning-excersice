/// 第 6 章: ロジスティック回帰。
/// パーセプトロンの「0 か 1 か」という硬い予測を、シグモイド関数で
/// 「0 から 1 の確率」という連続的な予測に置き換える。
module GrokkingMl.Ch06LogisticRegression

open System

/// 特徴量ベクトル。
type Point = float list

/// シグモイド関数。実数を 0 から 1 の範囲へ押し込む。
let sigmoid (x: float) =
    if x >= 0.0 then
        1.0 / (1.0 + exp (-x))
    else
        // x が大きな負の数のとき exp(-x) が溢れるため、数学的に等価な式へ切り替える
        let exponential = exp x
        exponential / (1.0 + exponential)

/// ロジスティック分類器。予測は 0 から 1 の確率。
type LogisticClassifier =
    { Weights: float list
      Bias: float }

/// 重み付き和。
let score (model: LogisticClassifier) (point: Point) =
    List.map2 (*) model.Weights point
    |> List.sum
    |> (+) model.Bias

/// 0 から 1 の確率としての予測。
let predictProbability (model: LogisticClassifier) (point: Point) = sigmoid (score model point)

/// 閾値による 0 / 1 の分類。
let predictWith (threshold: float) (model: LogisticClassifier) (point: Point) =
    if predictProbability model point >= threshold then 1 else 0

/// 閾値 0.5 による分類。
let predict (model: LogisticClassifier) (point: Point) = predictWith 0.5 model point

/// 1 点分の対数損失。予測確率が正解から離れるほど大きくなる。
let logLoss (model: LogisticClassifier) (point: Point) (label: int) =
    // log(0) を避けるためにごくわずかに内側へ丸める
    let epsilon = 1e-15

    let probability =
        predictProbability model point
        |> max epsilon
        |> min (1.0 - epsilon)

    if label = 1 then
        -log probability
    else
        -log (1.0 - probability)

/// 全点の平均対数損失。
let meanLogLoss (model: LogisticClassifier) (points: Point list) (labels: int list) =
    List.map2 (logLoss model) points labels |> List.average

/// ロジスティックトリック。すべての点を、確率の外れ具合に比例して動かす。
let logisticTrick (learningRate: float) (model: LogisticClassifier) (point: Point) (label: int) =
    let error = float label - predictProbability model point

    { Weights = List.map2 (fun w x -> w + learningRate * error * x) model.Weights point
      Bias = model.Bias + learningRate * error }

/// 正解率。
let accuracy (model: LogisticClassifier) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict model point = label then 1.0 else 0.0) points labels
    |> List.average

/// ロジスティック回帰。モデルとエポックごとの平均対数損失を返す。
let logisticRegression
    (learningRate: float)
    (epochs: int)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)
    let dimensions = List.length (List.head points)
    let initial = { Weights = List.replicate dimensions 0.0; Bias = 0.0 }

    let step (model, losses) _ =
        let losses = meanLogLoss model points labels :: losses
        let i = rng.Next(List.length points)
        let model = logisticTrick learningRate model points[i] labels[i]
        (model, losses)

    let model, losses = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev losses)
