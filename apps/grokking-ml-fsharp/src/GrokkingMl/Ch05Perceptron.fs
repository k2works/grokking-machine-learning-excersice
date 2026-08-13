/// 第 5 章: パーセプトロン。
/// 点を直線で 2 クラスに分ける。予測が外れた点だけを使って境界線を動かす
/// パーセプトロントリックを実装する。
module GrokkingMl.Ch05Perceptron

open System

/// 特徴量ベクトル。
type Point = float list

/// 線形分類器 score = w・x + bias。score >= 0 なら 1、そうでなければ 0。
type Perceptron =
    { Weights: float list
      Bias: float }

/// 重み付き和。
let score (model: Perceptron) (point: Point) =
    List.map2 (*) model.Weights point
    |> List.sum
    |> (+) model.Bias

/// スコアの符号による分類。境界線上は正のクラスとして扱う。
let predict (model: Perceptron) (point: Point) = if score model point >= 0.0 then 1 else 0

/// パーセプトロントリック。誤分類した点だけモデルを動かす。
let perceptronTrick (learningRate: float) (model: Perceptron) (point: Point) (label: int) =
    let error = float (label - predict model point)

    if error = 0.0 then
        model
    else
        { Weights = List.map2 (fun w x -> w + learningRate * error * x) model.Weights point
          Bias = model.Bias + learningRate * error }

/// 1 点分の誤差。正しく分類していれば 0、誤っていればスコアの絶対値。
let perceptronError (model: Perceptron) (point: Point) (label: int) =
    if predict model point = label then
        0.0
    else
        abs (score model point)

/// 全点の平均誤差。
let meanPerceptronError (model: Perceptron) (points: Point list) (labels: int list) =
    List.map2 (perceptronError model) points labels
    |> List.average

/// 正解率。
let accuracy (model: Perceptron) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict model point = label then 1.0 else 0.0) points labels
    |> List.average

/// パーセプトロンアルゴリズム。モデルとエポックごとの平均誤差を返す。
let perceptronAlgorithm
    (learningRate: float)
    (epochs: int)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)
    let dimensions = List.length (List.head points)
    let initial = { Weights = List.replicate dimensions 0.0; Bias = 0.0 }

    let step (model, errors) _ =
        let errors = meanPerceptronError model points labels :: errors
        let i = rng.Next(List.length points)
        let model = perceptronTrick learningRate model points[i] labels[i]
        (model, errors)

    let model, errors = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev errors)
