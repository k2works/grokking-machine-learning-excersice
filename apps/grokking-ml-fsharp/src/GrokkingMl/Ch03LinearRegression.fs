/// 第 3 章: 線形回帰。
/// 部屋数から住宅価格を予測する 1 次元の線形回帰を、
/// simple / absolute / square の 3 つのトリックで学習する。
module GrokkingMl.Ch03LinearRegression

open System

/// 1 次元の線形モデル price = slope * rooms + intercept。
type Model =
    { Slope: float
      Intercept: float }

/// モデルによる予測。
let predict (model: Model) (rooms: float) = model.Slope * rooms + model.Intercept

/// 単純なトリック。予測の上下だけを見て、ランダムな微小量だけ動かす。
let simpleTrick (rng: Random) (model: Model) (rooms: float) (price: float) =
    let stepSlope = rng.NextDouble() * 0.1
    let stepIntercept = rng.NextDouble() * 0.1
    let predicted = predict model rooms

    if price > predicted then
        { Slope = (if rooms > 0.0 then model.Slope + stepSlope else model.Slope - stepSlope)
          Intercept = model.Intercept + stepIntercept }
    else
        { Slope = (if rooms > 0.0 then model.Slope - stepSlope else model.Slope + stepSlope)
          Intercept = model.Intercept - stepIntercept }

/// 絶対トリック。誤差の符号のみを使い、特徴量に比例した量だけ動かす。
let absoluteTrick (learningRate: float) (model: Model) (rooms: float) (price: float) =
    let sign = if price > predict model rooms then 1.0 else -1.0

    { Slope = model.Slope + sign * learningRate * rooms
      Intercept = model.Intercept + sign * learningRate }

/// 二乗トリック。誤差の大きさに比例した量だけ動かす（二乗誤差の勾配降下法）。
let squareTrick (learningRate: float) (model: Model) (rooms: float) (price: float) =
    let error = price - predict model rooms

    { Slope = model.Slope + learningRate * rooms * error
      Intercept = model.Intercept + learningRate * error }

/// 二乗平均平方根誤差。
let rmse (labels: float list) (predictions: float list) =
    let n = float (List.length labels)

    List.zip labels predictions
    |> List.sumBy (fun (label, prediction) -> (label - prediction) ** 2.0)
    |> fun total -> sqrt (total / n)

/// モデルの RMSE。
let modelRmse (model: Model) (features: float list) (labels: float list) =
    rmse labels (features |> List.map (predict model))

/// 確率的勾配降下法で線形回帰を学習し、モデルとエポックごとの RMSE を返す。
let linearRegression (learningRate: float) (epochs: int) (seed: int) (features: float list) (labels: float list) =
    let rng = Random(seed)
    let initial = { Slope = rng.NextDouble(); Intercept = rng.NextDouble() }

    let step (model, errors) _ =
        let errors = modelRmse model features labels :: errors
        let i = rng.Next(List.length features)
        let model = squareTrick learningRate model features[i] labels[i]
        (model, errors)

    let model, errors = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev errors)
