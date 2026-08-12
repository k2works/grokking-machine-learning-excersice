/// 第 4 章: 過学習・未学習と正則化。
/// 多項式回帰で次数を変えながら、訓練データとテストデータの誤差を比べる。
/// さらに L1 / L2 正則化で係数を抑え、複雑すぎるモデルを緩める。
module GrokkingMl.Ch04Regularization

open System

/// 正則化の種類。
type Regularization =
    | NoRegularization
    | L1
    | L2

/// 多項式モデル y = bias + w1*x + w2*x^2 + ... + wn*x^n。
type PolynomialModel =
    { Weights: float list
      Bias: float }

    member this.Degree = List.length this.Weights

/// x から [x; x^2; ...; x^degree] を作る。
let polynomialFeatures (x: float) (degree: int) =
    [ for power in 1..degree -> x ** float power ]

/// モデルによる予測。
let predict (model: PolynomialModel) (x: float) =
    model.Weights
    |> List.mapi (fun index w -> w * x ** float (index + 1))
    |> List.sum
    |> (+) model.Bias

/// 正則化項の勾配。重みを 0 に引き戻す向きの力を返す。
let regularizationGradient (weight: float) (kind: Regularization) (strength: float) =
    match kind with
    | NoRegularization -> 0.0
    | L1 -> strength * (if weight > 0.0 then 1.0 elif weight < 0.0 then -1.0 else 0.0)
    | L2 -> strength * 2.0 * weight

/// 二乗トリックに正則化項を加えた 1 点分の更新。
let squareTrick
    (learningRate: float)
    (kind: Regularization)
    (strength: float)
    (model: PolynomialModel)
    (x: float)
    (y: float)
    =
    let error = y - predict model x
    let features = polynomialFeatures x model.Degree

    let weights =
        List.map2
            (fun w feature ->
                w + learningRate * (error * feature - regularizationGradient w kind strength))
            model.Weights
            features

    { Weights = weights
      Bias = model.Bias + learningRate * error }

/// 二乗平均平方根誤差。
let rmse (labels: float list) (predictions: float list) =
    let n = float (List.length labels)

    List.zip labels predictions
    |> List.sumBy (fun (label, prediction) -> (label - prediction) ** 2.0)
    |> fun total -> sqrt (total / n)

/// モデルの RMSE。
let modelRmse (model: PolynomialModel) (features: float list) (labels: float list) =
    rmse labels (features |> List.map (predict model))

/// 訓練用とテスト用に分割したデータセット。
type Split =
    { TrainFeatures: float list
      TrainLabels: float list
      TestFeatures: float list
      TestLabels: float list }

/// データを訓練用とテスト用に分割する。
let trainTestSplit (testRatio: float) (seed: int) (features: float list) (labels: float list) =
    let rng = Random(seed)

    let indices =
        List.init (List.length features) id
        |> List.sortBy (fun _ -> rng.Next())

    let testSize = int (float (List.length features) * testRatio)
    let testIndices = List.truncate testSize indices
    let trainIndices = List.skip testSize indices

    { TrainFeatures = trainIndices |> List.map (fun i -> features[i])
      TrainLabels = trainIndices |> List.map (fun i -> labels[i])
      TestFeatures = testIndices |> List.map (fun i -> features[i])
      TestLabels = testIndices |> List.map (fun i -> labels[i]) }

/// 確率的勾配降下法で多項式回帰を学習する。
let polynomialRegression
    (degree: int)
    (learningRate: float)
    (epochs: int)
    (kind: Regularization)
    (strength: float)
    (seed: int)
    (features: float list)
    (labels: float list)
    =
    let rng = Random(seed)
    let initial = { Weights = List.replicate degree 0.0; Bias = 0.0 }

    let step model _ =
        let i = rng.Next(List.length features)
        squareTrick learningRate kind strength model features[i] labels[i]

    List.fold step initial [ 1..epochs ]

/// 既定のハイパーパラメータ（学習率 0.01、20000 エポック、シード 0）で学習する。
let train (degree: int) (kind: Regularization) (strength: float) =
    polynomialRegression degree 0.01 20000 kind strength 0

/// 重みの絶対値の合計。モデルの複雑さの目安。
let weightMagnitude (model: PolynomialModel) = model.Weights |> List.sumBy abs
