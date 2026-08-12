/// 第 11 章: サポートベクターマシンとカーネル法。
/// 第 5 章のパーセプトロンは「分ければどこでもよい」だった。SVM は
/// 2 クラスの間にできるだけ広い余白（マージン）を空ける境界線を選ぶ。
module GrokkingMl.Ch11Svm

open System

/// 特徴量ベクトル。
type Point = float list

/// 2 点の類似度を測る関数。
type Kernel = Point -> Point -> float

/// 内積。
let dot (a: Point) (b: Point) = List.map2 (*) a b |> List.sum

/// 線形カーネル。ただの内積。
let linearKernel: Kernel = dot

/// 多項式カーネル。特徴量の積を暗黙のうちに作る。
let polynomialKernel (degree: int) (constant: float) : Kernel =
    fun a b -> (dot a b + constant) ** float degree

/// RBF（ガウシアン）カーネル。距離が近いほど 1 に近づく。
let rbfKernel (gamma: float) : Kernel =
    fun a b ->
        let squaredDistance = List.map2 (fun x y -> (x - y) ** 2.0) a b |> List.sum
        exp (-gamma * squaredDistance)

/// 線形 SVM。ラベルは +1 と -1 を使う。
type SupportVectorMachine =
    { Weights: float list
      Bias: float }

/// 判別スコア。
let score (model: SupportVectorMachine) (point: Point) = model.Bias + dot model.Weights point

/// +1 / -1 による分類。
let predict (model: SupportVectorMachine) (point: Point) =
    if score model point >= 0.0 then 1 else -1

/// マージン幅。境界線からもっとも近い点までの距離の 2 倍。
let margin (model: SupportVectorMachine) (points: Point list) =
    let norm = sqrt (model.Weights |> List.sumBy (fun w -> w * w))

    if norm = 0.0 then
        0.0
    else
        2.0 * (points |> List.map (score model >> abs) |> List.min) / norm

/// ヒンジ損失。マージンの内側に入った分だけ罰する。
let hingeLoss (model: SupportVectorMachine) (point: Point) (label: int) =
    max 0.0 (1.0 - float label * score model point)

/// SVM の目的関数。ヒンジ損失の平均 + 重みの大きさへの罰。
let svmError
    (regularization: float)
    (model: SupportVectorMachine)
    (points: Point list)
    (labels: int list)
    =
    let losses = List.map2 (hingeLoss model) points labels |> List.average
    losses + regularization * (model.Weights |> List.sumBy (fun w -> w * w))

/// 1 点分の更新。マージンの内側なら押し返し、常に重みを縮める。
let svmStep
    (learningRate: float)
    (regularization: float)
    (model: SupportVectorMachine)
    (point: Point)
    (label: int)
    =
    let insideMargin = float label * score model point < 1.0

    let weights =
        List.map2
            (fun w x ->
                let gradient =
                    2.0 * regularization * w - (if insideMargin then float label * x else 0.0)

                w - learningRate * gradient)
            model.Weights
            point

    { Weights = weights
      Bias = model.Bias + (if insideMargin then learningRate * float label else 0.0) }

/// SVM を学習する。モデルとエポックごとの目的関数値を返す。
let trainSvmWith
    (learningRate: float)
    (epochs: int)
    (regularization: float)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)

    let initial =
        { Weights = List.replicate (List.length (List.head points)) 0.0
          Bias = 0.0 }

    let step (model, errors) _ =
        let errors = svmError regularization model points labels :: errors
        let i = rng.Next(List.length points)
        let model = svmStep learningRate regularization model points[i] labels[i]
        (model, errors)

    let model, errors = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev errors)

/// 既定の設定（学習率 0.01、5000 エポック、正則化 0.1、シード 0）で学習する。
let trainSvm (points: Point list) (labels: int list) =
    trainSvmWith 0.01 5000 0.1 0 points labels

/// 正解率。
let accuracy (model: SupportVectorMachine) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict model point = label then 1.0 else 0.0) points labels
    |> List.average

/// カーネル分類器。訓練点そのものを重み付きで覚えておく。
type KernelClassifier =
    { Points: Point list
      Labels: int list
      Weights: float list
      Bias: float
      Kernel: Kernel }

/// カーネルによる判別スコア。
let kernelScore (model: KernelClassifier) (point: Point) =
    List.zip3 model.Weights model.Labels model.Points
    |> List.sumBy (fun (weight, label, support) -> weight * float label * model.Kernel support point)
    |> (+) model.Bias

/// カーネル分類器による分類。
let kernelPredict (model: KernelClassifier) (point: Point) =
    if kernelScore model point >= 0.0 then 1 else -1

/// カーネル版パーセプトロン。誤分類した点の重みだけを増やす。
let trainKernelClassifierWith
    (kernel: Kernel)
    (learningRate: float)
    (epochs: int)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)

    let initial =
        { Points = points
          Labels = labels
          Weights = List.replicate (List.length points) 0.0
          Bias = 0.0
          Kernel = kernel }

    let step (model: KernelClassifier) _ =
        let i = rng.Next(List.length points)

        if float labels[i] * kernelScore model points[i] <= 0.0 then
            { model with
                Weights = model.Weights |> List.mapi (fun j w -> if j = i then w + learningRate else w)
                Bias = model.Bias + learningRate * float labels[i] }
        else
            model

    List.fold step initial [ 1..epochs ]

/// 既定の設定（学習率 0.1、2000 エポック、シード 0）で学習する。
let trainKernelClassifier (kernel: Kernel) (points: Point list) (labels: int list) =
    trainKernelClassifierWith kernel 0.1 2000 0 points labels

/// カーネル分類器の正解率。
let kernelAccuracy (model: KernelClassifier) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if kernelPredict model point = label then 1.0 else 0.0) points labels
    |> List.average
