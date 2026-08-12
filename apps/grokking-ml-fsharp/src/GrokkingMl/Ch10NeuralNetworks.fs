/// 第 10 章: ニューラルネットワーク。
/// パーセプトロンを積み重ねて、直線では分けられないデータを分ける。
/// 学習は誤差逆伝播法（連鎖律による勾配の伝播）で行う。
module GrokkingMl.Ch10NeuralNetworks

open System

/// 特徴量ベクトル。
type Point = float list

/// シグモイド関数。第 6 章と同じ数値的に安定な実装。
let sigmoid (x: float) =
    if x >= 0.0 then
        1.0 / (1.0 + exp (-x))
    else
        let exponential = exp x
        exponential / (1.0 + exponential)

/// シグモイドの微分。出力そのものから計算できる。
let sigmoidDerivative (output: float) = output * (1.0 - output)

/// 全結合層。Weights[j][i] は入力 i から出力 j への重み。
type Layer =
    { Weights: float list list
      Biases: float list }

    member this.InputSize = List.length (List.head this.Weights)
    member this.OutputSize = List.length this.Weights

/// 順伝播。重み付き和にシグモイドを適用する。
let forward (layer: Layer) (inputs: float list) =
    List.map2
        (fun row bias -> sigmoid (bias + (List.map2 (*) row inputs |> List.sum)))
        layer.Weights
        layer.Biases

/// 多層パーセプトロン。層を順に適用する。
type NeuralNetwork = { Layers: Layer list }

/// 各層の出力を順に記録する。逆伝播で必要になる。
let forwardAll (model: NeuralNetwork) (inputs: float list) =
    model.Layers
    |> List.fold (fun activations layer -> activations @ [ forward layer (List.last activations) ]) [ inputs ]

/// 出力層の最初のニューロンの値を確率として返す。
let predictProbability (model: NeuralNetwork) (inputs: float list) =
    forwardAll model inputs |> List.last |> List.head

/// 閾値による 0 / 1 の分類。
let predictWith (threshold: float) (model: NeuralNetwork) (inputs: float list) =
    if predictProbability model inputs >= threshold then 1 else 0

/// 閾値 0.5 による分類。
let predict (model: NeuralNetwork) (inputs: float list) = predictWith 0.5 model inputs

/// 指定した層構成のネットワークを乱数で初期化する。
let initialNetwork (sizes: int list) (seed: int) =
    let rng = Random(seed)
    let nextWeight () = rng.NextDouble() * 2.0 - 1.0

    { Layers =
        List.pairwise sizes
        |> List.map (fun (inputSize, outputSize) ->
            { Weights = List.init outputSize (fun _ -> List.init inputSize (fun _ -> nextWeight ()))
              Biases = List.init outputSize (fun _ -> nextWeight ()) }) }

/// 1 点分の対数損失。第 6 章と同じ。
let logLoss (model: NeuralNetwork) (inputs: float list) (label: int) =
    let epsilon = 1e-15

    let probability =
        predictProbability model inputs |> max epsilon |> min (1.0 - epsilon)

    if label = 1 then -log probability else -log (1.0 - probability)

/// 全点の平均対数損失。
let meanLogLoss (model: NeuralNetwork) (points: Point list) (labels: int list) =
    List.map2 (logLoss model) points labels |> List.average

/// 誤差逆伝播法による 1 点分の更新。
let backpropagate (learningRate: float) (model: NeuralNetwork) (inputs: float list) (label: int) =
    let activations = forwardAll model inputs
    let layerCount = List.length model.Layers

    // 出力層の誤差（対数損失 × シグモイドの微分が predicted - label に簡約される）
    let outputDelta = [ (List.last activations |> List.head) - float label ]

    // 出力層から入力側へ、連鎖律で誤差を遡らせる
    let deltas =
        [ layerCount - 1 .. -1 .. 1 ]
        |> List.fold
            (fun accumulated index ->
                let layer = model.Layers[index]
                let downstream = List.head accumulated
                let outputs = activations[index]

                let delta =
                    List.init layer.InputSize (fun i ->
                        let weighted =
                            List.map2 (fun (row: float list) d -> row[i] * d) layer.Weights downstream
                            |> List.sum

                        weighted * sigmoidDerivative outputs[i])

                delta :: accumulated)
            [ outputDelta ]

    { Layers =
        model.Layers
        |> List.mapi (fun index layer ->
            let delta = deltas[index]
            let previous = activations[index]

            { Weights =
                layer.Weights
                |> List.mapi (fun j row ->
                    row |> List.mapi (fun i w -> w - learningRate * delta[j] * previous[i]))
              Biases = layer.Biases |> List.mapi (fun j bias -> bias - learningRate * delta[j]) }) }

/// ネットワークを学習する。モデルとエポックごとの平均損失を返す。
let trainWith
    (hiddenSize: int)
    (learningRate: float)
    (epochs: int)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)
    let initial = initialNetwork [ List.length (List.head points); hiddenSize; 1 ] seed

    let step (model, losses) _ =
        let losses = meanLogLoss model points labels :: losses
        let i = rng.Next(List.length points)
        let model = backpropagate learningRate model points[i] labels[i]
        (model, losses)

    let model, losses = List.fold step (initial, []) [ 1..epochs ]
    (model, List.rev losses)

/// 既定の設定（隠れ層 4、学習率 0.5、5000 エポック、シード 0）で学習する。
let train (points: Point list) (labels: int list) = trainWith 4 0.5 5000 0 points labels

/// 正解率。
let accuracy (model: NeuralNetwork) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict model point = label then 1.0 else 0.0) points labels
    |> List.average
