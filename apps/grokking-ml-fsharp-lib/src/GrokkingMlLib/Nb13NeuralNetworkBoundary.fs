/// 原著ノートブック #13 `Chapter_10_Neural_Networks/Graphical_example.ipynb`。
///
/// ニューラルネットワークの章に入る。円形に分布した 110 点を、
/// **2 層の隠れ層を持つネットワーク** で分類し、決定境界を見る。
///
/// 決定木（[#10](Nb10DecisionTreeBoundary.fs)）の境界が軸に平行な長方形だったのに対し、
/// ニューラルネットワークの境界は **曲線** になる。
///
/// 原著は Keras で `Dense(128, relu)` → `Dropout` → `Dense(64, relu)` → `Dropout`
/// → `Dense(2, softmax)` と組む。Accord.Neuro には **Dropout も ReLU も無い** ので、
/// シグモイドの隠れ層 2 つで代用する。
module GrokkingMlLib.Nb13NeuralNetworkBoundary

open Accord.Neuro
open Accord.Neuro.Learning
open Deedle

/// 原著のネットワークの隠れ層
let hiddenUnits = [| 128; 64 |]

/// 原著の学習回数
[<Literal>]
let OriginalEpochs = 100

/// このネットワークで十分に学習が進む回数
[<Literal>]
let Epochs = 1000

/// 学習率。Accord の既定（0.1）をそのまま使う
[<Literal>]
let LearningRate = 0.1

/// 原著の `plot_model` が使う格子の刻み幅
[<Literal>]
let PlotStep = 0.2

/// 読み込んだ円形データ
type Circle =
    { X: float[][]
      Y: int[] }

    member this.Size = this.Y.Length

/// 円形に分布したデータを読み込む。110 点
let loadCircle () =
    let frame = Datasets.loadFrame "one_circle.csv"
    let x1 = frame.GetColumn<float>("x_1") |> Series.values |> Array.ofSeq
    let x2 = frame.GetColumn<float>("x_2") |> Series.values |> Array.ofSeq
    let y = frame.GetColumn<int>("y") |> Series.values |> Array.ofSeq

    { X = Array.map2 (fun a b -> [| a; b |]) x1 x2
      Y = y }

/// 原著と同じ隠れ層を持つネットワークを学習する。
///
/// Accord の `ActivationNetwork` は「活性化関数・入力数・各層のユニット数」を取る。
/// **層ごとに活性化関数を変えられない** ので、隠れ層も出力層もシグモイドになる。
/// Keras のように ReLU と softmax を混ぜることはできない。
///
/// Accord には Dropout も無い。原著が過学習を抑えるために入れている 2 か所は、
/// ここでは省くことになる。
///
/// **`NguyenWidrow(...).Randomize()` が必須である。** これを呼ばないと重みが
/// 初期化されず、何エポック回しても学習が進まない（常に同じクラスを答え、
/// 正解率 0.236 のまま）。Keras は層を作った時点で自動的に初期化するので、
/// 移植のときに見落としやすい。
let fit (data: Circle) (epochs: int) (seed: int) =
    // Accord は内部で乱数を使うので、種を固定して再現性を確保する
    Accord.Math.Random.Generator.Seed <- System.Nullable seed

    let network =
        ActivationNetwork(SigmoidFunction(), 2, hiddenUnits.[0], hiddenUnits.[1], 1)

    // 重みの初期化。これが無いと学習が始まらない
    NguyenWidrow(network).Randomize()

    let teacher = BackPropagationLearning(network)
    teacher.LearningRate <- LearningRate
    let outputs = data.Y |> Array.map (fun label -> [| float label |])

    for _ in 1..epochs do
        teacher.RunEpoch(data.X, outputs) |> ignore

    network

/// 重みを初期化せずに学習する。比較用。
///
/// `NguyenWidrow` を呼ばない場合にどうなるかを示すために残してある。
let fitWithoutInitialization (data: Circle) (epochs: int) (seed: int) =
    Accord.Math.Random.Generator.Seed <- System.Nullable seed

    let network =
        ActivationNetwork(SigmoidFunction(), 2, hiddenUnits.[0], hiddenUnits.[1], 1)

    let teacher = BackPropagationLearning(network)
    teacher.LearningRate <- LearningRate
    let outputs = data.Y |> Array.map (fun label -> [| float label |])

    for _ in 1..epochs do
        teacher.RunEpoch(data.X, outputs) |> ignore

    network

/// 1 点を予測する。出力が 0.5 以上なら 1
let predict (network: ActivationNetwork) (point: float[]) =
    if network.Compute(point).[0] >= 0.5 then 1 else 0

/// 学習データに対する正解率
let accuracy (network: ActivationNetwork) (data: Circle) =
    let correct =
        Array.map2 (fun point label -> predict network point = label) data.X data.Y
        |> Array.filter id
        |> Array.length

    float correct / float data.Size

/// NumPy の `np.arange` と同じく、終端を含まない等差数列を作る
let arange (start: float) (stop: float) (step: float) =
    let count = int (ceil ((stop - start) / step))
    Array.init count (fun i -> start + float i * step)

/// 原著の `plot_model` と同じ格子を作り、各点の予測クラスを返す
let decisionGrid (network: ActivationNetwork) (data: Circle) (step: float) =
    let xs = data.X |> Array.map (fun p -> p.[0])
    let ys = data.X |> Array.map (fun p -> p.[1])
    let xValues = arange (Array.min xs - 1.0) (Array.max xs + 1.0) step
    let yValues = arange (Array.min ys - 1.0) (Array.max ys + 1.0) step

    yValues
    |> Array.map (fun yValue -> xValues |> Array.map (fun xValue -> predict network [| xValue; yValue |]))

/// 各行で予測が切り替わった回数を返す。
///
/// 決定木なら軸に平行な境界なので、切り替わる位置は行によらず同じだった。
/// ニューラルネットワークは曲線を引けるので、行ごとに変わる。
let boundaryChangesPerRow (grid: int[][]) =
    grid
    |> Array.map (fun row -> row |> Array.pairwise |> Array.filter (fun (a, b) -> a <> b) |> Array.length)
    |> List.ofArray
