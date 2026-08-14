/// 原著ノートブック #14
/// `Chapter_10_Neural_Networks/House_price_predictions_neural_network.ipynb`。
///
/// ニューラルネットワークを **回帰** に使う回。[#02](Nb02HousePrices.fs) と同じ
/// ハイデラバードの住宅データを、今度は 3 層のネットワークで予測する。
///
/// Accord.Neuro は出力が 0〜1 のシグモイドしか持たないので、
/// **価格を標準化してから学習し、予測を戻す** 形にする。
module GrokkingMlLib.Nb14HousePriceNetwork

open Accord.Neuro
open Accord.Neuro.Learning
open Deedle

/// 原著のネットワークの形。入力 38 に対して 38 → 128 → 64 → 1
let hiddenUnits = [| 38; 128; 64 |]

/// 原著の学習回数
[<Literal>]
let OriginalEpochs = 10

/// このネットワークで十分に学習が進む回数
[<Literal>]
let Epochs = 300

/// 読み込んだ住宅データ
type Housing =
    { X: float[][]
      Prices: float[] }

    member this.Size = this.Prices.Length
    member this.FeatureCount = this.X.[0].Length

/// 平均と標準偏差。学習時と予測時で同じ値を使う必要がある
type Scaler =
    { Mean: float; Std: float }

    member this.Scale(value: float) = if this.Std = 0.0 then 0.0 else (value - this.Mean) / this.Std
    member this.Unscale(value: float) = value * this.Std + this.Mean

/// 標準化した特徴量と価格、そして元に戻すための係数
type Scaled =
    { X: float[][]
      Prices: float[]
      PriceScaler: Scaler }

let private scalerOf (values: float[]) =
    let mean = Array.average values
    let variance = values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float values.Length
    { Mean = mean; Std = sqrt variance }

/// ハイデラバードの住宅データを読み込む。
///
/// `Location`（文字列）と `Price`（目的変数）を落として 38 列にする。
let loadHousing () =
    let frame = Datasets.loadFrame "Hyderabad.csv"

    let featureNames =
        frame.ColumnKeys |> Seq.filter (fun name -> name <> "Location" && name <> "Price") |> List.ofSeq

    // Deedle の `Series.values` は欠損を飛ばすので、列ごとに長さが変わりうる。
    // このデータには欠損があるため、0 で埋めてから配列にする
    let columns =
        featureNames
        |> List.map (fun name ->
            frame.GetColumn<float>(name) |> Series.fillMissingWith 0.0 |> Series.values |> Array.ofSeq)

    let prices = frame.GetColumn<float>("Price") |> Series.values |> Array.ofSeq

    { X = Array.init prices.Length (fun row -> columns |> List.map (fun column -> column.[row]) |> Array.ofList)
      Prices = prices }

/// 特徴量と価格をそれぞれ標準化する。
///
/// **原著は標準化していない。** Keras の Adam は勾配の大きさを自動で調整するので、
/// 桁の違う特徴量（`Area` は 4 桁、`Resale` は 0 か 1）でも学習できる。
/// Accord のシグモイド出力は 0〜1 しか返せないため、1000 万の桁の価格を
/// そのまま学習させることはそもそもできない。
let standardize (data: Housing) =
    let featureScalers =
        [| for column in 0 .. data.FeatureCount - 1 ->
            scalerOf (Array.init data.Size (fun row -> data.X.[row].[column])) |]

    let priceScaler = scalerOf data.Prices

    { X =
        Array.init data.Size (fun row ->
            Array.init data.FeatureCount (fun column -> featureScalers.[column].Scale data.X.[row].[column]))
      Prices = data.Prices |> Array.map priceScaler.Scale
      PriceScaler = priceScaler }

/// 標準化した価格を 0〜1 に写す。シグモイド出力が返せる範囲に合わせるため
let private toUnitRange (values: float[]) =
    let low = Array.min values
    let high = Array.max values
    values |> Array.map (fun v -> (v - low) / (high - low)), low, high

/// 原著と同じ隠れ層を持つネットワークを学習する。
///
/// [#13](Nb13NeuralNetworkBoundary.fs) と同じく `NguyenWidrow` による
/// 重みの初期化が必須である。出力はシグモイドなので 0〜1 に収まる。
/// 価格を 0〜1 に写してから学習し、予測時に戻す。
let fit (scaled: Scaled) (epochs: int) (seed: int) =
    Accord.Math.Random.Generator.Seed <- System.Nullable seed

    let network =
        ActivationNetwork(
            SigmoidFunction(),
            scaled.X.[0].Length,
            hiddenUnits.[0],
            hiddenUnits.[1],
            hiddenUnits.[2],
            1
        )

    NguyenWidrow(network).Randomize()

    let unitPrices, low, high = toUnitRange scaled.Prices
    let teacher = BackPropagationLearning(network)
    let outputs = unitPrices |> Array.map (fun price -> [| price |])

    for _ in 1..epochs do
        teacher.RunEpoch(scaled.X, outputs) |> ignore

    network, low, high

/// 全物件の価格を予測する。標準化と 0〜1 の写像を戻して円単位にする
let predict (network: ActivationNetwork, low: float, high: float) (scaled: Scaled) =
    scaled.X
    |> Array.map (fun row ->
        let unit = network.Compute(row).[0]
        scaled.PriceScaler.Unscale(unit * (high - low) + low))

/// 学習データに対する RMSE。円単位で返す
let rmse (model: ActivationNetwork * float * float) (data: Housing) (scaled: Scaled) =
    let predictions = predict model scaled

    Array.map2 (fun actual predicted -> (actual - predicted) ** 2.0) data.Prices predictions
    |> Array.average
    |> sqrt

/// 常に平均価格を答えたときの RMSE。
///
/// これを下回れなければ、ネットワークは何も学習できていない。
let baselineRmse (data: Housing) =
    let mean = Array.average data.Prices
    data.Prices |> Array.sumBy (fun p -> (p - mean) ** 2.0) |> fun s -> sqrt (s / float data.Size)
