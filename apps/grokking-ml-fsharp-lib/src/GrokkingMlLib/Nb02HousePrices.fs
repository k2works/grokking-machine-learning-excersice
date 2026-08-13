/// 原著ノートブック #02 `Chapter_03_Linear_Regression/House_price_predictions.ipynb`。
///
/// ハイデラバードの住宅データ 2518 件から価格を予測する。面積 1 つだけを使う単回帰と、
/// 全 40 列を前処理してから使う重回帰の 2 本立てになっている。
///
/// 前処理は原著の手順をそのまま踏襲する。
/// 1. 欠損（`9` で符号化されている）を含む末尾の行を落とす
/// 2. 数値列（`Area` と `No. of Bedrooms`）を標準化する
/// 3. カテゴリ列（`Location`）を one-hot 符号化する
///
/// pandas の `get_dummies` にあたるものが Deedle には無いので、そこは自前で書く。
module GrokkingMlLib.Nb02HousePrices

open Deedle
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

/// 原著が欠損ありとして切り落とす位置。これ以降の行は `9` で符号化された欠損を含む
[<Literal>]
let ValidRows = 2434

/// 標準化に使った平均と標準偏差。新しい物件も同じ値で変換する必要がある
type Standardizer =
    { AreaMean: float
      AreaStd: float
      BedroomsMean: float
      BedroomsStd: float }

    member this.Transform(area: float, bedrooms: float) =
        (area - this.AreaMean) / this.AreaStd, (bedrooms - this.BedroomsMean) / this.BedroomsStd

/// 前処理を通したあとのデータ一式
type Preprocessed =
    { /// 列名。順序が予測時の行の組み立てと一致している必要がある
      ColumnNames: string list
      /// 特徴量行列。行が物件、列が特徴量
      Features: Matrix<float>
      Labels: Vector<float>
      Standardizer: Standardizer }

    member this.RowCount = this.Features.RowCount

/// 学習済みの線形モデル。切片と係数を分けて持つ
type LinearModel =
    { Intercept: float
      Coefficients: Vector<float> }

    member this.Predict(row: Vector<float>) = this.Intercept + this.Coefficients * row

/// ハイデラバードの住宅データを読み込む。2518 行 × 40 列
let loadData () = Datasets.loadFrame "Hyderabad.csv"

let private column (frame: Frame<int, string>) (name: string) =
    frame.GetColumn<float>(name) |> Series.values |> Array.ofSeq

let private stringColumn (frame: Frame<int, string>) (name: string) =
    frame.GetColumn<string>(name) |> Series.values |> Array.ofSeq

/// 面積 1 列だけで価格を予測する単回帰。
///
/// scikit-learn は 2 次元に整形した特徴量を渡すが、Math.NET の `Fit.Line` は
/// x と y の 1 次元配列をそのまま受け取る。
let fitAreaOnly (frame: Frame<int, string>) =
    let struct (intercept, slope) =
        MathNet.Numerics.Fit.Line(column frame "Area", column frame "Price")

    { Intercept = intercept
      Coefficients = DenseVector.ofArray [| slope |] }

/// 不偏分散（ddof = 1）の標準偏差。pandas の `std()` の既定に合わせる
let private sampleStd (values: float[]) =
    let mean = Array.average values
    sqrt (values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float (values.Length - 1))

/// 原著の前処理 3 手順をそのまま行う
let preprocess (frame: Frame<int, string>) =
    let take (values: 'a[]) = Array.sub values 0 ValidRows

    let area = take (column frame "Area")
    let bedrooms = take (column frame "No. of Bedrooms")
    let price = take (column frame "Price")
    let location = take (stringColumn frame "Location")

    let standardizer =
        { AreaMean = Array.average area
          AreaStd = sampleStd area
          BedroomsMean = Array.average bedrooms
          BedroomsStd = sampleStd bedrooms }

    // 列の順序は Python 版・Kotlin 版と合わせる。
    // Area, No. of Bedrooms, 残りの数値列, 地域の one-hot
    let passthroughNames =
        frame.ColumnKeys
        |> Seq.filter (fun name ->
            not (List.contains name [ "Price"; "Area"; "No. of Bedrooms"; "Location" ]))
        |> List.ofSeq

    let locationNames = location |> Array.distinct |> Array.sort |> List.ofArray

    let columnNames =
        [ "Area"; "No. of Bedrooms" ]
        @ passthroughNames
        @ (locationNames |> List.map (sprintf "Location_%s"))

    let columns =
        [ area |> Array.map (fun v -> (v - standardizer.AreaMean) / standardizer.AreaStd)
          bedrooms
          |> Array.map (fun v -> (v - standardizer.BedroomsMean) / standardizer.BedroomsStd) ]
        @ (passthroughNames |> List.map (fun name -> take (column frame name)))
        @ (locationNames
           |> List.map (fun name -> location |> Array.map (fun v -> if v = name then 1.0 else 0.0)))

    { ColumnNames = columnNames
      Features = DenseMatrix.ofColumnArrays (Array.ofList columns)
      Labels = DenseVector.ofArray price
      Standardizer = standardizer }

/// 前処理済みの全特徴量で重回帰を学習する。
///
/// one-hot 符号化した地域の列は合計すると常に 1 になり、切片と線形従属になる。
/// 正規方程式では解けないので、擬似逆行列を求める SVD を使う。Math.NET の
/// `MultipleRegression.Svd` は最小ノルム解を返す。
let fitAllFeatures (prepared: Preprocessed) =
    // 切片を推定させるため、値が 1 の列を先頭に足した設計行列を作る
    let rows = prepared.RowCount
    let design =
        DenseMatrix.init rows (prepared.Features.ColumnCount + 1) (fun row col ->
            if col = 0 then 1.0 else prepared.Features.[row, col - 1])

    let solution =
        MathNet.Numerics.LinearRegression.MultipleRegression.Svd(design, prepared.Labels)

    { Intercept = solution.[0]
      Coefficients = solution.SubVector(1, solution.Count - 1) }

/// 二乗平均平方根誤差
let rmse (labels: Vector<float>) (predictions: Vector<float>) =
    let differences = labels - predictions
    sqrt (differences * differences / float labels.Count)

/// 学習データ全体に対する予測を返す
let predictAll (model: LinearModel) (prepared: Preprocessed) =
    prepared.Features * model.Coefficients
    |> Vector.map (fun value -> value + model.Intercept)

/// 新しい物件の価格を予測する。
///
/// 学習時と同じ列・同じ順序の 1 行を組み立てるのが要点である。地域の列は
/// 1 つだけを 1 にして残りは 0 にする。
let predictNewHouse
    (model: LinearModel)
    (prepared: Preprocessed)
    (area: float)
    (bedrooms: int)
    (location: string)
    =
    let locationColumn = $"Location_{location}"

    if not (List.contains locationColumn prepared.ColumnNames) then
        invalidArg "location" $"学習データに無い地域です: {location}"

    let scaledArea, scaledBedrooms = prepared.Standardizer.Transform(area, float bedrooms)

    let row =
        prepared.ColumnNames
        |> List.map (fun name ->
            if name = "Area" then scaledArea
            elif name = "No. of Bedrooms" then scaledBedrooms
            elif name = locationColumn then 1.0
            else 0.0)
        |> DenseVector.ofList

    model.Predict row
