/// 原著ノートブック #01 `Chapter_03_Linear_Regression/Coding_linear_regression.ipynb`。
///
/// 部屋数から住宅価格を予測する線形回帰を、3 つのトリック（simple / absolute / square）で
/// 学習したあと、同じ問題を Math.NET の `Fit.Line` に解かせて突き合わせる。
/// scikit-learn の `LinearRegression` に相当するのがこれである。
///
/// ML.NET にも回帰トレーナーはあるが、厳密な最小二乗を行う `Ols` は MKL ネイティブを
/// 要求して macOS x64 では動かない。閉じた式の解は Math.NET 側で求める。
module GrokkingMlLib.Nb01LinearRegression

open System

/// 原著が使うデータセット。部屋数と価格
let features = [| 1; 2; 3; 5; 6; 7 |]
let labels = [| 155; 197; 244; 356; 407; 448 |]

/// 学習された直線。原著の `price_per_room` と `base_price` に対応する
type Line =
    { PricePerRoom: float
      BasePrice: float }

    member this.Predict(numRooms: float) = this.BasePrice + this.PricePerRoom * numRooms

/// 学習の途中経過。原著が学習ループの中で描いていたものを記録する
type TrainingLog =
    { Line: Line
      /// 各エポック開始時点の直線
      History: Line list
      /// 各エポック開始時点の RMSE
      Errors: float list }

/// どのトリックで学習するか。原著はコメントアウトで切り替えていた
type Trick =
    | Simple
    | Absolute
    | Square

/// シンプルトリック。予測が外れた向きに、小さな乱数だけ直線を動かす。
///
/// 原著の 4 つの `if` をそのまま写している。3 番目の分岐だけ `BasePrice` を
/// 減らす非対称な書き方も原著のままである。
let simpleTrick (random: Random) (line: Line) (numRooms: float) (price: float) =
    let smallRandom1 = random.NextDouble() * 0.1
    let smallRandom2 = random.NextDouble() * 0.1
    let predictedPrice = line.BasePrice + line.PricePerRoom * numRooms

    let slope, intercept = line.PricePerRoom, line.BasePrice

    let slope, intercept =
        if price > predictedPrice && numRooms > 0.0 then
            slope + smallRandom1, intercept + smallRandom2
        else
            slope, intercept

    let slope, intercept =
        if price > predictedPrice && numRooms < 0.0 then
            slope - smallRandom1, intercept + smallRandom2
        else
            slope, intercept

    let slope, intercept =
        if price < predictedPrice && numRooms > 0.0 then
            slope - smallRandom1, intercept - smallRandom2
        else
            slope, intercept

    let slope, intercept =
        if price < predictedPrice && numRooms < 0.0 then
            slope - smallRandom1, intercept + smallRandom2
        else
            slope, intercept

    { PricePerRoom = slope; BasePrice = intercept }

/// 絶対トリック。外れた向きへ、学習率と部屋数に比例した幅で動かす
let absoluteTrick (learningRate: float) (line: Line) (numRooms: float) (price: float) =
    let predictedPrice = line.BasePrice + line.PricePerRoom * numRooms

    if price > predictedPrice then
        { PricePerRoom = line.PricePerRoom + learningRate * numRooms
          BasePrice = line.BasePrice + learningRate }
    else
        { PricePerRoom = line.PricePerRoom - learningRate * numRooms
          BasePrice = line.BasePrice - learningRate }

/// 二乗トリック。誤差の大きさにも比例して動かす。分岐が要らなくなる
let squareTrick (learningRate: float) (line: Line) (numRooms: float) (price: float) =
    let predictedPrice = line.BasePrice + line.PricePerRoom * numRooms
    let error = price - predictedPrice

    { PricePerRoom = line.PricePerRoom + learningRate * numRooms * error
      BasePrice = line.BasePrice + learningRate * error }

/// 二乗平均平方根誤差。原著は NumPy の内積で書いていた
let rmse (labels: int[]) (prediction: float) =
    let squaredSum =
        labels |> Array.sumBy (fun label -> (float label - prediction) ** 2.0)

    sqrt (squaredSum / float labels.Length)

/// トリックを繰り返して直線を学習する。
///
/// 乱数の消費順序（重みの初期化 2 回 → 毎エポックの添字 1 回）は原著と同じに
/// してあるが、.NET の `Random` は Python と別のアルゴリズムなので、同じ種でも
/// 同じ数列にはならない。数値の一致ではなく収束で検証する。
let linearRegression
    (features: int[])
    (labels: int[])
    (learningRate: float)
    (epochs: int)
    (trick: Trick)
    (seed: int)
    =
    let random = Random(seed)

    let initial =
        { PricePerRoom = random.NextDouble()
          BasePrice = random.NextDouble() }

    // 履歴は先頭追加で積み、最後に 1 度だけ反転する。末尾追加を繰り返すより安い
    let mutable line = initial
    let mutable history = []
    let mutable errors = []

    for _ in 1..epochs do
        history <- line :: history
        // 原著は features[0] だけを使って誤差を測っている
        errors <- rmse labels (float features.[0] * line.PricePerRoom + line.BasePrice) :: errors

        let i = random.Next(features.Length)
        let numRooms = float features.[i]
        let price = float labels.[i]

        line <-
            match trick with
            | Square -> squareTrick learningRate line numRooms price
            | Absolute -> absoluteTrick learningRate line numRooms price
            | Simple -> simpleTrick random line numRooms price

    { Line = line
      History = List.rev history
      Errors = List.rev errors }

/// 既定の設定で学習する。原著のセルと同じ引数
let linearRegressionDefault () =
    linearRegression features labels 0.01 1000 Square 0

/// 同じ問題を Math.NET の最小二乗に解かせる。
///
/// scikit-learn は特徴量を 2 次元配列に `reshape` させるが、`Fit.Line` は
/// x と y の 1 次元配列をそのまま受け取り、切片と傾きを構造体タプルで返す。
let fitWithMathNet (features: int[]) (labels: int[]) =
    let xs = features |> Array.map float
    let ys = labels |> Array.map float
    let struct (intercept, slope) = MathNet.Numerics.Fit.Line(xs, ys)

    { PricePerRoom = slope; BasePrice = intercept }
