/// 原著ノートブック #03
/// `Chapter_04_Testing_Overfitting_Underfitting/Polynomial_regression_regularization.ipynb`。
///
/// 二次関数 -x^2 + 2 の周りに散らした 40 点へ、**次数 20** の多項式を当てはめる。
/// 正則化なしでは激しく過学習し、L1（Lasso）・L2（Ridge）を入れると収まる。
///
/// scikit-learn の `PolynomialFeatures` にあたるものが .NET には無いので、
/// 多項式特徴量の展開は自前で書く。
/// 正則化は Math.NET に無いため、Ridge は解析解を、Lasso は座標降下を自分で実装する。
module GrokkingMlLib.Nb03PolynomialRegularization

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

/// 元にした多項式 -x^2 + 2 の係数。添字が次数に対応する
let polynomialCoefficients = [| 2; 0; -1 |]

/// 原著が使う点の数とノイズの大きさ
[<Literal>]
let SampleSize = 40

[<Literal>]
let NoiseStd = 0.1

/// 原著が当てはめる多項式の次数。40 点に対して 20 次は明らかに過剰
[<Literal>]
let Degree = 20

/// 正則化の種類。原著は文字列 'L1' / 'L2' / None で切り替えていた
type Regularization =
    | NoRegularization
    | L1
    | L2

/// 生成したデータと、その訓練／テスト分割
type Dataset =
    { X: float[]
      Y: float[]
      XTrain: float[]
      YTrain: float[]
      XTest: float[]
      YTest: float[] }

/// 学習済みモデル。切片と各次数の係数を持つ
type PolynomialModel =
    { Intercept: float
      Coefficients: float[] }

    /// x に対する予測値。Coefficients.[i] は x^(i+1) の係数
    member this.Predict(x: float) =
        this.Intercept
        + (this.Coefficients |> Array.mapi (fun i c -> c * x ** float (i + 1)) |> Array.sum)

/// 多項式の値を求める。`coefficients.[i]` が x^i の係数
let polynomial (coefficients: int[]) (x: float) =
    coefficients |> Array.mapi (fun i c -> float c * x ** float i) |> Array.sum

/// -x^2 + 2 の周りにガウスノイズを載せた点を生成し、訓練とテストに分ける。
///
/// .NET の `Random` は Python の Mersenne Twister と別物なので、同じ種を渡しても
/// 原著と同じ点にはならない。テストは数値の一致ではなく性質で検証する。
let generateDataset (size: int) (seed: int) (testRatio: float) =
    let random = Random(seed)

    // Box-Muller 法で正規乱数を作る。.NET の Random は一様分布しか持たない
    let nextGaussian () =
        let u1 = random.NextDouble()
        let u2 = random.NextDouble()
        sqrt (-2.0 * log u1) * cos (2.0 * Math.PI * u2)

    let x = Array.init size (fun _ -> random.NextDouble() * 2.0 - 1.0)
    let y =
        x |> Array.map (fun v -> polynomial polynomialCoefficients v + nextGaussian () * NoiseStd)

    // train_test_split と同じく、添字をシャッフルしてから分ける
    let indices = Array.init size id
    for i in size - 1 .. -1 .. 1 do
        let j = random.Next(i + 1)
        let tmp = indices.[i]
        indices.[i] <- indices.[j]
        indices.[j] <- tmp

    let testSize = int (float size * testRatio)
    let testIndices = Array.sub indices 0 testSize
    let trainIndices = Array.sub indices testSize (size - testSize)

    { X = x
      Y = y
      XTrain = trainIndices |> Array.map (fun i -> x.[i])
      YTrain = trainIndices |> Array.map (fun i -> y.[i])
      XTest = testIndices |> Array.map (fun i -> x.[i])
      YTest = testIndices |> Array.map (fun i -> y.[i]) }

/// x を x, x^2, ..., x^degree の列に展開する。
///
/// scikit-learn の `PolynomialFeatures(include_bias=False)` にあたる。
/// 定数 1 の列を作らないのは、切片を回帰側に任せるためである。
let polynomialFeatures (x: float[]) (degree: int) : Matrix<float> =
    DenseMatrix.init x.Length degree (fun row col -> x.[row] ** float (col + 1))

/// 中心化した特徴量とラベル。正則化つき回帰は切片を罰則の対象にしないので、
/// 平均を引いてから解き、最後に切片を復元する
let private center (features: Matrix<float>) (y: float[]) =
    let means = Array.init features.ColumnCount (fun c -> features.Column(c) |> Vector.sum |> fun s -> s / float features.RowCount)
    let yMean = Array.average y
    let centered =
        DenseMatrix.init features.RowCount features.ColumnCount (fun r c ->
            features.[r, c] - means.[c])

    centered, DenseVector.ofArray (y |> Array.map (fun v -> v - yMean)), means, yMean

let private restoreIntercept (means: float[]) (yMean: float) (coefficients: Vector<float>) =
    { Intercept = yMean - (Array.mapi (fun i m -> m * coefficients.[i]) means |> Array.sum)
      Coefficients = coefficients.ToArray() }

/// L2 正則化（Ridge）の解析解。(X^T X + alpha I)^-1 X^T y
let private fitRidge (features: Matrix<float>) (y: float[]) (alpha: float) =
    let centered, centeredY, means, yMean = center features y
    let gram = centered.TransposeThisAndMultiply centered
    let regularized = gram + DenseMatrix.identity<float> gram.ColumnCount * alpha
    let coefficients = regularized.Solve(centered.TransposeThisAndMultiply(centeredY))
    restoreIntercept means yMean coefficients

/// L1 正則化（Lasso）を座標降下で解く。
///
/// scikit-learn の `Lasso` と同じアルゴリズムで、軟判定しきい値関数によって
/// 係数が **厳密にゼロ** になる。1 つずつ座標を順に更新するので実装も短い。
let private fitLasso (features: Matrix<float>) (y: float[]) (alpha: float) =
    let centered, centeredY, means, yMean = center features y
    let n = float centered.RowCount
    let columns = centered.ColumnCount
    let coefficients = DenseVector.zero<float> columns
    // 各列のノルムは毎回同じなので先に計算しておく
    let normalizers =
        Array.init columns (fun c -> let col = centered.Column c in col * col / n)

    /// 軟判定しきい値関数。|value| が threshold 以下なら 0 を返す
    let softThreshold value threshold =
        if value > threshold then value - threshold
        elif value < -threshold then value + threshold
        else 0.0

    let mutable residual = centeredY - centered * coefficients

    for _ in 1..1000 do
        for c in 0 .. columns - 1 do
            if normalizers.[c] > 1e-12 then
                let column = centered.Column c
                // c 番目の寄与を残差に戻してから、その座標だけを解き直す
                let partial = residual + column * coefficients.[c]
                let rho = column * partial / n
                let updated = softThreshold rho alpha / normalizers.[c]
                residual <- partial - column * updated
                coefficients.[c] <- updated

    restoreIntercept means yMean coefficients

/// 正則化なしの最小二乗。次数 20 の特徴量は条件数が悪いので SVD で解く
let private fitOls (features: Matrix<float>) (y: float[]) =
    let design =
        DenseMatrix.init features.RowCount (features.ColumnCount + 1) (fun r c ->
            if c = 0 then 1.0 else features.[r, c - 1])

    let solution =
        MathNet.Numerics.LinearRegression.MultipleRegression.Svd(design, DenseVector.ofArray y)

    { Intercept = solution.[0]
      Coefficients = (solution.SubVector(1, solution.Count - 1)).ToArray() }

/// 多項式回帰を学習する。正則化は L1（Lasso）・L2（Ridge）から選ぶ
let trainPolynomialRegression
    (x: float[])
    (y: float[])
    (degree: int)
    (regularization: Regularization)
    (alpha: float)
    =
    let features = polynomialFeatures x degree

    match regularization with
    | NoRegularization -> fitOls features y
    | L1 -> fitLasso features y alpha
    | L2 -> fitRidge features y alpha

/// テストセットに対する RMSE を返す
let evaluateModel (model: PolynomialModel) (x: float[]) (y: float[]) =
    let squaredSum =
        Array.map2 (fun xi yi -> (yi - model.Predict xi) ** 2.0) x y |> Array.sum

    sqrt (squaredSum / float x.Length)
