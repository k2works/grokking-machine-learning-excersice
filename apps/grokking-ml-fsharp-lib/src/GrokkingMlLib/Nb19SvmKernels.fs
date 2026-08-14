/// 原著ノートブック #19 `Chapter_11_Support_Vector_Machines/SVM_graphical_example.ipynb`。
///
/// [#17](Nb17BuildingDatasets.fs) で作った 3 つのデータセットに、
/// いろいろな SVM を当てる回。
///
/// - 直線データに線形カーネル。`C` を変えて正則化の効き方を見る
/// - 円データに多項式カーネル。`degree` を変える
/// - 二重円データに RBF カーネル。`gamma` を変える
///
/// [#18](Nb18CalculatingSimilarities.fs) では ML.NET を使ったが、
/// ML.NET はカーネル SVM を持たない。ここは **Accord.MachineLearning** の
/// `SequentialMinimalOptimization` を使う。scikit-learn が内部で呼ぶ libsvm と
/// **同じ SMO アルゴリズム** なので、いちばん近い対応になる。
module GrokkingMlLib.Nb19SvmKernels

open Accord.MachineLearning.VectorMachines.Learning
open Accord.Statistics.Kernels
open Deedle

/// 読み込んだデータ
type Dataset =
    { X: float[][]
      Y: int[] }

    member this.Size = this.Y.Length

    /// Accord は真偽値でラベルを扱う
    member this.Decisions = this.Y |> Array.map (fun label -> label = 1)

/// [#17](Nb17BuildingDatasets.fs) が作った CSV を読む
let load (name: string) =
    let frame = Datasets.loadFrame $"{name}.csv"
    let x1 = frame.GetColumn<float>("x_1") |> Series.values |> Array.ofSeq
    let x2 = frame.GetColumn<float>("x_2") |> Series.values |> Array.ofSeq
    let y = frame.GetColumn<int>("y") |> Series.values |> Array.ofSeq

    { X = Array.init x1.Length (fun i -> [| x1.[i]; x2.[i] |])
      Y = y }

/// scikit-learn の `gamma='scale'` を計算する。
///
/// `1 / (n_features * X.var())`。**分散は列ごとではなく行列全体** で取る。
let scaleOf (data: Dataset) =
    let values = data.X |> Array.collect id
    let mean = Array.average values
    let variance = values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float values.Length
    1.0 / (2.0 * variance)

/// 既定の乱数の種。
///
/// **Accord の学習器は種を固定しないと実行のたびに結果が変わる。**
/// `LinearDualCoordinateDescent` を `C = 100` で走らせたところ、
/// 同じデータで 0.917 と 0.683 が出た。scikit-learn の `SVC` は決定的なので、
/// この違いは移植で必ず踏む。
[<Literal>]
let Seed = 0

let private useSeed () =
    Accord.Math.Random.Generator.Seed <- System.Nullable Seed

/// 線形 SVM。scikit-learn の `SVC(kernel='linear', C=...)` に対応する
let fitLinear (data: Dataset) (c: float) =
    useSeed ()
    let teacher = LinearDualCoordinateDescent(Complexity = c)
    teacher.Learn(data.X, data.Decisions)

/// scikit-learn の `kernel='poly', degree=d` と同じ式のカーネル。
///
/// scikit-learn は `(gamma * <x, y> + coef0)^degree`、`coef0` の既定は 0。
/// Accord の `Polynomial(degree, constant)` は `(<x, y> + constant)^degree` で
/// **gamma に相当する係数を持たない**。
let polynomialKernel (degree: int) = Polynomial(degree, 0.0)

/// 多項式カーネルの gamma を、**特徴量側に畳み込む**。
///
/// `(γ<x, y>)^d = (<√γ x, √γ y>)^d` なので、座標を √γ 倍しておけば
/// gamma を持たないカーネルでも scikit-learn と同じ式になる。
///
/// これをしないと `degree = 4` で正解率が 0.318 まで落ちる。
/// 座標は -3〜3 なので内積は最大 18 になり、その 4 乗は 10 万を超える。
/// カーネル行列の値が大きくなりすぎて SMO が解けなくなる。
let scaledForPolynomial (data: Dataset) =
    let factor = sqrt (scaleOf data)
    { data with X = data.X |> Array.map (Array.map (fun v -> v * factor)) }

/// scikit-learn の `kernel='rbf', gamma=g` に対応する
let rbfKernel (gamma: float) = Gaussian(Gamma = gamma)

/// カーネルを指定した SVM を学習する
let fitKernel (data: Dataset) (kernel: 'K when 'K :> IKernel<float[]>) (c: float) =
    useSeed ()
    let teacher = SequentialMinimalOptimization<'K>(Kernel = kernel, Complexity = c)
    teacher.Learn(data.X, data.Decisions)

/// 学習データに対する正解率
let accuracy (predict: float[] -> bool) (data: Dataset) =
    let correct =
        Array.map2 (fun row decision -> predict row = decision) data.X data.Decisions
        |> Array.filter id
        |> Array.length

    float correct / float data.Size
