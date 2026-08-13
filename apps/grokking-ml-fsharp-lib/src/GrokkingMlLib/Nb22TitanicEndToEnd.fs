/// 原著ノートブック #22 `Chapter_13_End_to_end_example/End_to_end_example.ipynb`。
///
/// タイタニックの生存予測を、前処理からモデル選択まで通しでやる回である。
/// 本の最終章で、これまでのアルゴリズムが一堂に会する。
///
/// **移植で難しいのは学習ではなく前処理と分割** である。
/// 欠損の埋め方・one-hot の列順・訓練とテストの切り分け方が合わなければ、
/// どんなモデルを使っても数字は合わない。
///
/// scikit-learn の `train_test_split(random_state=100)` の並べ替えは
/// .NET では再現できない。その代わり、原著と同じ件数（534 / 178 / 179）に
/// なることと、分割が全体を覆うことを確かめる。
module GrokkingMlLib.Nb22TitanicEndToEnd

open Deedle

/// 年齢の離散化の区切り。10 歳刻みで 8 区間
let ageBins = [| 0; 10; 20; 30; 40; 50; 60; 70; 80 |]

/// 学習に使わない列
let droppedColumns = [ "Name"; "Ticket"; "PassengerId" ]

/// 1 人ぶんの生データ（必要な列だけ）
type Passenger =
    { Survived: int
      Pclass: int
      Sex: string
      Age: float option
      SibSp: int
      Parch: int
      Fare: float
      Embarked: string option }

let private optionalFloat (series: Series<int, float>) index =
    match series.TryGet index with
    | OptionalValue.Present value -> Some value
    | _ -> None

let private optionalString (series: Series<int, string>) index =
    match series.TryGet index with
    | OptionalValue.Present value when not (System.String.IsNullOrWhiteSpace value) -> Some value
    | _ -> None

/// 生のタイタニックデータ 891 行
let loadRaw () =
    let frame = Datasets.loadFrame "titanic.csv"
    let survived = frame.GetColumn<int>("Survived")
    let pclass = frame.GetColumn<int>("Pclass")
    let sex = frame.GetColumn<string>("Sex")
    let age = frame.GetColumn<float>("Age")
    let sibSp = frame.GetColumn<int>("SibSp")
    let parch = frame.GetColumn<int>("Parch")
    let fare = frame.GetColumn<float>("Fare")
    let embarked = frame.GetColumn<string>("Embarked")

    [| for index in frame.RowKeys ->
        { Survived = survived.[index]
          Pclass = pclass.[index]
          Sex = sex.[index]
          Age = optionalFloat age index
          SibSp = sibSp.[index]
          Parch = parch.[index]
          Fare = fare.[index]
          Embarked = optionalString embarked index } |]

/// `Cabin` は列ごと落とすので、欠損数だけ数える
let cabinMissingCount () =
    let frame = Datasets.loadFrame "titanic.csv"
    let cabin = frame.GetColumn<string>("Cabin")

    frame.RowKeys
    |> Seq.filter (fun index -> (optionalString cabin index).IsNone)
    |> Seq.length

/// 欠損のある列と、その数
let missingCounts (data: Passenger[]) =
    dict
        [ "Age", data |> Array.filter (fun p -> p.Age.IsNone) |> Array.length
          "Cabin", cabinMissingCount ()
          "Embarked", data |> Array.filter (fun p -> p.Embarked.IsNone) |> Array.length ]

/// 年齢の中央値。原著は 28.0
let medianAge (data: Passenger[]) =
    let ages = data |> Array.choose (fun p -> p.Age) |> Array.sort
    let middle = ages.Length / 2

    if ages.Length % 2 = 0 then
        (ages.[middle - 1] + ages.[middle]) / 2.0
    else
        ages.[middle]

/// 欠損を片付ける。
///
/// - `Age` は中央値（28.0）で埋める
/// - `Embarked` は `U`（Unknown）という新しい区分にする
let clean (raw: Passenger[]) =
    let median = medianAge raw

    raw
    |> Array.map (fun p ->
        { p with
            Age = Some(defaultArg p.Age median)
            Embarked = Some(defaultArg p.Embarked "U") })

/// 年齢がどの区間に入るか。`pandas.cut` と同じ **左を開き右を閉じる** 区間
let ageBin (age: float) =
    [ 1 .. ageBins.Length - 1 ] |> List.find (fun i -> age <= float ageBins.[i]) |> fun i -> i - 1

/// one-hot 符号化した特徴量の列名。原著の列順をそのまま保つ
let featureNames =
    [| yield "SibSp"
       yield "Parch"
       yield "Fare"
       yield! [ "Sex_female"; "Sex_male" ]
       yield! [ "Embarked_C"; "Embarked_Q"; "Embarked_S"; "Embarked_U" ]
       yield! [ "Pclass_1"; "Pclass_2"; "Pclass_3" ]
       for index in 0 .. ageBins.Length - 2 ->
           $"Categorized_age_({ageBins.[index]}, {ageBins.[index + 1]}]" |]

let private indexOf name =
    featureNames |> Array.findIndex ((=) name)

/// 1 人ぶんの特徴量ベクトル
let featuresOf (p: Passenger) =
    let values = Array.zeroCreate featureNames.Length
    values.[0] <- float p.SibSp
    values.[1] <- float p.Parch
    values.[2] <- p.Fare
    values.[indexOf $"Sex_{p.Sex}"] <- 1.0
    values.[indexOf $"Embarked_{p.Embarked.Value}"] <- 1.0
    values.[indexOf $"Pclass_{p.Pclass}"] <- 1.0
    let bin = ageBin p.Age.Value
    values.[indexOf $"Categorized_age_({ageBins.[bin]}, {ageBins.[bin + 1]}]"] <- 1.0
    values

/// 前処理した特徴量とラベル
type Prepared =
    { X: float[][]
      Y: int[] }

    member this.Size = this.Y.Length
    member this.FeatureCount = if this.X.Length = 0 then 0 else this.X.[0].Length

/// 生データを特徴量に直す
let preprocess (data: Passenger[]) =
    { X = data |> Array.map featuresOf
      Y = data |> Array.map (fun p -> p.Survived) }

/// 訓練 / 検証 / テストの 3 分割。
///
/// 原著と同じ件数（534 / 178 / 179）にするが、**同じ行が同じ組に入る
/// わけではない**。scikit-learn の並べ替えは .NET では再現できない。
type Split =
    { Train: Prepared
      Validation: Prepared
      Test: Prepared }

let split (prepared: Prepared) (seed: int) =
    let random = System.Random(seed)
    let order = Array.init prepared.Size id |> Array.sortBy (fun _ -> random.Next())

    // 原著は 6:4 に切り、その 4 をさらに半分ずつにする。
    // scikit-learn は `test_size` の側を **切り上げる**（ceil）ので、
    // 891 -> 534 / 357、357 -> 178 / 179 になる
    let restSize = int (ceil (float prepared.Size * 0.4))
    let trainSize = prepared.Size - restSize
    let testSize = int (ceil (float restSize * 0.5))
    let validationSize = restSize - testSize

    let slice from count =
        let indices = order.[from .. from + count - 1]

        { X = indices |> Array.map (fun i -> prepared.X.[i])
          Y = indices |> Array.map (fun i -> prepared.Y.[i]) }

    { Train = slice 0 trainSize
      Validation = slice trainSize validationSize
      Test = slice (trainSize + validationSize) testSize }

/// 「全員死亡」と答えたときの正解率
let majorityBaseline (labels: int[]) =
    float (labels |> Array.filter ((=) 0) |> Array.length) / float labels.Length

/// 正解率
let accuracy (predicted: int[]) (actual: int[]) =
    let correct = Array.map2 (=) predicted actual |> Array.filter id |> Array.length
    float correct / float actual.Length

/// F1 スコア。適合率と再現率の調和平均。
///
/// 正解率だけでは不十分な理由を見せるための指標である。
/// 生存者が少数派なので、「全員死亡」と答えても正解率は 6 割を超える。
let f1Score (predicted: int[]) (actual: int[]) =
    let count p = Array.map2 p predicted actual |> Array.filter id |> Array.length
    let truePositive = count (fun p a -> p = 1 && a = 1)
    let falsePositive = count (fun p a -> p = 1 && a = 0)
    let falseNegative = count (fun p a -> p = 0 && a = 1)

    if truePositive = 0 then
        0.0
    else
        let precision = float truePositive / float (truePositive + falsePositive)
        let recall = float truePositive / float (truePositive + falseNegative)
        2.0 * precision * recall / (precision + recall)

/// ガウシアン素朴ベイズ。
///
/// scikit-learn の `GaussianNB` に対応する。
/// **`var_smoothing`（全特徴量の最大分散 × 1e-9）を足すのが要点**である。
/// one-hot の列は分散がごく小さいので、固定値を足すだけでは正解率が崩れる。
let gaussianNaiveBayes (x: float[][]) (y: int[]) =
    let varianceOfColumn column =
        let values = x |> Array.map (fun row -> row.[column])
        let mean = Array.average values
        values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float values.Length

    let smoothing =
        1e-9 * ([ 0 .. featureNames.Length - 1 ] |> List.map varianceOfColumn |> List.max)

    let stats =
        [ 0; 1 ]
        |> List.map (fun label ->
            let rows = x |> Array.mapi (fun i row -> i, row) |> Array.filter (fun (i, _) -> y.[i] = label)

            let perColumn =
                [| for column in 0 .. featureNames.Length - 1 ->
                    let values = rows |> Array.map (fun (_, row) -> row.[column])
                    let mean = Array.average values

                    let variance =
                        (values |> Array.sumBy (fun v -> (v - mean) ** 2.0)) / float values.Length + smoothing

                    mean, variance |]

            let prior = log (float (y |> Array.filter ((=) label) |> Array.length) / float y.Length)
            label, prior, perColumn)

    fun (row: float[]) ->
        stats
        |> List.maxBy (fun (_, prior, perColumn) ->
            prior
            + (perColumn
               |> Array.mapi (fun column (mean, variance) ->
                   -0.5 * log (2.0 * System.Math.PI * variance)
                   - (row.[column] - mean) ** 2.0 / (2.0 * variance))
               |> Array.sum))
        |> fun (label, _, _) -> label

/// 全要素の分散。scikit-learn の `gamma='scale'` が使うもの
let varianceOf (x: float[][]) =
    let values = x |> Array.collect id
    let mean = Array.average values
    values |> Array.sumBy (fun v -> (v - mean) ** 2.0) |> fun s -> s / float values.Length

/// 原著が試す 7 つのうち **3 つ** を学習する。名前 -> 予測関数を返す。
///
/// 木系の 4 つ（決定木・ランダムフォレスト・勾配ブースティング・AdaBoost）を
/// 落としたのは、[#20](Nb20EnsembleSpam.fs)・[#21](Nb21GradientBoosting.fs) の
/// 自前実装が **2 特徴量ぶんの決め打ち** で、20 列には広げていないためである。
/// ここで確かめたいのは前処理と分割の再現なので、そちらに絞った。
///
/// ロジスティック回帰と SVM は Accord、素朴ベイズは自前
/// （Accord の `NaiveBayes` は離散値しか扱えない）。
let fitAll (data: Split) =
    let trainX = data.Train.X
    let trainY = data.Train.Y
    let decisions = trainY |> Array.map (fun label -> label = 1)
    Accord.Math.Random.Generator.Seed <- System.Nullable 0

    let logistic =
        let teacher =
            Accord.Statistics.Models.Regression.Fitting.IterativeReweightedLeastSquares<Accord.Statistics.Models.Regression.LogisticRegression>(
                Tolerance = 1e-4,
                MaxIterations = 100,
                Regularization = 1e-6
            )

        let model = teacher.Learn(trainX, decisions)
        fun (rows: float[][]) -> rows |> Array.map (fun row -> if model.Decide row then 1 else 0)

    let naiveBayes =
        let model = gaussianNaiveBayes trainX trainY
        fun (rows: float[][]) -> rows |> Array.map model

    let svm =
        let gamma = 1.0 / (float featureNames.Length * varianceOf trainX)
        let kernel = Accord.Statistics.Kernels.Gaussian(Gamma = gamma)

        let teacher =
            Accord.MachineLearning.VectorMachines.Learning.SequentialMinimalOptimization<Accord.Statistics.Kernels.Gaussian>(
                Kernel = kernel,
                Complexity = 1.0
            )

        let model = teacher.Learn(trainX, decisions)
        fun (rows: float[][]) -> rows |> Array.map (fun row -> if model.Decide row then 1 else 0)

    [ "Logistic regression", logistic
      "Naive Bayes", naiveBayes
      "SVM", svm ]
