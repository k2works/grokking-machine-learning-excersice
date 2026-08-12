/// 第 13 章: エンドツーエンドの実例。
/// 生データの前処理から、複数モデルの比較、評価までを 1 本のパイプラインに
/// つなぐ。これまでの章で作った部品を組み合わせるだけで実現できる。
module GrokkingMl.Ch13EndToEnd

open System

/// 生データの 1 行。
type Row = Map<string, string>

/// 特徴量ベクトル。
type Point = float list

/// 前処理を終えた特徴量とラベル。
type Dataset =
    { Points: Point list
      Labels: int list
      FeatureNames: string list }

/// 数値に見えない値は既定値に落とす。欠損への最初の砦。
let parseNumber (defaultValue: float) (text: string) =
    match Double.TryParse(text) with
    | true, value -> value
    | _ -> defaultValue

/// 中央値。欠損の穴埋めに使う。
let median (values: float list) =
    if List.isEmpty values then
        0.0
    else
        let ordered = List.sort values
        let middle = List.length ordered / 2

        if List.length ordered % 2 = 1 then
            ordered[middle]
        else
            (ordered[middle - 1] + ordered[middle]) / 2.0

/// 欠損を中央値で埋める。平均より外れ値に強い。
let imputeMissing (column: float option list) =
    let filler = column |> List.choose id |> median
    column |> List.map (Option.defaultValue filler)

/// 最小 0・最大 1 に揃える。値の幅が違う特徴量を対等に扱うため。
let normalize (column: float list) =
    let low = List.min column
    let high = List.max column

    if high = low then
        column |> List.map (fun _ -> 0.0)
    else
        column |> List.map (fun value -> (value - low) / (high - low))

/// カテゴリ列を 0/1 の列に展開する。
let oneHot (column: string list) =
    let categories = column |> List.distinct |> List.sort

    let expanded =
        column
        |> List.map (fun value -> categories |> List.map (fun category -> if value = category then 1.0 else 0.0))

    (expanded, categories)

/// 生の行データを、数値の特徴量ベクトルとラベルに変換する。
let buildDataset (labelColumn: string) (rows: Row list) =
    let labels =
        rows
        |> List.map (fun row -> if Map.tryFind labelColumn row = Some "yes" then 1 else 0)

    let valueOf name (row: Row) = Map.tryFind name row |> Option.defaultValue ""

    let numericColumn name =
        let raw =
            rows
            |> List.map (fun row ->
                match valueOf name row with
                | "" -> None
                | text -> Some(parseNumber 0.0 text))

        ([ normalize (imputeMissing raw) ], [ name ])

    let categoricalColumn name =
        let expanded, categories = oneHot (rows |> List.map (valueOf name))

        let columns =
            categories
            |> List.mapi (fun index _ -> expanded |> List.map (fun row -> row[index]))

        (columns, categories |> List.map (fun category -> $"{name}={category}"))

    let parts =
        [ numericColumn "age"; numericColumn "income"; categoricalColumn "city" ]

    let columns = parts |> List.collect fst
    let names = parts |> List.collect snd

    { Points = List.init (List.length rows) (fun i -> columns |> List.map (fun column -> column[i]))
      Labels = labels
      FeatureNames = names }

/// 1 つのモデルの評価結果。
type Evaluation =
    { Name: string
      Accuracy: float
      Precision: float
      Recall: float
      F1: float
      Auc: float }

/// 予測関数と確率関数から、第 7 章の指標をまとめて算出する。
let evaluate
    (name: string)
    (predictFn: Point -> int)
    (probabilityFn: Point -> float)
    (points: Point list)
    (labels: int list)
    =
    let matrix = GrokkingMl.Ch07Metrics.confusionMatrix labels (points |> List.map predictFn)

    { Name = name
      Accuracy = GrokkingMl.Ch07Metrics.accuracy matrix
      Precision = GrokkingMl.Ch07Metrics.precision matrix
      Recall = GrokkingMl.Ch07Metrics.recall matrix
      F1 = GrokkingMl.Ch07Metrics.f1Score matrix
      Auc = GrokkingMl.Ch07Metrics.auc labels (points |> List.map probabilityFn) }

/// 分割されたデータセット。第 4 章の Split を多次元の特徴量へ一般化したもの。
type DataSplit =
    { TrainPoints: Point list
      TrainLabels: int list
      TestPoints: Point list
      TestLabels: int list }

/// 訓練用とテスト用に分割する。第 4 章と同じ手順を多次元の点に対して行う。
let splitDataset (testRatio: float) (seed: int) (dataset: Dataset) =
    let rng = Random(seed)

    let indices =
        List.init (List.length dataset.Points) id
        |> List.sortBy (fun _ -> rng.Next())

    let testSize = int (float (List.length dataset.Points) * testRatio)
    let testIndices = List.truncate testSize indices
    let trainIndices = List.skip testSize indices

    { TrainPoints = trainIndices |> List.map (fun i -> dataset.Points[i])
      TrainLabels = trainIndices |> List.map (fun i -> dataset.Labels[i])
      TestPoints = testIndices |> List.map (fun i -> dataset.Points[i])
      TestLabels = testIndices |> List.map (fun i -> dataset.Labels[i]) }

/// 前処理 → 分割 → 3 モデルの学習 → 評価までを一気に通す。
let runPipeline (labelColumn: string) (rows: Row list) =
    let dataset = buildDataset labelColumn rows
    let split = splitDataset 0.3 0 dataset

    let logistic, _ =
        GrokkingMl.Ch06LogisticRegression.logisticRegression 0.5 2000 0 split.TrainPoints split.TrainLabels

    let tree =
        GrokkingMl.Ch09DecisionTrees.buildTreeWith
            GrokkingMl.Ch09DecisionTrees.giniImpurity
            3
            1
            split.TrainPoints
            split.TrainLabels

    let boosted =
        GrokkingMl.Ch12Ensembles.trainAdaBoostWith
            5
            1
            GrokkingMl.Ch09DecisionTrees.giniImpurity
            split.TrainPoints
            (split.TrainLabels |> List.map (fun label -> if label = 1 then 1 else -1))

    [ evaluate
          "logistic"
          (GrokkingMl.Ch06LogisticRegression.predict logistic)
          (GrokkingMl.Ch06LogisticRegression.predictProbability logistic)
          split.TestPoints
          split.TestLabels
      evaluate
          "tree"
          (GrokkingMl.Ch09DecisionTrees.predict tree)
          (fun point -> float (GrokkingMl.Ch09DecisionTrees.predict tree point))
          split.TestPoints
          split.TestLabels
      evaluate
          "adaboost"
          (fun point -> if GrokkingMl.Ch12Ensembles.boostPredict boosted point = 1 then 1 else 0)
          (GrokkingMl.Ch12Ensembles.boostScore boosted)
          split.TestPoints
          split.TestLabels ]

/// F1 スコアがもっとも高いモデルを選ぶ。
let bestByF1 (evaluations: Evaluation list) =
    evaluations |> List.maxBy (fun evaluation -> evaluation.F1)
