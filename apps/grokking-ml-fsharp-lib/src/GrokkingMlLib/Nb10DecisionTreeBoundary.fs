/// 原著ノートブック #10 `Chapter_09_Decision_Trees/Graphical_example.ipynb`。
///
/// 2 次元の 12 点を決定木で分け、**決定境界を図で見る** 回。原著は 3 つのモデルを並べる。
///
/// 1. ジニ不純度で分割した木
/// 2. エントロピーで分割した木
/// 3. 深さ 1 に制限した木（1 本の直線になる）
///
/// 決定木の境界は必ず **軸に平行な長方形の集まり** になる。図そのものは記事の対象外なので、
/// **境界を格子上の予測ラベルとして取り出し**、性質をテストで確かめる。
///
/// [#09](Nb09AppRecommendations.fs) で書いた CART を、
/// 不純度の指標と深さ制限を選べるように拡張して使う。
module GrokkingMlLib.Nb10DecisionTreeBoundary

open GrokkingMlLib.Nb08GiniEntropy

/// 原著が使う 12 点
let x0 = [| 7.0; 3.0; 2.0; 1.0; 2.0; 4.0; 1.0; 8.0; 6.0; 7.0; 8.0; 9.0 |]
let x1 = [| 1.0; 2.0; 3.0; 5.0; 6.0; 7.0; 9.0; 10.0; 5.0; 8.0; 4.0; 6.0 |]
let y = [| "0"; "0"; "0"; "0"; "0"; "0"; "1"; "1"; "1"; "1"; "1"; "1" |]

/// 原著の `plot_model` が使う格子の刻み幅
[<Literal>]
let PlotStep = 0.2

/// 特徴量の名前。添字が行ベクトルの添字に対応する
let featureNames = [ "x_0"; "x_1" ]

/// 行ごとの特徴量ベクトル
let rows = Array.map2 (fun a b -> [| a; b |]) x0 x1

/// 分割の基準。原著の `criterion='gini'` / `'entropy'` に対応する
type Criterion =
    | Gini
    | Entropy

/// 決定木。#09 と同じ判別共用体
type Tree =
    | Leaf of prediction: string
    | Node of feature: string * threshold: float * left: Tree * right: Tree

/// 指標に応じた不純度。[Nb08GiniEntropy] の関数をそのまま使う
let impurityOf criterion (labels: string list) =
    match criterion with
    | Gini -> gini labels
    | Entropy -> entropy labels

/// 隣り合う値の中点を分割候補にする
let private candidateThresholds (values: float[]) =
    values |> Array.distinct |> Array.sort |> Array.pairwise |> Array.map (fun (a, b) -> (a + b) / 2.0)

let private majority (labels: string[]) =
    labels |> Array.countBy id |> Array.maxBy snd |> fst

/// ある特徴量としきい値で分けたときの、重み付き不純度
let private weightedImpurity criterion (rows: float[][]) (labels: string[]) column threshold =
    let indices = Array.init rows.Length id
    let leftIndices = indices |> Array.filter (fun i -> rows.[i].[column] <= threshold)
    let rightIndices = indices |> Array.filter (fun i -> rows.[i].[column] > threshold)

    if leftIndices.Length = 0 || rightIndices.Length = 0 then
        None
    else
        let labelsOf (indices: int[]) = indices |> Array.map (fun i -> labels.[i]) |> List.ofArray
        let leftLabels = labelsOf leftIndices
        let rightLabels = labelsOf rightIndices
        let n = float rows.Length

        let impurity =
            (impurityOf criterion leftLabels * float leftLabels.Length
             + impurityOf criterion rightLabels * float rightLabels.Length)
            / n

        Some(impurity, leftIndices, rightIndices)

/// CART を再帰的に構築する。
///
/// `maxDepth` は **scikit-learn と同じ数え方** で、根を深さ 0 とする。
/// `maxDepth = 1` なら分割が 1 つ入る。
let rec private build criterion maxDepth depth (rows: float[][]) (labels: string[]) =
    let pure' = (labels |> Array.distinct |> Array.length) <= 1

    if pure' then
        Leaf labels.[0]
    elif depth >= maxDepth then
        Leaf(majority labels)
    else
        let best =
            featureNames
            |> List.indexed
            |> List.collect (fun (column, name) ->
                rows
                |> Array.map (fun row -> row.[column])
                |> candidateThresholds
                |> Array.choose (fun threshold ->
                    weightedImpurity criterion rows labels column threshold
                    |> Option.map (fun (impurity, left, right) -> impurity, name, threshold, left, right))
                |> List.ofArray)
            |> List.sortBy (fun (impurity, _, _, _, _) -> impurity)
            |> List.tryHead

        match best with
        | None -> Leaf(majority labels)
        | Some(_, name, threshold, leftIndices, rightIndices) ->
            let subset (indices: int[]) =
                indices |> Array.map (fun i -> rows.[i]), indices |> Array.map (fun i -> labels.[i])

            let leftRows, leftLabels = subset leftIndices
            let rightRows, rightLabels = subset rightIndices

            Node(
                name,
                threshold,
                build criterion maxDepth (depth + 1) leftRows leftLabels,
                build criterion maxDepth (depth + 1) rightRows rightLabels
            )

/// 決定木を学習する。`maxDepth` の既定は実質無制限
let fit (criterion: Criterion) (maxDepth: int) = build criterion maxDepth 0 rows y

/// 1 点を予測する
let rec predict (tree: Tree) (point: float[]) =
    match tree with
    | Leaf prediction -> prediction
    | Node(feature, threshold, left, right) ->
        let column = List.findIndex ((=) feature) featureNames

        if point.[column] <= threshold then
            predict left point
        else
            predict right point

/// 学習データに対する正解率
let accuracy (tree: Tree) =
    let correct =
        Array.map2 (fun row label -> predict tree row = label) rows y
        |> Array.filter id
        |> Array.length

    float correct / float y.Length

/// 分割に使われた条件を、根から深さ優先で並べる
let rec splitConditions (tree: Tree) =
    match tree with
    | Leaf _ -> []
    | Node(feature, threshold, left, right) ->
        (feature, threshold) :: splitConditions left @ splitConditions right

/// NumPy の `np.arange` と同じく、終端を含まない等差数列を作る
let arange (start: float) (stop: float) (step: float) =
    let count = int (ceil ((stop - start) / step))
    Array.init count (fun i -> start + float i * step)

/// 決定境界を格子上の予測ラベルとして持ったもの
type DecisionGrid =
    { XValues: float[]
      YValues: float[]
      /// `Predictions.[row].[column]` が `(XValues.[column], YValues.[row])` に対応
      Predictions: string[][] }

    member this.RowCount = this.Predictions.Length
    member this.ColumnCount = this.Predictions.[0].Length

/// 原著の `plot_model` と同じ格子を作り、各点の予測ラベルを返す
let decisionGrid (tree: Tree) (step: float) =
    let xValues = arange (Array.min x0 - 1.0) (Array.max x0 + 1.0) step
    let yValues = arange (Array.min x1 - 1.0) (Array.max x1 + 1.0) step

    { XValues = xValues
      YValues = yValues
      Predictions =
        yValues
        |> Array.map (fun yValue -> xValues |> Array.map (fun xValue -> predict tree [| xValue; yValue |])) }

/// 左右で予測が変わる x 座標を集める。
///
/// 決定木の境界は軸に平行なので、変わる位置は分割しきい値の近くに限られる。
let boundaryColumns (grid: DecisionGrid) =
    grid.Predictions
    |> Array.collect (fun row ->
        row
        |> Array.pairwise
        |> Array.indexed
        |> Array.choose (fun (i, (previous, current)) ->
            if previous <> current then Some grid.XValues.[i + 1] else None))
    |> Array.distinct
    |> Array.sort
