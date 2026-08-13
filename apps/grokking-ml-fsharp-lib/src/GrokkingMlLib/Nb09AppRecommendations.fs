/// 原著ノートブック #09 `Chapter_09_Decision_Trees/App_recommendations.ipynb`。
///
/// 6 人のユーザーの「使っている端末」と「年齢」から、おすすめアプリを決定木で当てる。
/// 原著は同じデータを 2 通りの形で学習させ、木の形がどう変わるかを見せている。
///
/// 1. 年齢を **カテゴリ**（若者 / 大人）に潰して one-hot 符号化する
/// 2. 年齢を **数値** のまま渡す
///
/// **ML.NET の決定木は木の構造を外に出さない。** `FastTree` は勾配ブースティングの
/// 実装で、単体の決定木として使うには木を 1 本に制限する必要があり、それでも
/// 分割条件を読み出す API が無い。原著の見どころが「木の形」なので、
/// ここでは **CART を自前で実装** して形を取り出せるようにした。
module GrokkingMlLib.Nb09AppRecommendations

open System

/// 原著の元データ。この 6 人ぶんの情報しかない
let platforms = [| "iPhone"; "iPhone"; "Android"; "iPhone"; "Android"; "Android" |]
let ages = [| 15; 25; 32; 35; 12; 14 |]

let apps =
    [| "Atom Count"; "Check Mate Mate"; "Beehive Finder"
       "Check Mate Mate"; "Atom Count"; "Atom Count" |]

/// 決定木。判別共用体で葉と節を表す
type Tree =
    /// 葉。予測するクラス
    | Leaf of prediction: string
    /// 節。特徴量としきい値、そして左右の部分木
    | Node of feature: string * threshold: float * left: Tree * right: Tree

/// 学習に使う表。列名と値の対応
type Dataset =
    { FeatureNames: string list
      /// 行ごとの特徴量ベクトル
      Rows: float[][]
      Labels: string[] }

/// 年齢をカテゴリに潰して one-hot 符号化した表
let categoricalDataset =
    { FeatureNames = [ "Platform_iPhone"; "Platform_Android"; "Age_Adult"; "Age_Young" ]
      Rows =
        [| [| 1.0; 0.0; 0.0; 1.0 |]
           [| 1.0; 0.0; 1.0; 0.0 |]
           [| 0.0; 1.0; 1.0; 0.0 |]
           [| 1.0; 0.0; 1.0; 0.0 |]
           [| 0.0; 1.0; 0.0; 1.0 |]
           [| 0.0; 1.0; 0.0; 1.0 |] |]
      Labels = apps }

/// 年齢を数値のまま残した表
let numericDataset =
    { FeatureNames = [ "Age"; "Platform_iPhone"; "Platform_Android" ]
      Rows =
        [| [| 15.0; 1.0; 0.0 |]
           [| 25.0; 1.0; 0.0 |]
           [| 32.0; 0.0; 1.0 |]
           [| 35.0; 1.0; 0.0 |]
           [| 12.0; 0.0; 1.0 |]
           [| 14.0; 0.0; 1.0 |] |]
      Labels = apps }

/// ジニ不純度。#08 で書いたものと同じ定義
let gini (labels: string[]) =
    if labels.Length = 0 then
        0.0
    else
        let n = float labels.Length

        labels
        |> Array.countBy id
        |> Array.sumBy (fun (_, count) -> (float count / n) ** 2.0)
        |> fun sumOfSquares -> 1.0 - sumOfSquares

/// もっとも多いクラス。葉の予測に使う
let private majority (labels: string[]) =
    labels |> Array.countBy id |> Array.maxBy snd |> fst

/// ある特徴量としきい値で分けたときの、重み付きジニ不純度
let private weightedGini (rows: float[][]) (labels: string[]) (column: int) (threshold: float) =
    let indices = Array.init rows.Length id
    let leftIndices = indices |> Array.filter (fun i -> rows.[i].[column] <= threshold)
    let rightIndices = indices |> Array.filter (fun i -> rows.[i].[column] > threshold)

    if leftIndices.Length = 0 || rightIndices.Length = 0 then
        None
    else
        let leftLabels = leftIndices |> Array.map (fun i -> labels.[i])
        let rightLabels = rightIndices |> Array.map (fun i -> labels.[i])
        let n = float rows.Length

        Some(
            (gini leftLabels * float leftLabels.Length + gini rightLabels * float rightLabels.Length)
            / n,
            leftIndices,
            rightIndices
        )

/// 分割の候補となるしきい値。隣り合う値の中点を使う。
///
/// scikit-learn も同じ規則で、`Age` が 15 と 25 のときに 20.0 を選ぶ。
let candidateThresholds (values: float[]) =
    values
    |> Array.distinct
    |> Array.sort
    |> Array.pairwise
    |> Array.map (fun (a, b) -> (a + b) / 2.0)

/// CART を再帰的に構築する。
///
/// 同点の分割候補が複数あるとき、**列の順に最初のものを選ぶ**。
/// scikit-learn は無作為に選ぶので、そこだけ規則が違う。決定的なぶん
/// 実行するたびに同じ木になる。
let rec private build (dataset: Dataset) (rows: float[][]) (labels: string[]) =
    if gini labels = 0.0 then
        Leaf(labels.[0])
    else
        let best =
            dataset.FeatureNames
            |> List.indexed
            |> List.collect (fun (column, name) ->
                let values = rows |> Array.map (fun row -> row.[column])

                candidateThresholds values
                |> Array.choose (fun threshold ->
                    weightedGini rows labels column threshold
                    |> Option.map (fun (impurity, left, right) ->
                        impurity, name, column, threshold, left, right))
                |> List.ofArray)
            |> List.sortBy (fun (impurity, _, _, _, _, _) -> impurity)
            |> List.tryHead

        match best with
        | None -> Leaf(majority labels)
        | Some(_, name, _, threshold, leftIndices, rightIndices) ->
            let subset (indices: int[]) =
                indices |> Array.map (fun i -> rows.[i]), indices |> Array.map (fun i -> labels.[i])

            let leftRows, leftLabels = subset leftIndices
            let rightRows, rightLabels = subset rightIndices

            Node(
                name,
                threshold,
                build dataset leftRows leftLabels,
                build dataset rightRows rightLabels
            )

/// 決定木を学習する
let fit (dataset: Dataset) = build dataset dataset.Rows dataset.Labels

/// 1 行を予測する
let rec predict (tree: Tree) (dataset: Dataset) (row: float[]) =
    match tree with
    | Leaf prediction -> prediction
    | Node(feature, threshold, left, right) ->
        let column = List.findIndex ((=) feature) dataset.FeatureNames

        if row.[column] <= threshold then
            predict left dataset row
        else
            predict right dataset row

/// 訓練データに対する正解率
let accuracy (tree: Tree) (dataset: Dataset) =
    let correct =
        Array.map2 (fun row label -> predict tree dataset row = label) dataset.Rows dataset.Labels
        |> Array.filter id
        |> Array.length

    float correct / float dataset.Labels.Length

/// 分割に使われた特徴量としきい値を、根から深さ優先で並べる
let rec splits (tree: Tree) =
    match tree with
    | Leaf _ -> []
    | Node(feature, threshold, left, right) ->
        (feature, threshold) :: splits left @ splits right

/// 葉が予測するクラスを、左から順に並べる
let rec leafPredictions (tree: Tree) =
    match tree with
    | Leaf prediction -> [ prediction ]
    | Node(_, _, left, right) -> leafPredictions left @ leafPredictions right
