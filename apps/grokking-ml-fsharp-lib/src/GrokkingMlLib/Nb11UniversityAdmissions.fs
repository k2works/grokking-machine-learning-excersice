/// 原著ノートブック #11 `Chapter_09_Decision_Trees/University_Admissions.ipynb`。
///
/// 大学院の入学審査データ 400 件から、合格するかどうかを決定木で当てる。
/// 原著は **同じデータで木の大きさを変えて、過学習の様子を見せる** 構成になっている。
///
/// [#10](Nb10DecisionTreeBoundary.fs) の CART を、
/// 任意の特徴量・葉の最小件数・分割の最小件数に対応できるように一般化して使う。
/// 原著が掛ける 3 つの制限（深さ・葉の件数・分割の件数）をすべて再現する。
module GrokkingMlLib.Nb11UniversityAdmissions

open Deedle
open GrokkingMlLib.Nb08GiniEntropy

/// 原著が合格とみなす基準
[<Literal>]
let AdmissionThreshold = 0.75

/// 特徴量の列。原著の順序をそのまま保つ
let featureNames =
    [ "GRE Score"; "TOEFL Score"; "University Rating"; "SOP"; "LOR"; "CGPA"; "Research" ]

/// 2 特徴量だけで学習するときに使う列
let examFeatures = [ "GRE Score"; "TOEFL Score" ]

/// 読み込んだデータ。列名 -> 値と、合否ラベル
type Admissions =
    { Columns: Map<string, float[]>
      Admitted: string[] }

    member this.Size = this.Admitted.Length

    /// 指定した列だけを行ベクトルの配列にする
    member this.RowsOf(names: string list) =
        Array.init this.Size (fun row -> names |> List.map (fun name -> this.Columns.[name].[row]) |> Array.ofList)

/// 決定木。#09・#10 と同じ判別共用体
type Tree =
    | Leaf of prediction: string
    | Node of feature: string * threshold: float * left: Tree * right: Tree

/// 入学審査データを読み込み、合否のラベルを付ける。
///
/// 原著は `Chance of Admit`（合格確率）を 0.75 で切って 2 値にし、元の列を落としている。
let loadData () =
    let frame = Datasets.loadFrame "Admission_Predict.csv"

    let columns =
        featureNames
        |> List.map (fun name -> name, frame.GetColumn<float>(name) |> Series.values |> Array.ofSeq)
        |> Map.ofList

    let admitted =
        frame.GetColumn<float>("Chance of Admit")
        |> Series.values
        |> Seq.map (fun chance -> if chance >= AdmissionThreshold then "true" else "false")
        |> Array.ofSeq

    { Columns = columns; Admitted = admitted }

let private candidateThresholds (values: float[]) =
    values |> Array.distinct |> Array.sort |> Array.pairwise |> Array.map (fun (a, b) -> (a + b) / 2.0)

/// 葉の予測クラス。もっとも多いクラスを返す。
///
/// **同数のときはクラス名の小さいほうを選ぶ。** scikit-learn は `np.argmax` を
/// クラス順に並べた件数配列へ適用するので、同数なら先頭のクラス（ここでは
/// `"false"`）が選ばれる。原著が「白い（中立な）葉は False になる」と
/// コメントしているのはこの挙動である。出現順で選ぶと結果が変わる。
let private majority (labels: string[]) =
    labels
    |> Array.countBy id
    |> Array.sortBy (fun (label, count) -> -count, label)
    |> Array.head
    |> fst

/// 木を育てるときの制限。原著の 3 つの引数にそのまま対応する
type Limits =
    { /// `max_depth`。根を深さ 0 と数える
      MaxDepth: int
      /// `min_samples_leaf`。葉に残る件数の下限
      MinSamplesLeaf: int
      /// `min_samples_split`。分割してよい節の件数の下限
      MinSamplesSplit: int }

/// 制限なし
let unlimited =
    { MaxDepth = 100
      MinSamplesLeaf = 1
      MinSamplesSplit = 2 }

let private weightedGini (rows: float[][]) (labels: string[]) column threshold minSamplesLeaf =
    let indices = Array.init rows.Length id
    let leftIndices = indices |> Array.filter (fun i -> rows.[i].[column] <= threshold)
    let rightIndices = indices |> Array.filter (fun i -> rows.[i].[column] > threshold)

    if leftIndices.Length < minSamplesLeaf || rightIndices.Length < minSamplesLeaf then
        None
    else
        let labelsOf (indices: int[]) = indices |> Array.map (fun i -> labels.[i]) |> List.ofArray
        let leftLabels = labelsOf leftIndices
        let rightLabels = labelsOf rightIndices
        let n = float rows.Length

        let impurity =
            (gini leftLabels * float leftLabels.Length + gini rightLabels * float rightLabels.Length) / n

        Some(impurity, leftIndices, rightIndices)

/// CART を再帰的に構築する。原著の 3 つの制限をすべて見る
let rec private build (names: string list) limits depth (rows: float[][]) (labels: string[]) =
    let isPure = (labels |> Array.distinct |> Array.length) <= 1

    if isPure || depth >= limits.MaxDepth || labels.Length < limits.MinSamplesSplit then
        Leaf(majority labels)
    else
        let best =
            names
            |> List.indexed
            |> List.collect (fun (column, name) ->
                rows
                |> Array.map (fun row -> row.[column])
                |> candidateThresholds
                |> Array.choose (fun threshold ->
                    weightedGini rows labels column threshold limits.MinSamplesLeaf
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
                build names limits (depth + 1) leftRows leftLabels,
                build names limits (depth + 1) rightRows rightLabels
            )

/// 決定木を学習する
let fit (data: Admissions) (names: string list) (limits: Limits) =
    build names limits 0 (data.RowsOf names) data.Admitted

/// 制限なしの木。訓練データを完全に覚えてしまう
let fitFull (data: Admissions) = fit data featureNames unlimited

/// 原著が「過学習しない小さい木」として作る設定
let fitSmaller (data: Admissions) =
    fit data featureNames { MaxDepth = 3; MinSamplesLeaf = 10; MinSamplesSplit = 10 }

/// GRE と TOEFL の 2 特徴量だけで学習する
let fitExams (data: Admissions) (maxDepth: int) =
    fit data examFeatures { unlimited with MaxDepth = maxDepth }

/// 1 行を予測する
let rec predict (tree: Tree) (names: string list) (row: float[]) =
    match tree with
    | Leaf prediction -> prediction
    | Node(feature, threshold, left, right) ->
        let column = List.findIndex ((=) feature) names

        if row.[column] <= threshold then
            predict left names row
        else
            predict right names row

/// 学習データに対する正解率
let accuracy (tree: Tree) (data: Admissions) (names: string list) =
    let rows = data.RowsOf names

    let correct =
        Array.map2 (fun row label -> predict tree names row = label) rows data.Admitted
        |> Array.filter id
        |> Array.length

    float correct / float data.Size

/// 1 人ぶんの出願情報から合否を予測する
let predictApplicant (tree: Tree) (values: float list) =
    predict tree featureNames (Array.ofList values) = "true"

/// 分割に使われた条件を、根から深さ優先で並べる
let rec splitConditions (tree: Tree) =
    match tree with
    | Leaf _ -> []
    | Node(feature, threshold, left, right) ->
        (feature, threshold) :: splitConditions left @ splitConditions right

/// 節と葉を合わせた数
let rec nodeCount (tree: Tree) =
    match tree with
    | Leaf _ -> 1
    | Node(_, _, left, right) -> 1 + nodeCount left + nodeCount right
