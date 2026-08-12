/// 第 9 章: 決定木。
/// 「どの質問をすれば、もっともよくデータが分かれるか」を貪欲に選び続けて
/// 木を育てる。分割の良さはジニ不純度またはエントロピーで測る。
module GrokkingMl.Ch09DecisionTrees

/// 特徴量ベクトル。
type Point = float list

/// 不純度の測り方。
type Impurity = int list -> float

/// ラベルごとの割合。
let private shares (labels: int list) =
    let total = float (List.length labels)

    labels
    |> List.countBy id
    |> List.map (fun (_, count) -> float count / total)

/// ジニ不純度。ランダムに 2 つ選んだとき、ラベルが食い違う確率。
let giniImpurity: Impurity =
    fun labels ->
        if List.isEmpty labels then
            0.0
        else
            1.0 - (shares labels |> List.sumBy (fun share -> share * share))

/// エントロピー。ラベルの散らばりを情報量で測る。
let entropy: Impurity =
    fun labels ->
        if List.isEmpty labels then
            0.0
        else
            -(shares labels |> List.sumBy (fun share -> share * log share / log 2.0))

/// 1 つの質問による分割。「Feature 番目の特徴量が Threshold 未満か」を問う。
type Split =
    { Feature: int
      Threshold: float }

/// 左（true）へ進むか。
let matches (split: Split) (point: Point) = point[split.Feature] < split.Threshold

/// 分割で振り分けられた左右のデータ。
type Partition =
    { LeftPoints: Point list
      LeftLabels: int list
      RightPoints: Point list
      RightLabels: int list }

/// 分割を適用して左右に振り分ける。
let applySplit (points: Point list) (labels: int list) (split: Split) =
    let left, right =
        List.zip points labels
        |> List.partition (fun (point, _) -> matches split point)

    { LeftPoints = List.map fst left
      LeftLabels = List.map snd left
      RightPoints = List.map fst right
      RightLabels = List.map snd right }

/// 分割後の不純度。左右の大きさで重み付けして平均する。
let weightedImpurity (impurity: Impurity) (leftLabels: int list) (rightLabels: int list) =
    let leftSize = List.length leftLabels
    let rightSize = List.length rightLabels
    let total = float (leftSize + rightSize)

    if total = 0.0 then
        0.0
    else
        float leftSize / total * impurity leftLabels
        + float rightSize / total * impurity rightLabels

/// 情報利得。分割によって不純度がどれだけ下がったか。
let informationGain
    (impurity: Impurity)
    (labels: int list)
    (leftLabels: int list)
    (rightLabels: int list)
    =
    impurity labels - weightedImpurity impurity leftLabels rightLabels

/// 試す価値のある分割の候補。隣り合う値の中点を閾値にする。
let candidateSplits (points: Point list) =
    [ for feature in 0 .. List.length (List.head points) - 1 do
          let values = points |> List.map (fun point -> point[feature]) |> List.distinct |> List.sort

          for low, high in List.pairwise values do
              { Feature = feature; Threshold = (low + high) / 2.0 } ]

/// 情報利得がもっとも大きい分割。改善しないなら None。
let bestSplit (impurity: Impurity) (points: Point list) (labels: int list) =
    candidateSplits points
    |> List.choose (fun split ->
        let partition = applySplit points labels split

        if List.isEmpty partition.LeftLabels || List.isEmpty partition.RightLabels then
            None
        else
            Some(split, informationGain impurity labels partition.LeftLabels partition.RightLabels))
    |> function
        | [] -> None
        | candidates ->
            let best = candidates |> List.maxBy snd
            if snd best > 0.0 then Some best else None

/// 決定木。葉か内部ノードのいずれか。
type Tree =
    | Leaf of label: int
    | Node of split: Split * left: Tree * right: Tree

/// 木による予測。
let rec predict (tree: Tree) (point: Point) =
    match tree with
    | Leaf label -> label
    | Node(split, left, right) -> predict (if matches split point then left else right) point

/// 多数決。同数なら小さいラベルを選ぶ。
let majorityLabel (labels: int list) =
    let counts = List.countBy id labels
    let top = counts |> List.map snd |> List.max

    counts
    |> List.filter (fun (_, count) -> count = top)
    |> List.map fst
    |> List.min

/// 決定木を再帰的に構築する。
let rec buildTreeWith
    (impurity: Impurity)
    (maxDepth: int)
    (minSamples: int)
    (points: Point list)
    (labels: int list)
    =
    if maxDepth <= 0
       || List.length labels <= minSamples
       || (labels |> List.distinct |> List.length) = 1 then
        Leaf(majorityLabel labels)
    else
        match bestSplit impurity points labels with
        | None -> Leaf(majorityLabel labels)
        | Some(split, _) ->
            let partition = applySplit points labels split

            Node(
                split,
                buildTreeWith impurity (maxDepth - 1) minSamples partition.LeftPoints partition.LeftLabels,
                buildTreeWith impurity (maxDepth - 1) minSamples partition.RightPoints partition.RightLabels
            )

/// 既定の設定（ジニ不純度、深さ 5、最小サンプル 1）で木を構築する。
let buildTree (points: Point list) (labels: int list) =
    buildTreeWith giniImpurity 5 1 points labels

/// 木の深さ。葉だけなら 0。
let rec depth (tree: Tree) =
    match tree with
    | Leaf _ -> 0
    | Node(_, left, right) -> 1 + max (depth left) (depth right)

/// 葉の数。
let rec leafCount (tree: Tree) =
    match tree with
    | Leaf _ -> 1
    | Node(_, left, right) -> leafCount left + leafCount right

/// 正解率。
let accuracy (tree: Tree) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict tree point = label then 1.0 else 0.0) points labels
    |> List.average
