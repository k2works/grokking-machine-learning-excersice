/// 原著ノートブック #12 `Chapter_09_Decision_Trees/Regression_decision_tree.ipynb`。
///
/// 決定木を **回帰** に使う回。年齢からアプリの利用日数を予測する 8 点のデータに、
/// 深さ 2 の回帰木を当てはめる。
///
/// 分類木がジニ不純度を最小にする分割を探したのに対し、回帰木は
/// **平均二乗誤差（MSE）を最小にする分割** を探す。
/// [#09](Nb09AppRecommendations.fs) からの CART を回帰用に書き換える。
module GrokkingMlLib.Nb12RegressionTree

/// 原著が使う 8 点。年齢と、週あたりの利用日数
let ages = [| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0; 70.0; 80.0 |]
let days = [| 7.0; 5.0; 7.0; 1.0; 2.0; 1.0; 5.0; 4.0 |]

/// ある分割位置での、左右の平均と重み付き MSE
type SplitMse =
    { Index: int
      Left: float list
      Right: float list
      /// 左の平均。左が空なら NaN
      LeftMean: float
      RightMean: float
      /// 全体を分母にした重み付き MSE
      WeightedMse: float }

/// 回帰木。葉が **平均値** を返すのが分類木との違い
type Tree =
    | Leaf of prediction: float
    | Node of threshold: float * left: Tree * right: Tree

/// 空の配列の平均は NaN。原著も NumPy の警告つきで NaN を出している
let private meanOrNaN (values: float[]) =
    if Array.isEmpty values then nan else Array.average values

/// 分割位置を 0 から n まで動かし、それぞれの重み付き MSE を求める。
///
/// 原著は `range(0, 9)` と、要素数 8 に対して **9 通り** 回している。
/// 先頭（左が空）と末尾（右が空）の両方が含まれる。
let splitMses (labels: float[]) =
    let total = labels.Length

    [ for index in 0..total ->
        let left = Array.sub labels 0 index
        let right = Array.sub labels index (total - index)
        let leftMean = meanOrNaN left
        let rightMean = meanOrNaN right

        let squaredErrors =
            (left |> Array.sumBy (fun v -> (v - leftMean) ** 2.0))
            + (right |> Array.sumBy (fun v -> (v - rightMean) ** 2.0))

        { Index = index
          Left = List.ofArray left
          Right = List.ofArray right
          LeftMean = leftMean
          RightMean = rightMean
          WeightedMse = squaredErrors / float total } ]

/// 重み付き MSE がもっとも小さい分割を返す
let bestSplit (labels: float[]) =
    splitMses labels |> List.minBy (fun split -> split.WeightedMse)

/// 隣り合う値の中点を分割候補にする
let private candidateThresholds (values: float[]) =
    values |> Array.distinct |> Array.sort |> Array.pairwise |> Array.map (fun (a, b) -> (a + b) / 2.0)

/// 二乗誤差の合計。回帰木ではこれを最小にする分割を選ぶ
let private squaredError (values: float[]) =
    if Array.isEmpty values then
        0.0
    else
        let mean = Array.average values
        values |> Array.sumBy (fun v -> (v - mean) ** 2.0)

/// CART を回帰用に構築する。`maxDepth` は根を深さ 0 と数える
let rec private build maxDepth depth (xs: float[]) (ys: float[]) =
    // 値がすべて同じか、深さの上限に達したら葉にする
    if depth >= maxDepth || (ys |> Array.distinct |> Array.length) <= 1 then
        Leaf(Array.average ys)
    else
        let best =
            candidateThresholds xs
            |> Array.choose (fun threshold ->
                let indices = Array.init xs.Length id
                let leftIndices = indices |> Array.filter (fun i -> xs.[i] <= threshold)
                let rightIndices = indices |> Array.filter (fun i -> xs.[i] > threshold)

                if leftIndices.Length = 0 || rightIndices.Length = 0 then
                    None
                else
                    let leftYs = leftIndices |> Array.map (fun i -> ys.[i])
                    let rightYs = rightIndices |> Array.map (fun i -> ys.[i])
                    let total = squaredError leftYs + squaredError rightYs
                    Some(total, threshold, leftIndices, rightIndices))
            |> Array.sortBy (fun (total, _, _, _) -> total)
            |> Array.tryHead

        match best with
        | None -> Leaf(Array.average ys)
        | Some(_, threshold, leftIndices, rightIndices) ->
            let subset (indices: int[]) =
                indices |> Array.map (fun i -> xs.[i]), indices |> Array.map (fun i -> ys.[i])

            let leftXs, leftYs = subset leftIndices
            let rightXs, rightYs = subset rightIndices

            Node(threshold, build maxDepth (depth + 1) leftXs leftYs, build maxDepth (depth + 1) rightXs rightYs)

/// 回帰木を学習する。原著は深さ 2 に制限している
let fit (maxDepth: int) = build maxDepth 0 ages days

/// 1 点を予測する
let rec predict (tree: Tree) (age: float) =
    match tree with
    | Leaf prediction -> prediction
    | Node(threshold, left, right) ->
        if age <= threshold then predict left age else predict right age

/// 分割に使われたしきい値を、根から深さ優先で並べる
let rec splitThresholds (tree: Tree) =
    match tree with
    | Leaf _ -> []
    | Node(threshold, left, right) -> threshold :: splitThresholds left @ splitThresholds right

/// 葉が返す予測値を、左から順に並べる
let rec leafValues (tree: Tree) =
    match tree with
    | Leaf prediction -> [ prediction ]
    | Node(_, left, right) -> leafValues left @ leafValues right
