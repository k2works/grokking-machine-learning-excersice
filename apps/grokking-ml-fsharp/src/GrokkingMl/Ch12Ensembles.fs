/// 第 12 章: アンサンブル学習。
/// 弱い学習器を集めて強い学習器を作る。多数決（バギング）と
/// 逐次的な重み付け（AdaBoost）の 2 つの流儀を実装する。
module GrokkingMl.Ch12Ensembles

open System
open GrokkingMl.Ch09DecisionTrees

/// 復元抽出で元と同じ大きさの標本を作る。
let bootstrapSample (rng: Random) (points: Point list) (labels: int list) =
    let indices = List.init (List.length points) (fun _ -> rng.Next(List.length points))
    (indices |> List.map (fun i -> points[i]), indices |> List.map (fun i -> labels[i]))

/// 多数決で予測する木の集まり。
type Forest = { Trees: Tree list }

/// 各木の投票。
let votes (forest: Forest) (point: Point) =
    forest.Trees |> List.map (fun tree -> predict tree point)

/// 多数決による予測。
let forestPredict (forest: Forest) (point: Point) = majorityLabel (votes forest point)

/// バギング。復元抽出した標本ごとに木を育て、多数決で予測する。
let trainForestWith
    (treeCount: int)
    (maxDepth: int)
    (impurity: Impurity)
    (seed: int)
    (points: Point list)
    (labels: int list)
    =
    let rng = Random(seed)

    { Trees =
        List.init treeCount (fun _ ->
            let samplePoints, sampleLabels = bootstrapSample rng points labels
            buildTreeWith impurity maxDepth 1 samplePoints sampleLabels) }

/// 既定の設定（木 10 本、深さ 1、ジニ不純度、シード 0）でバギングする。
let trainForest (points: Point list) (labels: int list) =
    trainForestWith 10 1 giniImpurity 0 points labels

/// AdaBoost の弱学習器。発言権（Weight）を持つ。
type WeightedTree = { Tree: Tree; Weight: float }

/// 重み付き多数決で予測する学習器の列。ラベルは +1 / -1。
type AdaBoost = { Learners: WeightedTree list }

/// 重み付き投票のスコア。
let boostScore (model: AdaBoost) (point: Point) =
    model.Learners
    |> List.sumBy (fun learner -> learner.Weight * float (predict learner.Tree point))

/// AdaBoost による予測。
let boostPredict (model: AdaBoost) (point: Point) =
    if boostScore model point >= 0.0 then 1 else -1

/// 重み付き誤り率。重みの大きい点を間違えるほど大きくなる。
let weightedError (tree: Tree) (points: Point list) (labels: int list) (weights: float list) =
    let total = List.sum weights

    if total = 0.0 then
        0.0
    else
        List.zip3 points labels weights
        |> List.sumBy (fun (point, label, weight) -> if predict tree point <> label then weight else 0.0)
        |> fun wrong -> wrong / total

/// 弱学習器の発言権。誤り率が小さいほど大きい。
let learnerWeight (error: float) =
    let epsilon = 1e-10
    let clamped = error |> max epsilon |> min (1.0 - epsilon)
    0.5 * log ((1.0 - clamped) / clamped)

/// 重みを反映した木。重みに比例して点を複製してから学習する。
let buildTreeWithWeights
    (impurity: Impurity)
    (maxDepth: int)
    (points: Point list)
    (labels: int list)
    (weights: float list)
    =
    let scale = 100.0

    let replicated =
        List.zip3 points labels weights
        |> List.collect (fun (point, label, weight) ->
            let count = max 1 (int (Math.Round(weight * scale)))
            List.replicate count (point, label))

    let replicatedLabels = replicated |> List.map snd

    if (replicatedLabels |> List.distinct |> List.length) = 1 then
        Leaf(List.head replicatedLabels)
    else
        buildTreeWith impurity maxDepth 1 (replicated |> List.map fst) replicatedLabels

/// AdaBoost。間違えた点の重みを上げながら弱学習器を足していく。
let trainAdaBoostWith
    (rounds: int)
    (maxDepth: int)
    (impurity: Impurity)
    (points: Point list)
    (labels: int list)
    =
    let initialWeights = List.replicate (List.length points) (1.0 / float (List.length points))

    let step (learners, weights) _ =
        let tree = buildTreeWithWeights impurity maxDepth points labels weights
        let error = weightedError tree points labels weights

        // 当てずっぽう以下の学習器は採用しない
        if error >= 0.5 then
            (learners, weights)
        else
            let alpha = learnerWeight error

            let updated =
                List.zip3 points labels weights
                |> List.map (fun (point, label, weight) ->
                    weight * exp (-alpha * float label * float (predict tree point)))

            let total = List.sum updated
            (learners @ [ { Tree = tree; Weight = alpha } ], updated |> List.map (fun w -> w / total))

    let learners, _ = List.fold step ([], initialWeights) [ 1..rounds ]
    { Learners = learners }

/// 既定の設定（5 ラウンド、深さ 1、ジニ不純度）で学習する。
let trainAdaBoost (points: Point list) (labels: int list) =
    trainAdaBoostWith 5 1 giniImpurity points labels

/// 森の正解率。
let forestAccuracy (forest: Forest) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if forestPredict forest point = label then 1.0 else 0.0) points labels
    |> List.average

/// AdaBoost の正解率。
let boostAccuracy (model: AdaBoost) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if boostPredict model point = label then 1.0 else 0.0) points labels
    |> List.average

/// 1 本の木の正解率。
let treeAccuracy (tree: Tree) (points: Point list) (labels: int list) =
    List.map2 (fun point label -> if predict tree point = label then 1.0 else 0.0) points labels
    |> List.average
