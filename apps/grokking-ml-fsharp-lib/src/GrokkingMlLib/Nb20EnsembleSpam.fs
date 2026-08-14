/// 原著ノートブック #20 `Chapter_12_Ensemble_Methods/Random_forests_and_AdaBoost.ipynb`。
///
/// 18 通のメールを 2 特徴量（`Lottery`・`Sale`）で振り分ける小さなデータに、
/// アンサンブル学習を次々と当てる回である。
///
/// **「正解率が 1.0 になるのは良いことではない」** を見せるのが狙い。
///
/// F# には決定木のライブラリとして Accord.MachineLearning がある。
/// ランダムフォレストと決定木は持っているが、**AdaBoost は分類に使える形で
/// 揃っていない**（`Boost<TModel>` はあるが弱学習器の重み付き学習が要る）。
/// そこは [#11](Nb11UniversityAdmissions.fs) の自前 CART を土台に
/// **AdaBoost を書き下ろす**。原著のアルゴリズムをそのまま写せるので、
/// かえって中身が見える形になった。
module GrokkingMlLib.Nb20EnsembleSpam

/// 原著の 18 通。[Lottery; Sale; Spam]
let emails =
    [| [| 7; 8; 1 |]
       [| 3; 2; 0 |]
       [| 8; 4; 1 |]
       [| 2; 6; 0 |]
       [| 6; 5; 1 |]
       [| 9; 6; 1 |]
       [| 8; 5; 0 |]
       [| 7; 1; 0 |]
       [| 1; 9; 1 |]
       [| 4; 7; 0 |]
       [| 1; 3; 0 |]
       [| 3; 10; 1 |]
       [| 2; 2; 1 |]
       [| 9; 3; 0 |]
       [| 5; 3; 0 |]
       [| 10; 1; 0 |]
       [| 5; 9; 1 |]
       [| 10; 8; 1 |] |]

/// 原著が手作業で切り分ける 3 組。6 通ずつ **並び順のまま** 分ける
let batches =
    [ [ 0; 1; 2; 3; 4; 5 ]; [ 6; 7; 8; 9; 10; 11 ]; [ 12; 13; 14; 15; 16; 17 ] ]

/// 特徴量の列名。原著の順序をそのまま保つ
let featureNames = [| "Lottery"; "Sale" |]

/// 学習に渡すデータ
type Dataset =
    { X: float[][]
      Y: int[] }

    member this.Size = this.Y.Length

/// 指定した行だけを取り出す
let subset (indices: int list) =
    let rows = indices |> List.map (fun i -> emails.[i]) |> Array.ofList

    { X = rows |> Array.map (fun row -> [| float row.[0]; float row.[1] |])
      Y = rows |> Array.map (fun row -> row.[2]) }

/// 18 通すべて
let allData = subset [ 0 .. emails.Length - 1 ]

/// 3 組のうち 1 つ
let batch index = subset batches.[index]

/// 決定木。深さ 1 なら 1 回だけ分割する切り株になる
type Tree =
    | Leaf of prediction: int
    | Node of feature: int * threshold: float * left: Tree * right: Tree

/// 重み付きのジニ不純度。AdaBoost では点ごとに重みが付く
let private weightedGini (labels: int[]) (weights: float[]) (indices: int[]) =
    let total = indices |> Array.sumBy (fun i -> weights.[i])

    if total = 0.0 then
        0.0
    else
        let ofLabel label =
            indices |> Array.filter (fun i -> labels.[i] = label) |> Array.sumBy (fun i -> weights.[i])

        let p1 = ofLabel 1 / total
        let p0 = ofLabel 0 / total
        1.0 - p1 * p1 - p0 * p0

/// 重みつきの多数決。同数ならラベルの小さいほう（scikit-learn の argmax と同じ）
let private weightedMajority (labels: int[]) (weights: float[]) (indices: int[]) =
    let weightOf label =
        indices |> Array.filter (fun i -> labels.[i] = label) |> Array.sumBy (fun i -> weights.[i])

    if weightOf 1 > weightOf 0 then 1 else 0

/// 分割の候補。scikit-learn と同じく **隣り合う値の中点** を使う
let private candidateThresholds (values: float[]) =
    values |> Array.distinct |> Array.sort |> Array.pairwise |> Array.map (fun (a, b) -> (a + b) / 2.0)

/// CART を再帰的に構築する。重みを渡せるようにしてある
let rec private build (data: Dataset) (weights: float[]) (indices: int[]) depth maxDepth =
    let labels = indices |> Array.map (fun i -> data.Y.[i]) |> Array.distinct

    if labels.Length <= 1 || depth >= maxDepth then
        Leaf(weightedMajority data.Y weights indices)
    else
        let best =
            [| 0 .. featureNames.Length - 1 |]
            |> Array.collect (fun column ->
                indices
                |> Array.map (fun i -> data.X.[i].[column])
                |> candidateThresholds
                |> Array.choose (fun threshold ->
                    let left = indices |> Array.filter (fun i -> data.X.[i].[column] <= threshold)
                    let right = indices |> Array.filter (fun i -> data.X.[i].[column] > threshold)

                    if left.Length = 0 || right.Length = 0 then
                        None
                    else
                        let total = indices |> Array.sumBy (fun i -> weights.[i])

                        let impurity =
                            (weightedGini data.Y weights left * (left |> Array.sumBy (fun i -> weights.[i]))
                             + weightedGini data.Y weights right * (right |> Array.sumBy (fun i -> weights.[i])))
                            / total

                        Some(impurity, column, threshold, left, right)))
            |> Array.sortBy (fun (impurity, column, threshold, _, _) -> impurity, column, threshold)
            |> Array.tryHead

        match best with
        | None -> Leaf(weightedMajority data.Y weights indices)
        | Some(_, column, threshold, left, right) ->
            Node(
                column,
                threshold,
                build data weights left (depth + 1) maxDepth,
                build data weights right (depth + 1) maxDepth
            )

/// 決定木を学習する
let fitTree (data: Dataset) (maxDepth: int) =
    let weights = Array.create data.Size 1.0
    build data weights [| 0 .. data.Size - 1 |] 0 maxDepth

/// 重みを指定して決定木を学習する（AdaBoost 用）
let fitWeightedTree (data: Dataset) (weights: float[]) (maxDepth: int) =
    build data weights [| 0 .. data.Size - 1 |] 0 maxDepth

/// 1 行を予測する
let rec predictTree (tree: Tree) (row: float[]) =
    match tree with
    | Leaf prediction -> prediction
    | Node(column, threshold, left, right) ->
        if row.[column] <= threshold then
            predictTree left row
        else
            predictTree right row

/// 学習データに対する正解率
let accuracy (predict: float[] -> int) (data: Dataset) =
    let correct =
        Array.map2 (fun row label -> predict row = label) data.X data.Y
        |> Array.filter id
        |> Array.length

    float correct / float data.Size

/// 深さ 1 の木が使った特徴量としきい値
let splitOf (tree: Tree) =
    match tree with
    | Node(column, threshold, _, _) -> Some(featureNames.[column], threshold)
    | Leaf _ -> None

/// AdaBoost の 1 段。弱学習器とその発言力
type Stage = { Tree: Tree; Alpha: float }

/// AdaBoost（SAMME）を原著のアルゴリズムどおりに実装する。
///
/// 1. すべての点に等しい重みを与える
/// 2. 重み付きで切り株を学習する
/// 3. 誤り率 ε から発言力 α = log((1-ε)/ε) を計算する
/// 4. 外した点の重みを exp(α) 倍する
/// 5. 重みを正規化して 2 に戻る
///
/// scikit-learn が 1.6 で削除した `SAMME.R`（確率出力を使う実数版）ではなく、
/// **いま scikit-learn に残っているのと同じ離散版** である。
let fitAdaBoost (data: Dataset) (rounds: int) =
    let weights = Array.create data.Size (1.0 / float data.Size)
    let stages = ResizeArray<Stage>()

    for _ in 1..rounds do
        let tree = fitWeightedTree data weights 1

        let errorRate =
            Array.init data.Size (fun i ->
                if predictTree tree data.X.[i] <> data.Y.[i] then weights.[i] else 0.0)
            |> Array.sum

        // 完全に当ててしまうと α が無限大になる。scikit-learn も同じ扱いで打ち切る
        if errorRate <= 0.0 || errorRate >= 0.5 then
            stages.Add { Tree = tree; Alpha = 1.0 }
        else
            let alpha = log ((1.0 - errorRate) / errorRate)
            stages.Add { Tree = tree; Alpha = alpha }

            for i in 0 .. data.Size - 1 do
                if predictTree tree data.X.[i] <> data.Y.[i] then
                    weights.[i] <- weights.[i] * exp alpha

            let total = Array.sum weights

            for i in 0 .. data.Size - 1 do
                weights.[i] <- weights.[i] / total

    List.ofSeq stages

/// AdaBoost の予測。各段の予測を発言力で重み付けして多数決する
let predictAdaBoost (stages: Stage list) (row: float[]) =
    let score =
        stages
        |> List.sumBy (fun stage -> if predictTree stage.Tree row = 1 then stage.Alpha else -stage.Alpha)

    if score > 0.0 then 1 else 0

/// 原著の「手で作るランダムフォレスト」。
///
/// 原著は 6 通ずつ 3 組に分けて切り株を 3 本作り、図で並べて見せる。
/// **多数決を取るところまでは書いていない** ので、ここで補った。
///
/// Accord のランダムフォレストも使えるが、[Kotlin 版](Nb20EnsembleSpam.kt) の
/// Smile と同じくスレッド並列で部分標本を取るため結果が固定できない。
/// 原著が手で切った 3 組なら **決定的** で、章の主旨もそのまま伝わる。
let handMadeForest () =
    [ 0; 1; 2 ] |> List.map (fun index -> fitTree (batch index) 1)

/// 手で作ったランダムフォレストの予測。3 本の多数決
let predictHandMadeForest (trees: Tree list) (row: float[]) =
    let votes = trees |> List.sumBy (fun tree -> predictTree tree row)
    if votes * 2 > trees.Length then 1 else 0
