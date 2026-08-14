/// 原著ノートブック #21 `Chapter_12_Ensemble_Methods/Gradient_boosting_and_XGBoost.ipynb`。
///
/// 8 人の年齢から「週に何日アプリを使うか」を当てる回帰の回である。
///
/// 前半は **勾配ブースティングを手で追う**。弱学習器を 1 本ずつ取り出し、
/// 残差がどう縮んでいくかを段ごとに見る。
///
/// 後半は **XGBoost の類似度スコアを手で計算する**。
/// ここはライブラリに依存しないので、原著の式をそのまま写せる。
///
/// [#20](Nb20EnsembleSpam.fs) と同じく回帰木を自前で持つ。
/// 分類の多数決を **平均** に、ジニ不純度を **分散** に置き換えるだけで済む。
module GrokkingMlLib.Nb21GradientBoosting

/// 原著の 8 人。年齢
let ages = [| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0; 70.0; 80.0 |]

/// 週あたりの利用日数
let days = [| 7.0; 5.0; 7.0; 1.0; 2.0; 1.0; 5.0; 4.0 |]

/// 原著の勾配ブースティングの設定
[<Literal>]
let MaxDepth = 2

[<Literal>]
let NEstimators = 4

[<Literal>]
let LearningRate = 0.8

/// XGBoost 側の設定。最初の予測はラベルの平均
[<Literal>]
let InitialPrediction = 4.0

[<Literal>]
let XgbLambda = 3.0

/// 回帰木
type Tree =
    | Leaf of value: float
    | Node of threshold: float * left: Tree * right: Tree

/// 分割の候補。scikit-learn と同じく **隣り合う値の中点** を使う
let private candidateThresholds (values: float[]) =
    values |> Array.distinct |> Array.sort |> Array.pairwise |> Array.map (fun (a, b) -> (a + b) / 2.0)

/// 重み付き二乗誤差。回帰木では分散の減少で分割を決める
let private sumSquaredError (targets: float[]) =
    if targets.Length = 0 then
        0.0
    else
        let mean = Array.average targets
        targets |> Array.sumBy (fun v -> (v - mean) ** 2.0)

let rec private build (x: float[]) (y: float[]) (indices: int[]) depth maxDepth =
    let targets = indices |> Array.map (fun i -> y.[i])

    if depth >= maxDepth || indices.Length <= 1 || (targets |> Array.distinct |> Array.length) <= 1 then
        Leaf(Array.average targets)
    else
        let best =
            indices
            |> Array.map (fun i -> x.[i])
            |> candidateThresholds
            |> Array.choose (fun threshold ->
                let left = indices |> Array.filter (fun i -> x.[i] <= threshold)
                let right = indices |> Array.filter (fun i -> x.[i] > threshold)

                if left.Length = 0 || right.Length = 0 then
                    None
                else
                    let error =
                        sumSquaredError (left |> Array.map (fun i -> y.[i]))
                        + sumSquaredError (right |> Array.map (fun i -> y.[i]))

                    Some(error, threshold, left, right))
            |> Array.sortBy (fun (error, threshold, _, _) -> error, threshold)
            |> Array.tryHead

        match best with
        | None -> Leaf(Array.average targets)
        | Some(_, threshold, left, right) ->
            Node(threshold, build x y left (depth + 1) maxDepth, build x y right (depth + 1) maxDepth)

/// 回帰木を学習する
let fitTree (x: float[]) (y: float[]) (maxDepth: int) =
    build x y [| 0 .. x.Length - 1 |] 0 maxDepth

/// 1 点を予測する
let rec predictTree (tree: Tree) (value: float) =
    match tree with
    | Leaf prediction -> prediction
    | Node(threshold, left, right) ->
        if value <= threshold then
            predictTree left value
        else
            predictTree right value

/// ラベルから平均を引いたもの。勾配ブースティングが最初に予測する対象
let centeredLabels () =
    let mean = Array.average days
    days |> Array.map (fun v -> v - mean)

/// 1 段ぶんの記録。原著が段ごとに印刷するもの
type Stage =
    { Residuals: float[]
      Predictions: float[]
      NewResiduals: float[] }

/// 勾配ブースティングを原著の手順どおりに回す。
///
/// 1. ラベルから平均を引く（中心化）
/// 2. 残差に回帰木を当てる
/// 3. 予測に学習率を掛けて足し込む
/// 4. 新しい残差を計算して 2 に戻る
///
/// scikit-learn の `GradientBoostingRegressor(max_depth=2, learning_rate=0.8)`
/// と同じ手順である。
let fitGradientBoosting (rounds: int) =
    let centered = centeredLabels ()
    let predictions = Array.zeroCreate days.Length
    let stages = ResizeArray<Stage * Tree>()
    let mutable residuals = centered

    for _ in 1..rounds do
        let tree = fitTree ages residuals MaxDepth
        let stagePredictions = ages |> Array.map (predictTree tree)

        for i in 0 .. days.Length - 1 do
            predictions.[i] <- predictions.[i] + stagePredictions.[i] * LearningRate

        let newResiduals = Array.map2 (-) centered predictions

        stages.Add(
            { Residuals = residuals
              Predictions = stagePredictions
              NewResiduals = newResiduals },
            tree
        )

        residuals <- newResiduals

    List.ofSeq stages

/// 勾配ブースティングの予測。平均を足して元の尺度に戻す
let predictGradientBoosting (stages: (Stage * Tree) list) (value: float) =
    let mean = Array.average days
    mean + (stages |> List.sumBy (fun (_, tree) -> predictTree tree value * LearningRate))

/// XGBoost の類似度スコア。原著の実装をそのまま写した。
///
/// `sum(l)^2 / (len(l) + lambda)`。**分子は和の 2 乗であって
/// 2 乗の和ではない。** だから符号がばらけるほど値が小さくなり、
/// 「似ている点が集まっているか」の尺度になる。
let similarityScore (values: float[]) (lambda: float) =
    if values.Length = 0 then
        0.0
    else
        let total = Array.sum values
        total * total / (float values.Length + lambda)

/// 切れ目ごとの類似度スコアの和。原著が 1 行ずつ印刷しているもの
let splitScores (residuals: float[]) (lambda: float) =
    [ for index in 1 .. residuals.Length - 1 ->
        similarityScore residuals.[.. index - 1] lambda
        + similarityScore residuals.[index..] lambda ]

/// 分割の結果
type Split =
    { Left: float[]
      Right: float[]
      Score: float }

/// 原著の `find_best_split`。
///
/// **並び順のまま前から切る** だけで、しきい値を探しているわけではない。
/// データが年齢順に並んでいるから成り立つ簡略版である。
///
/// なお **原著の実装にはバグがある。** 返り値の 3 つ目に `best_score` ではなく
/// `new_score`（ループの最後の切れ目のスコア）を渡している。
/// ここでは最良のスコアを返す。詳しくは記事を参照。
let findBestSplit (residuals: float[]) (lambda: float) =
    let scores = splitScores residuals lambda

    if List.isEmpty scores then
        { Left = [||]; Right = residuals; Score = 0.0 }
    else
        let bestIndex = scores |> List.mapi (fun i score -> i, score) |> List.maxBy snd |> fst
        let cut = bestIndex + 1

        { Left = residuals.[.. cut - 1]
          Right = residuals.[cut..]
          Score = scores.[bestIndex] }

/// 最初の予測（平均 4.0）からの残差
let xgboostResiduals () = days |> Array.map (fun v -> v - InitialPrediction)
