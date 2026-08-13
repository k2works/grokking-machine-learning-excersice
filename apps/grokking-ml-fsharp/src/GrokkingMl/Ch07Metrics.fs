/// 第 7 章: 分類モデルの評価指標。
/// 正解率だけでは分類モデルの良し悪しを測れない。混同行列を土台に、
/// 適合率・再現率・F1 スコア・ROC 曲線下面積（AUC）を実装する。
module GrokkingMl.Ch07Metrics

/// 混同行列。すべての指標の土台になる 4 つの数。
type ConfusionMatrix =
    { TruePositives: int
      FalsePositives: int
      FalseNegatives: int
      TrueNegatives: int }

    member this.Total =
        this.TruePositives
        + this.FalsePositives
        + this.FalseNegatives
        + this.TrueNegatives

/// 正解ラベルと予測から混同行列を作る。
let confusionMatrix (labels: int list) (predictions: int list) =
    let counts =
        List.zip labels predictions
        |> List.countBy id
        |> Map.ofList

    let count key = Map.tryFind key counts |> Option.defaultValue 0

    { TruePositives = count (1, 1)
      FalsePositives = count (0, 1)
      FalseNegatives = count (1, 0)
      TrueNegatives = count (0, 0) }

/// 0 除算を避ける割り算。分母が 0 なら 0 を返す。
let private safeDivide (numerator: int) (denominator: int) =
    if denominator = 0 then
        0.0
    else
        float numerator / float denominator

/// 正解率。全体のうち正しく当てた割合。
let accuracy (matrix: ConfusionMatrix) =
    safeDivide (matrix.TruePositives + matrix.TrueNegatives) matrix.Total

/// 適合率。陽性と予測したもののうち、本当に陽性だった割合。
let precision (matrix: ConfusionMatrix) =
    safeDivide matrix.TruePositives (matrix.TruePositives + matrix.FalsePositives)

/// 再現率。本当に陽性のもののうち、拾えた割合。
let recall (matrix: ConfusionMatrix) =
    safeDivide matrix.TruePositives (matrix.TruePositives + matrix.FalseNegatives)

/// F ベータスコア。beta が大きいほど再現率を重視する。
let fBetaScore (beta: float) (matrix: ConfusionMatrix) =
    let p = precision matrix
    let r = recall matrix

    if p = 0.0 && r = 0.0 then
        0.0
    else
        let betaSquared = beta * beta
        (1.0 + betaSquared) * p * r / (betaSquared * p + r)

/// F1 スコア。適合率と再現率の調和平均。
let f1Score (matrix: ConfusionMatrix) = fBetaScore 1.0 matrix

/// 確率と閾値から 0 / 1 の予測を作る。
let predictionsAtThreshold (threshold: float) (probabilities: float list) =
    probabilities |> List.map (fun p -> if p >= threshold then 1 else 0)

/// ROC 曲線の点列。閾値を動かしたときの (偽陽性率, 真陽性率) を返す。
let rocPoints (labels: int list) (probabilities: float list) =
    let thresholds =
        probabilities @ [ 0.0; 1.0 + 1e-9 ]
        |> List.distinct
        |> List.sortDescending

    thresholds
    |> List.map (fun threshold ->
        let matrix = confusionMatrix labels (predictionsAtThreshold threshold probabilities)

        let falsePositiveRate =
            safeDivide matrix.FalsePositives (matrix.FalsePositives + matrix.TrueNegatives)

        (falsePositiveRate, recall matrix))
    |> List.sort

/// ROC 曲線下の面積。台形則で積分する。
let auc (labels: int list) (probabilities: float list) =
    rocPoints labels probabilities
    |> List.pairwise
    |> List.sumBy (fun ((x1, y1), (x2, y2)) -> (x2 - x1) * (y1 + y2) / 2.0)
