/// 原著ノートブック #17 `Chapter_11_Support_Vector_Machines/Building_the_datasets.ipynb`。
///
/// 第 11 章（SVM）で使う 3 つのデータセットを **乱数で作る** 回である。
///
/// 原著は種を固定していないので、**生成そのものは再現できない**。
/// できるのは 2 つ。
///
/// 1. 同じ規則で生成器を書き、規則どおりの点が出ることを確かめる
/// 2. 原著が生成して配布した CSV を読み、**その規則で作られたことを検証する**
///
/// 2 つ目が本命である。
module GrokkingMlLib.Nb17BuildingDatasets

open Deedle

/// 座標の範囲。原著の `6 * random.random() - 3`
[<Literal>]
let CoordScale = 6.0

[<Literal>]
let CoordOffset = -3.0

/// 2 次元の点とラベル
type Point = { X1: float; X2: float; Y: int }

/// 1 つのデータセットの作り方
type Spec =
    { Name: string
      /// 規則どおりに作る点の数
      Points: int
      /// ラベルを乱数にする点（ノイズ）の数
      Noise: int
      /// ラベルを決める規則
      Rule: float -> float -> int }

    member this.Total = this.Points + this.Noise

/// 直線 `x + y = 0.5` の上側なら 1
let linearRule (x: float) (y: float) = if x + y > 0.5 then 1 else 0

/// 原点を中心とする半径 √2.8 の円の内側なら 1
let oneCircleRule (x: float) (y: float) = if x * x + y * y < 2.8 then 1 else 0

/// (1, 0) と (-1, 0) を中心とする 2 つの円の **どちらか** の内側なら 1
let twoCirclesRule (x: float) (y: float) =
    let left = (x - 1.0) ** 2.0 + y * y < 2.0
    let right = (x + 1.0) ** 2.0 + y * y < 2.0
    if left || right then 1 else 0

let linear =
    { Name = "linear"
      Points = 50
      Noise = 10
      Rule = linearRule }

let oneCircle =
    { Name = "one_circle"
      Points = 100
      Noise = 10
      Rule = oneCircleRule }

let twoCircles =
    { Name = "two_circles"
      Points = 200
      Noise = 20
      Rule = twoCirclesRule }

let specs = [ linear; oneCircle; twoCircles ]

/// 原著と同じ手順でデータセットを作る。
///
/// 原著は種を固定していないので実行のたびに違うものが出る。
/// ここでは `seed` を渡せるようにして、テストで扱えるようにした。
/// **原著が配布している CSV とは一致しない**（一致しようがない）。
let generate (spec: Spec) (seed: int) =
    let random = System.Random(seed)
    let coordinate () = CoordScale * random.NextDouble() + CoordOffset

    let points =
        List.init spec.Points (fun _ ->
            let x = coordinate ()
            let y = coordinate ()
            { X1 = x; X2 = y; Y = spec.Rule x y })

    let noise =
        List.init spec.Noise (fun _ ->
            let x = coordinate ()
            let y = coordinate ()
            { X1 = x; X2 = y; Y = random.Next(0, 2) })

    points @ noise

/// 原著が生成して配布している CSV を読む
let load (spec: Spec) =
    let frame = Datasets.loadFrame $"{spec.Name}.csv"
    let x1 = frame.GetColumn<float>("x_1") |> Series.values |> Array.ofSeq
    let x2 = frame.GetColumn<float>("x_2") |> Series.values |> Array.ofSeq
    let y = frame.GetColumn<int>("y") |> Series.values |> Array.ofSeq
    List.init x1.Length (fun i -> { X1 = x1.[i]; X2 = x2.[i]; Y = y.[i] })

/// 規則とラベルが食い違う行の添字。
///
/// ノイズとして入れた点は規則と無関係にラベルを振っているので、
/// **約半分がここに現れる**。規則どおりに作った先頭の点は 1 つも現れない。
let ruleViolations (spec: Spec) (data: Point list) =
    data
    |> List.indexed
    |> List.filter (fun (_, point) -> spec.Rule point.X1 point.X2 <> point.Y)
    |> List.map fst
