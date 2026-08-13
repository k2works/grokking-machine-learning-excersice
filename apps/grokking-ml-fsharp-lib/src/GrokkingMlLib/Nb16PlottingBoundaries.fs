/// 原著ノートブック #16 `Chapter_10_Neural_Networks/Plotting_Boundaries.ipynb`。
///
/// **学習をしない回** である。重みを手で決めたネットワークを 2 つ作り、
/// その境界を描いて「1 層目は直線、2 層目は曲がる」ことを見せる。
///
/// 題材は [#13](Nb13NeuralNetworkBoundary.fs) と同じ「エイリアンが幸せかどうか」の 8 点。
/// ライブラリは要らない。**関数を値として渡す** 書き方が原著とそのまま重なる回である。
module GrokkingMlLib.Nb16PlottingBoundaries

/// 原著が図を描く範囲。境界の比較もこの範囲で行う
[<Literal>]
let GridMin = -0.5

[<Literal>]
let GridMax = 3.0

[<Literal>]
let GridStep = 0.005

/// 出力を 1 と見なすしきい値。原著の `f(x, y) >= 0.5`
[<Literal>]
let DecisionThreshold = 0.5

/// エイリアン 1 匹ぶんの観測
type Alien = { Aack: int; Beep: int; Happy: int }

/// 原著の 8 件
let alienDataset =
    [ { Aack = 1; Beep = 0; Happy = 0 }
      { Aack = 2; Beep = 0; Happy = 0 }
      { Aack = 0; Beep = 1; Happy = 0 }
      { Aack = 0; Beep = 2; Happy = 0 }
      { Aack = 1; Beep = 1; Happy = 1 }
      { Aack = 1; Beep = 2; Happy = 1 }
      { Aack = 2; Beep = 1; Happy = 1 }
      { Aack = 2; Beep = 2; Happy = 1 } ]

/// 階段関数。0 以上なら 1、そうでなければ 0
let step (x: float) = if x >= 0.0 then 1.0 else 0.0

/// 原著の書き方 `exp(x) / (1 + exp(x))` をそのまま使う。
///
/// 数学的には `1 / (1 + exp(-x))` と同じだが、**桁あふれの向きが逆** になる。
/// こちらは x が大きいときに `exp x` が無限大になり、`infinity / infinity` で
/// nan を返す。原著が図を描く範囲では起きない。
let sigmoid (x: float) = exp x / (1.0 + exp x)

/// 1 層目の 1 つ目のニューロン。重み (6, 10)、バイアス -15
let line1 (a: float) (b: float) = step (6.0 * a + 10.0 * b - 15.0)

/// 1 層目の 2 つ目のニューロン。重み (10, 6)、バイアス -15
let line2 (a: float) (b: float) = step (10.0 * a + 6.0 * b - 15.0)

/// 常に 1 を返すニューロン。入力を一切見ない
let bias (_: float) (_: float) = 1.0

/// 階段関数だけで組んだネットワーク。
///
/// 2 層目は `line1 + line2 - 1.5 >= 0`。1 層目の出力は 0 か 1 なので、
/// 和が 1.5 以上になるのは **両方とも 1 のときだけ**。つまり AND である。
let nnWithStep (a: float) (b: float) =
    step (step (6.0 * a + 10.0 * b - 15.0) + step (10.0 * a + 6.0 * b - 15.0) - 1.5)

/// 同じ重みでシグモイドに置き換えたネットワーク。出力は連続値になる
let nnWithSigmoid (a: float) (b: float) =
    sigmoid (
        1.0 * sigmoid (6.0 * a + 10.0 * b - 15.0)
        + 1.0 * sigmoid (10.0 * a + 6.0 * b - 15.0)
        - 1.5
    )

/// 原著の `h(x, y) = f(x, y) >= 0.5`。境界はこの判定で決まる
let classify (f: float -> float -> float) (a: float) (b: float) =
    if f a b >= DecisionThreshold then 1 else 0

/// 8 点それぞれの予測
let predictions (f: float -> float -> float) =
    alienDataset |> List.map (fun alien -> classify f (float alien.Aack) (float alien.Beep))

/// 8 点に対する正解率
let accuracy (f: float -> float -> float) =
    let correct =
        List.map2 (fun predicted alien -> predicted = alien.Happy) (predictions f) alienDataset
        |> List.filter id
        |> List.length

    float correct / float alienDataset.Length

/// 原著の `np.arange(-0.5, 3, 0.005)` にあたる格子の軸
let axis =
    let count = int (ceil ((GridMax - GridMin) / GridStep))
    Array.init count (fun i -> GridMin + float i * GridStep)

/// 格子のうち、1 と判定される点の割合。
///
/// 図を見なくても「境界がどこにあるか」を数値で比べられる。
let regionRatio (f: float -> float -> float) =
    let mutable ones = 0L

    for x in axis do
        for y in axis do
            ones <- ones + int64 (classify f x y)

    float ones / (float axis.Length * float axis.Length)

/// 2 つの関数の判定が食い違う格子点の割合
let disagreementRatio (f: float -> float -> float) (g: float -> float -> float) =
    let mutable different = 0L

    for x in axis do
        for y in axis do
            if classify f x y <> classify g x y then
                different <- different + 1L

    float different / (float axis.Length * float axis.Length)
