/// 原著ノートブック #08 `Chapter_09_Decision_Trees/Gini_entropy_calculations.ipynb`。
///
/// 決定木がどこで分割するかを決めるための 2 つの不純度、ジニ不純度とエントロピーを
/// 手で計算する。原著は NumPy しか使わない小さな回で、
/// **決定木の中身を理解するための下準備** にあたる。
///
/// 分割位置を 1 つずつずらしながら重み付き不純度を見ると、
/// `['A'; 'A'; 'A'] | ['C'; 'B'; 'C']` でどちらの指標も最小になる。
module GrokkingMlLib.Nb08GiniEntropy

open System.Collections.Generic
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

/// 原著が使う 6 要素
let elements = [ "A"; "A"; "A"; "C"; "B"; "C" ]

/// ある分割位置での、左右と重み付き不純度
type SplitImpurity =
    { Index: int
      Left: string list
      Right: string list
      WeightedGini: float
      WeightedEntropy: float }

/// 要素ごとの個数を、**初めて現れた順** に返す。
///
/// 原著は辞書に数えてから取り出しており、Python 3.7 以降の辞書は挿入順を保つ。
/// F# の `List.countBy` も **初出順** を保つので、そのまま対応する。
/// `Map` に入れると辞書順に並び替わってしまい、原著と結果がずれる。
let counts (items: string list) =
    items |> List.countBy id |> List.map snd

/// ジニ不純度。1 から「同じクラスを 2 回続けて引く確率」を引いたもの。
///
/// 空のリストに対しては 1 を返す。原著は特別扱いしておらず、
/// 空の合計が 0 になって `1 - 0` がそのまま 1 になる。
/// 重み付けのときは要素数 0 が掛かるので結果に影響しない。
let gini (items: string list) =
    let classCounts = counts items
    let n = List.sum classCounts

    1.0
    - (classCounts
       |> List.sumBy (fun count -> float count * float count / (float n * float n)))

/// 情報エントロピー。原著はこちらだけ空のリストを明示的に 0 にしている。
///
/// `log2 0` が発散するので、空を通すと NaN になる。
/// ジニ不純度が特別扱い不要だったのと対照的である。
///
/// NumPy の `np.log2` と `np.dot` にあたる部分を Math.NET のベクトルで書いた。
let entropy (items: string list) =
    if List.isEmpty items then
        0.0
    else
        let classCounts = counts items
        let n = List.sum classCounts
        let proportions: Vector<float> =
            classCounts |> List.map (fun count -> 1.0 / float n * float count) |> DenseVector.ofList
        let logs = proportions |> Vector.map (fun p -> log p / log 2.0)
        -(logs * proportions)

/// 左右の不純度を、要素数で重み付けして平均する
let private weighted impurity (left: string list) (right: string list) (total: int) =
    1.0 / float total * (impurity left * float left.Length + impurity right * float right.Length)

/// 先頭から順に分割位置をずらし、それぞれの重み付き不純度を求める。
///
/// 原著は 0 から `length - 1` まで回している。**右端では分割しない** ので、
/// 「左が全部・右が空」の場合は出てこない。
let splitImpurities (items: string list) =
    let total = items.Length

    [ for index in 0 .. total - 1 ->
        let left = List.truncate index items
        let right = List.skip index items

        { Index = index
          Left = left
          Right = right
          WeightedGini = weighted gini left right total
          WeightedEntropy = weighted entropy left right total } ]

/// 重み付きジニ不純度がもっとも小さい分割を返す
let bestSplit (items: string list) =
    splitImpurities items |> List.minBy (fun split -> split.WeightedGini)
