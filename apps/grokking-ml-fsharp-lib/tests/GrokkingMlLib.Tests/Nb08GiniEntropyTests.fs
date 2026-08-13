/// 原著ノートブック #08 の再現テスト。
///
/// NumPy しか使わない小さな回なので、原著の数値をすべて再現できる。
module GrokkingMlLib.Tests.Nb08GiniEntropyTests

open Xunit
open GrokkingMlLib.Nb08GiniEntropy

[<Fact>]
let ``個数は初出順に並ぶ`` () =
    // 原著の出力: [3, 2, 1]。A が 3、C が 2、B が 1 の順
    Assert.Equal<int list>([ 3; 2; 1 ], counts elements)

[<Fact>]
let ``ジニ不純度は原著と同じ`` () =
    // 原著の出力: 0.6111111111111112
    Assert.Equal(0.6111111111111112, gini elements, 15)

[<Fact>]
let ``エントロピーは原著と同じ`` () =
    // 原著の出力: 1.4591479170272448
    Assert.Equal(1.4591479170272448, entropy elements, 15)

[<Fact>]
let ``同じ要素だけなら不純度は0`` () =
    Assert.Equal(0.0, gini [ "A"; "A"; "A" ], 15)
    Assert.Equal(0.0, entropy [ "A"; "A"; "A" ], 15)

[<Fact>]
let ``2クラスが半々ならジニは0_5でエントロピーは1`` () =
    // 情報量 1 ビットぶん。コイン投げと同じ
    Assert.Equal(0.5, gini [ "A"; "B" ], 15)
    Assert.Equal(1.0, entropy [ "A"; "B" ], 15)

[<Fact>]
let ``空のリストの扱いは2つで違う`` () =
    // 原著はエントロピーだけ明示的に 0 を返す。ジニは 1 - 0 で 1 になる
    Assert.Equal(1.0, gini [], 15)
    Assert.Equal(0.0, entropy [], 15)

[<Fact>]
let ``各分割の重み付き不純度は原著と同じ`` () =
    // 原著のセル出力をそのまま期待値にしている
    let expected =
        [ 0, 0.6111111111111112, 1.4591479170272446
          1, 0.5333333333333333, 1.268273412406135
          2, 0.41666666666666663, 1.0
          3, 0.2222222222222222, 0.4591479170272448
          4, 0.41666666666666663, 0.8741854163060886
          5, 0.4666666666666667, 1.1424588287122237 ]

    let splits = splitImpurities elements

    for (index, expectedGini, expectedEntropy) in expected do
        Assert.Equal(expectedGini, splits.[index].WeightedGini, 15)
        Assert.Equal(expectedEntropy, splits.[index].WeightedEntropy, 15)

[<Fact>]
let ``分割は6通り試される`` () =
    // 0 から length - 1 まで。「左が全部・右が空」は試されない
    let splits = splitImpurities elements

    Assert.Equal(6, splits.Length)
    Assert.Empty(splits.[0].Left)
    Assert.Equal<string list>([ "C" ], splits.[5].Right)

[<Fact>]
let ``最良の分割はAのかたまりを切り離す`` () =
    // ['A'; 'A'; 'A'] | ['C'; 'B'; 'C'] で両方の指標が最小になる
    let best = bestSplit elements

    Assert.Equal(3, best.Index)
    Assert.Equal<string list>([ "A"; "A"; "A" ], best.Left)
    Assert.Equal<string list>([ "C"; "B"; "C" ], best.Right)
    Assert.Equal(0.2222222222222222, best.WeightedGini, 15)

[<Fact>]
let ``ジニとエントロピーは同じ分割を選ぶ`` () =
    let splits = splitImpurities elements

    Assert.Equal(
        (splits |> List.minBy (fun s -> s.WeightedGini)).Index,
        (splits |> List.minBy (fun s -> s.WeightedEntropy)).Index
    )

[<Fact>]
let ``分割しない場合の重み付き不純度は全体の不純度と一致する`` () =
    // index 0 は左が空なので、右がそのまま全体になる。
    // ただしエントロピーは 1.4591479170272446 で、entropy elements の
    // 1.4591479170272448 と最下位ビットだけ違う。重み付けの掛け算と割り算で
    // 丸めが 1 度多く入るため
    let split = (splitImpurities elements).[0]

    Assert.Equal(gini elements, split.WeightedGini, 15)
    Assert.NotEqual(entropy elements, split.WeightedEntropy)
    Assert.True(abs (entropy elements - split.WeightedEntropy) < 1e-15)
