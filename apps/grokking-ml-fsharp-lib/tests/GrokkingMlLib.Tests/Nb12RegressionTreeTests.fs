/// 原著ノートブック #12 の再現テスト。
///
/// 手計算の 9 通りの重み付き MSE、木のしきい値、葉が返す平均値まで
/// **すべて原著と一致** した。
module GrokkingMlLib.Tests.Nb12RegressionTreeTests

open Xunit
open GrokkingMlLib.Nb12RegressionTree

[<Fact>]
let ``データセットは8点`` () =
    Assert.Equal<float[]>([| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0; 70.0; 80.0 |], ages)
    Assert.Equal<float[]>([| 7.0; 5.0; 7.0; 1.0; 2.0; 1.0; 5.0; 4.0 |], days)

[<Fact>]
let ``全体の平均は4`` () =
    // 原著の出力: np.array([7,5,7,1,2,1,5,4]).mean() -> 4.0
    Assert.Equal(4.0, Array.average days, 12)

[<Fact>]
let ``分割は9通り試される`` () =
    // 要素は 8 個だが range(0, 9) なので 9 通り
    let splits = splitMses days

    Assert.Equal(9, splits.Length)
    Assert.Empty(splits.[0].Left)
    Assert.Empty(splits.[8].Right)

[<Fact>]
let ``空の側の平均はNaNになる`` () =
    // 原著も NumPy の RuntimeWarning つきで nan を出している
    Assert.True(System.Double.IsNaN((splitMses days).[0].LeftMean))
    Assert.True(System.Double.IsNaN((splitMses days).[8].RightMean))

[<Fact>]
let ``各分割の重み付きMSEは原著と同じ`` () =
    // 原著のセル出力をそのまま期待値にしている
    let expected =
        [ 5.25; 3.9642857142857144; 3.916666666666667; 1.9833333333333334
          4.25; 4.983333333333333; 5.166666666666667; 5.25; 5.25 ]

    let splits = splitMses days

    expected
    |> List.iteri (fun index value -> Assert.Equal(value, splits.[index].WeightedMse, 14))

[<Fact>]
let ``最良の分割は3番目`` () =
    // 原著の一覧で 1.9833 がもっとも小さい
    let best = bestSplit days

    Assert.Equal(3, best.Index)
    Assert.Equal<float list>([ 7.0; 5.0; 7.0 ], best.Left)
    Assert.Equal(6.333333333333333, best.LeftMean, 12)
    Assert.Equal(2.6, best.RightMean, 12)

[<Fact>]
let ``深さ2の木は scikit-learn と同じ3回の分割になる`` () =
    // 根で 35.0、左で 15.0、右で 65.0
    Assert.Equal<float list>([ 35.0; 15.0; 65.0 ], splitThresholds (fit 2))

[<Fact>]
let ``根の分割は手計算の最小値と一致する`` () =
    // 手計算の最小は 3 番目、つまり 30 歳と 40 歳の間。中点の 35.0
    Assert.Equal(35.0, List.head (splitThresholds (fit 2)), 12)

[<Fact>]
let ``葉は平均値を返す`` () =
    // 分類木は多数決だったが、回帰木は葉に落ちた点の平均を返す
    Assert.Equal<float list>([ 7.0; 6.0; 4.0 / 3.0; 4.5 ], leafValues (fit 2))

[<Fact>]
let ``予測は scikit-learn と完全に一致する`` () =
    // 10 歳 -> 7、20〜30 歳 -> 6、40〜60 歳 -> 1.333、70〜80 歳 -> 4.5
    let expected = [ 7.0; 6.0; 6.0; 4.0 / 3.0; 4.0 / 3.0; 4.0 / 3.0; 4.5; 4.5 ]
    let tree = fit 2

    expected
    |> List.iteri (fun index value -> Assert.Equal(value, predict tree ages.[index], 12))

[<Fact>]
let ``予測は階段状になる`` () =
    // 同じ葉に落ちる点は同じ値を返す。回帰木の予測は連続にならない
    let tree = fit 2

    Assert.Equal(predict tree 20.0, predict tree 30.0, 12)
    Assert.Equal(predict tree 40.0, predict tree 60.0, 12)

[<Fact>]
let ``判別共用体は回帰木でも木の形をそのまま表す`` () =
    // 分類木の Leaf は string、回帰木の Leaf は float。型だけ差し替えれば済む
    match fit 2 with
    | Node(35.0, Node(15.0, Leaf 7.0, Leaf 6.0), Node(65.0, left, Leaf 4.5)) ->
        Assert.Equal(Leaf(4.0 / 3.0), left)
    | other -> failwith $"想定と違う木になった: {other}"

[<Fact>]
let ``深さ0の木は全体の平均を返す`` () =
    // 分割が 1 つも入らないので、どの年齢でも 4.0
    Assert.Equal(4.0, predict (fit 0) 10.0, 12)
    Assert.Equal(4.0, predict (fit 0) 80.0, 12)
