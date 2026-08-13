module GrokkingMl.Tests.Ch09DecisionTreesTests

open Xunit
open GrokkingMl.Ch09DecisionTrees

// 原著と同じアプリ推薦データ
// 特徴量 = (性別: 0=女性 1=男性, 年齢)、ラベル = 1 が推薦
let points: Point list =
    [ [ 1.0; 15.0 ]
      [ 0.0; 25.0 ]
      [ 0.0; 32.0 ]
      [ 1.0; 35.0 ]
      [ 0.0; 12.0 ]
      [ 1.0; 14.0 ]
      [ 1.0; 55.0 ]
      [ 0.0; 40.0 ] ]

let labels = [ 1; 0; 0; 0; 1; 1; 0; 0 ]

let log2 (x: float) = log x / log 2.0

[<Fact>]
let ``純粋な集合のジニ不純度は 0`` () =
    Assert.Equal(0.0, giniImpurity [ 1; 1; 1 ], 6)
    Assert.Equal(0.0, giniImpurity [ 0; 0 ], 6)

[<Fact>]
let ``均等な集合のジニ不純度は 0.5`` () =
    Assert.Equal(0.5, giniImpurity [ 0; 1 ], 6)
    Assert.Equal(0.5, giniImpurity [ 0; 0; 1; 1 ], 6)

[<Fact>]
let ``空集合のジニ不純度は 0`` () = Assert.Equal(0.0, giniImpurity [], 6)

[<Fact>]
let ``純粋な集合のエントロピーは 0`` () = Assert.Equal(0.0, entropy [ 1; 1; 1 ], 6)

[<Fact>]
let ``均等な 2 クラスのエントロピーは 1 ビット`` () = Assert.Equal(1.0, entropy [ 0; 1 ], 6)

[<Fact>]
let ``均等な 4 クラスのエントロピーは 2 ビット`` () = Assert.Equal(2.0, entropy [ 0; 1; 2; 3 ], 6)

[<Fact>]
let ``エントロピーは定義どおり`` () =
    let expected = -(0.75 * log2 0.75 + 0.25 * log2 0.25)
    Assert.Equal(expected, entropy [ 1; 1; 1; 0 ], 6)

[<Fact>]
let ``ジニとエントロピーは同じ順序を与える`` () =
    let pure' = [ 1; 1; 1; 1 ]
    let skewed = [ 1; 1; 1; 0 ]
    let balanced = [ 1; 1; 0; 0 ]
    Assert.True(giniImpurity pure' < giniImpurity skewed)
    Assert.True(giniImpurity skewed < giniImpurity balanced)
    Assert.True(entropy pure' < entropy skewed)
    Assert.True(entropy skewed < entropy balanced)

[<Fact>]
let ``split は小さい値を左へ送る`` () =
    let split = { Feature = 1; Threshold = 20.0 }
    Assert.True(matches split [ 0.0; 15.0 ])
    Assert.False(matches split [ 0.0; 25.0 ])

[<Fact>]
let ``applySplit はデータを失わずに振り分ける`` () =
    let partition = applySplit points labels { Feature = 1; Threshold = 20.0 }

    Assert.Equal(
        List.length points,
        List.length partition.LeftPoints + List.length partition.RightPoints
    )

    Assert.All(partition.LeftPoints, fun point -> Assert.True(point[1] < 20.0))
    Assert.All(partition.RightPoints, fun point -> Assert.True(point[1] >= 20.0))

[<Fact>]
let ``weightedImpurity は純粋な子を好む`` () =
    Assert.Equal(0.0, weightedImpurity giniImpurity [ 1; 1 ] [ 0; 0 ], 6)
    Assert.Equal(0.5, weightedImpurity giniImpurity [ 1; 0 ] [ 1; 0 ], 6)

[<Fact>]
let ``役に立たない分割の情報利得は 0`` () =
    Assert.Equal(0.0, informationGain giniImpurity [ 1; 1; 0; 0 ] [ 1; 0 ] [ 1; 0 ], 6)

[<Fact>]
let ``完全な分割の情報利得は最大`` () =
    Assert.Equal(0.5, informationGain giniImpurity [ 1; 1; 0; 0 ] [ 1; 1 ] [ 0; 0 ], 6)

[<Fact>]
let ``candidateSplits は中点を使う`` () =
    let sample: Point list = [ [ 0.0; 10.0 ]; [ 0.0; 20.0 ]; [ 1.0; 30.0 ] ]
    let splits = candidateSplits sample

    let thresholdsOf feature =
        splits
        |> List.filter (fun split -> split.Feature = feature)
        |> List.map (fun split -> split.Threshold)
        |> List.sort

    Assert.Equal<float list>([ 15.0; 25.0 ], thresholdsOf 1)
    Assert.Equal<float list>([ 0.5 ], thresholdsOf 0)

[<Fact>]
let ``bestSplit は年齢の境界を見つける`` () =
    match bestSplit giniImpurity points labels with
    | None -> failwith "分割が見つからなかった"
    | Some(split, gain) ->
        Assert.Equal(1, split.Feature)
        Assert.InRange(split.Threshold, 15.0, 25.0)
        Assert.True(gain > 0.0)

[<Fact>]
let ``改善しないとき bestSplit は None`` () =
    Assert.Equal(None, bestSplit giniImpurity [ [ 1.0 ]; [ 2.0 ]; [ 3.0 ] ] [ 1; 1; 1 ])

[<Fact>]
let ``majorityLabel は同数なら小さいラベルを選ぶ`` () =
    Assert.Equal(1, majorityLabel [ 1; 1; 0 ])
    Assert.Equal(0, majorityLabel [ 0; 1 ])

[<Fact>]
let ``buildTree は訓練データに適合する`` () =
    Assert.Equal(1.0, accuracy (buildTree points labels) points labels, 6)

[<Fact>]
let ``buildTree は maxDepth で止まる`` () =
    let shallow = buildTreeWith giniImpurity 1 1 points labels
    Assert.Equal(1, depth shallow)
    Assert.Equal(2, leafCount shallow)

[<Fact>]
let ``純粋なデータは 1 枚の葉になる`` () =
    let tree = buildTree [ [ 1.0 ]; [ 2.0 ] ] [ 1; 1 ]
    Assert.Equal(Leaf 1, tree)
    Assert.Equal(0, depth tree)

[<Fact>]
let ``深い木は浅い木より浅くならない`` () =
    let shallow = buildTreeWith giniImpurity 1 1 points labels
    let deep = buildTreeWith giniImpurity 5 1 points labels
    Assert.True(depth deep >= depth shallow)
    Assert.True(accuracy deep points labels >= accuracy shallow points labels)

[<Fact>]
let ``minSamples は木の成長を止める`` () =
    match buildTreeWith giniImpurity 5 (List.length labels) points labels with
    | Leaf _ -> ()
    | Node _ -> failwith "葉になるはずだった"

[<Fact>]
let ``ジニとエントロピーはこのデータで同じ木を作る`` () =
    Assert.Equal(
        buildTreeWith giniImpurity 5 1 points labels,
        buildTreeWith entropy 5 1 points labels
    )

[<Fact>]
let ``木の構造をパターンマッチで検査できる`` () =
    match buildTreeWith giniImpurity 1 1 points labels with
    | Node(_, Leaf _, Leaf _) -> ()
    | _ -> failwith "深さ 1 の木は葉 2 枚を持つはずだった"
