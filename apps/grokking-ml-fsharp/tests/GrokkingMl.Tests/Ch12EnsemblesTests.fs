module GrokkingMl.Tests.Ch12EnsemblesTests

open System
open Xunit
open GrokkingMl.Ch09DecisionTrees
open GrokkingMl.Ch12Ensembles

// 3 つの領域に分かれるデータ。深さ 1 の木（切り株）1 本では解けない
let points: Point list =
    [ [ 1.0; 1.0 ]
      [ 2.0; 1.0 ]
      [ 3.0; 1.0 ]
      [ 4.0; 1.0 ]
      [ 5.0; 1.0 ]
      [ 6.0; 1.0 ]
      [ 7.0; 1.0 ]
      [ 8.0; 1.0 ] ]

let labels = [ 1; 1; -1; -1; -1; -1; 1; 1 ]

[<Fact>]
let ``bootstrapSample は大きさを保つ`` () =
    let samplePoints, sampleLabels = bootstrapSample (Random(0)) points labels
    Assert.Equal(List.length points, List.length samplePoints)
    Assert.Equal(List.length labels, List.length sampleLabels)

[<Fact>]
let ``bootstrapSample は特徴量とラベルの対応を保つ`` () =
    let samplePoints, sampleLabels = bootstrapSample (Random(0)) points labels

    List.zip samplePoints sampleLabels
    |> List.iter (fun (point, label) ->
        Assert.Equal(labels[List.findIndex ((=) point) points], label))

[<Fact>]
let ``bootstrapSample は復元抽出`` () =
    let samplePoints, _ = bootstrapSample (Random(0)) points labels
    // 復元抽出なので、同じ点が複数回選ばれ、選ばれない点も出る
    Assert.True((samplePoints |> List.distinct |> List.length) < List.length points)

[<Fact>]
let ``切り株 1 本では 3 領域を分けられない`` () =
    let stump = buildTreeWith giniImpurity 1 1 points labels
    Assert.True(treeAccuracy stump points labels < 1.0)

[<Fact>]
let ``forest は多数決で予測する`` () =
    let forest = { Trees = [ Leaf 1; Leaf 1; Leaf -1 ] }
    Assert.Equal(1, forestPredict forest [ 0.0 ])
    Assert.Equal<int list>([ 1; 1; -1 ], votes forest [ 0.0 ])

[<Fact>]
let ``切り株のバギングはこの問題では改善しない`` () =
    let stump = buildTreeWith giniImpurity 1 1 points labels
    let forest = trainForestWith 10 1 giniImpurity 0 points labels
    Assert.True(forestAccuracy forest points labels <= treeAccuracy stump points labels + 1e-9)

[<Fact>]
let ``誤り率 0.5 の学習器には発言権がない`` () = Assert.Equal(0.0, learnerWeight 0.5, 6)

[<Fact>]
let ``誤り率が小さいほど発言権は大きい`` () =
    Assert.True(learnerWeight 0.4 < learnerWeight 0.2)
    Assert.True(learnerWeight 0.2 < learnerWeight 0.05)

[<Fact>]
let ``当てずっぽうより悪い学習器の発言権は負`` () = Assert.True(learnerWeight 0.7 < 0.0)

[<Fact>]
let ``誤り率 0 でも発言権は有限`` () =
    Assert.True(learnerWeight 0.0 > 0.0)
    Assert.True(learnerWeight 0.0 < 100.0)

[<Fact>]
let ``weightedError は個数ではなく重みを数える`` () =
    let tree = Leaf 1
    let sample: Point list = [ [ 0.0 ]; [ 1.0 ]; [ 2.0 ] ]
    let sampleLabels = [ 1; 1; -1 ]
    Assert.Equal(0.8, weightedError tree sample sampleLabels [ 0.1; 0.1; 0.8 ], 6)
    Assert.Equal(0.1, weightedError tree sample sampleLabels [ 0.45; 0.45; 0.1 ], 6)

[<Fact>]
let ``完璧な木の重み付き誤り率は 0`` () =
    let tree = buildTreeWith giniImpurity 5 1 points labels
    let weights = List.replicate (List.length points) (1.0 / float (List.length points))
    Assert.Equal(0.0, weightedError tree points labels weights, 6)

[<Fact>]
let ``adaBoost は重み付き投票で判定する`` () =
    let model =
        { Learners = [ { Tree = Leaf 1; Weight = 2.0 }; { Tree = Leaf -1; Weight = 1.0 } ] }

    // 2*1 + 1*(-1) = 1 なので正のクラス
    Assert.Equal(1.0, boostScore model [ 0.0 ], 6)
    Assert.Equal(1, boostPredict model [ 0.0 ])

[<Fact>]
let ``AdaBoost は切り株 1 本では解けない問題を解く`` () =
    let stump = buildTreeWith giniImpurity 1 1 points labels
    let boosted = trainAdaBoostWith 10 1 giniImpurity points labels
    Assert.True(treeAccuracy stump points labels < 1.0)
    Assert.Equal(1.0, boostAccuracy boosted points labels, 6)

[<Fact>]
let ``この問題では AdaBoost がバギングに勝つ`` () =
    let forest = trainForestWith 10 1 giniImpurity 0 points labels
    let boosted = trainAdaBoostWith 10 1 giniImpurity points labels
    Assert.True(boostAccuracy boosted points labels > forestAccuracy forest points labels)

[<Fact>]
let ``AdaBoost の学習器は正の発言権を持つ`` () =
    let boosted = trainAdaBoostWith 10 1 giniImpurity points labels
    Assert.NotEmpty(boosted.Learners)
    Assert.All(boosted.Learners, fun learner -> Assert.True(learner.Weight > 0.0))

[<Fact>]
let ``ラウンドを増やしても訓練精度は下がらない`` () =
    let few = trainAdaBoostWith 2 1 giniImpurity points labels
    let many = trainAdaBoostWith 10 1 giniImpurity points labels
    Assert.True(boostAccuracy many points labels >= boostAccuracy few points labels)
