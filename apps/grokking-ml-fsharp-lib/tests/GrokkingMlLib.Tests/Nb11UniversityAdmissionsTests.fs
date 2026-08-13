/// 原著ノートブック #11 の再現テスト。
///
/// 自前の CART が **原著の出力をすべて再現** した。節の数（15）、正解率（0.885）、
/// 2 人の出願者の合否、2 特徴量の木の正解率（0.8525 / 0.8625 / 0.93）まで一致する。
module GrokkingMlLib.Tests.Nb11UniversityAdmissionsTests

open Xunit
open GrokkingMlLib.Nb11UniversityAdmissions

let private data = loadData ()

[<Fact>]
let ``データセットは400件7特徴量になる`` () =
    Assert.Equal(400, data.Size)
    Assert.Equal(7, featureNames.Length)
    Assert.Equal("GRE Score", List.head featureNames)

[<Fact>]
let ``合格ラベルは合格確率0_75で切る`` () =
    // 原著は Chance of Admit >= 0.75 を合格とする
    Assert.Equal(0.75, AdmissionThreshold, 12)
    // 400 件中 180 件が合格
    Assert.Equal(180, data.Admitted |> Array.filter ((=) "true") |> Array.length)

[<Fact>]
let ``制限なしの木は訓練データを完全に覚える`` () =
    // 原著の出力: dt.score(features, labels) -> 1.0
    Assert.Equal(1.0, accuracy (fitFull data) data featureNames, 12)

[<Fact>]
let ``小さい木は原著と同じ15節になる`` () =
    // 原著の木も 15 節（内部 7 + 葉 8）
    Assert.Equal(15, nodeCount (fitSmaller data))

[<Fact>]
let ``小さい木の正解率は原著と同じ`` () =
    // 原著の出力: dt_smaller.score(features, labels) -> 0.885
    Assert.Equal(0.885, accuracy (fitSmaller data) data featureNames, 12)

[<Fact>]
let ``小さい木の根はCGPAで分割する`` () =
    // 7 つの特徴量のうち、成績（CGPA）がもっとも効く
    Assert.Equal(("CGPA", 8.735), List.head (splitConditions (fitSmaller data)))

[<Fact>]
let ``CGPAが高い出願者は合格と予測される`` () =
    // 原著の出力: dt_smaller.predict([[320, 110, 3, 4.0, 3.5, 8.9, 0]]) -> True
    Assert.True(predictApplicant (fitSmaller data) [ 320.0; 110.0; 3.0; 4.0; 3.5; 8.9; 0.0 ])

[<Fact>]
let ``CGPAだけ下げると不合格に変わる`` () =
    // 原著の出力: 8.9 を 8.0 にすると False。
    // 原著は「白い（中立な）葉は False になる」とコメントしている。
    // 同数の葉でクラス名の小さいほうを選ぶ規則にしないと、ここが True になる
    Assert.False(predictApplicant (fitSmaller data) [ 320.0; 110.0; 3.0; 4.0; 3.5; 8.0; 0.0 ])

[<Fact>]
let ``2特徴量の木の正解率は scikit-learn と完全に一致する`` () =
    // 原著は図でしか見せていないが、深さごとの正解率まで一致した
    Assert.Equal(0.8525, accuracy (fitExams data 1) data examFeatures, 12)
    Assert.Equal(0.8625, accuracy (fitExams data 2) data examFeatures, 12)
    Assert.Equal(0.93, accuracy (fitExams data 20) data examFeatures, 12)

[<Fact>]
let ``2特徴量の木の節の数も scikit-learn と一致する`` () =
    // 深さ 1 は 3 節、深さ 2 は 7 節、制限なしは 197 節
    Assert.Equal(3, nodeCount (fitExams data 1))
    Assert.Equal(7, nodeCount (fitExams data 2))
    Assert.Equal(197, nodeCount (fitExams data 20))

[<Fact>]
let ``2特徴量ではどの深さでもGREで分割し始める`` () =
    // GRE Score <= 319.5。TOEFL より GRE のほうが効く
    for depth in [ 1; 2; 20 ] do
        Assert.Equal(("GRE Score", 319.5), List.head (splitConditions (fitExams data depth)))

[<Fact>]
let ``2特徴量では制限なしでも完全には分けられない`` () =
    // 同じ GRE・TOEFL で合否が違う出願者がいるため
    Assert.True(accuracy (fitExams data 20) data examFeatures < 1.0)

[<Fact>]
let ``節が28倍になっても正解率は7ポイントしか上がらない`` () =
    // 深さ 2 の 7 節（0.8625）から制限なしの 197 節（0.93）へ。過学習の割に合わない
    let shallow = fitExams data 2
    let unbounded = fitExams data 20

    Assert.Equal(7, nodeCount shallow)
    Assert.Equal(197, nodeCount unbounded)
    Assert.True(accuracy unbounded data examFeatures - accuracy shallow data examFeatures < 0.08)

[<Fact>]
let ``3つの制限はそれぞれ独立に効く`` () =
    // 深さだけ、葉の件数だけ、分割の件数だけを緩めると、いずれも木が大きくなる
    let baseline = fitSmaller data
    let deeper = fit data featureNames { MaxDepth = 5; MinSamplesLeaf = 10; MinSamplesSplit = 10 }
    let smallerLeaves = fit data featureNames { MaxDepth = 3; MinSamplesLeaf = 1; MinSamplesSplit = 10 }

    Assert.True(nodeCount deeper > nodeCount baseline)
    Assert.True(nodeCount smallerLeaves >= nodeCount baseline)
