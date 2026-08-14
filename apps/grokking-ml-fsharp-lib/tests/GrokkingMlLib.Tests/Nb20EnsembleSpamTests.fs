/// 原著ノートブック #20 の再現テスト。
///
/// 弱学習器 3 本は **選んだ特徴量としきい値まで** scikit-learn と一致する。
/// 自前で書いた AdaBoost は、**いまの scikit-learn と同じ 0.7778** を出す。
module GrokkingMlLib.Tests.Nb20EnsembleSpamTests

open Xunit
open GrokkingMlLib.Nb20EnsembleSpam

[<Fact>]
let ``データセットは18通`` () =
    Assert.Equal(18, emails.Length)
    // スパムとそうでないものが 9 通ずつ
    Assert.Equal(9, emails |> Array.filter (fun row -> row.[2] = 1) |> Array.length)

[<Fact>]
let ``制限なしの決定木は丸暗記する`` () =
    // 原著の出力: 1.0
    // 18 点を完全に分けきる。良い結果に見えるが過学習そのもの
    Assert.Equal(1.0, accuracy (predictTree (fitTree allData 20)) allData, 15)

[<Fact>]
let ``深さ1に制限すると丸暗記できない`` () =
    // 分割 1 つでは 18 点を分けられない。Kotlin 版と同じ 0.7778
    Assert.Equal(0.7777777777777778, accuracy (predictTree (fitTree allData 1)) allData, 15)

[<Fact>]
let ``3組は6通ずつ重複なく分かれる`` () =
    Assert.Equal<int list>([ 6; 6; 6 ], batches |> List.map List.length)
    Assert.Equal<int list>([ 0..17 ], batches |> List.concat |> List.sort)

[<Fact>]
let ``弱学習器の正解率は原著と一致する`` () =
    // 原著の出力
    //   Weak learner 1 training accuracy: 1.0
    //   Weak learner 2 training accuracy: 1.0
    //   Weak learner 3 training accuracy: 0.8333333333333334
    let expected = [ 1.0; 1.0; 0.8333333333333334 ]

    for index in 0..2 do
        let tree = fitTree (batch index) 1
        Assert.Equal(expected.[index], accuracy (predictTree tree) (batch index), 15)

[<Fact>]
let ``弱学習器が選ぶ分割はscikit-learnと完全に一致する`` () =
    // 特徴量もしきい値も同じところを選んだ。CART の分割規則が同じなら、
    // 6 点しかないデータでは解が一意に決まる。3 言語すべてで一致する
    let splits = [ 0..2 ] |> List.map (fun index -> splitOf (fitTree (batch index) 1))

    Assert.Equal<(string * float) option list>(
        [ Some("Lottery", 4.5); Some("Sale", 8.0); Some("Sale", 5.5) ],
        splits
    )

[<Fact>]
let ``手で作るランダムフォレストは原著と完全に一致する`` () =
    // 原著の RandomForestClassifier は 0.8333333333333334。
    // 原著が手で切った 3 組の切り株を多数決すると、同じ値になる
    Assert.Equal(0.8333333333333334, accuracy (predictHandMadeForest (handMadeForest ())) allData, 15)

[<Fact>]
let ``AdaBoostはいまのscikit-learnと一致する`` () =
    // 原著の出力は 0.8888888888888888 だが、それは削除された SAMME.R の値。
    // 離散版 SAMME をアルゴリズムどおりに書くと 0.7777777777777778 になり、
    // **いまの scikit-learn の出力と一致する**
    Assert.Equal(0.7777777777777778, accuracy (predictAdaBoost (fitAdaBoost allData 6)) allData, 15)

[<Fact>]
let ``AdaBoostは6段になる`` () =
    Assert.Equal(6, (fitAdaBoost allData 6).Length)

[<Fact>]
let ``AdaBoostの発言力は段を追うごとに下がる`` () =
    // 前の段が外した点を重点的に見るので、後の段ほど難しい問題を解く。
    // 誤り率が上がるぶん発言力 α は小さくなる
    let alphas = fitAdaBoost allData 6 |> List.map (fun stage -> stage.Alpha)

    Assert.True(List.head alphas > List.last alphas)
    Assert.All(alphas, fun alpha -> Assert.True(alpha > 0.0))

[<Fact>]
let ``アンサンブルは1本の木より正解率が低い`` () =
    // 原著が「ブースティングは正確だが過学習からは遠い」と書いているところ。
    // 学習データの正解率だけを見れば、1 本の木（1.0）が最も高い
    let single = accuracy (predictTree (fitTree allData 20)) allData

    Assert.True(accuracy (predictHandMadeForest (handMadeForest ())) allData < single)
    Assert.True(accuracy (predictAdaBoost (fitAdaBoost allData 6)) allData < single)

[<Fact>]
let ``自前実装は決定的`` () =
    // Kotlin 版（Smile）のランダムフォレストは種を固定しても結果が変わる。
    // 自前で書けばその心配が無い
    let score () = accuracy (predictAdaBoost (fitAdaBoost allData 6)) allData

    Assert.Equal(score (), score (), 15)
