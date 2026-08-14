/// 原著ノートブック #04 の再現テスト。
///
/// トリックの単発の挙動は原著の数値と完全に一致する。学習ループは原著が乱数の種を
/// 与えていないため出力自体が実行のたびに変わるので、収束で検証する。
module GrokkingMlLib.Tests.Nb04PerceptronTests

open Xunit
open GrokkingMlLib.Nb04Perceptron

let private start = { Weights = [| 1.0; 2.0 |]; Bias = -4.0 }

[<Fact>]
let ``データセットは原著と同じ`` () =
    Assert.Equal<int[][]>(
        [| [| 1; 0 |]; [| 0; 2 |]; [| 1; 1 |]; [| 1; 2 |]
           [| 1; 3 |]; [| 2; 2 |]; [| 2; 3 |]; [| 3; 2 |] |],
        features
    )
    Assert.Equal<int[]>([| 0; 0; 0; 0; 1; 1; 1; 1 |], labels)

[<Fact>]
let ``ステップ関数は0で1を返す`` () =
    // 境界をどちらに含めるかで結果が変わる。原著は 0 以上を 1 とする
    Assert.Equal(1, step 0.0)
    Assert.Equal(0, step -1e-12)
    Assert.Equal(1, step 1.0)

[<Fact>]
let ``スコアは重み付き和にバイアスを足す`` () =
    // [2, 3] . [1, 2] - 4 = 2 + 6 - 4 = 4
    Assert.Equal(4.0, start.Score [| 2; 3 |], 12)

[<Fact>]
let ``誤差は当たれば0外れればスコアの絶対値`` () =
    Assert.Equal(0.0, start.Error([| 2; 3 |], 1), 12)
    Assert.Equal(4.0, start.Error([| 2; 3 |], 0), 12)

[<Fact>]
let ``重み1と2バイアス-4の予測は原著と同じ`` () =
    // 原著の出力
    //   0 0 / 1 0 / 0 0 / 1 1 / 1 0 / 1 0 / 1 0 / 1 0
    Assert.Equal<int[]>([| 0; 1; 0; 1; 1; 1; 1; 1 |], features |> Array.map start.Predict)
    Assert.Equal<float[]>(
        [| 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0 |],
        Array.map2 (fun f l -> start.Error(f, l)) features labels
    )

[<Fact>]
let ``平均パーセプトロン誤差は誤差の平均`` () =
    Assert.Equal(1.0 / 8.0, meanPerceptronError start features labels, 12)

[<Fact>]
let ``短く書いた版のトリックは原著と同じ数値になる`` () =
    // 原著の出力: ([0.9, 1.85], -4.1)
    let updated = perceptronTrick 0.05 start features.[6] 0

    Assert.Equal(0.9, updated.Weights.[0], 12)
    Assert.Equal(1.85, updated.Weights.[1], 12)
    Assert.Equal(-4.1, updated.Bias, 12)

[<Fact>]
let ``短く書いた版はバイアスを特徴量の数だけ動かす`` () =
    // 原著の 2 つの実装は挙動が違う。短く書いた版はバイアスの更新が
    // 重みのループの内側にあり、特徴量が 2 つなので 2 回適用される
    let explicit = perceptronTrickExplicit 0.05 start features.[6] 0
    let short = perceptronTrick 0.05 start features.[6] 0

    // 重みの更新は一致する
    Assert.Equal<float[]>(explicit.Weights, short.Weights)
    // バイアスだけ 2 倍動く
    Assert.Equal(-4.05, explicit.Bias, 12)
    Assert.Equal(-4.1, short.Bias, 12)

[<Fact>]
let ``当たっているときは何も動かない`` () =
    // features.[4] = [1, 3] は重み [1, 2] バイアス -4 で予測 1、ラベルも 1
    // レコードなので構造的等価性でそのまま比べられる
    Assert.Equal(start, perceptronTrick 0.05 start features.[4] 1)

[<Fact>]
let ``学習は誤差を下げる`` () =
    let result = perceptronAlgorithm features labels 0.01 200 0

    Assert.Equal(200, result.Errors.Length)
    Assert.True(List.last result.Errors < List.head result.Errors)

[<Fact>]
let ``学習後は全点を正しく分類できる`` () =
    let result = perceptronAlgorithm features labels 0.01 500 0

    Assert.Equal<int[]>(labels, features |> Array.map result.Boundary.Predict)

[<Fact>]
let ``ML_NET の AveragedPerceptron も全点を正しく分類する`` () =
    // 原著 scikit-learn の出力: Predictions: [0 0 0 0 1 1 1 1]
    let predict = fitWithMlNet features labels

    Assert.Equal<int[]>(labels, features |> Array.map predict)
