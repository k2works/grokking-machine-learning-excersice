/// 原著ノートブック #14 の再現テスト。
///
/// 原著は特徴量を標準化していないが、Accord のシグモイド出力は 0〜1 しか返せない。
/// 価格を標準化してから学習し、予測を戻す形にすると原著（RMSE 約 554 万）より
/// 良い約 319 万に届く。
module GrokkingMlLib.Tests.Nb14HousePriceNetworkTests

open Xunit
open GrokkingMlLib.Nb14HousePriceNetwork

let private data = loadHousing ()
let private scaled = standardize data

[<Fact>]
let ``データセットは2518件38特徴量になる`` () =
    // Location（文字列）と Price（目的変数）を落とす
    Assert.Equal(2518, data.Size)
    Assert.Equal(38, data.FeatureCount)

[<Fact>]
let ``ネットワークの形は原著と同じ`` () =
    // Dense(38) -> Dense(128) -> Dense(64) -> Dense(1)
    Assert.Equal<int[]>([| 38; 128; 64 |], hiddenUnits)

[<Fact>]
let ``平均を答えるだけの基準は約877万`` () =
    // 価格の標準偏差。ネットワークはこれを下回る必要がある
    Assert.Equal(8775370.0, baselineRmse data, 0)

[<Fact>]
let ``標準化した特徴量は平均0になる`` () =
    for column in 0 .. data.FeatureCount - 1 do
        let mean = Array.init data.Size (fun row -> scaled.X.[row].[column]) |> Array.average
        Assert.True(abs mean < 1e-9, $"column {column} mean={mean}")

[<Fact>]
let ``標準化を戻すと元の価格になる`` () =
    // 予測を円単位に戻すために必要な操作
    for index in [ 0; 100; 2517 ] do
        Assert.Equal(data.Prices.[index], scaled.PriceScaler.Unscale scaled.Prices.[index], 6)

[<Fact>]
let ``原著の10エポックでも基準を大きく下回る`` () =
    // 原著と同じ 10 エポック。RMSE 約 492 万で、原著の 554 万より良い
    Assert.True(rmse (fit scaled OriginalEpochs 0) data scaled < baselineRmse data)

[<Fact>]
let ``300エポックなら原著より大きく良くなる`` () =
    // 原著の evaluate は RMSE 約 554 万。Accord は 300 エポックで約 319 万
    Assert.True(rmse (fit scaled Epochs 0) data scaled < 5_540_000.0)

[<Fact>]
let ``学習を進めるほどRMSEが下がる`` () =
    let early = rmse (fit scaled 10 0) data scaled
    let late = rmse (fit scaled Epochs 0) data scaled

    Assert.True(late < early, $"early={early} late={late}")

[<Fact>]
let ``予測は件数分の配列になる`` () =
    Assert.Equal(2518, (predict (fit scaled 10 0) scaled).Length)

[<Fact>]
let ``予測の平均は実際の平均に近い`` () =
    // 回帰なので、全体の水準は合ってくる
    let predictions = predict (fit scaled Epochs 0) scaled
    let actual = Array.average data.Prices

    Assert.InRange(Array.average predictions, actual * 0.5, actual * 1.5)

[<Fact>]
let ``種を固定すれば同じ結果になる`` () =
    Assert.Equal(rmse (fit scaled 20 7) data scaled, rmse (fit scaled 20 7) data scaled, 6)
