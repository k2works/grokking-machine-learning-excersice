/// 原著ノートブック #02 の再現テスト。
///
/// 単回帰は原著の数値と完全に一致する。全特徴量の重回帰は one-hot 符号化によって
/// 設計行列がランク落ちしており係数が一意に決まらないので、当てはまりの良さで検証する。
module GrokkingMlLib.Tests.Nb02HousePricesTests

open Xunit
open MathNet.Numerics.LinearAlgebra
open GrokkingMlLib.Nb02HousePrices

/// データの読み込みと SVD は重いので、クラスフィクスチャで 1 度だけ行う
type Fixture() =
    let frame = loadData ()
    let prepared = preprocess frame
    let model = fitAllFeatures prepared

    member _.Frame = frame
    member _.Prepared = prepared
    member _.Model = model

type Nb02HousePricesTests(fixture: Fixture) =
    interface Xunit.IClassFixture<Fixture>

    [<Fact>]
    member _.``データセットの形は原著と同じ``() =
        // 原著の出力: The dataset has 2518 rows, and 40 columns
        Assert.Equal(2518, fixture.Frame.RowCount)
        Assert.Equal(40, fixture.Frame.ColumnCount)

    [<Fact>]
    member _.``単回帰の係数は原著と同じ数値になる``() =
        // 原著 scikit-learn の出力
        //   y-intercept: -6222669.083283698
        //   slope (coefficient of Area): 9753.940608184039
        let model = fitAreaOnly fixture.Frame

        Assert.Equal(-6222669.083283698, model.Intercept, 6)
        Assert.Equal(9753.940608184039, model.Coefficients.[0], 9)

    [<Fact>]
    member _.``欠損を含む末尾の行を落とす``() =
        Assert.Equal(2434, ValidRows)
        Assert.Equal(ValidRows, fixture.Prepared.RowCount)
        Assert.True(fixture.Prepared.RowCount < fixture.Frame.RowCount)

    [<Fact>]
    member _.``標準化した列は平均0分散1になる``() =
        for index in 0..1 do
            let values = fixture.Prepared.Features.Column index
            let mean = Vector.sum values / float values.Count
            let variance =
                values |> Vector.map (fun v -> (v - mean) ** 2.0) |> Vector.sum
                |> fun s -> s / float (values.Count - 1)

            Assert.Equal(0.0, mean, 12)
            Assert.Equal(1.0, variance, 12)

    [<Fact>]
    member _.``標準化の統計量は Python 版と同じ``() =
        let standardizer = fixture.Prepared.Standardizer

        Assert.Equal(1644.1516023007396, standardizer.AreaMean, 9)
        Assert.Equal(748.1348121200747, standardizer.AreaStd, 9)
        Assert.Equal(2.6261298274445357, standardizer.BedroomsMean, 12)
        Assert.Equal(0.6850461155463963, standardizer.BedroomsStd, 12)

    [<Fact>]
    member _.``one-hot 符号化で277列になる``() =
        // 原著の出力: X_full.loc[0] ... Length: 277
        Assert.Equal(277, fixture.Prepared.ColumnNames.Length)
        Assert.Equal(277, fixture.Prepared.Features.ColumnCount)

        let locationColumns =
            fixture.Prepared.ColumnNames
            |> List.filter (fun name -> name.StartsWith "Location_")

        // 元の 40 列から Price と Location を除いた 38 列 + 地域の one-hot
        Assert.Equal(277 - 38, locationColumns.Length)

    [<Fact>]
    member _.``全特徴量モデルの RMSE は原著とほぼ同じ``() =
        // 原著の出力: Root Mean Squared Error (RMSE) of the model: 3981401.4927888927
        // 係数は一意でないが、当てはまりの良さは解の取り方によらずほぼ同じになる
        let error = rmse fixture.Prepared.Labels (predictAll fixture.Model fixture.Prepared)

        Assert.Equal(3981401.4927888927, error, 3981401.4927888927 * 1e-4)

    [<Fact>]
    member _.``全特徴量モデルは単回帰より当てはまりが良い``() =
        let simple = fitAreaOnly fixture.Frame
        let standardizer = fixture.Prepared.Standardizer
        // 単回帰は標準化前の面積で学習しているので、標準化を戻してから予測する
        let simplePredictions =
            fixture.Prepared.Features.Column 0
            |> Vector.map (fun scaled ->
                simple.Intercept
                + simple.Coefficients.[0] * (scaled * standardizer.AreaStd + standardizer.AreaMean))

        let simpleRmse = rmse fixture.Prepared.Labels simplePredictions
        let fullRmse = rmse fixture.Prepared.Labels (predictAll fixture.Model fixture.Prepared)

        Assert.True(fullRmse < simpleRmse, $"full={fullRmse} simple={simpleRmse}")

    [<Fact>]
    member _.``新しい物件の予測は原著とほぼ同じ``() =
        // 原著の出力
        //   Predicted price for a house with size 1000 and 3 bedrooms: 6,006,016.00
        let predicted = predictNewHouse fixture.Model fixture.Prepared 1000.0 3 "Gachibowli"

        Assert.Equal(6006016.00, predicted, 6006016.00 * 1e-2)

    [<Fact>]
    member _.``学習データに無い地域はエラーになる``() =
        let error =
            Assert.Throws<System.ArgumentException>(fun () ->
                predictNewHouse fixture.Model fixture.Prepared 1000.0 3 "Atlantis" |> ignore)

        Assert.Contains("学習データに無い地域", error.Message)
