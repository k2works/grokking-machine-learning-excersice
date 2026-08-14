/// 原著ノートブック #17 の再現テスト。
///
/// 原著は種を固定せずに乱数でデータを作るので、生成は再現できない。
/// 代わりに **配布 CSV が原著の規則で作られたこと** を検証する。
/// 検証結果は Python 版・Kotlin 版と完全に一致する。
module GrokkingMlLib.Tests.Nb17BuildingDatasetsTests

open Xunit
open GrokkingMlLib.Nb17BuildingDatasets

[<Fact>]
let ``配布CSVの行数は規則の点とノイズの合計`` () =
    // 原著は「規則どおりの点」と「ノイズ」を続けて追加している
    Assert.Equal(60, linear.Total)
    Assert.Equal(110, oneCircle.Total)
    Assert.Equal(220, twoCircles.Total)

    for spec in specs do
        Assert.Equal(spec.Total, (load spec).Length)

[<Fact>]
let ``先頭の点は規則に1つも違反しない`` () =
    // 配布 CSV の先頭 Points 件は規則どおりに作られた点。
    // ここに違反が 1 つも無いことが、規則を正しく読めた証拠になる
    for spec in specs do
        let violations = ruleViolations spec (load spec)
        Assert.Empty(violations |> List.filter (fun i -> i < spec.Points))

[<Fact>]
let ``違反はすべてノイズ部分にあり約半数`` () =
    // ノイズはラベルを 0 か 1 で振り直すので、規則と食い違うのは約半分
    let expected = dict [ "linear", 5; "one_circle", 7; "two_circles", 12 ]

    for spec in specs do
        let violations = ruleViolations spec (load spec)
        Assert.Equal(expected.[spec.Name], violations.Length)
        Assert.True(violations |> List.forall (fun i -> i >= spec.Points))
        let ratio = float violations.Length / float spec.Noise
        Assert.True(ratio >= 0.2 && ratio <= 0.8)

[<Fact>]
let ``座標はマイナス3から3の範囲`` () =
    // 原著の 6 * random() - 3
    for spec in specs do
        for point in load spec do
            Assert.InRange(point.X1, -3.0, 3.0)
            Assert.InRange(point.X2, -3.0, 3.0)

[<Fact>]
let ``ラベルは0か1だけ`` () =
    for spec in specs do
        let labels = load spec |> List.map (fun p -> p.Y) |> List.distinct |> List.sort
        Assert.Equal<int list>([ 0; 1 ], labels)

[<Fact>]
let ``直線の規則は境界のちょうど上を0にする`` () =
    // 原著は x + y > 0.5（等号を含まない）
    Assert.Equal(0, linearRule 0.25 0.25)
    Assert.Equal(1, linearRule 0.3 0.3)

[<Fact>]
let ``円の規則は境界のちょうど上を0にする`` () =
    // 原著は x^2 + y^2 < 2.8（等号を含まない）
    let radius = sqrt 2.8
    Assert.Equal(0, oneCircleRule radius 0.0)
    Assert.Equal(1, oneCircleRule (radius - 1e-9) 0.0)

[<Fact>]
let ``2つの円は重なりを持つ`` () =
    // 中心 (1, 0) と (-1, 0)、半径 √2 ≒ 1.414。中心間の距離 2 より大きいので重なる
    Assert.Equal(1, twoCirclesRule 0.0 0.0)
    Assert.Equal(0, twoCirclesRule 0.0 2.0)

[<Fact>]
let ``生成した点は規則どおりでノイズだけが外れる`` () =
    for spec in specs do
        let generated = generate spec 0
        Assert.Equal(spec.Total, generated.Length)
        Assert.Empty(ruleViolations spec generated |> List.filter (fun i -> i < spec.Points))

[<Fact>]
let ``生成は種を固定すれば再現する`` () =
    // 原著は種を固定していないので毎回違うものが出る
    Assert.Equal<Point list>(generate linear 42, generate linear 42)
    Assert.NotEqual<Point list>(generate linear 42, generate linear 43)

[<Fact>]
let ``生成したものは配布CSVとは一致しない`` () =
    // 原著が種を固定していない以上、これは一致しようがない
    let generated = generate linear 0
    let published = load linear

    Assert.Equal(published.Length, generated.Length)
    Assert.NotEqual(published.Head.X1, generated.Head.X1)

[<Fact>]
let ``同じ種でもPythonとは違う点が出る`` () =
    // .NET の Random と Python の random は別のアルゴリズム（Python は
    // メルセンヌツイスタ）。種を揃えても一致しない。
    // Python 版の同じ種での 1 点目は 2.0665311091502883
    Assert.NotEqual(2.0665311091502883, (generate linear 0).Head.X1)
