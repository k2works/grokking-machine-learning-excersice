# ライブラリ版 - F# 版

原著のノートブックを、.NET の ML ライブラリで再現します。
3 言語のなかでもっとも「1 つのライブラリでは足りない」構成になり、
**Math.NET・Deedle・ML.NET・Accord** を役割ごとに使い分けます。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | F#（.NET 10） |
| ビルド | dotnet CLI（ソリューション + classlib + xUnit） |
| 数値計算 | Math.NET Numerics（+ F# 拡張） |
| 表形式データ | Deedle |
| 機械学習 | ML.NET |
| ニューラルネットワーク | Accord.Neuro |
| 可視化 | Plotly.NET |
| テスト | xUnit |
| 実行環境 | Nix devShell `dotnet` |
| サンプル実装 | `apps/grokking-ml-fsharp-lib` |

### ライブラリの対応

| 原著（Python） | F# | 備考 |
| :--- | :--- | :--- |
| `numpy` | Math.NET Numerics | `DenseMatrix.ofRowList` などは `MathNet.Numerics.FSharp` が提供する |
| `pandas.read_csv` | Deedle | `Frame.ReadCsv` |
| `sklearn.linear_model`（回帰） | Math.NET `Fit.Line` / `Fit.MultiDim` | 厳密な最小二乗はこちらで行う |
| `sklearn.linear_model`（分類） | ML.NET `SdcaLogisticRegression` | 特徴量の正規化が事実上必須 |
| `sklearn.tree` / `ensemble` | ML.NET `FastTree` / `FastForest` | XGBoost の代替も FastTree が担う |
| `sklearn.svm` | ML.NET `LinearSvm` | 非線形カーネルは特徴量変換で表現する |
| `keras.Sequential` | Accord.Neuro | `ActivationNetwork` + `BackPropagationLearning` |
| `matplotlib` | Plotly.NET | ノートブック上で描画する |

### 動かなかった選択肢

- **ML.NET の `Ols` トレーナー** — MKL ネイティブライブラリ（`libMklImports`）を要求し、
  macOS x64 では実行時に `NotSupportedException` になります。厳密な最小二乗は Math.NET に寄せました。
- **TorchSharp** — `libtorch` の osx-x64 ネイティブが提供されておらず、
  テンソルを 1 つ作った時点で初期化に失敗します。ニューラルネットワークは Accord.Neuro を使います。

### FSharp.Core の明示指定

Deedle 8.0.0 が `FSharp.Core >= 10.1.201` を要求する一方、.NET SDK 10.0.100 が暗黙参照するのは
10.0.100 です。そのままだと NU1605（ダウングレード）警告が出るため、
`DisableImplicitFSharpCoreReference` を有効にしてバージョンを明示的に固定しています。

### セットアップ

```bash
nix develop .#dotnet
cd apps/grokking-ml-fsharp-lib
dotnet build
```

### テストの実行

```bash
cd apps/grokking-ml-fsharp-lib
dotnet test
```

## 目次

| # | 章 | テーマ | 記事 |
| ---: | :--- | :--- | :--- |
| 01 | 03 | 線形回帰を 3 つのトリックで学習する | [nb01](nb01.md) |
| 02 | 03 | ハイデラバードの住宅価格を予測する | [nb02](nb02.md) |
| 03 | 04 | 多項式回帰と L1・L2 正則化 | [nb03](nb03.md) |
| 04 | 05 | パーセプトロンで直線を引く | [nb04](nb04.md) |
| 05 | 06 | ロジスティック回帰の学習 | [nb05](nb05.md) |
| 06 | 06 | IMDB レビューの感情分析 | [nb06](nb06.md) |
| 07 | 08 | ナイーブベイズでスパム判定 | [nb07](nb07.md) |
| 08 | 09 | ジニ不純度とエントロピーの計算 | [nb08](nb08.md) |
| 09 | 09 | アプリ推薦の決定木 | [nb09](nb09.md) |
| 10 | 09 | 決定木の境界を図示する | [nb10](nb10.md) |
| 11 | 09 | 大学院入学審査の決定木 | [nb11](nb11.md) |
| 12 | 09 | 回帰木 | [nb12](nb12.md) |
| 13 | 10 | ニューラルネットワークの境界 | [nb13](nb13.md) |
| 14 | 10 | ニューラルネットワークで住宅価格を予測 | [nb14](nb14.md) |
| 15 | 10 | 手書き数字を認識する | [nb15](nb15.md) |
| 16 | 10 | ネットワークの境界を描く | [nb16](nb16.md) |
| 17 | 11 | SVM 用のデータセットを作る | [nb17](nb17.md) |
| 18 | 11 | 類似度行列で RBF カーネルを分解する | [nb18](nb18.md) |
| 19 | 11 | カーネルとハイパーパラメータを比べる | [nb19](nb19.md) |
| 20 | 12 | ランダムフォレストと AdaBoost | [nb20](nb20.md) |
| 15 | 10 | MNIST の画像認識 | 執筆予定 |
| 16 | 10 | 決定境界の描画 | 執筆予定 |
| 17 | 11 | SVM 用データセットの生成 | 執筆予定 |
| 18 | 11 | RBF カーネルの類似度計算 | 執筆予定 |
| 19 | 11 | SVM の境界とカーネルの比較 | 執筆予定 |
| 20 | 12 | ランダムフォレストと AdaBoost | 執筆予定 |
| 21 | 12 | 勾配ブースティングと XGBoost | 執筆予定 |
| 22 | 13 | タイタニックのエンドツーエンド | 執筆予定 |

## 参照

- [ライブラリ版 索引](../index.md)
- [自前実装版 F#](../../fsharp/index.md)
- [Math.NET Numerics ドキュメント](https://numerics.mathdotnet.com/)
- [ML.NET ドキュメント](https://learn.microsoft.com/ja-jp/dotnet/machine-learning/)
- [Deedle ドキュメント](https://fslab.org/Deedle/)
- [Plotly.NET ドキュメント](https://plotly.net/)
