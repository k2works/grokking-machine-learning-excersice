# ライブラリ版 - Kotlin 版

原著のノートブックを、JVM の ML ライブラリで再現します。
scikit-learn ひとつで済んでいた処理が、Kotlin では **Smile・Multik・Kotlin DataFrame** の
3 つに分かれます。どの処理がどのライブラリの担当になるのかが、そのまま読みどころです。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | Kotlin 2.2（JVM 21） |
| ビルド | Gradle Wrapper（Kotlin DSL） |
| 数値計算 | Multik |
| 表形式データ | Kotlin DataFrame |
| 機械学習 | Smile（回帰・分類・木・SVM・MLP） |
| 勾配ブースティング | XGBoost4J |
| 可視化 | Kandy（lets-plot） |
| テスト | kotlin.test |
| 実行環境 | Nix devShell `kotlin` |
| サンプル実装 | `apps/grokking-ml-kotlin-lib` |

### ライブラリの対応

| 原著（Python） | Kotlin | 備考 |
| :--- | :--- | :--- |
| `numpy` | Multik | `mk.ndarray` で多次元配列を作る |
| `pandas.read_csv` | Kotlin DataFrame | `DataFrame.readCSV(File)` |
| `sklearn.linear_model` | Smile `OLS` / `LogisticRegression` | Smile は切片を `coefficients()` に含めず `intercept()` で返す |
| `sklearn.tree` / `ensemble` | Smile `DecisionTree` / `RandomForest` / `AdaBoost` | |
| `sklearn.svm` | Smile `SVM` | カーネルは `MercerKernel` で指定する |
| `keras.Sequential` | Smile `MLP` | 層は `Layer.input` / `Layer.sigmoid` / `Layer.mle` で組む |
| `xgboost` | XGBoost4J | 原著と同じ実装の JVM バインディング |
| `matplotlib` | Kandy | ノートブック上で描画する |

### セットアップ

```bash
nix develop .#kotlin
cd apps/grokking-ml-kotlin-lib
./gradlew build
```

### テストの実行

```bash
cd apps/grokking-ml-kotlin-lib
./gradlew test
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
- [自前実装版 Kotlin](../../kotlin/index.md)
- [Smile ドキュメント](https://haifengl.github.io/)
- [Kotlin DataFrame ドキュメント](https://kotlin.github.io/dataframe/)
- [Kandy ドキュメント](https://kotlin.github.io/kandy/)
