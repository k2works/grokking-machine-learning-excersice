# ライブラリ版 - 原著ノートブックの再現

本シリーズのもう一方の軸である [自前実装版](../index.md) は、`fit()` を呼ばずにアルゴリズムを手で書きました。
こちらの **ライブラリ版** は逆に、原著 [luisguiserrano/manning](https://github.com/luisguiserrano/manning) の
ノートブック **22 本** を、各言語の主流 ML ライブラリで **忠実に再現** します。

同じ問題に対して「自分で書いた場合」と「ライブラリに任せた場合」が並ぶので、
ライブラリが裏で何をしているのかを、自前実装のコードと突き合わせながら読めます。

## 何を「忠実」とするか

原著のノートブックはセル単位で「データを読む → 学習する → 図を描く → 数値を見る」と進みます。
ライブラリ版はこの **セルの流れと出力される数値** を再現の基準とします。

| 再現するもの | 再現しないもの |
| :--- | :--- |
| セルの並びと各セルの目的 | 図の見た目（配色・軸ラベルの細部） |
| 読み込むデータセット（同一ファイル） | 乱数列（言語ごとに異なる） |
| 学習結果の数値（係数・精度・誤差） | 実行時間 |
| ライブラリの選択（ライブラリに任せる範囲を原著に合わせる） | 原著のコメント文そのもの |

乱数を使う箇所は、学習結果が許容誤差の範囲で一致することをテストで確認します。

## 対応するライブラリ

原著は Python + NumPy / pandas / scikit-learn / Keras / XGBoost です。
Kotlin と F# では、同じ役割を担うライブラリに置き換えます。

| 原著（Python） | 役割 | Kotlin | F# |
| :--- | :--- | :--- | :--- |
| NumPy | 多次元配列と数値演算 | Multik | Math.NET Numerics |
| pandas | 表形式データの読み書き | Kotlin DataFrame | Deedle |
| scikit-learn | 回帰・分類・木・SVM | Smile | ML.NET / Smile 相当は Accord で補完 |
| matplotlib | 可視化 | Kandy | Plotly.NET |
| TensorFlow / Keras | ニューラルネットワーク | Smile MLP | Accord.Neuro |
| XGBoost | 勾配ブースティング | XGBoost4J | ML.NET FastTree |

すべて実際に学習まで動くことを、各実装のスモークテストで確認済みです。
選定の経緯と、動かなかった候補（TorchSharp・ML.NET の OLS トレーナーなど）は
[アウトライン](../outline.md) の「ライブラリ版の選定経緯」に記録しています。

## 言語別

| 言語 | 実行環境 | サンプル実装 | 解説 |
| :--- | :--- | :--- | :--- |
| Python | Nix devShell `python-ml`（Python 3.12） | `apps/grokking-ml-python-lib` | [Python 版](python/index.md) |
| Kotlin | Nix devShell `kotlin`（JVM 21） | `apps/grokking-ml-kotlin-lib` | [Kotlin 版](kotlin/index.md) |
| F# | Nix devShell `dotnet`（.NET 10） | `apps/grokking-ml-fsharp-lib` | [F# 版](fsharp/index.md) |

## ノートブック対応表

原著のノートブック 1 本が、記事 1 本に対応します。

| # | 章 | 原著のノートブック | テーマ |
| ---: | :--- | :--- | :--- |
| 01 | 03 | `Coding_linear_regression.ipynb` | 線形回帰を 3 つのトリックで学習する |
| 02 | 03 | `House_price_predictions.ipynb` | ハイデラバードの住宅価格を予測する |
| 03 | 04 | `Polynomial_regression_regularization.ipynb` | 多項式回帰と L1・L2 正則化 |
| 04 | 05 | `Coding_perceptron_algorithm.ipynb` | パーセプトロンで直線を引く |
| 05 | 06 | `Coding_logistic_regression.ipynb` | ロジスティック回帰の学習 |
| 06 | 06 | `Sentiment_analysis_IMDB.ipynb` | IMDB レビューの感情分析 |
| 07 | 08 | `Coding_naive_Bayes.ipynb` | ナイーブベイズでスパム判定 |
| 08 | 09 | `Gini_entropy_calculations.ipynb` | ジニ不純度とエントロピーの計算 |
| 09 | 09 | `App_recommendations.ipynb` | アプリ推薦の決定木 |
| 10 | 09 | `Graphical_example.ipynb` | 決定木の境界を図示する |
| 11 | 09 | `University_Admissions.ipynb` | 大学院入学審査の決定木 |
| 12 | 09 | `Regression_decision_tree.ipynb` | 回帰木 |
| 13 | 10 | `Graphical_example.ipynb` | ニューラルネットワークの境界 |
| 14 | 10 | `House_price_predictions_neural_network.ipynb` | ニューラルネットワークで住宅価格を予測 |
| 15 | 10 | `Image_recognition.ipynb` | MNIST の画像認識 |
| 16 | 10 | `Plotting_Boundaries.ipynb` | 決定境界の描画 |
| 17 | 11 | `Building_the_datasets.ipynb` | SVM 用データセットの生成 |
| 18 | 11 | `Calculating_similarities.ipynb` | RBF カーネルの類似度計算 |
| 19 | 11 | `SVM_graphical_example.ipynb` | SVM の境界とカーネルの比較 |
| 20 | 12 | `Random_forests_and_AdaBoost.ipynb` | ランダムフォレストと AdaBoost |
| 21 | 12 | `Gradient_boosting_and_XGBoost.ipynb` | 勾配ブースティングと XGBoost |
| 22 | 13 | `End_to_end_example.ipynb` | タイタニックのエンドツーエンド |

原著リポジトリには `DEPRECATED_` 接頭辞の付いた旧版ノートブックも 7 本ありますが、
著者自身が置き換えたものなので対象外とします。

## データセット

3 言語の実装が `apps/grokking-ml-datasets/` の同じ CSV を読みます。
同じ入力に対する結果を言語間で突き合わせるためです。詳細は
[データセットの README](https://github.com/k2works/grokking-machine-learning-excersice/tree/main/apps/grokking-ml-datasets)
を参照してください。

大きい 2 本（`emails.csv` 約 8.5 MB・`IMDB_Dataset.csv` 約 63 MB）だけは Git に含めず、
各言語のデータローダが初回参照時に原著リポジトリから取得します。

## 参照

- [シリーズ索引](../index.md)
- [アウトライン](../outline.md)
- [原著のサンプルコード（GitHub）](https://github.com/luisguiserrano/manning)
