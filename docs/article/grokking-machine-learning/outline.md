# Grokking Machine Learning 記事アウトライン

「Grokking Machine Learning」（Luis G. Serrano 著）の内容を、**Python・Kotlin・F# の 3 言語**で実装しながら解説する連載シリーズの執筆計画です。

本シリーズは 2 つの軸を持ちます。

| 軸 | 目的 | 記事 | サンプル実装 |
| :--- | :--- | :--- | :--- |
| **自前実装版** | アルゴリズムの中身を見せる。ML ライブラリを使わない | `python/` `kotlin/` `fsharp/` `all/` | `apps/grokking-ml-{python,kotlin,fsharp}` |
| **ライブラリ版** | 原著ノートブック 22 本を主流 ML ライブラリで忠実に再現する | `lib/{python,kotlin,fsharp}/` | `apps/grokking-ml-{python,kotlin,fsharp}-lib` |

以下、まず自前実装版の計画を述べ、後半に「ライブラリ版の計画」を置きます。

## 方針（自前実装版）

- 原著のコード（`tmp/manning`）は Python + NumPy/scikit-learn で書かれています。自前実装版では **アルゴリズムの中身を見せること** を目的とし、**3 言語とも ML ライブラリに依存しない自前実装** を行います。
- 3 言語で同じアルゴリズム・同じデータセット・同じ関数名（言語の命名規約に合わせた変形）を使い、章ごとに読み比べられるようにします。
- 可視化（matplotlib のプロット）は記事の対象外とし、代わりに **数値（学習後の係数・誤差の推移）をテストで検証** します。
- 乱数列は言語ごとに異なるため、学習結果の完全一致は求めません。テストは収束の範囲（許容誤差）で検証します。

## 対象言語

| 言語 | バージョン | 実行環境 | ビルド/テスト | サンプル実装 |
| :--- | :--- | :--- | :--- | :--- |
| Python | 3.11+ | Nix devShell `python` | uv + pytest + ruff | `apps/grokking-ml-python` |
| Kotlin | 2.2 (JVM 21) | Nix devShell `kotlin` | Gradle Wrapper + kotlin.test | `apps/grokking-ml-kotlin` |
| F# | .NET 10 | Nix devShell `dotnet` | dotnet + xUnit | `apps/grokking-ml-fsharp` |

## 章構成

原著は全 13 章です。うちコードを伴うのは第 3〜6 章・第 8〜13 章で、第 1・2・7 章は概念中心です。

| 章 | 原題 | テーマ | 実装 |
| :--- | :--- | :--- | :--- |
| 01 | What is machine learning? | 機械学習とは何か | なし（概念） |
| 02 | Types of machine learning | 教師あり／教師なし／強化学習 | なし（概念） |
| 03 | Drawing a line close to our points | 線形回帰、3 つのトリック、RMSE | あり |
| 04 | Optimizing the training process | 過学習・未学習、テスト、正則化 | あり |
| 05 | Using lines to split our points | パーセプトロン | あり |
| 06 | A continuous approach to splitting points | ロジスティック回帰 | あり |
| 07 | How do you measure classification models? | 混同行列、精度・再現率、ROC | あり（指標の実装） |
| 08 | Using probability to its maximum | ナイーブベイズ | あり |
| 09 | Splitting data by asking questions | 決定木、ジニ不純度・エントロピー | あり |
| 10 | Combining building blocks to gain more power | ニューラルネットワーク | あり |
| 11 | Finding boundaries with style | SVM とカーネル法 | あり |
| 12 | Combining models to maximize results | アンサンブル学習 | あり |
| 13 | Putting it all in practice | エンドツーエンドの実例 | あり |

## ファイル構成

```text
docs/article/grokking-machine-learning/
├── index.md          # シリーズ索引
├── outline.md        # 本ファイル
├── all/              # 3 言語統合比較
│   ├── index.md
│   └── ch03-linear-regression.md ...
├── python/
│   ├── index.md
│   └── ch01.md ... ch13.md
├── kotlin/
│   └── (同上)
├── fsharp/
│   └── (同上)
└── lib/              # ライブラリ版（原著ノートブックの再現）
    ├── index.md
    ├── python/
    ├── kotlin/
    └── fsharp/

apps/
├── grokking-ml-datasets/     # 3 言語のライブラリ版が共有する CSV
├── grokking-ml-python/       # src/grokking_ml/chNN_*.py, tests/
├── grokking-ml-kotlin/       # src/main/kotlin/chNN/, src/test/kotlin/chNN/
├── grokking-ml-fsharp/       # src/GrokkingMl/ChNN*.fs, tests/GrokkingMl.Tests/
├── grokking-ml-python-lib/   # src/grokking_ml_lib/, tests/
├── grokking-ml-kotlin-lib/   # src/main/kotlin/lib/, src/test/kotlin/lib/
└── grokking-ml-fsharp-lib/   # src/GrokkingMlLib/, tests/GrokkingMlLib.Tests/
```

## 前提整備

| 項目 | 内容 | 状態 |
| :--- | :--- | :--- |
| 実行環境 | Nix devShell `python` / `kotlin` / `dotnet`（`kotlin` は本シリーズで追加） | 完了 |
| Python 実装雛形 | uv プロジェクト、pytest、ruff | 完了 |
| Kotlin 実装雛形 | Gradle Kotlin DSL、Gradle Wrapper、kotlin.test、JVM 21 | 完了 |
| F# 実装雛形 | ソリューション + classlib + xUnit | 完了 |
| 記事ディレクトリ | `docs/article/grokking-machine-learning/` | 完了 |
| CI | サンプル実装 3 言語のテスト実行（`.github/workflows/grokking-ml.yml`） | 完了 |
| 全 13 章の執筆 | 第 1・2 章は `all/` に概念解説、第 3〜13 章は 3 言語 + 統合比較 | 完了 |
| テスト | Python 179・Kotlin 178・F# 176 がグリーン | 完了 |
| ノートブック | Jupyter / Kotlin Notebook / Polyglot Notebook を各 11 章分（計 33 本）、実行済み出力つき | 完了 |
| ノートブックのサイト掲載 | mkdocs-jupyter で記事として描画（`gulp notebooks:sync` で取り込み） | 完了 |

## 章別執筆計画

| 章 | Python の焦点 | Kotlin の焦点 | F# の焦点 |
| :--- | :--- | :--- | :--- |
| 01 | 3 言語共通の概念解説として `all/` に執筆（執筆済み） | 同左 | 同左 |
| 02 | 3 言語共通の概念解説として `all/` に執筆（執筆済み） | 同左 | 同左 |
| 03 | dataclass と純関数でトリックを表現（執筆済み） | data class と不変更新（執筆済み） | レコード型とパイプライン（執筆済み） |
| 04 | 訓練／テスト分割、L1・L2 正則化（執筆済み） | enum と when の網羅性検査（執筆済み） | 判別共用体と部分適用（執筆済み） |
| 05 | パーセプトロンの更新則（執筆済み） | typealias と data class の等価性（執筆済み） | 演算子を関数として渡す（執筆済み） |
| 06 | シグモイドと対数損失（執筆済み） | デフォルト引数で閾値を追加（執筆済み） | 部分適用で既定値を固定（執筆済み） |
| 07 | 混同行列と指標の実装（執筆済み） | groupingBy と zipWithNext（執筆済み） | フィールド名必須と構造的比較（執筆済み） |
| 08 | Counter によるカウント、ラプラス平滑化（執筆済み） | groupBy 2 引数版とローカル関数（執筆済み） | 関数合成とタプル畳み込み（執筆済み） |
| 09 | 再帰的な木構築（執筆済み） | sealed interface と網羅性検査（執筆済み） | 判別共用体の木と入れ子パターン（執筆済み） |
| 10 | 逆伝播の手計算実装（執筆済み） | zip の入れ子と ArrayDeque（執筆済み） | List.map2 と先頭追加（執筆済み） |
| 11 | マージン最大化とカーネル（執筆済み） | 関数型としてのカーネル注入（執筆済み） | カリー化された関数型への直接代入（執筆済み） |
| 12 | バギング・AdaBoost（執筆済み） | zip の制約と多重定義（執筆済み） | List.zip3 と fold（執筆済み） |
| 13 | データ前処理からモデル評価まで（執筆済み） | null 安全演算子と型の一般性（執筆済み） | 完全修飾名と部分適用（執筆済み） |

---

## ライブラリ版の計画

自前実装版が「ライブラリを使わずに書く」だったのに対し、ライブラリ版は
原著リポジトリ [luisguiserrano/manning](https://github.com/luisguiserrano/manning) の
ノートブック **22 本**（`DEPRECATED_` を除く）を、各言語の主流 ML ライブラリで **忠実に再現** します。

同じ問題に「自分で書いた場合」と「ライブラリに任せた場合」が並ぶことで、
ライブラリが裏で何をしているのかを、自前実装のコードと突き合わせて読めるようにするのが狙いです。

### 何を「忠実」とするか

原著のノートブックはセル単位で「データを読む → 学習する → 図を描く → 数値を見る」と進みます。
この **セルの流れと出力される数値** を再現の基準とします。

| 再現するもの | 再現しないもの |
| :--- | :--- |
| セルの並びと各セルの目的 | 図の見た目（配色・軸ラベルの細部） |
| 読み込むデータセット（同一ファイル） | 乱数列（言語ごとに異なる） |
| 学習結果の数値（係数・精度・誤差） | 実行時間 |
| ライブラリに任せる範囲 | 原著のコメント文そのもの |

乱数を使う箇所は、学習結果が許容誤差の範囲で一致することをテストで検証します。

### 対象言語とライブラリ

| 言語 | バージョン | 実行環境 | 主なライブラリ | サンプル実装 |
| :--- | :--- | :--- | :--- | :--- |
| Python | 3.12 固定 | Nix devShell `python-ml` | NumPy / pandas / scikit-learn / TensorFlow (Keras) / XGBoost / matplotlib | `apps/grokking-ml-python-lib` |
| Kotlin | 2.2（JVM 21） | Nix devShell `kotlin` | Multik / Kotlin DataFrame / Smile / XGBoost4J / Kandy | `apps/grokking-ml-kotlin-lib` |
| F# | .NET 10 | Nix devShell `dotnet` | Math.NET Numerics / Deedle / ML.NET / Accord.Neuro / Plotly.NET | `apps/grokking-ml-fsharp-lib` |

### ライブラリ版の選定経緯

3 言語とも、候補を実際にビルドして学習まで走らせたうえで選定しました。
以下は **動かなかったので採らなかった** ものと、その理由です。憶測ではなく実行して確認した結果です。

| 候補 | 結果 | 採った代替 |
| :--- | :--- | :--- |
| `tensorflow-cpu`（最新） | macOS x86_64 向け wheel が無い。2.17 以降は Linux / Windows のみ | `tensorflow==2.16.2` + Python 3.12 に固定 |
| Python 3.13 / 3.14 | TensorFlow 2.16 系の wheel が cp312 まで | `requires-python = ">=3.12,<3.13"` |
| ML.NET `Ols` トレーナー | MKL ネイティブ（`libMklImports`）が無く実行時に `NotSupportedException` | 厳密な最小二乗は Math.NET `Fit.Line` |
| ML.NET `Sdca`（回帰・正規化なし） | ラベルが大きいと重みが発散し `non-finite weights` で失敗 | 特徴量を `NormalizeMinMax` してから使う |
| TorchSharp / TorchSharp-cpu | `libtorch` の osx-x64 ネイティブが無く、テンソル生成時に初期化失敗 | Accord.Neuro（`ActivationNetwork`） |
| KotlinDL | 解決はできるが TensorFlow JNI に依存し、macOS x86_64 での動作が不確実 | Smile `MLP` |

Python の XGBoost は wheel 同梱の `libxgboost.dylib` が `libomp.dylib` を要求します。
devShell `python-ml` が `llvmPackages.openmp` を入れて `DYLD_LIBRARY_PATH` を通し、
CI では `libomp-dev` を入れて解決しています。

### 前提整備

| 項目 | 内容 | 状態 |
| :--- | :--- | :--- |
| 実行環境 | Nix devShell `python-ml` を新設（Python 3.12 + uv + libomp）。Kotlin・F# は既存を流用 | 完了 |
| 共有データセット | `apps/grokking-ml-datasets/`（小さい 9 本を同梱、大きい 2 本は初回ダウンロード） | 完了 |
| Python 実装雛形 | uv プロジェクト、データローダ、pytest、ruff | 完了 |
| Kotlin 実装雛形 | Gradle Kotlin DSL、Gradle Wrapper、データローダ、kotlin.test | 完了 |
| F# 実装雛形 | ソリューション + classlib + xUnit、データローダ | 完了 |
| スモークテスト | 各言語で主要ライブラリが学習まで走ることを検証（Python 5・Kotlin 6・F# 6） | 完了 |
| 記事ディレクトリ | `docs/article/grokking-machine-learning/lib/{python,kotlin,fsharp}/` | 完了 |
| CI | `.github/workflows/grokking-ml-lib.yml`（3 言語、Kotlin は JDK 21 / 25） | 完了 |
| 22 本の執筆 | 3 言語 × 22 本 | #01〜#17 完了（残り 5 本） |
| ノートブック | Jupyter / Kotlin Notebook / Polyglot Notebook | 未着手 |

### ノートブック別執筆計画

各行が記事 1 本に対応します。「焦点」はその言語で特に説明が要る箇所です。

| # | 章 | 原著のノートブック | Python の焦点 | Kotlin の焦点 | F# の焦点 |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 01 | 03 | `Coding_linear_regression` | `LinearRegression` と手書きトリックの対応（執筆済み） | Smile `OLS` は切片を別メソッドで返す（執筆済み） | `Fit.Line` の構造体タプル（執筆済み） |
| 02 | 03 | `House_price_predictions` | one-hot でランク落ちする設計行列（執筆済み） | `get_dummies` 相当を自前で書く（執筆済み） | Math.NET SVD の打ち切り閾値問題（執筆済み） |
| 03 | 04 | `Polynomial_regression_regularization` | 正則化が解を再現可能にする（執筆済み） | Smile の LASSO は厳密なゼロを出さない（執筆済み） | Ridge・Lasso を自前で実装（執筆済み） |
| 04 | 05 | `Coding_perceptron_algorithm` | 原著のトリック 2 版の挙動差（執筆済み） | Smile にパーセプトロンが無く線形 SVM で代替（執筆済み） | ML.NET `AveragedPerceptron`（執筆済み） |
| 05 | 06 | `Coding_logistic_regression` | 原著の対数損失の別表現が誤り（執筆済み） | `fit` が無く `binomial`・既定で正則化なし（執筆済み） | `LbfgsLogisticRegression`（執筆済み） |
| 06 | 06 | `Sentiment_analysis_IMDB` | `CountVectorizer` の 4 規則（執筆済み） | 自前ベクトル化 + 疎行列（執筆済み） | 自前ベクトル化 + 自前の勾配降下（執筆済み） |
| 07 | 08 | `Coding_naive_Bayes` | `np.compat.long` の切り捨て（執筆済み） | `split()` の意味の違いと多倍長整数（執筆済み） | `BigInteger` の整数除算（執筆済み） |
| 08 | 09 | `Gini_entropy_calculations` | 空リストの扱いが 2 指標で違う（執筆済み） | `groupingBy` が初出順を保つ（執筆済み） | Math.NET の `*` がそのまま内積（執筆済み） |
| 09 | 09 | `App_recommendations` | 同点の分割を無作為に選ぶため木が毎回変わる（執筆済み） | DOT 文字列・多出力なし・決定的（執筆済み） | 木を読み出せず CART を自前実装（執筆済み） |
| 10 | 09 | `Graphical_example`（決定木） | 境界を格子として取り出しテストする（執筆済み） | 深さの数え方が 1 つずれる（執筆済み） | 判別共用体で指標を差し替える（執筆済み） |
| 11 | 09 | `University_Admissions` | 3 つの制限と過学習（執筆済み） | 深さの対応則が通じない（執筆済み） | 自前 CART が 197 節まで完全一致（執筆済み） |
| 12 | 09 | `Regression_decision_tree` | 手計算の 9 通りの MSE（執筆済み） | 分類木と別パッケージ・完全一致（執筆済み） | `Leaf` の型差し替えだけ（執筆済み） |
| 13 | 10 | `Graphical_example`（NN） | パラメータ数 8770 が一致（執筆済み） | Adam が無く 20 倍のエポックが要る（執筆済み） | 重みの初期化が手動（執筆済み） |
| 14 | 10 | `House_price_predictions_neural_network` | パラメータ数 14795 が一致（執筆済み） | 標準化しないと 10^29 に発散（執筆済み） | 出力がシグモイドのみで 2 段の写像（執筆済み） |
| 15 | 10 | `Image_recognition` | Keras 同梱の MNIST ローダ | MNIST の取得方法（同梱ローダが無い） | 同左 |
| 16 | 10 | `Plotting_Boundaries` | matplotlib のみで完結する章 | Kandy への読み替え | Plotly.NET への読み替え |
| 17 | 11 | `Building_the_datasets` | `make_circles` などの生成器 | 生成器が無いので自前で書く範囲 | 同左 |
| 18 | 11 | `Calculating_similarities` | RBF カーネルの行列計算 | Multik での距離行列 | Math.NET での距離行列 |
| 19 | 11 | `SVM_graphical_example` | `SVC` のカーネル切り替え | Smile `SVM` + `MercerKernel` | ML.NET `LinearSvm` の制約 |
| 20 | 12 | `Random_forests_and_AdaBoost` | `RandomForestClassifier` / `AdaBoostClassifier` | Smile `RandomForest` / `AdaBoost` | ML.NET `FastForest` |
| 21 | 12 | `Gradient_boosting_and_XGBoost` | `xgboost` をそのまま使う | XGBoost4J（同じ実装の JVM バインディング） | ML.NET `FastTree` で代替する理由 |
| 22 | 13 | `End_to_end_example` | 前処理から評価までの流れ | 3 ライブラリをまたぐパイプライン | Deedle → ML.NET の受け渡し |

第 15 章（MNIST）は、Kotlin・F# に既製のローダが無く、データ取得を自前で書く範囲が広がります。
第 6 章（IMDB）は執筆済みで、scikit-learn のストップワード 318 語を共有データセットに書き出し、
3 言語で同じ語彙を作れるようにしました（語彙の添字まで一致）。
