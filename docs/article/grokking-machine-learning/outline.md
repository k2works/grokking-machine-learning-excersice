# Grokking Machine Learning 記事アウトライン

「Grokking Machine Learning」（Luis G. Serrano 著）の内容を、**Python・Kotlin・F# の 3 言語**で実装しながら解説する連載シリーズの執筆計画です。

## 方針

- 原著のコード（`tmp/manning`）は Python + NumPy/scikit-learn で書かれています。本シリーズでは **アルゴリズムの中身を見せること** を目的とし、**3 言語とも ML ライブラリに依存しない自前実装** を行います。
- 3 言語で同じアルゴリズム・同じデータセット・同じ関数名（言語の命名規約に合わせた変形）を使い、章ごとに読み比べられるようにします。
- 可視化（matplotlib のプロット）は記事の対象外とし、代わりに **数値（学習後の係数・誤差の推移）をテストで検証** します。
- 乱数列は言語ごとに異なるため、学習結果の完全一致は求めません。テストは収束の範囲（許容誤差）で検証します。

## 対象言語

| 言語 | バージョン | 実行環境 | ビルド/テスト | サンプル実装 |
| :--- | :--- | :--- | :--- | :--- |
| Python | 3.11+ | Nix devShell `python` | uv + pytest + ruff | `apps/grokking-ml-python` |
| Kotlin | 2.0 (JVM 21) | Nix devShell `kotlin` | Gradle Wrapper + kotlin.test | `apps/grokking-ml-kotlin` |
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
└── fsharp/
    └── (同上)

apps/
├── grokking-ml-python/   # src/grokking_ml/chNN_*.py, tests/
├── grokking-ml-kotlin/   # src/main/kotlin/chNN/, src/test/kotlin/chNN/
└── grokking-ml-fsharp/   # src/GrokkingMl/ChNN*.fs, tests/GrokkingMl.Tests/
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
