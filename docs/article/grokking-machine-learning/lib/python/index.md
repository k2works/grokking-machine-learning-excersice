# ライブラリ版 - Python 版

原著のノートブックをそのまま動かす構成です。原著が使うライブラリをそのまま使うので、
3 言語のなかでもっとも原著に近いコードになります。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | Python 3.12（3.13 以降は不可） |
| パッケージ管理 | uv |
| 数値計算 | NumPy / pandas |
| 機械学習 | scikit-learn / TensorFlow (Keras) / XGBoost |
| 可視化 | matplotlib |
| テスト | pytest |
| Lint / Format | ruff |
| 実行環境 | Nix devShell `python-ml` |
| サンプル実装 | `apps/grokking-ml-python-lib` |

### Python 3.12 に固定している理由

原著の第 10 章は Keras でネットワークを組みます。これを書き換えずに動かすには
TensorFlow が要りますが、macOS x86_64 向けの wheel を提供しているのは **2.16 系が最後** で、
その 2.16 系が対応する Python は **3.12 まで** です。そのため `requires-python` を
`>=3.12,<3.13` に固定しています。自前実装版（Python 3.11+、依存なし）とは別プロジェクトです。

### libomp が要る理由

XGBoost の wheel に同梱される `libxgboost.dylib` は `libomp.dylib` を動的に要求します。
これが無いと `import xgboost` の時点で `dlopen` に失敗します。devShell `python-ml` が
`llvmPackages.openmp` を入れ、`DYLD_LIBRARY_PATH` を通しています。

### セットアップ

```bash
nix develop .#python-ml
cd apps/grokking-ml-python-lib
uv sync
```

### テストの実行

```bash
cd apps/grokking-ml-python-lib
uv run pytest
uv run ruff check .
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
- [自前実装版 Python](../../python/index.md)
- [scikit-learn ドキュメント](https://scikit-learn.org/stable/)
- [Keras ドキュメント](https://keras.io/)
- [XGBoost ドキュメント](https://xgboost.readthedocs.io/)
