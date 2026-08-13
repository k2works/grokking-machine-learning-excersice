# Grokking Machine Learning - Python 版

Python 3.11+ を使い、機械学習アルゴリズムを **標準ライブラリのみ** で実装します。NumPy も scikit-learn も使いません。アルゴリズムの計算がすべてコード上に見えている状態を保つためです。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | Python 3.11 以上 |
| パッケージ管理 | uv |
| テスト | pytest |
| Lint / Format | ruff |
| 実行環境 | Nix devShell `python` |
| サンプル実装 | `apps/grokking-ml-python` |

### セットアップ

```bash
nix develop .#python
cd apps/grokking-ml-python
uv sync
```

### テストの実行

```bash
cd apps/grokking-ml-python
uv run pytest
uv run ruff check .
```

## 目次

| 章 | タイトル |
| :--- | :--- |
| 03 | [線形回帰](ch03.md) |
| 04 | [過学習・未学習と正則化](ch04.md) |
| 05 | [パーセプトロン](ch05.md) |
| 06 | [ロジスティック回帰](ch06.md) |
| 07 | [分類モデルの評価指標](ch07.md) |
| 08 | [ナイーブベイズ](ch08.md) |
| 09 | [決定木](ch09.md) |
| 10 | [ニューラルネットワーク](ch10.md) |
| 11 | [サポートベクターマシンとカーネル法](ch11.md) |
| 12 | [アンサンブル学習](ch12.md) |
| 13 | [エンドツーエンドの実例](ch13.md) |

## Jupyter Notebook

各章の実験を、**実装本体を読み込んで** その場で動かせるノートブックです。`../src/grokking_ml/` を `sys.path` に追加して読み込みます。
コードを複製していないので、記事・実装・ノートブックの 3 者が食い違いません。

```bash
cd apps/grokking-ml-python
uv sync && uv run jupyter lab notebooks/
```

| 章 | ノートブック |
| :--- | :--- |
| 第 3 章 線形回帰 | [ch03.ipynb](notebooks/ch03.ipynb) |
| 第 4 章 過学習・未学習と正則化 | [ch04.ipynb](notebooks/ch04.ipynb) |
| 第 5 章 パーセプトロン | [ch05.ipynb](notebooks/ch05.ipynb) |
| 第 6 章 ロジスティック回帰 | [ch06.ipynb](notebooks/ch06.ipynb) |
| 第 7 章 分類モデルの評価指標 | [ch07.ipynb](notebooks/ch07.ipynb) |
| 第 8 章 ナイーブベイズ | [ch08.ipynb](notebooks/ch08.ipynb) |
| 第 9 章 決定木 | [ch09.ipynb](notebooks/ch09.ipynb) |
| 第 10 章 ニューラルネットワーク | [ch10.ipynb](notebooks/ch10.ipynb) |
| 第 11 章 SVM とカーネル法 | [ch11.ipynb](notebooks/ch11.ipynb) |
| 第 12 章 アンサンブル学習 | [ch12.ipynb](notebooks/ch12.ipynb) |
| 第 13 章 エンドツーエンドの実例 | [ch13.ipynb](notebooks/ch13.ipynb) |

## 参照

- [シリーズ索引](../index.md)
- [3 言語統合比較](../all/index.md)
- [uv ドキュメント](https://docs.astral.sh/uv/)
- [pytest ドキュメント](https://docs.pytest.org/)
