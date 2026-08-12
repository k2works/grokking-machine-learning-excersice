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

## 参照

- [シリーズ索引](../index.md)
- [3 言語統合比較](../all/index.md)
- [uv ドキュメント](https://docs.astral.sh/uv/)
- [pytest ドキュメント](https://docs.pytest.org/)
