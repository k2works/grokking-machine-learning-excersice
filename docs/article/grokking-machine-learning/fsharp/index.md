# Grokking Machine Learning - F# 版

F#（.NET 10）を使い、機械学習アルゴリズムを **標準ライブラリのみ** で実装します。ML.NET は使いません。

レコード型による不変なモデル表現、部分適用によるハイパーパラメータの注入、`List.fold` による学習ループの表現が F# 版の焦点です。

## 開発環境

| 項目 | 内容 |
| :--- | :--- |
| 言語 | F#（.NET 10） |
| ビルド | dotnet CLI |
| テスト | xUnit |
| 実行環境 | Nix devShell `dotnet` |
| サンプル実装 | `apps/grokking-ml-fsharp` |

### セットアップ

```bash
nix develop .#dotnet
cd apps/grokking-ml-fsharp
dotnet restore
```

### テストの実行

```bash
cd apps/grokking-ml-fsharp
dotnet test
```

## プロジェクト構成

```text
apps/grokking-ml-fsharp/
├── GrokkingMl.sln
├── src/GrokkingMl/            # アルゴリズム本体（ChNN*.fs）
└── tests/GrokkingMl.Tests/    # xUnit テスト
```

F# はファイルのコンパイル順序が意味を持つため、章を追加するたびに `.fsproj` の `<Compile Include="..." />` を順番どおりに追記します。

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

## Polyglot Notebook

各章の実験を、**実装本体を読み込んで** その場で動かせるノートブックです。`#load` で `../src/GrokkingMl/*.fs` を直接読み込みます。VS Code の Polyglot Notebooks 拡張でも開けます。
コードを複製していないので、記事・実装・ノートブックの 3 者が食い違いません。

```bash
cd apps/grokking-ml-fsharp
dotnet interactive jupyter install
jupyter lab notebooks/
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
- [F# 公式ドキュメント](https://learn.microsoft.com/ja-jp/dotnet/fsharp/)
- [F# for Fun and Profit](https://fsharpforfunandprofit.com/)
