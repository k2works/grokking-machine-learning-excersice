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

## 参照

- [シリーズ索引](../index.md)
- [3 言語統合比較](../all/index.md)
- [F# 公式ドキュメント](https://learn.microsoft.com/ja-jp/dotnet/fsharp/)
- [F# for Fun and Profit](https://fsharpforfunandprofit.com/)
