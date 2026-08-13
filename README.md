# Grokking Machine Learning 学習リポジトリ

書籍「[Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning)」（Luis G. Serrano 著）の内容を、**Python・Kotlin・F# の 3 言語**で実装しながら学ぶ記事シリーズです。

## 概要

### 目的

ライブラリの `fit()` を呼ぶのではなく、**線形回帰の重み更新や決定木の分割基準を自分の手で書く**ことで、機械学習アルゴリズムの中身を理解します。

同じアルゴリズムを 3 言語で書き比べると、手続き型・オブジェクト指向・関数型それぞれの表現の違いも見えてきます。第 9 章の決定木では F# の判別共用体がアルゴリズムの構造そのものになり、第 10 章の逆伝播では不変リストの先頭追加が計算の向きに合致します。そうした「言語機能とアルゴリズムが噛み合う瞬間」を各章で記録しています。

3 言語とも **ML ライブラリを使わず標準ライブラリだけ**で実装しています。NumPy も scikit-learn も KotlinDL も ML.NET も使いません。

### 成果物

| 区分 | 数量 | 場所 |
| :--- | ---: | :--- |
| 記事 | 52 本 | `docs/article/grokking-machine-learning/` |
| サンプル実装 | 3 言語 | `apps/grokking-ml-{python,kotlin,fsharp}/` |
| ノートブック | 33 本 | `apps/grokking-ml-*/notebooks/` |
| テスト | 533 件 | Python 179 / Kotlin 178 / F# 176 |

記事に載せたコードと数値は、**すべて実装の実行結果からの転記**です。ノートブックは実装本体を読み込む構成にしてあり、コードを複製していません。記事・実装・ノートブックの三者が食い違わないようにしています。

### 章構成

原著は全 13 章です。第 1・2 章は概念中心のため 3 言語共通、第 3〜13 章は言語別に執筆しています。

| 章 | テーマ | 章 | テーマ |
| :--- | :--- | :--- | :--- |
| 01 | 機械学習とは何か | 08 | ナイーブベイズ |
| 02 | 機械学習の種類 | 09 | 決定木 |
| 03 | 線形回帰 | 10 | ニューラルネットワーク |
| 04 | 過学習・未学習と正則化 | 11 | SVM とカーネル法 |
| 05 | パーセプトロン | 12 | アンサンブル学習 |
| 06 | ロジスティック回帰 | 13 | エンドツーエンドの実例 |
| 07 | 分類モデルの評価指標 | | |

入口は [シリーズ索引](docs/article/grokking-machine-learning/index.md) です。

### 前提

| ソフトウェア | バージョン | 用途 |
| :----------- | :--------- | :--- |
| Node.js | 22.x | ドキュメント・運用タスク（Gulp） |
| Docker | — | ドキュメントサイトのビルド |
| uv | — | Python のサンプル実装 |
| JDK | **21 以上** | Kotlin のサンプル実装 |
| .NET SDK | 10.x | F# のサンプル実装 |

Nix を使う場合は言語ごとの環境が用意されています（[開発](#開発)を参照）。

## 構成

- [構築](#構築)
- [配置](#配置)
- [運用](#運用)
- [開発](#開発)

## 詳細

### Quick Start

```bash
npm install
npm run docs:serve   # http://localhost:8000 でドキュメントサイトを開く
```

サンプル実装を動かす場合は言語ごとに次を実行します。

```bash
# Python
cd apps/grokking-ml-python && uv sync && uv run pytest

# Kotlin（JDK 21 以上が必要）
cd apps/grokking-ml-kotlin && ./gradlew test

# F#
cd apps/grokking-ml-fsharp && dotnet test
```

3 言語をまとめて実行する場合はリポジトリのルートで `npm run test` を使います。

### ノートブック

各章の実験は 3 種類のノートブックでも試せます。**実装本体を読み込んで動く**ので、値を変えてその場で確かめられます。

| 言語 | 種類 | 実行環境 | 実装の読み込み方 |
| :--- | :--- | :--- | :--- |
| Python | Jupyter Notebook | `uv run jupyter lab notebooks/` | `sys.path` に `../src` を追加 |
| Kotlin | Kotlin Notebook | IntelliJ IDEA / [Kotlin Jupyter カーネル](https://github.com/Kotlin/kotlin-jupyter) | `@file:DependsOn` でビルド済み JAR |
| F# | Polyglot Notebook | VS Code Polyglot Notebooks / .NET Interactive | `#load` で `.fs` を直接読み込み |

Kotlin は先に `./gradlew jar` で JAR を作っておく必要があります。

ノートブックの正本は `apps/*/notebooks/` にあります（実装を相対パスで読み込むため）。ドキュメントサイトへはビルド時に `npm run docs:notebooks` で取り込み、[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) が記事として描画します。

**[⬆ back to top](#構成)**

### 構築

```bash
claude mcp add -s project memory -- npx @modelcontextprotocol/server-memory
claude mcp add -s project codex -- npx @openai/codex mcp-server
```

#### ralph-loop の導入

1. Claude Code 起動後、`/plugin` を実行
2. 検索ボックスで ralph-loop を探して選択
3. インストールするスコープを選ぶ（ユーザー / プロジェクト / ローカル）
4. Claude Code を再起動
5. コマンドで実行

```text
/ralph-loop "<プロンプト>" --max-iterations <数値> --completion-promise "<完了テキスト>"
```

#### AI アシスタント（Skills）

`.claude/skills/` ディレクトリに定義された Skills により、AI アシスタントがタスクに応じた専門的な指示を自動的に読み込みます。Progressive Disclosure（段階的開示）により、必要なスキルのみがコンテキストに展開されます。

Skills 一覧は [CLAUDE.md の Skills 体系](CLAUDE.md#skills-体系) を参照してください。

新しいスキルの追加・改善には `/skill-creator` プラグインを使用します。テスト・評価・最適化を含むスキル作成ワークフローが自動化されます。

**[⬆ back to top](#構成)**

### 配置

#### GitHub Pages セットアップ

1. **GitHub リポジトリの Settings を開く**
   - リポジトリページで `Settings` タブをクリック

2. **Pages 設定を開く**
   - 左サイドバーの `Pages` をクリック

3. **Source を設定**
   - `Source` で `Deploy from a branch` を選択
   - `Branch` で `gh-pages` を選択し、フォルダは `/ (root)` を選択
   - `Save` をクリック

4. **初回デプロイ**
   - main ブランチにプッシュすると GitHub Actions が自動実行
   - Actions タブでデプロイ状況を確認

#### リリース

品質ゲート → バージョンバンプ → CHANGELOG 生成 → commit + tag を一貫して実行します。

```bash
npm run release:dry-run    # CHANGELOG プレビューとバージョン計算
npm run release:preflight  # 品質ゲートのみ実行
npm run release:minor      # 新機能追加（0.1.0 → 0.2.0）
```

品質ゲートは次の 5 つを直列実行します。

| チェック | 内容 |
| :--- | :--- |
| clean | working tree に未コミットの変更がないこと |
| lint | ノートブックの実行済み検証 + Markdown Lint + ruff |
| test | Python・Kotlin・F# のユニットテスト |
| build | Kotlin JAR・F# ビルド・ドキュメントサイト |
| e2e | 本リポジトリには対象がないため未定義（明示的に報告） |

詳細は [リリースガイド](docs/reference/リリースガイド.md) を参照してください。

**[⬆ back to top](#構成)**

### 運用

#### ドキュメントの編集

1. ローカル環境で MkDocs サーバーを起動します。

   ```bash
   npm run docs:serve
   ```

2. ブラウザで <http://localhost:8000> にアクセスして編集結果をプレビューします。

3. `docs/` ディレクトリ内の Markdown ファイルを編集します。

4. 変更をコミットしてプッシュします。

`npm run docs:serve` と `npm run docs:build` は、実行前にノートブックを `docs/` へ取り込みます（`gulp notebooks:sync`）。取り込み先は生成物のため `.gitignore` に入っています。

#### Gulp タスク

| タスク | コマンド | 内容 |
| :--- | :--- | :--- |
| ドキュメント起動 | `npm run docs:serve` | ノートブックを取り込んで MkDocs サーバーを起動 |
| ドキュメント停止 | `npm run docs:stop` | MkDocs サーバーを停止 |
| ドキュメントビルド | `npm run docs:build` | `site/` へ静的サイトを出力 |
| ノートブック取り込み | `npm run docs:notebooks` | `apps/*/notebooks/` を `docs/` へ複製 |
| ノートブック検証 | `npm run notebooks:check` | 未実行セル・エラー出力がないか確認 |
| ユーザーマニュアル | `npm run manual:build` | `docs/manual/` を HTML へ変換（`apps/manual/` へ出力） |
| 作業履歴 | `npm run journal` | コミットから `docs/journal/` を生成 |

特定日の作業履歴だけを生成する場合は次を使います。

```bash
npx gulp journal:generate:date --date=2026-08-13
```

#### GitHub Container Registry

開発コンテナイメージを GitHub Container Registry（GHCR）で管理しています。タグをプッシュすると GitHub Actions が自動的にビルドしてプッシュします。

```bash
git tag 0.1.1
git push origin 0.1.1
```

イメージの取得と実行は次のとおりです。

```bash
docker pull ghcr.io/k2works/grokking-machine-learning-excersice:latest
docker run -it -v $(pwd):/srv ghcr.io/k2works/grokking-machine-learning-excersice:latest
```

ローカルでビルドして使う場合は docker compose を使います。

```bash
docker compose run --rm dev bash
```

認証が必要な場合は GitHub Personal Access Token でログインします。

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin
```

権限設定は次のとおりです。

- リポジトリの Settings → Actions → General で `Read and write permissions` を設定
- `GITHUB_TOKEN` に `packages: write` 権限が付与されています

#### Dev Container

VS Code で Dev Container を使用する場合は次を実行します。

1. 「Dev Containers: Reopen in Container」を実行
2. 再ビルドする場合は「Dev Containers: Rebuild and Reopen in Container」

**[⬆ back to top](#構成)**

### 開発

#### ディレクトリ構成

```text
apps/
├── grokking-ml-python/     # uv + pytest + ruff
│   ├── src/grokking_ml/    #   ch03_*.py 〜 ch13_*.py
│   ├── tests/              #   179 テスト
│   └── notebooks/          #   Jupyter Notebook 11 本
├── grokking-ml-kotlin/     # Gradle + kotlin.test
│   ├── src/main/kotlin/    #   ch03/ 〜 ch13/
│   ├── src/test/kotlin/    #   178 テスト
│   └── notebooks/          #   Kotlin Notebook 11 本
└── grokking-ml-fsharp/     # dotnet + xUnit
    ├── src/GrokkingMl/     #   Ch03*.fs 〜 Ch13*.fs
    ├── tests/              #   176 テスト
    └── notebooks/          #   Polyglot Notebook 11 本

docs/article/grokking-machine-learning/
├── index.md                # シリーズ索引
├── outline.md              # 執筆計画
├── all/                    # 概念章 2 本 + 3 言語統合比較 11 本 + 目次
├── python/                 # 言語別の章 11 本 + 索引
├── kotlin/
└── fsharp/
```

#### Nix による開発環境

Nix を使用して、再現可能な開発環境を構築できます。

1. [Nix をインストール](https://nixos.org/download.html)します。
2. Flakes を有効にします（`~/.config/nix/nix.conf` に `experimental-features = nix-command flakes` を追加）。

本シリーズで使う環境は次の 4 つです。

```bash
nix develop            # 共通ツール
nix develop .#python   # Python / MkDocs
nix develop .#kotlin   # JDK 21 + Kotlin + Gradle
nix develop .#dotnet   # .NET SDK
```

他に `node` `go` `rust` `java` `haskell` `ruby` `php` `clojure` `elixir` `scala` の環境も定義されています。環境から抜けるには `exit` を入力します。依存関係の更新は `nix flake update` です。

#### 継続的インテグレーション

`.github/workflows/grokking-ml.yml` が次を検証します。

| ジョブ | 内容 |
| :--- | :--- |
| python | `uv sync` → `ruff check` → `pytest` |
| kotlin (21) / kotlin (25) | `./gradlew clean test`（**JDK は 2 バージョンで検証**） |
| fsharp | `dotnet test` |
| notebooks | 33 本すべてが実行済みでエラー出力を含まないこと |

Kotlin を 2 つの JDK で検証しているのは、Kotlin コンパイラが新しい JDK のバージョン文字列を解釈できずビルドが壊れる問題を実際に踏んだためです。読者の環境は CI より新しいことがあります。

#### GitHub Codespaces に SSH 接続

外部ターミナルアプリから GitHub Codespaces に SSH 接続することで、VS Code のエディタスペースを広く使いながら別ウィンドウのターミナルで作業できます。

前提として [GitHub CLI](https://cli.github.com/) がインストール済みであることが必要です。

1. <https://github.com/codespaces> から Codespace を作成します。

2. ブラウザに表示される Codespace の URL から名前を取得します。

   例: URL が `https://upgraded-cod-rpxpjr97jrwcxxw7.github.dev/` の場合、Codespace 名は `upgraded-cod-rpxpjr97jrwcxxw7` です。

3. SSH 接続します。

   ```bash
   gh codespace ssh -c upgraded-cod-rpxpjr97jrwcxxw7
   ```

接続後は通常のターミナル操作が可能です。

参考: [GitHub Codespaces に SSH 接続する](https://zenn.dev/hirokisakabe/articles/fdd7eb730423c0)

**[⬆ back to top](#構成)**

## 参照

- [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) — 原著
- [原著のサンプルコード](https://github.com/luisguiserrano/manning)
- [シリーズ索引](docs/article/grokking-machine-learning/index.md)
- [3 言語統合比較](docs/article/grokking-machine-learning/all/index.md)
- [レビュー結果](docs/review/index.md)
- [開発ガイド](docs/reference/開発ガイド.md)
- [リリースガイド](docs/reference/リリースガイド.md)
