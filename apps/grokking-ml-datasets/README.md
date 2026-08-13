# Grokking ML データセット

ライブラリ版サンプル実装（`apps/grokking-ml-*-lib`）が共有して読み込むデータセットです。
原著のサンプルコードリポジトリ [luisguiserrano/manning](https://github.com/luisguiserrano/manning) に
同梱されている CSV をそのまま置いています。

3 言語の実装がこの 1 か所を参照するので、同じ入力に対する結果を言語間で突き合わせられます。

## リポジトリに入っているファイル

| ファイル | 出典（原著の章） | 用途 |
| :--- | :--- | :--- |
| `Hyderabad.csv` | 第 3 章 / 第 10 章 | ハイデラバードの住宅価格。線形回帰・ニューラルネットワークの回帰 |
| `one_circle.csv` | 第 10 章 / 第 11 章 | 円形に分布する 2 クラスデータ。非線形分類 |
| `linear.csv` | 第 11 章 | 線形分離可能な 2 クラスデータ。線形 SVM |
| `two_circles.csv` | 第 11 章 | 二重円のデータ。RBF カーネル |
| `Admission_Predict.csv` | 第 9 章 | 大学院入学審査データ。決定木の回帰・分類 |
| `titanic.csv` | 第 13 章 | タイタニック号の生存者データ（生データ） |
| `clean_titanic_data.csv` | 第 13 章 | 欠損値を処理した中間データ |
| `preprocessed_titanic_data.csv` | 第 13 章 | 特徴量化まで済ませたデータ |
| `titanic_test.csv` | 第 13 章 | 原著の `test.csv`。ファイル名が汎用すぎるので改名した |

`one_circle.csv` と `Hyderabad.csv` は原著では複数の章に同じ内容で重複して置かれています。
ここでは 1 本にまとめました（バイト単位で同一であることを確認済み）。

## リポジトリに入れていないファイル

サイズが大きいので Git には含めず、初回利用時にダウンロードします。

| ファイル | サイズ | 出典 | 用途 |
| :--- | ---: | :--- | :--- |
| `emails.csv` | 約 8.5 MB | 第 8 章 | スパムメール判定。ナイーブベイズ |
| `IMDB_Dataset.csv` | 約 63 MB | 第 6 章 | 映画レビューの感情分析。ロジスティック回帰 |

各言語のデータローダが、未取得ならこのディレクトリへ自動でダウンロードします。
手動で取得する場合は次のとおりです。

```bash
cd apps/grokking-ml-datasets
curl -L -o emails.csv \
  https://raw.githubusercontent.com/luisguiserrano/manning/master/Chapter_08_Naive_Bayes/emails.csv
curl -L -o IMDB_Dataset.csv \
  https://raw.githubusercontent.com/luisguiserrano/manning/master/Chapter_06_Logistic_Regression/IMDB_Dataset.csv
```

ダウンロードしたファイルは `.gitignore` で除外しています。

## ライセンス

原著のサンプルコードリポジトリの配布条件に従います。
`Hyderabad.csv`・`Admission_Predict.csv`・`titanic.csv` はいずれも Kaggle 由来の公開データセットです。
