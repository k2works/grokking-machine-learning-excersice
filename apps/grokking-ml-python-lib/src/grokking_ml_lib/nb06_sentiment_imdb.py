"""原著ノートブック #06 `Chapter_06_Logistic_Regression/Sentiment_analysis_IMDB.ipynb`。

IMDB の映画レビュー 50000 件を、単語の出現回数だけからロジスティック回帰で
肯定・否定に分類する。学習した係数がそのまま「単語の感情スコア」になるのが読みどころ。

`CountVectorizer` がテキストを語彙 2000 語の出現回数ベクトルに変える。
語彙の作り方（トークンの切り出し・ストップワード・上位 2000 語の選び方）が
そのまま結果を左右するので、その規則も外に出してある。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from grokking_ml_lib.datasets import dataset_path

#: 原著が使う語彙の上限
MAX_FEATURES = 2000

#: 原著が使う lbfgs の反復回数上限。既定の 100 では収束しない
MAX_ITER = 1000


def load_reviews() -> pd.DataFrame:
    """IMDB のレビューを読み込み、`sentiment` を 0 / 1 に置き換える。

    このファイルは約 63 MB あるためリポジトリには含めていない。
    未取得なら `dataset_path` が原著リポジトリから取得する。
    """
    movies = pd.read_csv(dataset_path("IMDB_Dataset.csv"))
    movies["sentiment"] = movies["sentiment"].map({"positive": 1, "negative": 0})
    return movies


def build_vectorizer(max_features: int = MAX_FEATURES) -> CountVectorizer:
    """原著と同じ設定の `CountVectorizer` を作る。

    既定の挙動が結果を決めているので、明示しておく。
    - 小文字化する
    - トークンは正規表現 `(?u)\\b\\w\\w+\\b`（2 文字以上の単語）
    - `stop_words='english'` で scikit-learn 内蔵の 318 語を除く
    - 残った語を **コーパス全体の出現回数** で並べ、上位 `max_features` 語を採る
    """
    return CountVectorizer(max_features=max_features, stop_words="english")


@dataclass
class SentimentModel:
    """学習済みモデルと、それを作るのに使った語彙。"""

    vectorizer: CountVectorizer
    model: LogisticRegression
    features: csr_matrix

    @property
    def vocabulary(self) -> NDArray[np.str_]:
        """語彙。添字がそのまま係数の添字に対応する。"""
        return self.vectorizer.get_feature_names_out()


def fit(movies: pd.DataFrame, max_features: int = MAX_FEATURES) -> SentimentModel:
    """レビューをベクトル化してロジスティック回帰を学習する。"""
    vectorizer = build_vectorizer(max_features)
    features = vectorizer.fit_transform(movies["review"])
    model = LogisticRegression(max_iter=MAX_ITER).fit(features, movies["sentiment"])
    return SentimentModel(vectorizer, model, features)


def word_sentiments(trained: SentimentModel) -> pd.DataFrame:
    """単語と、その係数（感情スコア）の対応表。

    ロジスティック回帰の係数がそのまま「この単語が出ると肯定に傾く度合い」になる。
    出現回数を特徴量にしているからこそ、係数を単語の重みとして読める。
    """
    return pd.DataFrame(
        {"word": trained.vocabulary, "weight": trained.model.coef_[0]}
    )


def most_positive_words(trained: SentimentModel, count: int = 10) -> pd.DataFrame:
    """係数が大きい順に単語を返す。"""
    return word_sentiments(trained).sort_values("weight", ascending=False).head(count)


def most_negative_words(trained: SentimentModel, count: int = 10) -> pd.DataFrame:
    """係数が小さい順に単語を返す。"""
    return word_sentiments(trained).sort_values("weight").head(count)


def predicted_probabilities(trained: SentimentModel) -> NDArray[np.float64]:
    """各レビューが肯定である予測確率。"""
    return trained.model.predict_proba(trained.features)[:, 1]


def extreme_review_indices(trained: SentimentModel) -> tuple[int, int]:
    """もっとも肯定的・否定的と判定されたレビューの行番号を返す。"""
    probabilities = predicted_probabilities(trained)
    return int(np.argmax(probabilities)), int(np.argmin(probabilities))
