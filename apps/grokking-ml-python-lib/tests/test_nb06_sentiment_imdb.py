"""原著ノートブック #06 の再現テスト。

IMDB のデータセットは約 63 MB あり、リポジトリに含めていない。
**テストからは自動ダウンロードしない**（CI で毎回 63 MB を落とすのは無駄なため）。
未取得ならスキップし、取得済みなら原著との一致を確認する。

    cd apps/grokking-ml-python-lib
    uv run python -c "from grokking_ml_lib.datasets import dataset_path; dataset_path('IMDB_Dataset.csv')"

語彙の作り方そのものは、小さなコーパスで常に検証している。
"""

import numpy as np
import pandas as pd
import pytest

from grokking_ml_lib.datasets import datasets_dir
from grokking_ml_lib.nb06_sentiment_imdb import (
    MAX_FEATURES,
    build_vectorizer,
    extreme_review_indices,
    fit,
    load_reviews,
    most_negative_words,
    most_positive_words,
    word_sentiments,
)

#: 語彙の作り方を確かめるための小さなコーパス
TINY_REVIEWS = pd.DataFrame(
    {
        "review": [
            "This movie was wonderful and the acting was superb",
            "A dreadful waste of time, truly awful acting",
            "Wonderful direction and a superb cast",
            "Awful script and a dreadful waste",
        ],
        "sentiment": [1, 0, 1, 0],
    }
)


def _imdb_available() -> bool:
    return (datasets_dir() / "IMDB_Dataset.csv").exists()


imdb_required = pytest.mark.skipif(
    not _imdb_available(),
    reason="IMDB_Dataset.csv（約 63 MB）が未取得です。datasets の README を参照してください",
)


@pytest.fixture(scope="module")
def imdb():
    """5 万件の読み込みと学習は数秒かかるので、モジュールで 1 度だけ行う。"""
    return load_reviews()


@pytest.fixture(scope="module")
def trained(imdb):
    return fit(imdb)


def test_ベクトル化は2文字以上の単語だけを拾う() -> None:
    # 既定のトークン正規表現は (?u)\b\w\w+\b なので 1 文字の語は落ちる
    vectorizer = build_vectorizer(max_features=50)
    vectorizer.fit(["a wonderful movie", "I saw it"])

    assert "wonderful" in vectorizer.get_feature_names_out()
    assert "a" not in vectorizer.get_feature_names_out()


def test_ベクトル化は小文字に揃える() -> None:
    vectorizer = build_vectorizer(max_features=50)
    vectorizer.fit(["Wonderful WONDERFUL wonderful"])

    assert vectorizer.get_feature_names_out().tolist() == ["wonderful"]


def test_ストップワードは語彙から除かれる() -> None:
    # 'the' や 'and' は scikit-learn 内蔵の英語ストップワード 318 語に含まれる
    vectorizer = build_vectorizer(max_features=50)
    vectorizer.fit(["the movie and the acting"])
    vocabulary = vectorizer.get_feature_names_out().tolist()

    assert "movie" in vocabulary
    assert "acting" in vocabulary
    assert "the" not in vocabulary
    assert "and" not in vocabulary


def test_max_featuresは出現回数の上位を採る() -> None:
    vectorizer = build_vectorizer(max_features=2)
    # rare は 1 回、common は 3 回、middle は 2 回
    vectorizer.fit(["common middle rare", "common middle", "common"])

    assert sorted(vectorizer.get_feature_names_out().tolist()) == ["common", "middle"]


def test_小さなコーパスでも単語の重みが感情を反映する() -> None:
    trained = fit(TINY_REVIEWS, max_features=20)
    weights = dict(zip(word_sentiments(trained)["word"], word_sentiments(trained)["weight"]))

    # 肯定的なレビューにだけ出る語は正、否定的なレビューにだけ出る語は負
    assert weights["wonderful"] > 0
    assert weights["superb"] > 0
    assert weights["dreadful"] < 0
    assert weights["waste"] < 0
    # 両方に出る語は 0 に近い
    assert abs(weights["acting"]) < abs(weights["wonderful"])


def test_係数の数は語彙の数と一致する() -> None:
    trained = fit(TINY_REVIEWS, max_features=20)

    assert len(trained.vocabulary) == trained.model.coef_[0].shape[0]
    assert len(word_sentiments(trained)) == len(trained.vocabulary)


@imdb_required
def test_データセットは5万件で0と1のラベルになる(imdb) -> None:
    movies = imdb

    assert len(movies) == 50000
    assert sorted(movies["sentiment"].unique().tolist()) == [0, 1]
    # 肯定と否定がちょうど半々
    assert movies["sentiment"].sum() == 25000


@imdb_required
def test_語彙は2000語になる(trained) -> None:

    assert trained.features.shape == (50000, MAX_FEATURES)
    assert len(trained.vocabulary) == MAX_FEATURES


@imdb_required
def test_もっとも肯定的な10語は原著と同じ(trained) -> None:
    # 原著の出力（上位から順に）
    #   wonderfully funniest gem brilliantly subtle superb excellent finest
    #   delightful underrated

    assert most_positive_words(trained)["word"].tolist() == [
        "wonderfully",
        "funniest",
        "gem",
        "brilliantly",
        "subtle",
        "superb",
        "excellent",
        "finest",
        "delightful",
        "underrated",
    ]


@imdb_required
def test_もっとも否定的な10語は原著と同じ集合になる(trained) -> None:
    # 原著の出力（上位から順に）
    #   waste disappointment worst unfunny dreadful laughable awful redeeming
    #   poorly tedious
    # 係数が -1.48 から -1.38 の狭い範囲に固まっているため、lbfgs の収束差で
    # 4 位以下の順序が入れ替わる。集合としては一致する

    assert sorted(most_negative_words(trained)["word"].tolist()) == sorted(
        [
            "waste",
            "disappointment",
            "worst",
            "unfunny",
            "dreadful",
            "laughable",
            "awful",
            "redeeming",
            "poorly",
            "tedious",
        ]
    )
    # 上位 3 語は順序まで一致する
    assert most_negative_words(trained)["word"].tolist()[:3] == [
        "waste",
        "disappointment",
        "worst",
    ]


@imdb_required
def test_極端なレビューの行番号は原著と同じ(trained) -> None:
    # 原著の出力
    #   Most Positive Review: 42946（ジブリ「天空の城ラピュタ」英語吹替のレビュー）
    #   Most Negative Review: 13452（ゾンビ映画 Zombi 3 のレビュー）
    positive, negative = extreme_review_indices(trained)

    assert positive == 42946
    assert negative == 13452


@imdb_required
def test_もっとも否定的なレビューの確率は極端に小さい(trained) -> None:
    # 原著の出力: 2.352703e-17
    probabilities = trained.model.predict_proba(trained.features)[:, 1]

    assert probabilities.min() == pytest.approx(2.35e-17, rel=0.5)
    assert np.isclose(probabilities.max(), 1.0)
