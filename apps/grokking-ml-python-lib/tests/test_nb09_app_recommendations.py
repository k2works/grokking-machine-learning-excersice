"""原著ノートブック #09 の再現テスト。

正解率・根の分割・ジニ不純度・節の数は再現できる。
一方 **木の形そのものは再現できない**。`DecisionTreeClassifier` は同点の分割候補から
無作為に 1 つ選ぶので、実行のたびに違う（しかし等価な）木ができる。
その様子も数えてテストにしてある。
"""

from collections import Counter

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from grokking_ml_lib.nb09_app_recommendations import (
    APP_DATASET,
    APP_DATASET_ONE_HOT,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    fit_categorical,
    fit_numeric,
    split_features,
    summarize,
)


def test_元データは6人ぶん() -> None:
    assert len(APP_DATASET) == 6
    assert APP_DATASET["Age"].tolist() == [15, 25, 32, 35, 12, 14]
    assert APP_DATASET["App"].nunique() == 3


def test_one_hot版は原著が手で書き下した表と同じ() -> None:
    # 原著は get_dummies を使わず、7 列を直接書いている
    assert list(APP_DATASET_ONE_HOT.columns) == [
        "Platform_iPhone",
        "Platform_Android",
        "Age_Young",
        "Age_Adult",
        "App_Atom_Count",
        "App_Beehive_Finder",
        "App_Check_Mate_Mate",
    ]
    # 各行でちょうど 1 つのアプリが選ばれている
    apps = APP_DATASET_ONE_HOT[
        ["App_Atom_Count", "App_Beehive_Finder", "App_Check_Mate_Mate"]
    ]
    assert apps.sum(axis=1).tolist() == [1] * 6


def test_カテゴリ版は全問正解する() -> None:
    # 原著の出力: dt.score(X, y) -> 1.0
    summary = summarize(fit_categorical(random_state=0), CATEGORICAL_FEATURES)

    assert summary.score == 1.0


def test_カテゴリ版の根は年齢で分割しジニは0_407() -> None:
    # 原著の出力: X[3] <= 0.5 / gini = 0.407 / samples = 6
    # X[3] は Age_Young。年齢が最初の分割に選ばれる
    summary = summarize(fit_categorical(random_state=0), CATEGORICAL_FEATURES)

    assert summary.root_feature in {"Age_Young", "Age_Adult"}
    assert summary.root_threshold == pytest.approx(0.5)
    assert summary.root_gini == pytest.approx(0.407, abs=5e-4)
    assert summary.node_count == 5


def test_数値版は全問正解する() -> None:
    # 原著の出力: app_model.score(features, labels) -> 1.0
    summary = summarize(fit_numeric(random_state=0), NUMERIC_FEATURES)

    assert summary.score == 1.0


def test_数値版の根は年齢20歳で分割しジニは0_611() -> None:
    # 原著の出力: X[0] <= 20.0 / gini = 0.611 / samples = 6
    # 15 と 25 の間ではなく、15 と 25 の中点 20 が選ばれている
    summary = summarize(fit_numeric(random_state=0), NUMERIC_FEATURES)

    assert summary.root_feature == "Age"
    assert summary.root_threshold == pytest.approx(20.0)
    assert summary.root_gini == pytest.approx(0.611, abs=5e-4)
    assert summary.node_count == 5


def test_しきい値は隣り合う年齢の中点になる() -> None:
    # 20 歳の人はいない。15（Atom Count）と 25（Check Mate Mate）の中点である
    ages = sorted(APP_DATASET["Age"])
    summary = summarize(fit_numeric(random_state=0), NUMERIC_FEATURES)

    assert 20.0 not in ages
    assert summary.root_threshold == pytest.approx((15 + 25) / 2)


def test_カテゴリ版のジニは数値版より小さい() -> None:
    # 原著の 0.407 と 0.611。多出力分類のジニは 3 列の平均なので値が下がる
    categorical = summarize(fit_categorical(random_state=0), CATEGORICAL_FEATURES)
    numeric = summarize(fit_numeric(random_state=0), NUMERIC_FEATURES)

    assert categorical.root_gini < numeric.root_gini


def test_数値版の木の形は2通りに割れる() -> None:
    # Platform_iPhone と Platform_Android は互いに裏返しなので、
    # 2 段目の分割はどちらを選んでも同じ結果になる。scikit-learn は
    # 同点の候補から無作為に選ぶため、種を変えると木の形が変わる
    structures = Counter(
        tuple(split_features(fit_numeric(random_state=seed), NUMERIC_FEATURES))
        for seed in range(100)
    )

    assert set(structures) == {
        ("Age", "Platform_iPhone"),
        ("Age", "Platform_Android"),
    }
    # どちらもおおむね半々に現れる
    for count in structures.values():
        assert 30 < count < 70


def test_カテゴリ版の木の形は4通りに割れる() -> None:
    # 年齢も端末も裏返しの列を持つので、組み合わせで 4 通りになる
    structures = Counter(
        tuple(split_features(fit_categorical(random_state=seed), CATEGORICAL_FEATURES))
        for seed in range(100)
    )

    assert len(structures) == 4
    for count in structures.values():
        assert 10 < count < 40


def test_木の形が変わっても正解率と節の数は変わらない() -> None:
    # 再現できないのは形だけで、性能は変わらない
    summaries = [
        summarize(fit_numeric(random_state=seed), NUMERIC_FEATURES) for seed in range(20)
    ]

    assert {summary.score for summary in summaries} == {1.0}
    assert {summary.node_count for summary in summaries} == {5}
    assert {round(summary.root_gini, 3) for summary in summaries} == {0.611}


def test_種を固定すれば同じ木になる() -> None:
    # 再現性が要るなら random_state を渡す。原著は渡していない
    first = split_features(fit_numeric(random_state=7), NUMERIC_FEATURES)
    second = split_features(fit_numeric(random_state=7), NUMERIC_FEATURES)

    assert first == second


def test_多出力分類では目的変数が3列になる() -> None:
    model = fit_categorical(random_state=0)

    assert model.n_outputs_ == 3
    assert isinstance(model.classes_, list)
    assert len(model.classes_) == 3


def test_未知の組み合わせも予測できる() -> None:
    # 学習データに無い「Android の 20 歳」を投げてみる。
    # 列名つきで渡さないと scikit-learn が警告を出す
    model: DecisionTreeClassifier = fit_numeric(random_state=0)
    unknown = pd.DataFrame([[20, 0, 1]], columns=NUMERIC_FEATURES)
    prediction = model.predict(unknown)

    assert prediction[0] in set(APP_DATASET["App"])
