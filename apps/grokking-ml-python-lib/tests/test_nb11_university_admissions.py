"""原著ノートブック #11 の再現テスト。

原著の出力（正解率 1.0 と 0.885、5 件の予測、2 人の出願者の合否）を
すべて再現できる。小さい木は制限が効いて **構造まで決定的** になった。
"""

import pytest

from grokking_ml_lib.nb11_university_admissions import (
    ADMISSION_THRESHOLD,
    EXAM_FEATURES,
    features,
    fit_exams,
    fit_full,
    fit_smaller,
    labels,
    load_data,
    predict_applicant,
    split_conditions,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def full(data):
    return fit_full(data)


@pytest.fixture(scope="module")
def smaller(data):
    return fit_smaller(data)


def test_データセットは400件7特徴量になる(data) -> None:
    assert len(data) == 400
    assert list(features(data).columns) == [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
    ]


def test_合格ラベルは合格確率0_75で切る(data) -> None:
    # 原著は Chance of Admit >= 0.75 を合格とする
    assert ADMISSION_THRESHOLD == 0.75
    # 400 件中 180 件が合格。おおむね 45%
    assert int(labels(data).sum()) == 180
    # 元の列は落とされている
    assert "Chance of Admit" not in data.columns


def test_制限なしの木は訓練データを完全に覚える(data, full) -> None:
    # 原著の出力: dt.score(features, labels) -> 1.0
    assert full.score(features(data), labels(data)) == 1.0


def test_最初の5件の予測は原著と同じ(data, full) -> None:
    # 原著の出力: array([ True,  True, False,  True, False])
    assert full.predict(features(data)[0:5]).tolist() == [True, True, False, True, False]


def test_小さい木の正解率は原著と同じ(data, smaller) -> None:
    # 原著の出力: dt_smaller.score(features, labels) -> 0.885
    assert smaller.score(features(data), labels(data)) == pytest.approx(0.885)


def test_小さい木は制限のぶんだけ節が減る(data, full, smaller) -> None:
    # 制限なしの木は 115 節前後まで育つ。小さい木は 15 節で収まる
    assert full.tree_.node_count > 100
    assert smaller.tree_.node_count == 15


def test_小さい木は種によらず同じ構造になる(data) -> None:
    # 制限が効いて同点の分割が起きにくくなり、木が決定的になる
    structures = {
        tuple(split_conditions(fit_smaller(data, random_state=seed), list(features(data).columns)))
        for seed in range(10)
    }

    assert len(structures) == 1


def test_小さい木の根はCGPAで分割する(data, smaller) -> None:
    # 7 つの特徴量のうち、成績（CGPA）がもっとも効く
    conditions = split_conditions(smaller, list(features(data).columns))

    assert conditions[0][0] == "CGPA"


def test_CGPAが高い出願者は合格と予測される(data, smaller) -> None:
    # 原著の出力: dt_smaller.predict([[320, 110, 3, 4.0, 3.5, 8.9, 0]]) -> array([ True])
    assert predict_applicant(smaller, data, [320, 110, 3, 4.0, 3.5, 8.9, 0]) is True


def test_CGPAだけ下げると不合格に変わる(data, smaller) -> None:
    # 原著の出力: 8.9 を 8.0 にすると array([False])
    # 他の 6 項目は同じ。根が CGPA なので、ここだけで判定が反転する
    assert predict_applicant(smaller, data, [320, 110, 3, 4.0, 3.5, 8.0, 0]) is False


def test_2特徴量の木は深さを増やすほど訓練データに当たる(data) -> None:
    # 原著は図でしか見せていないが、過学習の進み方が数値で追える
    scores = {
        depth: fit_exams(data, depth).score(data[EXAM_FEATURES], labels(data))
        for depth in [1, 2, None]
    }

    assert scores[1] == pytest.approx(0.8525)
    assert scores[2] == pytest.approx(0.8625)
    assert scores[None] == pytest.approx(0.93)


def test_2特徴量では制限なしでも1_0にならない(data) -> None:
    # 7 特徴量なら 1.0 になったが、2 特徴量では同じ点で違うラベルの出願者がいて
    # 完全には分けられない
    unbounded = fit_exams(data, None)

    assert unbounded.score(data[EXAM_FEATURES], labels(data)) < 1.0


def test_2特徴量の木はどの深さでもGREで分割し始める(data) -> None:
    # GRE Score <= 319.5。TOEFL より GRE のほうが効く
    for depth in [1, 2, None]:
        conditions = split_conditions(fit_exams(data, depth), EXAM_FEATURES)

        assert conditions[0] == ("GRE Score", 319.5)


def test_制限なしの木は節が増えても正解率は上がりきらない(data) -> None:
    # 深さ 2 の 7 節から制限なしの 197 節へ。節が 28 倍になっても
    # 正解率は 0.8625 から 0.93 にしか上がらない
    shallow = fit_exams(data, 2)
    unbounded = fit_exams(data, None)

    assert shallow.tree_.node_count == 7
    assert unbounded.tree_.node_count > 150
