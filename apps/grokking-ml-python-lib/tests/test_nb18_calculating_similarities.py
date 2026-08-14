"""原著ノートブック #18 の再現テスト。

類似度行列も SVM の係数も、**原著の出力と桁まで一致する**。
乱数が絡まないので、ここは完全に固定できる。
"""

import math

import numpy as np
import pytest

from grokking_ml_lib.nb18_calculating_similarities import (
    dataset,
    features,
    fit,
    similarity,
    similarity_matrix,
    svm_rbf_prediction,
    training_predictions,
    with_similarities,
)


@pytest.fixture(scope="module")
def data():
    return dataset()


def test_データセットは7点(data) -> None:
    # 原点と、その上下左右と斜め 2 点
    assert len(data) == 7
    assert list(data["x1"]) == [0, -1, 0, 0, 1, -1, 1]
    assert list(data["x2"]) == [0, 0, -1, 1, 0, 1, -1]
    assert list(data["y"]) == [0, 0, 0, 1, 1, 1, 1]


def test_同じ点の類似度は1() -> None:
    # exp(0) = 1
    assert similarity([0, 0], [0, 0]) == 1.0
    assert similarity([3, -2], [3, -2]) == 1.0


def test_類似度は距離の2乗で決まる() -> None:
    # exp(-||x-y||^2)。距離 1 なら exp(-1)
    assert similarity([0, 0], [1, 0]) == pytest.approx(math.exp(-1), rel=1e-15)
    # 距離 √2 なら exp(-2)
    assert similarity([0, 0], [1, 1]) == pytest.approx(math.exp(-2), rel=1e-15)
    # 原著のセルに残っている出力 1.522997974471263e-08 は (0,0) と (3,3) の類似度
    assert similarity([0, 0], [3, 3]) == pytest.approx(1.522997974471263e-08, rel=1e-15)


def test_類似度は対称(data) -> None:
    matrix = similarity_matrix(data)

    assert np.allclose(matrix, matrix.T, rtol=0, atol=0)


def test_対角成分はすべて1(data) -> None:
    # 自分自身との類似度
    assert list(np.diag(similarity_matrix(data))) == [1.0] * 7


def test_類似度行列は原著の表と一致する(data) -> None:
    matrix = similarity_matrix(data)

    # 原著の表の 1 行目: 1.000000 0.367879 0.367879 0.367879 0.367879 0.135335 ...
    assert matrix[0][1] == pytest.approx(0.36787944117144233, rel=1e-15)
    assert matrix[0][5] == pytest.approx(0.1353352832366127, rel=1e-15)
    # 2 行目の Sim4: 0.018316（距離の 2 乗が 4）
    assert matrix[1][4] == pytest.approx(0.01831563888873418, rel=1e-15)
    # 3 行目の Sim5: 0.006738（距離の 2 乗が 5）
    assert matrix[2][5] == pytest.approx(0.006737946999085467, rel=1e-15)


def test_特徴量は9列になる(data) -> None:
    # 元の x1, x2 に類似度 7 列を足す
    assert list(features(data).columns) == [
        "x1",
        "x2",
        "Sim0",
        "Sim1",
        "Sim2",
        "Sim3",
        "Sim4",
        "Sim5",
        "Sim6",
    ]


def test_ラベルの列は特徴量から落とす(data) -> None:
    assert "y" in with_similarities(data).columns
    assert "y" not in features(data).columns


def test_SVMの係数は原著と一致する(data) -> None:
    # 原著の svm.coef_
    #   [ 0.67476187, 0.67482825, -1.09720887, -0.64636729, -0.64708568,
    #     0.01538266, 0.01603589, 0.67770347, 0.6776795 ]
    expected = [
        0.67476187,
        0.67482825,
        -1.09720887,
        -0.64636729,
        -0.64708568,
        0.01538266,
        0.01603589,
        0.67770347,
        0.6776795,
    ]

    assert list(fit(data).coef_[0]) == pytest.approx(expected, abs=5e-9)


def test_線形カーネルを使っている(data) -> None:
    # カーネルは特徴量として展開済みなので、SVM 側は線形でよい
    assert fit(data).kernel == "linear"


def test_手書きの予測式は原著と桁まで一致する(data) -> None:
    # 原著の出力をそのまま並べたもの
    expected = [
        -0.7293294335267746,
        -0.9749464141121803,
        -0.9749464141121804,
        0.9884223081103513,
        0.9884223081103514,
        0.8650001793912898,
        0.8650001793912898,
    ]

    assert training_predictions(data) == pytest.approx(expected, rel=1e-15)


def test_予測の符号は正解ラベルと合う(data) -> None:
    # 原著のコメント「ラベルが 1 なら正、0 なら負になるはず」
    for value, label in zip(training_predictions(data), data["y"], strict=True):
        assert (value > 0) == (label == 1)


def test_対称な2点は同じ値になる(data) -> None:
    # (0,1) と (1,0)、(-1,1) と (1,-1) は x1 と x2 を入れ替えた関係。
    # データ全体もその入れ替えで不変なので、予測値も一致する。
    #
    # ただし **ビット単位で同じにはならない**。np.dot は足す順序が変わりうるので、
    # 数学的に等しくても最後の 1 桁がずれる。実際 CI（Linux）では
    # 0.8650001793912899 と 0.8650001793912898 になった（macOS では一致した）。
    # 対称性は 15 桁の一致で確かめる
    assert svm_rbf_prediction(data, [0, 1]) == pytest.approx(
        svm_rbf_prediction(data, [1, 0]), rel=1e-15
    )
    assert svm_rbf_prediction(data, [-1, 1]) == pytest.approx(
        svm_rbf_prediction(data, [1, -1]), rel=1e-15
    )


def test_遠く離れた点の予測は0に近づく(data) -> None:
    # RBF は距離とともに指数的に減る。データから離れると判断できなくなる
    assert abs(svm_rbf_prediction(data, [10, 10])) < 1e-30
