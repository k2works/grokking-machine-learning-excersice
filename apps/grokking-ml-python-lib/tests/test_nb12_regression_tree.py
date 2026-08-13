"""原著ノートブック #12 の再現テスト。

原著が手で書き下した 9 通りの分割の重み付き MSE を、すべて再現する。
scikit-learn が選ぶ分割も、その手計算の最小値と一致する。
"""

import math

import pytest

from grokking_ml_lib.nb12_regression_tree import (
    FEATURES,
    LABELS,
    best_split,
    fit,
    leaf_values,
    split_conditions,
    split_mses,
)


def test_データセットは8点() -> None:
    assert FEATURES == [[10], [20], [30], [40], [50], [60], [70], [80]]
    assert LABELS == [7, 5, 7, 1, 2, 1, 5, 4]


def test_全体の平均は4() -> None:
    # 原著の出力: np.array([7,5,7,1,2,1,5,4]).mean() -> 4.0
    assert sum(LABELS) / len(LABELS) == pytest.approx(4.0)


def test_分割は9通り試される() -> None:
    # 要素は 8 個だが range(0, 9) なので 9 通り。
    # 左が空の場合と右が空の場合の両方が含まれる
    splits = split_mses()

    assert len(splits) == 9
    assert splits[0].left == []
    assert splits[8].right == []


def test_左が空のときの平均はNaNになる() -> None:
    # 原著も NumPy の RuntimeWarning つきで nan を出している
    first = split_mses()[0]

    assert math.isnan(first.left_mean)
    assert first.right_mean == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("index", "left_mean", "right_mean", "weighted_mse"),
    [
        (1, 7.0, 3.5714285714285716, 3.9642857142857144),
        (2, 6.0, 3.3333333333333335, 3.916666666666667),
        (3, 6.333333333333333, 2.6, 1.9833333333333334),
        (4, 5.0, 3.0, 4.25),
        (5, 4.4, 3.3333333333333335, 4.983333333333333),
        (6, 3.8333333333333335, 4.5, 5.166666666666667),
        (7, 4.0, 4.0, 5.25),
    ],
)
def test_各分割の重み付きMSEは原著と同じ(
    index: int, left_mean: float, right_mean: float, weighted_mse: float
) -> None:
    # 原著のセル出力をそのまま期待値にしている
    split = split_mses()[index]

    assert split.left_mean == pytest.approx(left_mean)
    assert split.right_mean == pytest.approx(right_mean)
    assert split.weighted_mse == pytest.approx(weighted_mse)


def test_分割しない場合のMSEは全体の分散になる() -> None:
    # 原著の出力: 5.25。左が空なので右がそのまま全体になる
    assert split_mses()[0].weighted_mse == pytest.approx(5.25)


def test_最良の分割は3番目() -> None:
    # 原著の一覧で 1.9833 がもっとも小さい
    best = best_split()

    assert best.index == 3
    assert best.left == [7, 5, 7]
    assert best.right == [1, 2, 1, 5, 4]
    assert best.weighted_mse == pytest.approx(1.9833333333333334)


def test_scikit_learnも同じ位置で分割する() -> None:
    # 手計算の最小値は 3 番目、つまり 30 歳と 40 歳の間。中点の 35.0 が選ばれる
    conditions = split_conditions(fit())

    assert conditions[0] == pytest.approx(35.0)


def test_深さ2の木は3回分割する() -> None:
    # 根で 35.0、左で 15.0、右で 65.0
    assert split_conditions(fit()) == pytest.approx([35.0, 15.0, 65.0])


def test_葉は平均値を返す() -> None:
    # 分類木は多数決だったが、回帰木は葉に落ちた点の平均を返す
    # 10 歳 -> 7、20〜30 歳 -> 6、40〜60 歳 -> 1.333、70〜80 歳 -> 4.5
    assert leaf_values(fit()) == pytest.approx([7.0, 6.0, 4 / 3, 4.5])


def test_予測は階段状になる() -> None:
    # 同じ葉に落ちる点は同じ値を返す。回帰木の予測は連続にならない
    predictions = fit().predict(FEATURES).tolist()

    assert predictions == pytest.approx([7.0, 6.0, 6.0, 4 / 3, 4 / 3, 4 / 3, 4.5, 4.5])


def test_葉の値は元のラベルの平均になっている() -> None:
    # 40, 50, 60 歳のラベルは 1, 2, 1。その平均 4/3 が葉の値
    assert (1 + 2 + 1) / 3 == pytest.approx(4 / 3)
    assert leaf_values(fit())[2] == pytest.approx(4 / 3)


def test_根のMSEは全体の分散と一致する() -> None:
    # scikit-learn の tree_.impurity[0] は MSE。手計算の 5.25 と同じ
    assert fit().tree_.impurity[0] == pytest.approx(5.25)
