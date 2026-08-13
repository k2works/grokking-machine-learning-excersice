"""原著ノートブック #19 の再現テスト。

原著が印刷する 9 つの正解率が **すべて桁まで一致** する。
SVM は凸最適化で解が一意に決まるので、版が変わっても値が動かない。
"""

import pytest

from grokking_ml_lib.nb19_svm_kernels import fit, fit_and_score, labels, load


@pytest.mark.parametrize(
    ("name", "size"), [("linear", 60), ("one_circle", 110), ("two_circles", 220)]
)
def test_データセットは17で作ったもの(name, size) -> None:
    # [#17](nb17.md) が生成した CSV をそのまま使う
    assert len(load(name)) == size


def test_線形カーネルの正解率は原著と一致する() -> None:
    # 原著の出力: Accuracy: 0.9333333333333333
    assert fit_and_score("linear", kernel="linear") == pytest.approx(
        0.9333333333333333, rel=1e-15
    )


@pytest.mark.parametrize(
    ("c", "expected"), [(0.01, 0.8666666666666667), (100, 0.9166666666666666)]
)
def test_Cを変えると正解率が変わる(c, expected) -> None:
    # 原著の出力
    #   C = 0.01  -> 0.8666666666666667
    #   C = 100   -> 0.9166666666666666
    assert fit_and_score("linear", kernel="linear", C=c) == pytest.approx(
        expected, rel=1e-15
    )


def test_Cは既定値が一番良い() -> None:
    # C = 1（既定）が 0.933 で、0.01 と 100 のどちらより良い。
    # 小さすぎると正則化が強すぎ、大きすぎると外れ値に引きずられる
    default = fit_and_score("linear", kernel="linear")

    assert default > fit_and_score("linear", kernel="linear", C=0.01)
    assert default > fit_and_score("linear", kernel="linear", C=100)


@pytest.mark.parametrize(("degree", "expected"), [(2, 0.8909090909090909), (4, 0.9)])
def test_多項式カーネルの次数を変える(degree, expected) -> None:
    # 原著の出力
    #   Polynomial kernel of degree = 2 -> 0.8909090909090909
    #   Polynomial kernel of degree = 4 -> 0.9
    assert fit_and_score("one_circle", kernel="poly", degree=degree) == pytest.approx(
        expected, rel=1e-15
    )


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [
        (0.1, 0.8772727272727273),
        (1, 0.9045454545454545),
        (10, 0.9636363636363636),
        (100, 0.990909090909091),
    ],
)
def test_RBFのgammaを変える(gamma, expected) -> None:
    # 原著の出力: gamma を 0.1 -> 1 -> 10 -> 100 と上げると正解率も上がる
    assert fit_and_score("two_circles", kernel="rbf", gamma=gamma) == pytest.approx(
        expected, rel=1e-15
    )


def test_gammaを上げるほど正解率が単調に上がる() -> None:
    # 0.877 -> 0.905 -> 0.964 -> 0.991。
    # ただしこれは **学習データに対する** 正解率で、過学習が進んでいるだけ
    scores = [
        fit_and_score("two_circles", kernel="rbf", gamma=gamma)
        for gamma in (0.1, 1, 10, 100)
    ]

    assert scores == sorted(scores)


def test_gamma100は221件中2件しか外さない() -> None:
    # 0.990909... = 218/220。データにはノイズを 20 件混ぜてあるのに、
    # そのほとんどを覚えてしまっている
    data = load("two_circles")
    model = fit(data, kernel="rbf", gamma=100)
    wrong = (model.predict(data[["x_1", "x_2"]].to_numpy()) != labels(data)).sum()

    assert wrong == 2


def test_gamma100はデータの外側では何も答えられない() -> None:
    # RBF は局所的で、gamma が大きいほど影響範囲が狭い。
    # データの範囲（-3〜3）から離れると、決定関数はほぼ 0 になる
    data = load("two_circles")
    model = fit(data, kernel="rbf", gamma=100)

    assert abs(model.decision_function([[10.0, 10.0]])[0]) < 1.0
