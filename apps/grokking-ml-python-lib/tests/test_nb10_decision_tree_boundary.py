"""原著ノートブック #10 の再現テスト。

原著は図を 3 枚見せるだけで、数値の出力は正解率しかない。
ここでは **決定境界を格子上の予測ラベルとして取り出し**、
「軸に平行な長方形になる」という決定木の性質までテストにしている。

ノートブック #09 と違い、このデータには同点の分割が無いので **木は決定的** になる。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb10_decision_tree_boundary import (
    DATASET,
    PLOT_STEP,
    boundary_columns,
    decision_grid,
    features,
    fit,
    labels,
    split_conditions,
)


def test_データセットは12点で半々に分かれる() -> None:
    assert len(DATASET) == 12
    assert DATASET["y"].tolist() == [0] * 6 + [1] * 6


def test_ジニの木は全問正解する() -> None:
    # 原著の出力: decision_tree.score(features, labels) -> 1.0
    assert fit().score(features(), labels()) == 1.0


def test_エントロピーの木も全問正解する() -> None:
    # 原著の出力: decision_tree_entropy.score(features, labels) -> 1.0
    assert fit(criterion="entropy").score(features(), labels()) == 1.0


def test_ジニとエントロピーは同じ木になる() -> None:
    # 分割の候補に同点が無いので、どちらの指標でも同じ順序で選ばれる。
    # #09 では同点があって木の形が変わったのと対照的
    assert split_conditions(fit()) == split_conditions(fit(criterion="entropy"))


def test_木は3つの条件で分割する() -> None:
    # 根で x_0 <= 5.0、左で x_1 <= 8.0、右で x_1 <= 2.5
    assert split_conditions(fit()) == [
        ("x_0", 5.0),
        ("x_1", 8.0),
        ("x_1", 2.5),
    ]


def test_木は同点が無いので種によらず同じ() -> None:
    # #09 は種を変えると木の形が変わった。こちらは変わらない
    structures = {tuple(split_conditions(fit(random_state=seed))) for seed in range(20)}

    assert len(structures) == 1


def test_深さ1の木は1本の直線になる() -> None:
    # 原著の「1 本の縦線または横線」。分割は 1 つだけ
    shallow = fit(max_depth=1)

    assert split_conditions(shallow) == [("x_0", 5.0)]


def test_深さ1の木は12点中10点しか当てられない() -> None:
    # 直線 1 本では 2 点を取り違える
    shallow = fit(max_depth=1)

    assert shallow.score(features(), labels()) == pytest.approx(10 / 12)


def test_格子は原著と同じ大きさになる() -> None:
    # x は 0 から 10 まで、y は 0 から 11 まで、刻みは 0.2。
    # arange は終端を含まないので 50 × 55 になる
    grid = decision_grid(fit())

    assert PLOT_STEP == 0.2
    assert grid.x_values[0] == pytest.approx(0.0)
    assert grid.y_values[0] == pytest.approx(0.0)
    assert grid.shape == (55, 50)


def test_格子の予測は0か1しかない() -> None:
    grid = decision_grid(fit())

    assert set(np.unique(grid.predictions)) == {0, 1}


def test_境界は軸に平行になる() -> None:
    # 決定木の境界は長方形の集まりなので、予測が変わる x 座標は
    # 分割しきい値の直後だけに限られる
    grid = decision_grid(fit())
    changes = sorted(boundary_columns(grid))

    # x_0 <= 5.0 の境界。刻み 0.2 の格子なので 5.0 の次の点で変わる
    assert changes == pytest.approx([5.2])


def test_深さ1の境界は1本だけ() -> None:
    grid = decision_grid(fit(max_depth=1))
    changes = sorted(boundary_columns(grid))

    assert changes == pytest.approx([5.2])


def test_深さ1の境界は縦線なので全行で同じ() -> None:
    # 1 本の縦線なので、どの行も同じパターンになる
    grid = decision_grid(fit(max_depth=1))

    assert all(np.array_equal(row, grid.predictions[0]) for row in grid.predictions)


def test_深い木の境界は行によって変わる() -> None:
    # x_1 での分割が入るので、行ごとにパターンが違う
    grid = decision_grid(fit())
    patterns = {tuple(row) for row in grid.predictions}

    assert len(patterns) > 1


def test_学習点はすべて正しく塗り分けられる() -> None:
    # 格子の話とは別に、元の 12 点が正しく予測できることを直接確かめる
    model = fit()
    predictions = model.predict(features())

    assert predictions.tolist() == labels().tolist()
