"""原著ノートブック #17 `Chapter_11_Support_Vector_Machines/Building_the_datasets.ipynb`。

第 11 章（SVM）で使う 3 つのデータセットを **乱数で作る** 回である。

原著は種を固定していないので、**生成そのものは再現できない**。
できるのは 2 つ。

1. 同じ規則で生成器を書き、規則どおりの点が出ることを確かめる
2. 原著が生成して配布した CSV を読み、**その規則で作られたことを検証する**

2 つ目が本命である。配布 CSV の各行が規則に合うかを数えれば、
「先頭 n 件は規則どおり、末尾は乱数ラベルのノイズ」という構造が見える。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from grokking_ml_lib.datasets import load_csv

#: 座標の範囲。原著の `6 * random.random() - 3`
COORD_SCALE = 6.0
COORD_OFFSET = -3.0


@dataclass(frozen=True)
class Spec:
    """1 つのデータセットの作り方。"""

    name: str
    #: 規則どおりに作る点の数
    points: int
    #: ラベルを乱数にする点（ノイズ）の数
    noise: int

    @property
    def total(self) -> int:
        return self.points + self.noise


LINEAR = Spec("linear", 50, 10)
ONE_CIRCLE = Spec("one_circle", 100, 10)
TWO_CIRCLES = Spec("two_circles", 200, 20)

SPECS = (LINEAR, ONE_CIRCLE, TWO_CIRCLES)


def linear_rule(x: float, y: float) -> int:
    """直線 `x + y = 0.5` の上側なら 1。"""
    return int(x + y > 0.5)


def one_circle_rule(x: float, y: float) -> int:
    """原点を中心とする半径 √2.8 の円の内側なら 1。"""
    return int(x**2 + y**2 < 2.8)


def two_circles_rule(x: float, y: float) -> int:
    """(1, 0) と (-1, 0) を中心とする 2 つの円の **どちらか** の内側なら 1。"""
    return int(((x - 1) ** 2 + y**2 < 2) or ((x + 1) ** 2 + y**2 < 2))


RULES = {
    "linear": linear_rule,
    "one_circle": one_circle_rule,
    "two_circles": two_circles_rule,
}


def generate(spec: Spec, seed: int | None = None) -> pd.DataFrame:
    """原著と同じ手順でデータセットを作る。

    原著は種を固定していないので実行のたびに違うものが出る。
    ここでは `seed` を渡せるようにして、テストで扱えるようにした。
    **原著が配布している CSV とは一致しない**（一致しようがない）。
    """
    generator = random.Random(seed)
    rule = RULES[spec.name]
    rows = []

    for _ in range(spec.points):
        x = COORD_SCALE * generator.random() + COORD_OFFSET
        y = COORD_SCALE * generator.random() + COORD_OFFSET
        rows.append([x, y, rule(x, y)])

    for _ in range(spec.noise):
        x = COORD_SCALE * generator.random() + COORD_OFFSET
        y = COORD_SCALE * generator.random() + COORD_OFFSET
        rows.append([x, y, generator.randint(0, 1)])

    return pd.DataFrame(rows, columns=["x_1", "x_2", "y"])


def load(spec: Spec) -> pd.DataFrame:
    """原著が生成して配布している CSV を読む。

    先頭に無名の添字列が付いているので落とす。
    """
    return load_csv(f"{spec.name}.csv").drop(columns=["Unnamed: 0"])


def rule_violations(spec: Spec, data: pd.DataFrame) -> list[int]:
    """規則とラベルが食い違う行の添字。

    ノイズとして入れた点は、規則と無関係にラベルを振っているので、
    **約半分がここに現れる**。規則どおりに作った先頭の点は 1 つも現れない。
    """
    rule = RULES[spec.name]
    return [
        index
        for index, row in enumerate(data.itertuples())
        if rule(row.x_1, row.x_2) != row.y
    ]
