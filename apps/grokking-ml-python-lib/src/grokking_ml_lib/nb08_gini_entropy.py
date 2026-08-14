"""原著ノートブック #08 `Chapter_09_Decision_Trees/Gini_entropy_calculations.ipynb`。

決定木がどこで分割するかを決めるための 2 つの不純度、ジニ不純度とエントロピーを
手で計算する。ライブラリは NumPy しか使わない小さな回で、
**決定木の中身を理解するための下準備** にあたる。

分割位置を 1 つずつずらしながら重み付き不純度を見ると、
`['A', 'A', 'A'] | ['C', 'B', 'C']` でどちらの指標も最小になる。
決定木はこれを全特徴量・全分割点について繰り返している。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 原著が使う 6 要素
ELEMENTS = ["A", "A", "A", "C", "B", "C"]


def counts(elements: list[str]) -> list[int]:
    """要素ごとの個数を、**初めて現れた順** に返す。

    原著は辞書に数えてから `[classes[e] for e in classes]` で取り出している。
    Python 3.7 以降の辞書は挿入順を保つので、結果は初出順になる。
    `['A', 'A', 'A', 'C', 'B', 'C']` なら A=3, C=2, B=1 で `[3, 2, 1]`。
    """
    classes: dict[str, int] = {}
    for element in elements:
        classes[element] = classes.get(element, 0) + 1
    return list(classes.values())


def gini(elements: list[str]) -> float:
    """ジニ不純度。1 から「同じクラスを 2 回続けて引く確率」を引いたもの。

    空のリストに対しては 1 を返す。原著は特別扱いしておらず、
    `1 - sum([])` がそのまま 1 になる。重み付けのときは要素数 0 が掛かるので
    結果に影響しない。
    """
    class_counts = counts(elements)
    n = sum(class_counts)
    return 1 - sum(count**2 / n**2 for count in class_counts)


def entropy(elements: list[str]) -> float:
    """情報エントロピー。原著はこちらだけ空のリストを明示的に 0 にしている。

    `log2(0)` が発散するので、空を通すと NumPy の警告が出る。
    ジニ不純度が特別扱い不要だったのと対照的である。
    """
    if len(elements) == 0:
        return 0
    class_counts = counts(elements)
    n = sum(class_counts)
    proportions = 1 / n * np.array(class_counts)
    return float(-np.dot(np.log2(proportions), proportions))


@dataclass
class SplitImpurity:
    """ある分割位置での、左右と重み付き不純度。"""

    index: int
    left: list[str]
    right: list[str]
    weighted_gini: float
    weighted_entropy: float


def _weighted(
    impurity, left: list[str], right: list[str], total: int
) -> float:
    """左右の不純度を、要素数で重み付けして平均する。"""
    return 1 / total * (impurity(left) * len(left) + impurity(right) * len(right))


def split_impurities(elements: list[str] = ELEMENTS) -> list[SplitImpurity]:
    """先頭から順に分割位置をずらし、それぞれの重み付き不純度を求める。

    原著は 0 から `len(elements) - 1` まで回している。**右端では分割しない**
    ので、「左が全部・右が空」の場合は出てこない。
    """
    total = len(elements)
    results = []
    for index in range(total):
        left, right = elements[:index], elements[index:]
        results.append(
            SplitImpurity(
                index=index,
                left=left,
                right=right,
                weighted_gini=_weighted(gini, left, right, total),
                weighted_entropy=_weighted(entropy, left, right, total),
            )
        )
    return results


def best_split(elements: list[str] = ELEMENTS) -> SplitImpurity:
    """重み付きジニ不純度がもっとも小さい分割を返す。"""
    return min(split_impurities(elements), key=lambda split: split.weighted_gini)
