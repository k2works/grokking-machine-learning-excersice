"""原著ノートブック #08 の再現テスト。

NumPy しか使わない小さな回なので、原著の数値をすべて再現できる。
"""

import pytest

from grokking_ml_lib.nb08_gini_entropy import (
    ELEMENTS,
    best_split,
    counts,
    entropy,
    gini,
    split_impurities,
)


def test_個数は初出順に並ぶ() -> None:
    # 原著の出力: [3, 2, 1]。A が 3、C が 2、B が 1 の順
    assert counts(ELEMENTS) == [3, 2, 1]


def test_ジニ不純度は原著と同じ() -> None:
    # 原著の出力: 0.6111111111111112
    assert gini(ELEMENTS) == pytest.approx(0.6111111111111112, rel=1e-15)


def test_エントロピーは原著と同じ() -> None:
    # 原著の出力: 1.4591479170272448
    assert entropy(ELEMENTS) == pytest.approx(1.4591479170272448, rel=1e-15)


def test_同じ要素だけなら不純度は0() -> None:
    assert gini(["A", "A", "A"]) == pytest.approx(0.0)
    assert entropy(["A", "A", "A"]) == pytest.approx(0.0)


def test_2クラスが半々ならジニは0_5でエントロピーは1() -> None:
    # 情報量 1 ビットぶん。コイン投げと同じ
    assert gini(["A", "B"]) == pytest.approx(0.5)
    assert entropy(["A", "B"]) == pytest.approx(1.0)


def test_空のリストの扱いは2つで違う() -> None:
    # 原著はエントロピーだけ明示的に 0 を返す。ジニは 1 - sum([]) で 1 になる
    assert gini([]) == 1
    assert entropy([]) == 0


@pytest.mark.parametrize(
    ("index", "expected_gini", "expected_entropy"),
    [
        (0, 0.6111111111111112, 1.4591479170272446),
        (1, 0.5333333333333333, 1.268273412406135),
        (2, 0.41666666666666663, 1.0),
        (3, 0.2222222222222222, 0.4591479170272448),
        (4, 0.41666666666666663, 0.8741854163060886),
        (5, 0.4666666666666667, 1.1424588287122237),
    ],
)
def test_各分割の重み付き不純度は原著と同じ(
    index: int, expected_gini: float, expected_entropy: float
) -> None:
    # 原著のセル出力をそのまま期待値にしている
    split = split_impurities()[index]

    assert split.weighted_gini == pytest.approx(expected_gini, rel=1e-15)
    assert split.weighted_entropy == pytest.approx(expected_entropy, rel=1e-15)


def test_分割は6通り試される() -> None:
    # 0 から len - 1 まで。「左が全部・右が空」は試されない
    splits = split_impurities()

    assert len(splits) == 6
    assert splits[0].left == []
    assert splits[5].right == ["C"]


def test_最良の分割はAのかたまりを切り離す() -> None:
    # ['A', 'A', 'A'] | ['C', 'B', 'C'] で両方の指標が最小になる
    best = best_split()

    assert best.index == 3
    assert best.left == ["A", "A", "A"]
    assert best.right == ["C", "B", "C"]
    assert best.weighted_gini == pytest.approx(0.2222222222222222, rel=1e-15)


def test_ジニとエントロピーは同じ分割を選ぶ() -> None:
    splits = split_impurities()
    by_gini = min(splits, key=lambda split: split.weighted_gini)
    by_entropy = min(splits, key=lambda split: split.weighted_entropy)

    assert by_gini.index == by_entropy.index


def test_分割しない場合の重み付き不純度は全体の不純度と一致する() -> None:
    # index 0 は左が空なので、右がそのまま全体になる。
    # ただしエントロピーは 1.4591479170272446 で、entropy(ELEMENTS) の
    # 1.4591479170272448 と最下位ビットだけ違う。重み付けの掛け算と割り算で
    # 丸めが 1 度多く入るため
    split = split_impurities()[0]

    assert split.weighted_gini == pytest.approx(gini(ELEMENTS), rel=1e-15)
    assert split.weighted_entropy != entropy(ELEMENTS)
    assert split.weighted_entropy == pytest.approx(entropy(ELEMENTS), rel=1e-15)
