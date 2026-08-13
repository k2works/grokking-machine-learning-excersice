"""原著ノートブック #17 の再現テスト。

原著は種を固定せずに乱数でデータを作るので、生成は再現できない。
代わりに **配布 CSV が原著の規則で作られたこと** を検証する。
"""

import pytest

from grokking_ml_lib.nb17_building_datasets import (
    LINEAR,
    ONE_CIRCLE,
    SPECS,
    TWO_CIRCLES,
    generate,
    linear_rule,
    load,
    one_circle_rule,
    rule_violations,
    two_circles_rule,
)


@pytest.mark.parametrize(("spec", "total"), [(LINEAR, 60), (ONE_CIRCLE, 110), (TWO_CIRCLES, 220)])
def test_配布CSVの行数は規則の点とノイズの合計(spec, total) -> None:
    # 原著は「規則どおりの点」と「ノイズ」を続けて追加している
    assert spec.total == total
    assert len(load(spec)) == total


@pytest.mark.parametrize("spec", SPECS)
def test_先頭の点は規則に1つも違反しない(spec) -> None:
    # 配布 CSV の先頭 spec.points 件は規則どおりに作られた点。
    # ここに違反が 1 つも無いことが、規則を正しく読めた証拠になる
    violations = rule_violations(spec, load(spec))

    assert [index for index in violations if index < spec.points] == []


@pytest.mark.parametrize(
    ("spec", "count"), [(LINEAR, 5), (ONE_CIRCLE, 7), (TWO_CIRCLES, 12)]
)
def test_違反はすべてノイズ部分にあり約半数(spec, count) -> None:
    # ノイズはラベルを 0 か 1 で振り直すので、規則と食い違うのは約半分。
    # 実測: 10 個中 5 個 / 10 個中 7 個 / 20 個中 12 個
    violations = rule_violations(spec, load(spec))

    assert len(violations) == count
    assert all(index >= spec.points for index in violations)
    assert 0.2 <= len(violations) / spec.noise <= 0.8


@pytest.mark.parametrize("spec", SPECS)
def test_座標はマイナス3から3の範囲(spec) -> None:
    # 原著の 6 * random() - 3
    data = load(spec)

    assert data["x_1"].between(-3, 3).all()
    assert data["x_2"].between(-3, 3).all()


@pytest.mark.parametrize("spec", SPECS)
def test_ラベルは0か1だけ(spec) -> None:
    assert sorted(load(spec)["y"].unique()) == [0, 1]


def test_直線の規則は境界のちょうど上を0にする() -> None:
    # 原著は x + y > 0.5（等号を含まない）
    assert linear_rule(0.25, 0.25) == 0
    assert linear_rule(0.3, 0.3) == 1


def test_円の規則は境界のちょうど上を0にする() -> None:
    # 原著は x^2 + y^2 < 2.8（等号を含まない）
    import math

    radius = math.sqrt(2.8)
    assert one_circle_rule(radius, 0.0) == 0
    assert one_circle_rule(radius - 1e-9, 0.0) == 1


def test_2つの円は重なりを持つ() -> None:
    # 中心 (1, 0) と (-1, 0)、半径 √2 ≒ 1.414。中心間の距離 2 より大きいので重なる。
    # 原点は両方の内側に入る
    assert two_circles_rule(0.0, 0.0) == 1
    # どちらの円からも外れる点
    assert two_circles_rule(0.0, 2.0) == 0


@pytest.mark.parametrize("spec", SPECS)
def test_生成した点は規則どおりでノイズだけが外れる(spec) -> None:
    # 生成器は原著と同じ手順。種を固定しているので毎回同じものが出る
    generated = generate(spec, seed=0)

    assert len(generated) == spec.total
    assert [index for index in rule_violations(spec, generated) if index < spec.points] == []


def test_生成は種を固定すれば再現する() -> None:
    # 原著は種を固定していないので毎回違うものが出る。
    # ここは種を渡せるようにしてテストできる形にした
    assert generate(LINEAR, seed=42).equals(generate(LINEAR, seed=42))
    assert not generate(LINEAR, seed=42).equals(generate(LINEAR, seed=43))


def test_生成したものは配布CSVとは一致しない() -> None:
    # 原著が種を固定していない以上、これは一致しようがない。
    # 「再現できない」ことを明示的に記録しておく
    generated = generate(LINEAR, seed=0)
    published = load(LINEAR)

    assert len(generated) == len(published)
    assert generated["x_1"].iloc[0] != published["x_1"].iloc[0]
