"""原著ノートブック #16 の再現テスト。

原著は図しか出さないので、突き合わせる数値が無い。
そこで **境界を数値に直して** 検証する。
格子上で 1 と判定される割合と、8 点それぞれの判定である。
"""

import pytest

from grokking_ml_lib.nb16_plotting_boundaries import (
    accuracy,
    alien_dataset,
    bias,
    classify,
    disagreement_ratio,
    features,
    labels,
    line_1,
    line_2,
    nn_with_sigmoid,
    nn_with_step,
    predictions,
    region_ratio,
    sigmoid,
    step,
)


@pytest.fixture(scope="module")
def dataset():
    return alien_dataset()


def test_データセットは8点(dataset) -> None:
    assert len(dataset) == 8
    assert list(features(dataset).columns) == ["aack", "beep"]
    # 幸せなエイリアンと不幸せなエイリアンが半分ずつ
    assert list(labels(dataset)) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_階段関数は0で1になる() -> None:
    # 原著は x >= 0。0 ちょうどは 1 の側に入る
    assert step(0) == 1
    assert step(-1e-15) == 0


def test_シグモイドは0で半分() -> None:
    assert sigmoid(0) == pytest.approx(0.5, rel=1e-15)
    # 原著の書き方は exp(x)/(1+exp(x))。1/(1+exp(-x)) と同じ値になる
    assert sigmoid(2.0) == pytest.approx(0.8807970779778823, rel=1e-15)


def test_1層目の2つの直線は対称(dataset) -> None:
    # 重みが (6, 10) と (10, 6) なので、aack と beep を入れ替えた関係
    assert predictions(line_1, dataset) == [0, 0, 0, 1, 1, 1, 1, 1]
    assert predictions(line_2, dataset) == [0, 1, 0, 0, 1, 1, 1, 1]
    # 領域の広さは同じ
    assert region_ratio(line_1) == pytest.approx(region_ratio(line_2), rel=1e-15)


def test_1層目だけでは8点を分けられない(dataset) -> None:
    # 直線 1 本では 1 点ずつ間違える
    assert accuracy(line_1, dataset) == 0.875
    assert accuracy(line_2, dataset) == 0.875


def test_バイアスは入力を見ない(dataset) -> None:
    # 常に 1。格子のすべてが 1 の領域になる
    assert predictions(bias, dataset) == [1] * 8
    assert region_ratio(bias) == 1.0


def test_2層目はANDになっている(dataset) -> None:
    # 1 層目の出力は 0 か 1。その和が 1.5 以上になるのは両方 1 のときだけ
    assert nn_with_step(1, 1) == 1
    assert nn_with_step(2, 0) == 0  # line_2 だけが 1
    assert nn_with_step(0, 2) == 0  # line_1 だけが 1


def test_階段関数のネットワークは8点すべて正解(dataset) -> None:
    assert predictions(nn_with_step, dataset) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert accuracy(nn_with_step, dataset) == 1.0


def test_シグモイド版は1点だけ外す(dataset) -> None:
    # aack=1 beep=1 の点だけ 0 と答える。正解は 1
    assert predictions(nn_with_sigmoid, dataset) == [0, 0, 0, 0, 0, 1, 1, 1]
    assert accuracy(nn_with_sigmoid, dataset) == 0.875


def test_外した点は判定の境目にある() -> None:
    # 0.4905 で、しきい値 0.5 をわずかに下回る
    assert nn_with_sigmoid(1, 1) == pytest.approx(0.4905304218, rel=1e-9)
    assert classify(nn_with_sigmoid, 1, 1) == 0


def test_シグモイドは出力が飽和する() -> None:
    # 内側のシグモイドが 1 に近づくので、外側は sigmoid(0.5) で頭打ちになる
    assert nn_with_sigmoid(2, 2) == pytest.approx(0.6224593117, rel=1e-9)
    assert nn_with_sigmoid(3, 3) == pytest.approx(0.6224593312, rel=1e-9)
    # 3 倍離れてもほとんど変わらない
    assert nn_with_sigmoid(3, 3) - nn_with_sigmoid(2, 2) < 1e-7


def test_2つのネットワークの境界はほぼ重なる() -> None:
    # 図では見分けが付かないが、格子の 0.48% だけ判定が食い違う
    assert disagreement_ratio(nn_with_step, nn_with_sigmoid) == pytest.approx(0.0048, abs=1e-4)


def test_ANDの領域は各直線より狭い() -> None:
    # AND なので、それぞれの直線の領域の共通部分になる。
    # Python 版は 0.5544327、Kotlin 版・F# 版は 0.5543918 になる。
    # NumPy の arange は誤差を累積し、-0.5 + 100 * 0.005 が 0.0 ではなく
    # 4.44e-16 になる。境界ちょうどの点の判定がそこで分かれる
    assert region_ratio(nn_with_step) == pytest.approx(0.5544, abs=1e-4)
    assert region_ratio(nn_with_step) < region_ratio(line_1)
