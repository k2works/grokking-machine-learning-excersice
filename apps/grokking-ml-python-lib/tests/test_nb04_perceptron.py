"""原著ノートブック #04 の再現テスト。

scikit-learn の `Perceptron` は原著の数値と完全に一致する。手書きの学習ループは、
原著が乱数の種を与えていないため出力自体が実行のたびに変わる。そこは再現せず、
**トリックが持つ性質** と収束で検証する。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb04_perceptron import (
    FEATURES,
    LABELS,
    error,
    fit_with_scikit_learn,
    mean_perceptron_error,
    perceptron_algorithm,
    perceptron_trick,
    perceptron_trick_explicit,
    prediction,
    score,
    step,
)


def test_データセットは原著と同じ() -> None:
    assert FEATURES.tolist() == [[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2]]
    assert LABELS.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_ステップ関数は0で1を返す() -> None:
    # 境界をどちらに含めるかで結果が変わる。原著は 0 以上を 1 とする
    assert step(0) == 1
    assert step(-1e-12) == 0
    assert step(1) == 1


def test_スコアは重み付き和にバイアスを足す() -> None:
    # [2, 3] . [1, 2] - 4 = 2 + 6 - 4 = 4
    assert score([1, 2], -4, np.array([2, 3])) == pytest.approx(4.0)


def test_誤差は当たれば0外れればスコアの絶対値() -> None:
    # 予測 1、ラベル 1 なので誤差なし
    assert error([1, 2], -4, np.array([2, 3]), 1) == 0.0
    # 予測 1、ラベル 0 なのでスコアの絶対値 4 が誤差になる
    assert error([1, 2], -4, np.array([2, 3]), 0) == pytest.approx(4.0)


def test_重み1と2バイアスマイナス4の予測は原著と同じ() -> None:
    # 原著の出力
    #   0 0 / 1 0 / 0 0 / 1 1 / 1 0 / 1 0 / 1 0 / 1 0
    weights, bias = [1, 2], -4
    predictions = [prediction(weights, bias, FEATURES[i]) for i in range(len(FEATURES))]
    errors = [error(weights, bias, FEATURES[i], LABELS[i]) for i in range(len(FEATURES))]

    assert predictions == [0, 1, 0, 1, 1, 1, 1, 1]
    assert errors == pytest.approx([0, 0, 0, 1, 0, 0, 0, 0])


def test_平均パーセプトロン誤差は誤差の平均() -> None:
    # 上の 8 点で誤差が 1 になるのは 1 点だけ
    assert mean_perceptron_error([1, 2], -4, FEATURES, LABELS) == pytest.approx(1 / 8)


def test_短く書いた版のトリックは原著と同じ数値になる() -> None:
    # 原著の出力: ([0.9, 1.85], -4.1)
    weights, bias = perceptron_trick([1, 2], -4, FEATURES[6], 0)

    assert weights == pytest.approx([0.9, 1.85])
    assert bias == pytest.approx(-4.1)


def test_短く書いた版はバイアスを特徴量の数だけ動かす() -> None:
    # 原著の 2 つの実装は挙動が違う。短く書いた版はバイアスの更新が
    # 重みのループの内側にあり、特徴量が 2 つなので 2 回適用される
    explicit_weights, explicit_bias = perceptron_trick_explicit([1, 2], -4, FEATURES[6], 0)
    short_weights, short_bias = perceptron_trick([1, 2], -4, FEATURES[6], 0)

    # 重みの更新は一致する
    assert short_weights == pytest.approx(explicit_weights)
    # バイアスだけ 2 倍動く
    assert explicit_bias == pytest.approx(-4.05)
    assert short_bias == pytest.approx(-4.1)


def test_当たっているときは何も動かない() -> None:
    # FEATURES[4] = [1, 3] は重み [1, 2] バイアス -4 で予測 1、ラベルも 1
    weights, bias = perceptron_trick([1, 2], -4, FEATURES[4], 1)

    assert weights == pytest.approx([1, 2])
    assert bias == pytest.approx(-4)


def test_学習は誤差を下げる() -> None:
    # 原著の出力 ([0.55, 0.25], -1.1) は、点の選択に使う random に種が
    # 与えられていないため実行のたびに変わる。数値ではなく収束で検証する
    result = perceptron_algorithm(epochs=200, seed=0)

    assert len(result.errors) == 200
    assert result.errors[-1] < result.errors[0]


def test_学習後は全点を正しく分類できる() -> None:
    result = perceptron_algorithm(epochs=500, seed=0)
    predictions = [
        prediction(result.weights, result.bias, FEATURES[i]) for i in range(len(FEATURES))
    ]

    assert predictions == LABELS.tolist()


def test_scikit_learnの係数は原著と同じ数値になる() -> None:
    # 原著の出力
    #   Coefficients: [4. 2.]
    #   Intercept: -9.0
    model = fit_with_scikit_learn()

    assert model.coef_[0].tolist() == [4.0, 2.0]
    assert float(model.intercept_[0]) == -9.0


def test_scikit_learnは全点を正しく分類する() -> None:
    # 原著の出力: Predictions: [0 0 0 0 1 1 1 1]
    model = fit_with_scikit_learn()

    assert model.predict(FEATURES).tolist() == LABELS.tolist()
