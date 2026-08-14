"""原著ノートブック #05 の再現テスト。

scikit-learn の `LogisticRegression` は原著と 4 桁まで一致する。手書きの学習ループは
#04 と同じく種が与えられていないため再現できない。

あわせて、原著が示す「対数損失の別の書き方」が実際には対数損失と一致しないことを
テストで固定してある。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb05_logistic_regression import (
    FEATURES,
    LABELS,
    alternate_log_loss,
    alternate_log_loss_original,
    fit_with_scikit_learn,
    log_loss,
    logistic_regression_algorithm,
    logistic_trick,
    prediction,
    score,
    sigmoid,
    soft_relu,
    total_log_loss,
)


def test_データセットは原著と同じ() -> None:
    # #04 と似ているが、最後の 2 点が入れ替わっている
    assert FEATURES.tolist() == [[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [3, 2], [2, 3]]
    assert LABELS.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_シグモイドは0で0_5を返す() -> None:
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(100) == pytest.approx(1.0)
    assert sigmoid(-100) == pytest.approx(0.0)


def test_シグモイドの2つの書き方は一致する() -> None:
    # 原著が使う exp(x)/(1+exp(x)) と、教科書の 1/(1+exp(-x))
    for x in [-3.0, -0.5, 0.0, 1.7, 4.2]:
        assert sigmoid(x) == pytest.approx(1 / (1 + np.exp(-x)))


def test_ソフトreluはreluをなめらかにしたもの() -> None:
    # 大きな正の値では x にほぼ等しく、大きな負の値では 0 に近づく
    assert soft_relu(20) == pytest.approx(20.0, abs=1e-8)
    assert soft_relu(-20) == pytest.approx(0.0, abs=1e-8)
    assert soft_relu(0) == pytest.approx(np.log(2))


def test_対数損失は当たっていても0にはならない() -> None:
    # パーセプトロン誤差は当たれば 0 だったが、対数損失は確信の度合いで変わる
    weights, bias = [1.0, 1.0], 0.0
    # FEATURES[4] = [1, 3] はラベル 1。スコア 4 なのでよく当たっている
    confident = log_loss(weights, bias, FEATURES[4], 1)
    # FEATURES[0] = [1, 0] はラベル 0。スコア 1 なので外している
    wrong = log_loss(weights, bias, FEATURES[0], 0)

    assert 0 < confident < 0.1
    assert wrong > 1.0


def test_原著の別の書き方は対数損失と一致しない() -> None:
    # 原著は「対数損失の別の書き方」として soft_relu((pred - label) * score) を
    # 示すが、pred は 0〜1 の確率なので (pred - label) は ±1 にならない
    weights, bias = [1.0, 1.0], 0.0
    for i in range(len(FEATURES)):
        original = alternate_log_loss_original(weights, bias, FEATURES[i], LABELS[i])
        exact = log_loss(weights, bias, FEATURES[i], LABELS[i])
        if abs(score(weights, bias, FEATURES[i])) > 1e-9:
            assert original != pytest.approx(exact)


def test_スコアが0のときだけ両者は一致する() -> None:
    # (pred - label) * 0 も (1 - 2 * label) * 0 も 0 になるため
    weights, bias = [1.0, 1.0], -1.0
    features = np.array([1, 0])  # スコア = 1 - 1 = 0

    assert score(weights, bias, features) == pytest.approx(0.0)
    assert alternate_log_loss_original(weights, bias, features, 1) == pytest.approx(
        log_loss(weights, bias, features, 1)
    )


def test_正しい別の書き方は対数損失と厳密に一致する() -> None:
    # ラベルが 0 なら +1、1 なら -1 を掛ける。つまり (1 - 2 * label)
    weights, bias = [1.0, 1.0], 0.0
    for i in range(len(FEATURES)):
        for label in [0, 1]:
            assert alternate_log_loss(weights, bias, FEATURES[i], label) == pytest.approx(
                log_loss(weights, bias, FEATURES[i], label)
            )


def test_ロジスティックトリックはバイアスを1回だけ動かす() -> None:
    # #04 の「短く書いた版」はバイアスをループの内側で更新していたが、
    # こちらは外側にあるので 1 回だけ適用される
    _, bias = logistic_trick([1.0, 1.0], 0.0, FEATURES[0], 0, learning_rate=0.05)
    pred = prediction([1.0, 1.0], 0.0, FEATURES[0])

    assert bias == pytest.approx(0.0 + (0 - pred) * 0.05)


def test_確信を持って間違えた点ほど大きく動く() -> None:
    # パーセプトロンのトリックは当たり外れの 2 値でしか動かなかったが、
    # ロジスティックトリックは pred が連続値なので度合いが効く
    close = logistic_trick([1.0, 1.0], -3.0, np.array([1, 2]), 1)
    far = logistic_trick([1.0, 1.0], -10.0, np.array([1, 2]), 1)

    assert far[0][0] > close[0][0]


def test_学習は誤差を下げる() -> None:
    # 原著の出力 ([1.2019, 0.7009], -2.7884) は #04 と同じ理由で再現できない
    result = logistic_regression_algorithm(epochs=500, seed=0)

    assert len(result.errors) == 500
    assert result.errors[-1] < result.errors[0]


def test_学習後は全点を正しく分類できる() -> None:
    # 500 エポックでは 1 点（[1, 2]）を取り違える。対数損失は当たっていても
    # 0 にならないので、パーセプトロンより収束に時間がかかる
    result = logistic_regression_algorithm(epochs=5000, seed=0)
    predictions = [
        1 if prediction(result.weights, result.bias, FEATURES[i]) >= 0.5 else 0
        for i in range(len(FEATURES))
    ]

    assert predictions == LABELS.tolist()


def test_合計対数損失は各点の合計() -> None:
    weights, bias = [1.0, 1.0], 0.0
    expected = sum(log_loss(weights, bias, FEATURES[i], LABELS[i]) for i in range(len(FEATURES)))

    assert total_log_loss(weights, bias, FEATURES, LABELS) == pytest.approx(expected)


def test_scikit_learnの係数は原著と4桁一致する() -> None:
    # 原著の出力
    #   Coefficients: [1.00458154 0.93718206]
    #   Intercept: -3.1600974622062727
    # lbfgs ソルバの収束判定が変わっているため、5 桁目から分かれる
    model = fit_with_scikit_learn()

    assert model.coef_[0][0] == pytest.approx(1.00458154, rel=1e-4)
    assert model.coef_[0][1] == pytest.approx(0.93718206, rel=1e-4)
    assert float(model.intercept_[0]) == pytest.approx(-3.1600974622062727, rel=1e-4)


def test_scikit_learnは全点を正しく分類する() -> None:
    # 原著の出力: Logistic Regression Predictions: [0 0 0 0 1 1 1 1]
    model = fit_with_scikit_learn()

    assert model.predict(FEATURES).tolist() == LABELS.tolist()
