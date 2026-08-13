"""原著ノートブック #21 の再現テスト。

勾配ブースティングの段ごとの残差も、XGBoost の葉の値も、
**原著の出力と桁まで一致** する。唯一ずれるのは `score` の 8 桁目で、
XGBoost が単精度で計算しているためである。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb21_gradient_boosting import (
    FEATURES,
    INITIAL_PREDICTION,
    LABELS,
    centered_labels,
    find_best_split,
    fit_decision_tree,
    fit_gradient_boosting,
    similarity_score,
    split_scores,
    weak_learner_trace,
    xgboost_residuals,
)


def _xgboost_available() -> bool:
    # macOS では libomp が要る。Nix の python-ml シェルの外では読み込めない
    try:
        import xgboost  # noqa: F401
    except Exception:  # noqa: BLE001 - 読み込み失敗の理由は問わない
        return False
    return True


xgboost_required = pytest.mark.skipif(
    not _xgboost_available(),
    reason="xgboost の読み込みに失敗しました（macOS では libomp が要ります）",
)


def test_データセットは8人() -> None:
    assert FEATURES.ravel().tolist() == [10, 20, 30, 40, 50, 60, 70, 80]
    assert LABELS.tolist() == [7, 5, 7, 1, 2, 1, 5, 4]
    # 平均は 4.0。XGBoost の base_score に使う
    assert LABELS.mean() == 4.0


def test_決定木は深さ2で4つの葉を持つ() -> None:
    tree = fit_decision_tree()

    assert tree.get_depth() == 2
    assert tree.get_n_leaves() == 4


def test_中心化したラベルが最初の残差() -> None:
    # 原著の出力: Residuals to predict: [ 3.  1.  3. -3. -2. -3.  1.  0.]
    assert centered_labels().tolist() == [3.0, 1.0, 3.0, -3.0, -2.0, -3.0, 1.0, 0.0]


def test_勾配ブースティングの予測は原著と一致する() -> None:
    # 原著の出力
    #   array([6.87466667, 5.11466667, 6.71466667, 1.43466667, 1.43466667,
    #          1.43466667, 4.896     , 4.096     ])
    expected = [
        6.87466667,
        5.11466667,
        6.71466667,
        1.43466667,
        1.43466667,
        1.43466667,
        4.896,
        4.096,
    ]

    assert fit_gradient_boosting().predict(FEATURES) == pytest.approx(expected, abs=5e-9)


def test_弱学習器は4本() -> None:
    assert len(fit_gradient_boosting().estimators_) == 4


def test_1本目の弱学習器の予測は原著と一致する() -> None:
    # 原著の出力
    #   Predictions: [ 3. 2. 2. -2.66666667 -2.66666667 -2.66666667 0.5 0.5 ]
    trace = weak_learner_trace(fit_gradient_boosting())
    expected = [3.0, 2.0, 2.0, -2.66666667, -2.66666667, -2.66666667, 0.5, 0.5]

    assert trace[0]["predictions"] == pytest.approx(expected, abs=5e-9)


def test_1段目の後の残差は原著と一致する() -> None:
    # 原著の出力
    #   New residuals: [ 0.6 -0.6 1.4 -0.86666667 0.13333333 -0.86666667 0.6 -0.4 ]
    trace = weak_learner_trace(fit_gradient_boosting())
    expected = [0.6, -0.6, 1.4, -0.86666667, 0.13333333, -0.86666667, 0.6, -0.4]

    assert trace[0]["new_residuals"] == pytest.approx(expected, abs=5e-9)


def test_2段目の後の残差は原著と一致する() -> None:
    # 原著の出力: [ 0.6 -0.6 0.28 -0.44 0.56 -0.44 0.52 -0.48]
    trace = weak_learner_trace(fit_gradient_boosting())
    expected = [0.6, -0.6, 0.28, -0.44, 0.56, -0.44, 0.52, -0.48]

    assert trace[1]["new_residuals"] == pytest.approx(expected, rel=1e-12)


def test_3段目の予測はほとんど0になる() -> None:
    # 原著の出力に -7.40148683e-17 が並ぶ。
    # 「もう説明できるものが残っていない」を浮動小数点の誤差が示している
    trace = weak_learner_trace(fit_gradient_boosting())
    predictions = trace[2]["predictions"]

    assert predictions[0] == pytest.approx(0.6, rel=1e-12)
    assert predictions[1] == pytest.approx(-0.6, rel=1e-12)
    assert all(abs(value) < 1e-15 for value in predictions[2:])


def test_残差は段を追うごとに小さくなる() -> None:
    # ブースティングの定義そのもの。二乗和で測る
    trace = weak_learner_trace(fit_gradient_boosting())
    magnitudes = [float(np.sum(stage["residuals"] ** 2)) for stage in trace]

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_類似度スコアは和の2乗を使う() -> None:
    # sum(l)**2 / (len(l) + lam)。2 乗の和ではない
    assert similarity_score([3.0], 3) == pytest.approx(2.25, rel=1e-15)
    assert similarity_score([3.0, 1.0], 3) == pytest.approx(3.2, rel=1e-15)
    # 符号が打ち消し合うと 0 になる
    assert similarity_score([3.0, -3.0], 3) == 0.0


def test_空の集合の類似度は0() -> None:
    assert similarity_score([], 3) == 0.0


def test_全体の類似度は0になる() -> None:
    # 残差の総和が 0 なので、根の類似度スコアは 0。
    # 原著が「similarity_score(residuals, lam) -> 0.0」と出しているところ
    assert similarity_score(xgboost_residuals(), 3) == 0.0


def test_残差は原著と一致する() -> None:
    # 原著の出力: array([ 3.,  1.,  3., -3., -2., -3.,  1.,  0.])
    assert xgboost_residuals().tolist() == [3.0, 1.0, 3.0, -3.0, -2.0, -3.0, 1.0, 0.0]


def test_切れ目ごとのスコアは原著と一致する() -> None:
    # 原著が 1 行ずつ印刷している「Sum of similarity scores」
    scores = split_scores(xgboost_residuals(), 3)

    assert scores[0] == pytest.approx(3.15, rel=1e-15)
    assert scores[1] == pytest.approx(4.977777777777778, rel=1e-15)
    # 3 番目が最大
    assert scores[2] == pytest.approx(14.291666666666666, rel=1e-15)
    assert max(scores) == scores[2]


def test_最良の分割は原著と一致する() -> None:
    # 原著の結論: Left tree: [3. 1. 3.] / Right tree: [-3. -2. -3.  1.  0.]
    left, right, score = find_best_split(xgboost_residuals(), 3)

    assert left.tolist() == [3.0, 1.0, 3.0]
    assert right.tolist() == [-3.0, -2.0, -3.0, 1.0, 0.0]
    assert score == pytest.approx(14.291666666666666, rel=1e-15)


def test_左の部分木の最良の分割は原著と一致する() -> None:
    # 原著の出力: (array([3.]), array([1., 3.]), 5.45, ...)
    # 同じ 5.45 になる切り方が 2 通りあり、先に見つけたほうが選ばれる
    left, _, _ = find_best_split(xgboost_residuals(), 3)
    sub_left, sub_right, score = find_best_split(left, 3)

    assert sub_left.tolist() == [3.0]
    assert sub_right.tolist() == [1.0, 3.0]
    assert score == pytest.approx(5.45, rel=1e-15)


def test_右の部分木の分割は一致するがスコアは原著と違う() -> None:
    # 原著の出力: (array([-3., -2., -3.]), array([1., 0.]), 7.0, ...)
    # 左右の木は一致するが、3 つ目の 7.0 は **原著の実装のバグ** による。
    # 原著は best_score ではなく new_score（最後の切れ目のスコア）を返している
    _, right, _ = find_best_split(xgboost_residuals(), 3)
    sub_left, sub_right, score = find_best_split(right, 3)

    assert sub_left.tolist() == [-3.0, -2.0, -3.0]
    assert sub_right.tolist() == [1.0, 0.0]
    # 正しい最良スコア
    assert score == pytest.approx(10.866666666666665, rel=1e-15)


def test_原著が返す7という値は最後の切れ目のスコア() -> None:
    # 原著のバグを再現して確かめる。右の木 [-3,-2,-3,1,0] の最後の切れ目は
    # [-3,-2,-3,1] / [0] で、(-7)^2/(4+3) + 0^2/(1+3) = 7.0 + 0.0 = 7.0。
    # これが原著の返り値 7.0 の正体である
    _, right, _ = find_best_split(xgboost_residuals(), 3)
    last_split_score = split_scores(right, 3)[-1]

    assert last_split_score == pytest.approx(7.0, rel=1e-15)
    # 最良（10.87）とは違う。原著は最良でないほうを返している
    assert last_split_score != max(split_scores(right, 3))


def test_左の部分木ではバグが表に出ない() -> None:
    # 左の木 [3,1,3] は最後の切れ目 [3,1] / [3] が最良と同点（5.45）なので、
    # new_score を返しても best_score を返しても同じ値になる。
    # **バグが隠れる条件が揃っていた**
    left, _, _ = find_best_split(xgboost_residuals(), 3)
    scores = split_scores(left, 3)

    assert scores[-1] == pytest.approx(max(scores), rel=1e-15)


@xgboost_required
def test_XGBoostの葉の値は原著と一致する() -> None:
    # 原著の get_dump() の出力
    #   1 本目: leaf=0.816666603 / leaf=-0.933333337 / leaf=0.140000001
    #   2 本目: leaf=0.530833364 / leaf=-0.606666625 / leaf=0.100800037
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost, xgboost_leaf_values

    leaves = xgboost_leaf_values(fit_xgboost())

    assert leaves[0] == pytest.approx([0.816666603, -0.933333337, 0.140000001], abs=1e-9)
    assert leaves[1] == pytest.approx([0.530833364, -0.606666625, 0.100800037], abs=1e-9)
    assert leaves[2][0] == pytest.approx(0.345041722, abs=1e-9)


@xgboost_required
def test_20歳の予測は葉の値の足し算になる() -> None:
    # 原著が手で確かめているところ
    #   4 + 0.816666603 + 0.530833364 + 0.345041722 = 5.6925416890000005
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost, xgboost_leaf_values

    model = fit_xgboost()
    leaves = xgboost_leaf_values(model)
    by_hand = INITIAL_PREDICTION + leaves[0][0] + leaves[1][0] + leaves[2][0]

    assert by_hand == pytest.approx(5.6925416890000005, rel=1e-12)
    # モデルの予測とも一致する（単精度なので 7 桁まで）
    assert float(model.predict(FEATURES)[1]) == pytest.approx(by_hand, rel=1e-7)


@xgboost_required
def test_XGBoostの予測は3種類の値しか取らない() -> None:
    # 原著の出力
    #   array([5.6925416, 5.6925416, 5.6925416, 2.20961, 2.20961, 2.20961,
    #          3.99041, 3.99041], dtype=float32)
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost

    predictions = fit_xgboost().predict(FEATURES)

    assert predictions[:3] == pytest.approx([5.6925416] * 3, rel=1e-7)
    assert predictions[3:6] == pytest.approx([2.20961] * 3, rel=1e-7)
    assert predictions[6:] == pytest.approx([3.99041] * 2, rel=1e-7)


@xgboost_required
def test_XGBoostのスコアは8桁目でずれる() -> None:
    # 原著の出力は 0.8121875824035912。いまは 0.8121875524520874。
    # XGBoost は内部を単精度（float32）で計算するので、
    # 有効桁 7 桁を超えたところは版や環境で変わりうる
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost

    score = fit_xgboost().score(FEATURES, LABELS)

    assert score == pytest.approx(0.8121875824035912, rel=1e-7)
    assert score != 0.8121875824035912


@xgboost_required
def test_lambdaを上げると予測が平均に近づく() -> None:
    # 原著が「λ を上げるほど平均に近づく」と書いているところ。
    # 予測のばらつきで測る
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost

    spreads = [
        float(np.std(fit_xgboost(reg_lambda=lam, min_split_loss=0).predict(FEATURES)))
        for lam in (0, 10, 20, 100)
    ]

    assert spreads == sorted(spreads, reverse=True)


@xgboost_required
def test_gammaを上げると予測が平均に近づく() -> None:
    # 原著が「γ を上げるほど平均に近づく」と書いているところ。
    # γ = 100 では分割が 1 つも許されず、予測が平均 4.0 だけになる
    from grokking_ml_lib.nb21_gradient_boosting import fit_xgboost

    predictions = fit_xgboost(reg_lambda=0, min_split_loss=100).predict(FEATURES)

    assert predictions == pytest.approx([INITIAL_PREDICTION] * 8, rel=1e-6)
