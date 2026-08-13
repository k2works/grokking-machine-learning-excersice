"""原著ノートブック #21 `Chapter_12_Ensemble_Methods/Gradient_boosting_and_XGBoost.ipynb`。

8 人の年齢から「週に何日アプリを使うか」を当てる回帰の回である。

前半は **勾配ブースティングを手で追う**。弱学習器を 1 本ずつ取り出し、
残差がどう縮んでいくかを段ごとに印刷する。

後半は **XGBoost の類似度スコアを手で計算する**。
`similarity_score` と `find_best_split` を自分で書いて、
XGBoost が内部でやっていることを確かめる。

数値がいちばん多く印刷される回で、突き合わせの材料が豊富にある。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

#: 原著の 8 人。年齢
FEATURES = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])

#: 週あたりの利用日数
LABELS = np.array([7, 5, 7, 1, 2, 1, 5, 4])

#: 原著の勾配ブースティングの設定
MAX_DEPTH = 2
N_ESTIMATORS = 4
LEARNING_RATE = 0.8

#: XGBoost 側の設定。最初の予測はラベルの平均
INITIAL_PREDICTION = 4.0
XGB_N_ESTIMATORS = 3
XGB_LAMBDA = 3
XGB_MIN_SPLIT_LOSS = 2
XGB_LEARNING_RATE = 0.7


def fit_decision_tree(max_depth: int = MAX_DEPTH) -> DecisionTreeRegressor:
    """比較用の決定木 1 本。

    原著は `random_state` を渡していない。回帰木は分割が一意に決まるので
    種を渡さなくても結果は変わらない。
    """
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(FEATURES, LABELS)
    return model


def fit_gradient_boosting(n_estimators: int = N_ESTIMATORS) -> GradientBoostingRegressor:
    """原著と同じ設定の勾配ブースティング。"""
    model = GradientBoostingRegressor(
        max_depth=MAX_DEPTH, n_estimators=n_estimators, learning_rate=LEARNING_RATE
    )
    model.fit(FEATURES, LABELS)
    return model


def centered_labels() -> NDArray[np.float64]:
    """ラベルから平均を引いたもの。勾配ブースティングが最初に予測する対象。"""
    return LABELS - LABELS.mean()


def weak_learner_trace(model: GradientBoostingRegressor) -> list[dict]:
    """原著が段ごとに印刷する「残差 → 予測 → 新しい残差」を集める。

    原著は `predictions += preds * 0.8` と学習率を掛けて足し込み、
    `centered_labels - predictions` を次の残差にしている。
    """
    predictions = np.zeros(len(LABELS))
    residuals = centered_labels()
    trace = []

    for stage in model.estimators_:
        weak_learner = stage[0]
        stage_predictions = weak_learner.predict(FEATURES)
        predictions = predictions + stage_predictions * LEARNING_RATE
        new_residuals = centered_labels() - predictions
        trace.append(
            {
                "residuals": residuals,
                "predictions": stage_predictions,
                "new_residuals": new_residuals,
            }
        )
        residuals = new_residuals

    return trace


def similarity_score(values, lam: float = 0.0) -> float:
    """XGBoost の類似度スコア。原著の実装をそのまま写した。

    `sum(l)**2 / (len(l) + lam)`。**分子は和の 2 乗であって
    2 乗の和ではない。** だから符号がばらけるほど値が小さくなり、
    「似ている点が集まっているか」の尺度になる。
    """
    values = list(values)
    if len(values) == 0:
        return 0.0
    return float(sum(values) ** 2 / (len(values) + lam))


def find_best_split(residuals, lam: float) -> tuple:
    """原著の `find_best_split`。

    **並び順のまま前から切る** だけで、しきい値を探しているわけではない。
    データが年齢順に並んでいるから成り立つ簡略版である。
    左右の類似度スコアの和が最大になる切れ目を選ぶ。

    なお **原著の実装にはバグがある。** 最後に `return winning_left,
    winning_right, new_score, winning_split` としているが、`new_score` は
    ループの **最後の切れ目** のスコアであって、最良のものではない。
    返すべきは `best_score` である。左右の木は正しく選べているので、
    返り値の 3 つ目だけがおかしい。ここでは `best_score` を返す。
    """
    residuals = np.asarray(residuals, dtype=float)
    best_score = 0.0
    winning_left: NDArray[np.float64] = np.array([])
    winning_right = residuals

    for index in range(1, len(residuals)):
        left = residuals[:index]
        right = residuals[index:]
        new_score = similarity_score(left, lam) + similarity_score(right, lam)
        if new_score > best_score:
            best_score = new_score
            winning_left, winning_right = left, right

    return winning_left, winning_right, best_score


def split_scores(residuals, lam: float) -> list[float]:
    """切れ目ごとの類似度スコアの和。原著が 1 行ずつ印刷しているもの。"""
    residuals = np.asarray(residuals, dtype=float)
    return [
        similarity_score(residuals[:index], lam) + similarity_score(residuals[index:], lam)
        for index in range(1, len(residuals))
    ]


def fit_xgboost(reg_lambda: float = XGB_LAMBDA, min_split_loss: float = XGB_MIN_SPLIT_LOSS):
    """原著と同じ設定の XGBoost。

    `base_score` にラベルの平均 4.0 を渡すのが要点である。
    XGBoost はここから残差を縮めていく。
    """
    from xgboost import XGBRegressor

    model = XGBRegressor(
        random_state=0,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        reg_lambda=reg_lambda,
        min_split_loss=min_split_loss,
        learning_rate=XGB_LEARNING_RATE,
        base_score=INITIAL_PREDICTION,
    )
    model.fit(FEATURES, LABELS)
    return model


def xgboost_leaf_values(model) -> list[list[float]]:
    """各木の葉の値を、木ごとに取り出す。

    原著は `booster.get_dump()` のテキストをそのまま印刷している。
    そこから `leaf=` の数値だけを拾う。
    """
    import re

    dumps = model.get_booster().get_dump(with_stats=True)
    return [[float(value) for value in re.findall(r"leaf=(-?[\d.e+-]+)", dump)] for dump in dumps]


def xgboost_residuals() -> NDArray[np.float64]:
    """最初の予測（平均 4.0）からの残差。"""
    return LABELS - INITIAL_PREDICTION
