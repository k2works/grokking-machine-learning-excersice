"""原著ノートブック #20 `Chapter_12_Ensemble_Methods/Random_forests_and_AdaBoost.ipynb`。

18 通のメールを 2 特徴量（`Lottery`・`Sale`）で振り分ける小さなデータに、
アンサンブル学習を次々と当てる回である。

- 決定木 1 本（制限なし）→ 訓練データを丸暗記して 1.0
- 6 通ずつ 3 組に分けて、深さ 1 の弱学習器を手で作る
- ランダムフォレスト（5 本・深さ 1）
- AdaBoost（6 本）
- おまけとして勾配ブースティングと XGBoost

**「正解率が 1.0 になるのは良いことではない」** を見せるのが狙いである。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

#: 原著が冒頭で設定する種
NUMPY_SEED = 0

#: 原著の 18 通。[Lottery, Sale, Spam]
EMAILS = (
    (7, 8, 1),
    (3, 2, 0),
    (8, 4, 1),
    (2, 6, 0),
    (6, 5, 1),
    (9, 6, 1),
    (8, 5, 0),
    (7, 1, 0),
    (1, 9, 1),
    (4, 7, 0),
    (1, 3, 0),
    (3, 10, 1),
    (2, 2, 1),
    (9, 3, 0),
    (5, 3, 0),
    (10, 1, 0),
    (5, 9, 1),
    (10, 8, 1),
)

#: 原著が手作業で切り分ける 3 組。6 通ずつ **並び順のまま** 分ける
BATCHES = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11), (12, 13, 14, 15, 16, 17))


def spam_dataset() -> pd.DataFrame:
    """原著の 18 通。"""
    return pd.DataFrame(np.array(EMAILS), columns=["Lottery", "Sale", "Spam"])


def features(data: pd.DataFrame) -> pd.DataFrame:
    return data[["Lottery", "Sale"]]


def labels(data: pd.DataFrame) -> pd.Series:
    return data["Spam"]


def score(model, data: pd.DataFrame) -> float:
    """学習データに対する正解率。"""
    return float(model.score(features(data), labels(data)))


def fit_decision_tree(data: pd.DataFrame, **kwargs) -> DecisionTreeClassifier:
    """決定木 1 本。制限を掛けなければ訓練データを丸暗記する。"""
    model = DecisionTreeClassifier(random_state=NUMPY_SEED, **kwargs)
    model.fit(features(data), labels(data))
    return model


def batch(data: pd.DataFrame, index: int) -> pd.DataFrame:
    """原著が手で切り分ける 3 組のうち 1 つ。"""
    return data.loc[list(BATCHES[index])]


def fit_weak_learner(data: pd.DataFrame, index: int) -> DecisionTreeClassifier:
    """1 組ぶんのデータに深さ 1 の決定木（切り株）を当てる。"""
    return fit_decision_tree(batch(data, index), max_depth=1)


def fit_random_forest(data: pd.DataFrame) -> RandomForestClassifier:
    """原著と同じ設定（5 本・深さ 1）のランダムフォレスト。"""
    model = RandomForestClassifier(random_state=NUMPY_SEED, n_estimators=5, max_depth=1)
    model.fit(features(data), labels(data))
    return model


def fit_adaboost(data: pd.DataFrame) -> AdaBoostClassifier:
    """原著と同じ設定（6 本）の AdaBoost。

    **原著の 0.8889 は再現しない。0.7778 になる。**
    原著の実行時（scikit-learn 0.2x）の既定は `algorithm='SAMME.R'`
    （弱学習器の確率出力を使う実数版）だった。1.4 で非推奨になり、
    **1.6 で削除された** ので、いまは `SAMME`（離散版）しか選べない。
    `algorithm` 引数そのものが 1.9 の署名から消えている。
    """
    model = AdaBoostClassifier(random_state=NUMPY_SEED, n_estimators=6)
    model.fit(features(data), labels(data))
    return model


def fit_gradient_boosting(data: pd.DataFrame) -> GradientBoostingClassifier:
    """原著がおまけで試す勾配ブースティング（5 本）。"""
    model = GradientBoostingClassifier(random_state=NUMPY_SEED, n_estimators=5)
    model.fit(features(data), labels(data))
    return model


def fit_xgboost(data: pd.DataFrame):
    """原著がおまけで試す XGBoost（5 本）。

    原著は `np.array(features)` と DataFrame を配列に直して渡している。
    列名を持たない形で学習させると、予測時にも配列が要る。
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(random_state=NUMPY_SEED, n_estimators=5)
    model.fit(np.array(features(data)), labels(data))
    return model


def xgboost_score(model, data: pd.DataFrame) -> float:
    """XGBoost だけは配列で渡す必要がある。"""
    return float(model.score(np.array(features(data)), labels(data)))


def split_of(tree: DecisionTreeClassifier) -> tuple[str, float]:
    """深さ 1 の木が使った特徴量としきい値。

    `tree_.feature[0]` は根の分割に使った列番号、
    `tree_.threshold[0]` はそのしきい値。
    """
    column = int(tree.tree_.feature[0])
    return ("Lottery", "Sale")[column], float(tree.tree_.threshold[0])


def predictions(model, data: pd.DataFrame) -> NDArray[np.int_]:
    return np.asarray(model.predict(features(data)))
