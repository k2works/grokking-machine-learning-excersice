"""原著ノートブック #09 `Chapter_09_Decision_Trees/App_recommendations.ipynb`。

6 人のユーザーの「使っている端末」と「年齢」から、おすすめアプリを決定木で当てる。
原著は同じデータを 2 通りの形で学習させ、木の形がどう変わるかを見せている。

1. 年齢を **カテゴリ**（若者 / 大人）に潰して one-hot 符号化する
2. 年齢を **数値** のまま渡す

`DecisionTreeClassifier` は同点の分割候補から **無作為に 1 つ選ぶ** ので、
木の形は実行のたびに変わる。何が安定していて何が変わるのかを分けて扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#: 原著の元データ。この 6 人ぶんの情報しかない
APP_DATASET = pd.DataFrame(
    {
        "Platform": ["iPhone", "iPhone", "Android", "iPhone", "Android", "Android"],
        "Age": [15, 25, 32, 35, 12, 14],
        "App": [
            "Atom Count",
            "Check Mate Mate",
            "Beehive Finder",
            "Check Mate Mate",
            "Atom Count",
            "Atom Count",
        ],
    }
)

#: 年齢をカテゴリに潰して one-hot 符号化したもの。原著が手で書き下している
APP_DATASET_ONE_HOT = pd.DataFrame(
    {
        "Platform_iPhone": [1, 1, 0, 1, 0, 0],
        "Platform_Android": [0, 0, 1, 0, 1, 1],
        "Age_Young": [1, 0, 0, 0, 1, 1],
        "Age_Adult": [0, 1, 1, 1, 0, 0],
        "App_Atom_Count": [1, 0, 0, 0, 1, 1],
        "App_Beehive_Finder": [0, 0, 1, 0, 0, 0],
        "App_Check_Mate_Mate": [0, 1, 0, 1, 0, 0],
    }
)

#: カテゴリ版で使う特徴量の列。原著の順序をそのまま保つ
CATEGORICAL_FEATURES = ["Platform_iPhone", "Platform_Android", "Age_Adult", "Age_Young"]

#: カテゴリ版の目的変数。3 列を同時に予測する多出力分類になる
CATEGORICAL_LABELS = ["App_Atom_Count", "App_Beehive_Finder", "App_Check_Mate_Mate"]

#: 数値版で使う特徴量の列
NUMERIC_FEATURES = ["Age", "Platform_iPhone", "Platform_Android"]


@dataclass
class TreeSummary:
    """学習した木の形を、比較しやすい形にまとめたもの。"""

    #: 根で使われた特徴量の名前
    root_feature: str
    #: 根の分割しきい値
    root_threshold: float
    #: 根のジニ不純度
    root_gini: float
    #: 節と葉を合わせた数
    node_count: int
    #: 訓練データに対する正解率
    score: float


def _numeric_dataset() -> pd.DataFrame:
    """数値版のデータ。端末だけ one-hot にし、年齢は数値のまま残す。"""
    return pd.DataFrame(
        {
            "Platform_iPhone": [1, 1, 0, 1, 0, 0],
            "Platform_Android": [0, 0, 1, 0, 1, 1],
            "Age": APP_DATASET["Age"],
            "App": APP_DATASET["App"],
        }
    )


def fit_categorical(random_state: int | None = None) -> DecisionTreeClassifier:
    """年齢をカテゴリに潰した版を学習する。

    目的変数が 3 列あるので **多出力分類** になる。scikit-learn の決定木は
    複数の列を同時に予測でき、各節のジニ不純度は列ごとの平均になる。
    """
    features = APP_DATASET_ONE_HOT[CATEGORICAL_FEATURES]
    labels = APP_DATASET_ONE_HOT[CATEGORICAL_LABELS]
    return DecisionTreeClassifier(random_state=random_state).fit(features, labels)


def fit_numeric(random_state: int | None = None) -> DecisionTreeClassifier:
    """年齢を数値のまま渡した版を学習する。

    決定木がしきい値を自分で決めるので、`Age <= 20.0` のような分割が現れる。
    どこで切るかは、隣り合う値の中点から選ばれる（12, 14, 15 と 25 の間なら 20）。
    """
    dataset = _numeric_dataset()
    return DecisionTreeClassifier(random_state=random_state).fit(
        dataset[NUMERIC_FEATURES], dataset["App"]
    )


def summarize(model: DecisionTreeClassifier, feature_names: list[str]) -> TreeSummary:
    """木の形を [TreeSummary] にまとめる。"""
    tree = model.tree_
    if isinstance(model.classes_, list):
        # 多出力分類のときは classes_ がリストになる
        features = APP_DATASET_ONE_HOT[CATEGORICAL_FEATURES]
        labels = APP_DATASET_ONE_HOT[CATEGORICAL_LABELS]
    else:
        dataset = _numeric_dataset()
        features = dataset[NUMERIC_FEATURES]
        labels = dataset["App"]

    return TreeSummary(
        root_feature=feature_names[tree.feature[0]],
        root_threshold=float(tree.threshold[0]),
        root_gini=float(tree.impurity[0]),
        node_count=int(tree.node_count),
        score=float(model.score(features, labels)),
    )


def split_features(model: DecisionTreeClassifier, feature_names: list[str]) -> list[str]:
    """木のなかで実際に分割に使われた特徴量を、節の順に並べる。

    葉は `-2` で表されるので取り除く。同点の候補から無作為に選ばれるため、
    ここが実行のたびに変わる。
    """
    return [feature_names[index] for index in model.tree_.feature if index >= 0]
