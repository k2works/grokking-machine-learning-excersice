"""原著ノートブック #11 `Chapter_09_Decision_Trees/University_Admissions.ipynb`。

大学院の入学審査データ 400 件から、合格するかどうかを決定木で当てる。
原著は **同じデータで木の大きさを変えて、過学習の様子を見せる** 構成になっている。

1. 制限なしの木 — 訓練データを 100% 当てる（明らかに過学習）
2. 深さ 3・葉の最小件数 10 の木 — 88.5% に落ちるが、木が読める大きさになる
3. 2 特徴量（GRE と TOEFL）だけの木を深さ 1・2・無制限で比べる

[#10](nb10.md) が 12 点の作り物だったのに対し、こちらは実データである。
**訓練データの正解率が 1.0 になっても嬉しくない** ことがはっきり出る。
"""

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from grokking_ml_lib.datasets import dataset_path

#: 原著が合格とみなす基準
ADMISSION_THRESHOLD = 0.75

#: 2 特徴量だけで学習するときに使う列
EXAM_FEATURES = ["GRE Score", "TOEFL Score"]


def load_data() -> pd.DataFrame:
    """入学審査データを読み込み、合否のラベルを付ける。

    原著は `Chance of Admit`（合格確率）を 0.75 で切って 2 値にし、
    元の列を落としている。`index_col=0` で `Serial No.` を索引にする。
    """
    data = pd.read_csv(dataset_path("Admission_Predict.csv"), index_col=0)
    data["Admitted"] = data["Chance of Admit"] >= ADMISSION_THRESHOLD
    return data.drop(["Chance of Admit"], axis=1)


def features(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(["Admitted"], axis=1)


def labels(data: pd.DataFrame) -> pd.Series:
    return data["Admitted"]


def fit_full(data: pd.DataFrame, random_state: int | None = 0) -> DecisionTreeClassifier:
    """制限なしの木。訓練データを完全に覚えてしまう。"""
    return DecisionTreeClassifier(random_state=random_state).fit(
        features(data), labels(data)
    )


def fit_smaller(data: pd.DataFrame, random_state: int | None = 0) -> DecisionTreeClassifier:
    """原著が「過学習しない小さい木」として作る設定。

    3 つの制限を同時に掛けている。
    - `max_depth=3` — 木の深さ
    - `min_samples_leaf=10` — 葉に残る件数の下限
    - `min_samples_split=10` — 分割してよい節の件数の下限
    """
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=10,
        random_state=random_state,
    ).fit(features(data), labels(data))


def fit_exams(
    data: pd.DataFrame, max_depth: int | None, random_state: int | None = 0
) -> DecisionTreeClassifier:
    """GRE と TOEFL の 2 特徴量だけで学習する。境界を 2 次元で描けるようにするため。"""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state).fit(
        data[EXAM_FEATURES], labels(data)
    )


def predict_applicant(
    model: DecisionTreeClassifier, data: pd.DataFrame, values: list[float]
) -> bool:
    """1 人ぶんの出願情報から合否を予測する。

    列名つきの 1 行に組み立ててから渡す。生のリストを渡すと
    scikit-learn が「学習時は列名があったのに」と警告を出す。
    """
    row = pd.DataFrame([values], columns=features(data).columns)
    return bool(model.predict(row)[0])


def split_conditions(
    model: DecisionTreeClassifier, feature_names: list[str]
) -> list[tuple[str, float]]:
    """木のなかで実際に分割に使われた条件を、節の順に並べる。"""
    tree = model.tree_
    return [
        (feature_names[feature], float(threshold))
        for feature, threshold in zip(tree.feature, tree.threshold)
        if feature >= 0
    ]
