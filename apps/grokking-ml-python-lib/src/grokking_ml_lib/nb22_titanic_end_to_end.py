"""原著ノートブック #22 `Chapter_13_End_to_end_example/End_to_end_example.ipynb`。

タイタニックの生存予測を、前処理からモデル選択まで通しでやる回である。
本の最終章で、これまでのアルゴリズムが一堂に会する。

1. 欠損の処理（`Cabin` を捨て、`Age` は中央値、`Embarked` は `U`）
2. one-hot 符号化と離散化（`Age` を 10 歳刻みに）
3. 訓練 / 検証 / テストに 6:2:2 で分ける
4. 7 つのモデルを学習し、正解率と F1 スコアで比べる
5. SVM のグリッドサーチ

**再現の鍵は 3 分割の乱数** である。原著は `random_state=100` を
固定しているので、そこが合えばモデルの数字も合う。
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from grokking_ml_lib.datasets import load_csv

#: 原著が固定する分割の種
SPLIT_SEED = 100

#: 年齢の離散化の区切り。10 歳刻みで 8 区間
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80]

#: 学習に使わない列
DROPPED_COLUMNS = ["Name", "Ticket", "PassengerId"]


def load_raw() -> pd.DataFrame:
    """生のタイタニックデータ 891 行。"""
    return load_csv("titanic.csv")


def missing_counts(data: pd.DataFrame) -> dict[str, int]:
    """列ごとの欠損数。原著の `isna().sum()`。"""
    return {name: int(count) for name, count in data.isna().sum().items()}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """欠損を片付ける。

    - `Cabin` は 687 / 891 が欠損。**列ごと捨てる**
    - `Age` は中央値（28.0）で埋める
    - `Embarked` は `U`（Unknown）という新しい区分にする
    """
    data = raw.drop("Cabin", axis=1).copy()
    data["Age"] = data["Age"].fillna(raw["Age"].median())
    data["Embarked"] = data["Embarked"].fillna("U")
    return data


def preprocess(cleaned: pd.DataFrame) -> pd.DataFrame:
    """one-hot 符号化と離散化。

    `Sex`・`Embarked`・`Pclass` を one-hot にし、`Age` を 10 歳刻みに切って
    それも one-hot にする。最後に学習に使わない列を落とす。
    """
    data = cleaned.copy()

    for column in ("Sex", "Embarked", "Pclass"):
        dummies = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, dummies], axis=1).drop(column, axis=1)

    categorized = pd.cut(data["Age"], AGE_BINS)
    data = data.drop("Age", axis=1)
    age_dummies = pd.get_dummies(categorized, prefix="Categorized_age")
    data = pd.concat([data, age_dummies], axis=1)

    return data.drop(DROPPED_COLUMNS, axis=1)


class Split:
    """訓練 / 検証 / テストの 3 分割。"""

    def __init__(self, data: pd.DataFrame) -> None:
        features = data.drop(["Survived"], axis=1)
        labels = data["Survived"]

        # 原著は 2 段階で切る。まず 6:4、その 4 を半分ずつ
        train_x, rest_x, train_y, rest_y = train_test_split(
            features, labels, test_size=0.4, random_state=SPLIT_SEED
        )
        validation_x, test_x, validation_y, test_y = train_test_split(
            rest_x, rest_y, test_size=0.5, random_state=SPLIT_SEED
        )

        self.train_x, self.train_y = train_x, train_y
        self.validation_x, self.validation_y = validation_x, validation_y
        self.test_x, self.test_y = test_x, test_y


def model_factories() -> dict[str, callable]:
    """原著が試す 7 つのモデル。

    原著は種を渡していない。決定木・ランダムフォレスト・
    ブースティング系は乱数を使うので、**指定しないと毎回変わる**。
    ここでは再現のために 0 を渡す。
    """
    return {
        "Logistic regression": lambda: LogisticRegression(max_iter=1000),
        "Decision tree": lambda: DecisionTreeClassifier(random_state=0),
        "Naive Bayes": GaussianNB,
        "SVM": lambda: SVC(random_state=0),
        "Random forest": lambda: RandomForestClassifier(random_state=0),
        "Gradient boosting": lambda: GradientBoostingClassifier(random_state=0),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=0),
    }


def fit_all(split: Split) -> dict[str, object]:
    """7 つのモデルを訓練データで学習する。"""
    models = {}
    for name, factory in model_factories().items():
        model = factory()
        model.fit(split.train_x, split.train_y)
        models[name] = model
    return models


def accuracies(models: dict, split: Split) -> dict[str, float]:
    """検証データに対する正解率。"""
    return {
        name: float(model.score(split.validation_x, split.validation_y))
        for name, model in models.items()
    }


def f1_scores(models: dict, split: Split) -> dict[str, float]:
    """検証データに対する F1 スコア。

    正解率だけでは不十分な理由を見せるための指標である。
    生存者が少数派なので、「全員死亡」と答えても正解率は 6 割を超える。
    """
    return {
        name: float(f1_score(split.validation_y, model.predict(split.validation_x)))
        for name, model in models.items()
    }


def grid_search(split: Split, kernel: str = "rbf") -> dict[tuple[float, float], float]:
    """原著のグリッドサーチ。`C` と `gamma` の組み合わせを総当たりする。

    原著は `GridSearchCV` を import しているが、**実際には使わず
    手で 9 通り書き並べている**。ここは総当たりを 1 つのループにまとめた。
    """
    scores = {}
    for c in (1, 10, 100):
        for gamma in (0.1, 1, 10):
            model = SVC(kernel=kernel, C=c, gamma=gamma, random_state=0)
            model.fit(split.train_x, split.train_y)
            scores[(c, gamma)] = float(model.score(split.validation_x, split.validation_y))
    return scores


def majority_baseline(split: Split) -> float:
    """「全員死亡」と答えたときの正解率。

    原著は基準を出していない。正解率だけを見て良し悪しを言えないことは、
    この値と比べて初めて分かる。
    """
    return float((split.validation_y == 0).mean())
