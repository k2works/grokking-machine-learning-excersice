"""原著ノートブック #19 `Chapter_11_Support_Vector_Machines/SVM_graphical_example.ipynb`。

[#17](nb17.md) で作った 3 つのデータセットに、いろいろな SVM を当てる回である。

- 直線データに線形カーネル。`C` を変えて正則化の効き方を見る
- 円データに多項式カーネル。`degree` を変える
- 二重円データに RBF カーネル。`gamma` を変える

原著は図を並べて説明するが、**正解率は数字で印刷されている**。
9 つの数字がすべて突き合わせの対象になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.svm import SVC

from grokking_ml_lib.datasets import load_csv


def load(name: str) -> pd.DataFrame:
    """[#17](nb17.md) が作った CSV を読む。"""
    return load_csv(f"{name}.csv")


def features(data: pd.DataFrame) -> NDArray[np.float64]:
    return np.array(data[["x_1", "x_2"]])


def labels(data: pd.DataFrame) -> NDArray[np.int_]:
    return np.array(data["y"])


def fit(data: pd.DataFrame, **kwargs) -> SVC:
    """`SVC` に渡す引数だけを変えて学習する。

    原著は 9 通りをコピペで並べているので、ここでは 1 つにまとめた。
    """
    model = SVC(**kwargs)
    model.fit(features(data), labels(data))
    return model


def accuracy(model: SVC, data: pd.DataFrame) -> float:
    """学習データに対する正解率。原著の `score` と同じ。"""
    return float(model.score(features(data), labels(data)))


def fit_and_score(name: str, **kwargs) -> float:
    """データセットを読み、学習し、正解率を返すところまで。"""
    data = load(name)
    return accuracy(fit(data, **kwargs), data)
