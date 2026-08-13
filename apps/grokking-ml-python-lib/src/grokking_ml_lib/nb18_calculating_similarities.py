"""原著ノートブック #18 `Chapter_11_Support_Vector_Machines/Calculating_similarities.ipynb`。

RBF カーネルの正体を **手で計算して見せる** 回である。

7 点の小さなデータに対して、全対の類似度を計算して 7 列の特徴量を足し、
そこに **線形** SVM を当てる。カーネルトリックを使わず、
「カーネルとは特徴量を増やすことだ」を目に見える形にしている。

原著には Turi Create を使った古い版もあるが、こちらは scikit-learn 版である。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.svm import SVC

#: 原著が予測に使う符号。ラベル 0 を -1 に読み替えたもの
PREDICTION_SIGNS = (-1, -1, -1, 1, 1, 1, 1)


def dataset() -> pd.DataFrame:
    """原著の 7 点。原点とその周りに 6 点が並ぶ。"""
    return pd.DataFrame(
        {
            "x1": [0, -1, 0, 0, 1, -1, 1],
            "x2": [0, 0, -1, 1, 0, 1, -1],
            "y": [0, 0, 0, 1, 1, 1, 1],
        }
    )


def similarity(x, y) -> float:
    """原著の類似度。RBF（ガウス）カーネルそのもの。

    `exp(-(x1-y1)^2 - (x2-y2)^2)` は `exp(-||x-y||^2)`、
    つまり γ = 1 の RBF カーネルである。
    同じ点なら 1、離れるほど急速に 0 に近づく。
    """
    return float(np.exp(-((x[0] - y[0]) ** 2) - (x[1] - y[1]) ** 2))


def similarity_matrix(data: pd.DataFrame) -> NDArray[np.float64]:
    """全対の類似度を並べた 7 × 7 の行列。"""
    points = data[["x1", "x2"]].to_numpy(dtype=float)
    return np.array([[similarity(a, b) for b in points] for a in points])


def with_similarities(data: pd.DataFrame) -> pd.DataFrame:
    """元の 2 列に、類似度の 7 列（`Sim0`〜`Sim6`）を足したもの。

    原著は列ごとに `Sim{i}` を作る。行列としては対称なので、
    行で作っても列で作っても同じものになる。
    """
    result = data.copy()
    matrix = similarity_matrix(data)
    for index in range(len(data)):
        result[f"Sim{index}"] = matrix[index]
    return result


def features(data: pd.DataFrame) -> pd.DataFrame:
    """SVM に渡す特徴量。`y` を落とした 9 列。"""
    return with_similarities(data).drop(columns=["y"])


def fit(data: pd.DataFrame) -> SVC:
    """**線形** SVM を、類似度を足した特徴量に当てる。

    `kernel='rbf'` を指定していないのが要点である。
    カーネルは既に特徴量として展開済みなので、線形で足りる。
    """
    model = SVC(kernel="linear")
    model.fit(features(data), data["y"])
    return model


def svm_rbf_prediction(data: pd.DataFrame, new_point) -> float:
    """原著が手で書いた予測式。

    学習した SVM の係数ではなく、**ラベルの符号をそのまま重みにする**。
    `similarity` の重み付き和が正なら 1、負なら 0 と読む。
    """
    points = data[["x1", "x2"]].to_numpy(dtype=float)
    similarities = [similarity(new_point, point) for point in points]
    return float(np.dot(similarities, PREDICTION_SIGNS))


def training_predictions(data: pd.DataFrame) -> list[float]:
    """7 点それぞれに対する `svm_rbf_prediction` の値。"""
    points = data[["x1", "x2"]].to_numpy(dtype=float)
    return [svm_rbf_prediction(data, point) for point in points]
