"""原著ノートブック #02 `Chapter_03_Linear_Regression/House_price_predictions.ipynb`。

ハイデラバードの住宅データ 2518 件から価格を予測する。面積 1 つだけを使う単回帰と、
全 40 列を前処理してから使う重回帰の 2 本立てになっている。

前処理は原著の手順をそのまま踏襲する。
1. 欠損（`9` で符号化されている）を含む末尾の行を落とす
2. 数値列（`Area` と `No. of Bedrooms`）を標準化する
3. カテゴリ列（`Location`）を one-hot 符号化する
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from grokking_ml_lib.datasets import load_csv

#: 原著が欠損ありとして切り落とす位置。これ以降の行は `9` で符号化された欠損を含む
VALID_ROWS = 2434

#: 標準化する数値列
NUMERIC_COLUMNS = ["Area", "No. of Bedrooms"]


def load_data() -> pd.DataFrame:
    """ハイデラバードの住宅データを読み込む。2518 行 × 40 列。"""
    return load_csv("Hyderabad.csv")


def fit_area_only(data: pd.DataFrame) -> LinearRegression:
    """面積 1 列だけで価格を予測する単回帰。

    `data[['Area']]` と二重の括弧で取り出しているのは、scikit-learn が 2 次元の
    特徴量行列を要求するためである。`data['Area']` では 1 次元になって弾かれる。
    """
    return LinearRegression().fit(data[["Area"]], data["Price"])


@dataclass
class Standardizer:
    """数値列の標準化に使った平均と標準偏差。

    新しい物件を予測するときも、学習時とまったく同じ平均・標準偏差で変換する
    必要がある。ここに保持しておかないと、予測のたびに計算し直すことになり、
    データが変われば静かに食い違う。
    """

    area_mean: float
    area_std: float
    bedrooms_mean: float
    bedrooms_std: float

    def transform(self, area: float, bedrooms: float) -> tuple[float, float]:
        return (
            (area - self.area_mean) / self.area_std,
            (bedrooms - self.bedrooms_mean) / self.bedrooms_std,
        )


@dataclass
class Preprocessed:
    """前処理を通したあとのデータ一式。"""

    features: pd.DataFrame
    labels: pd.Series
    standardizer: Standardizer


def preprocess(data: pd.DataFrame) -> Preprocessed:
    """原著の前処理 3 手順をそのまま行う。"""
    truncated = data[:VALID_ROWS].copy()

    # pandas の std() は既定で不偏分散（ddof=1）を使う。原著もこの既定に乗っている
    standardizer = Standardizer(
        area_mean=truncated["Area"].mean(),
        area_std=truncated["Area"].std(),
        bedrooms_mean=truncated["No. of Bedrooms"].mean(),
        bedrooms_std=truncated["No. of Bedrooms"].std(),
    )
    truncated["Area"] = (truncated["Area"] - standardizer.area_mean) / standardizer.area_std
    truncated["No. of Bedrooms"] = (
        truncated["No. of Bedrooms"] - standardizer.bedrooms_mean
    ) / standardizer.bedrooms_std

    encoded = pd.get_dummies(truncated, columns=["Location"], prefix="Location", dtype=int)

    return Preprocessed(
        features=encoded.drop("Price", axis=1),
        labels=encoded["Price"],
        standardizer=standardizer,
    )


def fit_all_features(prepared: Preprocessed) -> LinearRegression:
    """前処理済みの全特徴量で重回帰を学習する。"""
    return LinearRegression().fit(prepared.features, prepared.labels)


def rmse(labels: pd.Series, predictions: NDArray[np.float64]) -> float:
    """二乗平均平方根誤差。原著は `mean_squared_error` の平方根で求めている。"""
    return float(np.sqrt(mean_squared_error(labels, predictions)))


def predict_new_house(
    model: LinearRegression,
    prepared: Preprocessed,
    area: float,
    bedrooms: int,
    location: str = "Gachibowli",
) -> float:
    """新しい物件の価格を予測する。

    学習時と同じ列・同じ順序の 1 行を組み立てるのが要点である。one-hot 符号化した
    地域の列は 277 列中 275 列あり、そのうち 1 つだけを 1 にして残りは 0 にする。
    列が 1 つでも欠けたり順序が違ったりすると、scikit-learn が弾くか、
    弾かずに別の特徴量として解釈してしまう。
    """
    columns = prepared.features.columns
    location_column = f"Location_{location}"
    if location_column not in columns:
        raise ValueError(f"学習データに無い地域です: {location}")

    scaled_area, scaled_bedrooms = prepared.standardizer.transform(area, bedrooms)
    row = pd.DataFrame({column: [0] for column in columns})
    row["Area"] = scaled_area
    row["No. of Bedrooms"] = scaled_bedrooms
    row[location_column] = 1

    return float(model.predict(row[columns])[0])
