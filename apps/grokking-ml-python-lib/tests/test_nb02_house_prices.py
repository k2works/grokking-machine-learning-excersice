"""原著ノートブック #02 の再現テスト。

単回帰は原著の数値と完全に一致する。一方で全特徴量の重回帰は、one-hot 符号化に
よって設計行列がランク落ちしており **係数が一意に決まらない**。そこは一致を
求めず、当てはまりの良さ（RMSE）と設計行列の性質で検証する。
"""

import numpy as np
import pytest

from grokking_ml_lib.nb02_house_prices import (
    VALID_ROWS,
    fit_all_features,
    fit_area_only,
    load_data,
    predict_new_house,
    preprocess,
    rmse,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def prepared(data):
    return preprocess(data)


@pytest.fixture(scope="module")
def full_model(prepared):
    return fit_all_features(prepared)


def test_データセットの形は原著と同じ(data) -> None:
    # 原著の出力: The dataset has 2518 rows, and 40 columns
    assert data.shape == (2518, 40)


def test_単回帰の係数は原著と同じ数値になる(data) -> None:
    # 原著の出力
    #   y-intercept: -6222669.083283698
    #   slope (coefficient of Area): 9753.940608184039
    model = fit_area_only(data)

    assert float(model.intercept_) == pytest.approx(-6222669.083283698, rel=1e-12)
    assert model.coef_[0] == pytest.approx(9753.940608184039, rel=1e-12)


def test_欠損を含む末尾の行を落とす(data, prepared) -> None:
    assert VALID_ROWS == 2434
    assert len(prepared.features) == VALID_ROWS
    assert len(prepared.features) < len(data)


def test_標準化した列は平均0分散1になる(prepared) -> None:
    for column in ["Area", "No. of Bedrooms"]:
        assert prepared.features[column].mean() == pytest.approx(0.0, abs=1e-12)
        # pandas の std() は不偏分散（ddof=1）なのでちょうど 1 になる
        assert prepared.features[column].std() == pytest.approx(1.0, rel=1e-12)


def test_標準化の統計量を保持している(prepared) -> None:
    standardizer = prepared.standardizer

    assert standardizer.area_mean == pytest.approx(1644.1516023007396)
    assert standardizer.area_std == pytest.approx(748.1348121200747)
    assert standardizer.bedrooms_mean == pytest.approx(2.6261298274445357)
    assert standardizer.bedrooms_std == pytest.approx(0.6850461155463963)


def test_one_hot符号化で277列になる(prepared) -> None:
    # 原著の出力: X_full.loc[0] ... Length: 277
    assert prepared.features.shape == (2434, 277)

    location_columns = [c for c in prepared.features.columns if c.startswith("Location_")]
    # 元の 40 列から Price と Location を除いた 38 列 + 地域の one-hot
    assert len(location_columns) == 277 - 38


def test_設計行列はランク落ちしている(prepared) -> None:
    # one-hot 符号化した地域の列は合計すると常に 1 になり、切片と線形従属になる。
    # そのため最小二乗解が一意に決まらない。原著の切片が 7.29e17 という
    # 非常識な値になっているのはこのため
    design = np.c_[np.ones(len(prepared.features)), prepared.features.values.astype(float)]

    assert np.linalg.matrix_rank(design) == design.shape[1] - 1


def test_最小特異値だけがゼロとみなされる(prepared) -> None:
    # F# 版（Math.NET）だけ数値が食い違う原因を固定しておく。
    # 理論上ゼロの特異値が丸め誤差で 1.44e-15 として残り、そのすぐ上は 0.87。
    # LAPACK はサイズとマシンイプシロンから閾値を決めて前者だけを切り落とすが、
    # 打ち切りを行わない実装では 1/1.44e-15 倍の増幅が係数に乗る
    design = np.c_[np.ones(len(prepared.features)), prepared.features.values.astype(float)]
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = singular_values[0] * max(design.shape) * np.finfo(float).eps

    assert singular_values[-1] < tolerance
    assert singular_values[-2] > tolerance
    assert singular_values[-1] == pytest.approx(1.44e-15, rel=1e-2)


def test_全特徴量モデルのRMSEは原著とほぼ同じ(prepared, full_model) -> None:
    # 原著の出力: Root Mean Squared Error (RMSE) of the model: 3981401.4927888927
    # 係数は一意でないが、当てはまりの良さは解の取り方によらずほぼ同じになる。
    # 完全一致しないのは最小二乗ソルバの実装差による
    predictions = full_model.predict(prepared.features)

    assert rmse(prepared.labels, predictions) == pytest.approx(3981401.4927888927, rel=1e-5)


def test_全特徴量モデルは単回帰より当てはまりが良い(data, prepared, full_model) -> None:
    simple = fit_area_only(data)
    simple_rmse = rmse(data["Price"], simple.predict(data[["Area"]]))
    full_rmse = rmse(prepared.labels, full_model.predict(prepared.features))

    assert full_rmse < simple_rmse


def test_新しい物件の予測は原著とほぼ同じ(prepared, full_model) -> None:
    # 原著の出力: Predicted price for a house with size 1000 and 3 bedrooms: 6,006,016.00
    # 係数が一意でないぶん、学習データに無い点の予測もわずかにぶれる
    predicted = predict_new_house(full_model, prepared, area=1000, bedrooms=3)

    assert predicted == pytest.approx(6006016.00, rel=1e-3)


def test_学習データに無い地域はエラーになる(prepared, full_model) -> None:
    with pytest.raises(ValueError, match="学習データに無い地域"):
        predict_new_house(full_model, prepared, area=1000, bedrooms=3, location="Atlantis")
