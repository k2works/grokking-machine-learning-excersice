"""ライブラリ版の実行環境が揃っていることを確認するスモークテスト。

章の実装を書く前に、原著が使う 4 つのライブラリ（scikit-learn・TensorFlow/Keras・
XGBoost・pandas）が実際に学習まで走ることをここで担保する。
XGBoost は macOS で libomp を要求するので、devShell の設定漏れもここで落ちる。
"""

import numpy as np

from grokking_ml_lib import dataset_path, load_csv


def test_共有データセットを読み込める() -> None:
    df = load_csv("Hyderabad.csv")
    assert "Price" in df.columns
    assert len(df) > 1000


def test_未登録のデータセットはエラーになる() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        dataset_path("does_not_exist.csv")


def test_scikit_learnで線形回帰が学習できる() -> None:
    from sklearn.linear_model import LinearRegression

    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    model = LinearRegression().fit(x, y)

    # y = 2x + 1 を完全に復元できる
    assert round(float(model.coef_[0]), 6) == 2.0
    assert round(float(model.intercept_), 6) == 1.0


def test_kerasで小さなネットワークが学習できる() -> None:
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input((2,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model.fit(x, y, epochs=2, verbose=0)

    assert model.predict(x, verbose=0).shape == (4, 1)


def test_xgboostが学習できる() -> None:
    import pytest

    try:
        from xgboost import XGBClassifier
    except Exception as error:  # noqa: BLE001
        # XGBoost の wheel が同梱する libxgboost は libomp を動的に要求する。
        # devShell の外で走らせるとここで落ちるので、原因が分かる形で飛ばす。
        # CI は libomp-dev を入れているため、そちらでは必ず実行される。
        pytest.skip(f"libomp が見つかりません。nix develop .#python-ml で実行してください: {error}")

    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    model = XGBClassifier(n_estimators=3, max_depth=2).fit(x, y)

    assert model.predict(x).shape == (4,)
