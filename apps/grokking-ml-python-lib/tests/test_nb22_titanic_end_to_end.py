"""原著ノートブック #22 の再現テスト。

前処理・分割・7 モデルの評価・グリッドサーチまで通しで再現する。
**7 つのうち 5 つが桁まで一致** し、残り 2 つは原著が乱数の種を
渡していないことによる。
"""

import pytest
from sklearn.metrics import f1_score

from grokking_ml_lib.nb22_titanic_end_to_end import (
    AGE_BINS,
    Split,
    accuracies,
    clean,
    f1_scores,
    fit_all,
    grid_search,
    load_raw,
    majority_baseline,
    missing_counts,
    preprocess,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def data(raw):
    return preprocess(clean(raw))


@pytest.fixture(scope="module")
def split(data):
    return Split(data)


@pytest.fixture(scope="module")
def models(split):
    return fit_all(split)


def test_データセットは891行(raw) -> None:
    # 原著の出力: The dataset has 891 rows
    assert len(raw) == 891


def test_生存者は342人(raw) -> None:
    # 原著の出力: 342 passengers survived out of 891
    assert int(raw["Survived"].sum()) == 342


def test_欠損のある列は3つ(raw) -> None:
    # 原著の isna().sum() の出力
    counts = missing_counts(raw)

    assert counts["Age"] == 177
    assert counts["Cabin"] == 687
    assert counts["Embarked"] == 2
    assert sum(1 for value in counts.values() if value > 0) == 3


def test_年齢の中央値は28(raw) -> None:
    # 原著の出力: 28.0
    assert raw["Age"].median() == 28.0


def test_Cabinは8割近くが欠損(raw) -> None:
    # 687 / 891 = 0.771。原著が列ごと捨てる根拠
    assert missing_counts(raw)["Cabin"] / len(raw) > 0.75


def test_前処理後は欠損がなくなる(raw) -> None:
    counts = missing_counts(clean(raw))

    assert all(value == 0 for value in counts.values())
    assert "Cabin" not in counts


def test_年齢の区切りは10歳刻みで8区間() -> None:
    assert AGE_BINS == [0, 10, 20, 30, 40, 50, 60, 70, 80]
    assert len(AGE_BINS) - 1 == 8


def test_one_hot後の列は21になる(data) -> None:
    # Survived + SibSp + Parch + Fare + Sex 2 + Embarked 4 + Pclass 3 + 年齢 8
    assert len(data.columns) == 21
    assert "Sex_female" in data.columns
    assert "Embarked_U" in data.columns
    assert "Categorized_age_(70, 80]" in data.columns


def test_学習に使わない列は落とす(data) -> None:
    for column in ("Name", "Ticket", "PassengerId", "Age", "Sex", "Cabin"):
        assert column not in data.columns


def test_分割の件数は原著と一致する(split) -> None:
    # 原著の出力: 534 / 178 / 179
    assert len(split.train_x) == 534
    assert len(split.validation_x) == 178
    assert len(split.test_x) == 179
    # ラベル側も同じ件数
    assert len(split.train_y) == 534
    assert len(split.validation_y) == 178
    assert len(split.test_y) == 179


def test_分割は全体を覆う(split) -> None:
    assert len(split.train_x) + len(split.validation_x) + len(split.test_x) == 891


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Logistic regression", 0.7696629213483146),
        ("Naive Bayes", 0.7471910112359551),
        ("SVM", 0.6797752808988764),
        ("Gradient boosting", 0.8089887640449438),
        ("AdaBoost", 0.7359550561797753),
    ],
)
def test_正解率が原著と一致する5つ(models, split, name, expected) -> None:
    # 原著の「Scores of the models」の出力
    assert accuracies(models, split)[name] == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("name", "expected"), [("Decision tree", 0.7808988764044944), ("Random forest", 0.7808988764044944)]
)
def test_乱数を使うモデルは原著と一致しない(models, split, name, expected) -> None:
    # 原著はどちらも 0.7696629213483146。
    # 原著は random_state を渡していないので、そもそも実行のたびに変わる。
    # ここでは 0 を固定した値を記録する
    assert accuracies(models, split)[name] == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Logistic regression", 0.6870229007633588),
        ("Naive Bayes", 0.6808510638297872),
        ("SVM", 0.4),
        ("Gradient boosting", 0.7384615384615385),
        ("AdaBoost", 0.6466165413533834),
    ],
)
def test_F1スコアが原著と一致する5つ(models, split, name, expected) -> None:
    # 原著の「F1-scores of the models」の出力
    assert f1_scores(models, split)[name] == pytest.approx(expected, rel=1e-15)


def test_SVMは正解率とF1の差が大きい(models, split) -> None:
    # 正解率 0.680 に対して F1 は 0.400。
    # 多数派（死亡）に寄せた予測をしているので、正解率だけでは見抜けない。
    # 原著が F1 を持ち出す理由がこの差である
    accuracy = accuracies(models, split)["SVM"]
    f1 = f1_scores(models, split)["SVM"]

    assert accuracy - f1 > 0.25


def test_勾配ブースティングが両方の指標で最良(models, split) -> None:
    accuracy = accuracies(models, split)
    f1 = f1_scores(models, split)

    assert max(accuracy, key=accuracy.get) == "Gradient boosting"
    assert max(f1, key=f1.get) == "Gradient boosting"


def test_テストデータでの成績は原著と一致する(models, split) -> None:
    # 原著の出力: 0.8324022346368715 / 0.8026315789473685
    model = models["Gradient boosting"]

    assert model.score(split.test_x, split.test_y) == pytest.approx(
        0.8324022346368715, rel=1e-15
    )
    assert f1_score(split.test_y, model.predict(split.test_x)) == pytest.approx(
        0.8026315789473685, rel=1e-15
    )


def test_検証よりテストのほうが成績が良い(models, split) -> None:
    # 検証 0.809、テスト 0.832。**過学習していれば逆になる**。
    # 原著は触れていないが、モデル選択が妥当だったことの傍証になる
    model = models["Gradient boosting"]
    validation = accuracies(models, split)["Gradient boosting"]

    assert model.score(split.test_x, split.test_y) > validation


def test_全員死亡と答える基準は0_607(split) -> None:
    # 原著は基準を出していない。SVM の 0.680 は基準を 0.07 上回るだけで、
    # AdaBoost の 0.736 でも 0.13 の改善にすぎない
    assert majority_baseline(split) == pytest.approx(0.6067415730337079, rel=1e-15)


def test_全モデルが基準を上回る(models, split) -> None:
    baseline = majority_baseline(split)

    assert all(score > baseline for score in accuracies(models, split).values())


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ((1, 0.1), 0.702247191011236),
        ((1, 1), 0.6966292134831461),
        ((1, 10), 0.6685393258426966),
        ((10, 0.1), 0.7247191011235955),
        ((10, 1), 0.6910112359550562),
        ((10, 10), 0.651685393258427),
    ],
)
def test_グリッドサーチの結果は原著と一致する(split, params, expected) -> None:
    # 原著が印刷している 6 通り（C=1, 10 × gamma=0.1, 1, 10）
    assert grid_search(split)[params] == pytest.approx(expected, rel=1e-15)


def test_gammaは小さいほうが良い(split) -> None:
    # どの C でも gamma=0.1 が最良。データが 20 次元あるので、
    # gamma を上げると各点の影響範囲が狭くなりすぎる
    scores = grid_search(split)

    for c in (1, 10, 100):
        assert scores[(c, 0.1)] > scores[(c, 1)] > scores[(c, 10)]


def test_グリッドサーチの最良でも勾配ブースティングに届かない(models, split) -> None:
    # SVM は調整しても 0.725。勾配ブースティングの 0.809 には遠い。
    # 原著はグリッドサーチの結果を最終モデルに採用していない
    assert max(grid_search(split).values()) < accuracies(models, split)["Gradient boosting"]
