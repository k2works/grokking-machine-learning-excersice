"""原著ノートブック #20 の再現テスト。

原著が印刷する 6 つの正解率のうち 5 つが一致する。
唯一外れる AdaBoost は、**scikit-learn から実装が削除された** ためである。
"""

import pytest

from grokking_ml_lib.nb20_ensemble_spam import (
    BATCHES,
    batch,
    features,
    fit_adaboost,
    fit_decision_tree,
    fit_gradient_boosting,
    fit_random_forest,
    fit_weak_learner,
    labels,
    predictions,
    score,
    spam_dataset,
    split_of,
)


@pytest.fixture(scope="module")
def data():
    return spam_dataset()


def test_データセットは18通(data) -> None:
    assert len(data) == 18
    assert list(data.columns) == ["Lottery", "Sale", "Spam"]
    # スパムとそうでないものが 9 通ずつ
    assert int(labels(data).sum()) == 9


def test_特徴量は2列だけ(data) -> None:
    assert list(features(data).columns) == ["Lottery", "Sale"]


def test_制限なしの決定木は丸暗記する(data) -> None:
    # 原著の出力: 1.0
    # 18 点を完全に分けきる。良い結果に見えるが過学習そのもの
    assert score(fit_decision_tree(data), data) == 1.0


def test_深さ1に制限すると丸暗記できない(data) -> None:
    # 1 本の直線では 18 点を分けられない
    assert score(fit_decision_tree(data, max_depth=1), data) < 1.0


def test_3組は6通ずつ重複なく分かれる(data) -> None:
    # 原著は loc で添字を直に指定して切る。並び順のまま 6 通ずつ
    assert [len(indices) for indices in BATCHES] == [6, 6, 6]
    assert sorted(index for indices in BATCHES for index in indices) == list(range(18))


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 1.0), (1, 1.0), (2, 0.8333333333333334)]
)
def test_弱学習器の正解率は原著と一致する(data, index, expected) -> None:
    # 原著の出力
    #   Weak learner 1 training accuracy: 1.0
    #   Weak learner 2 training accuracy: 1.0
    #   Weak learner 3 training accuracy: 0.8333333333333334
    model = fit_weak_learner(data, index)

    assert score(model, batch(data, index)) == pytest.approx(expected, rel=1e-15)


def test_弱学習器はそれぞれ違う分割を選ぶ(data) -> None:
    # 6 通ずつのデータが違うので、選ぶ特徴量もしきい値も変わる。
    # これがランダムフォレストの「多様性」の源になる
    splits = [split_of(fit_weak_learner(data, index)) for index in range(3)]

    assert splits == [("Lottery", 4.5), ("Sale", 8.0), ("Sale", 5.5)]


def test_ランダムフォレストの正解率は原著と一致する(data) -> None:
    # 原著の出力: 0.8333333333333334
    assert score(fit_random_forest(data), data) == pytest.approx(
        0.8333333333333334, rel=1e-15
    )


def test_ランダムフォレストは丸暗記しない(data) -> None:
    # 深さ 1 の木を 5 本。1 本の制限なしの木（1.0）より低いが、
    # 過学習していないという意味ではこちらが健全
    assert score(fit_random_forest(data), data) < score(fit_decision_tree(data), data)


def test_ランダムフォレストは5本の木を持つ(data) -> None:
    assert len(fit_random_forest(data).estimators_) == 5


def test_AdaBoostは原著と一致しない(data) -> None:
    # 原著の出力は 0.8888888888888888。いまは 0.7777777777777778 になる。
    # 原著の既定 algorithm='SAMME.R'（実数版）が scikit-learn 1.6 で削除され、
    # SAMME（離散版）しか選べなくなったため
    assert score(fit_adaboost(data), data) == pytest.approx(
        0.7777777777777778, rel=1e-15
    )


def test_AdaBoostのalgorithm引数はもう存在しない() -> None:
    # 原著の版では algorithm='SAMME.R' が既定だった。
    # 1.9 では引数そのものが消えている
    import inspect

    from sklearn.ensemble import AdaBoostClassifier

    assert "algorithm" not in inspect.signature(AdaBoostClassifier.__init__).parameters


def test_AdaBoostは6本の木を持つ(data) -> None:
    assert len(fit_adaboost(data).estimators_) == 6


def test_勾配ブースティングの正解率は原著と一致する(data) -> None:
    # 原著の出力: 0.8888888888888888
    assert score(fit_gradient_boosting(data), data) == pytest.approx(
        0.8888888888888888, rel=1e-15
    )


def test_アンサンブルは1本の木より正解率が低い(data) -> None:
    # 原著が「ブースティングは正確だが過学習からは遠い」と書いているところ。
    # 学習データの正解率だけを見れば、1 本の木（1.0）が最も高い
    single = score(fit_decision_tree(data), data)

    assert score(fit_random_forest(data), data) < single
    assert score(fit_adaboost(data), data) < single
    assert score(fit_gradient_boosting(data), data) < single


def test_予測は0か1だけ(data) -> None:
    assert set(predictions(fit_random_forest(data), data)) <= {0, 1}


def _xgboost_available() -> bool:
    # macOS では libomp が要る。Nix の python-ml シェルの外では読み込めない
    try:
        import xgboost  # noqa: F401
    except Exception:  # noqa: BLE001 - 読み込み失敗の理由は問わない
        return False
    return True


@pytest.mark.skipif(
    not _xgboost_available(),
    reason="xgboost の読み込みに失敗しました（macOS では libomp が要ります）",
)
def test_XGBoostの正解率は原著と一致する(data) -> None:
    # 原著の出力: 0.8888888888888888
    from grokking_ml_lib.nb20_ensemble_spam import fit_xgboost, xgboost_score

    assert xgboost_score(fit_xgboost(data), data) == pytest.approx(
        0.8888888888888888, rel=1e-15
    )
