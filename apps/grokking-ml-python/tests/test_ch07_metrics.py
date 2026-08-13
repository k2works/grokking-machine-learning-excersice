import pytest

from grokking_ml.ch07_metrics import (
    ConfusionMatrix,
    accuracy,
    auc,
    confusion_matrix,
    f1_score,
    f_beta_score,
    precision,
    predictions_at_threshold,
    recall,
    roc_points,
)

# 1000 人に 10 人が罹る病気を、全員「陰性」と判定するモデル
SICK_LABELS = [1] * 10 + [0] * 990
ALWAYS_HEALTHY = [0] * 1000


def test_confusion_matrix_counts_four_cases():
    matrix = confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
    assert matrix.true_positives == 1
    assert matrix.false_negatives == 1
    assert matrix.false_positives == 1
    assert matrix.true_negatives == 1
    assert matrix.total == 4


def test_accuracy_can_be_high_while_the_model_is_useless():
    matrix = confusion_matrix(SICK_LABELS, ALWAYS_HEALTHY)
    # 990/1000 を当てているので正解率は 99%
    assert accuracy(matrix) == pytest.approx(0.99)
    # しかし病人を 1 人も見つけられていない
    assert recall(matrix) == pytest.approx(0.0)


def test_precision_measures_how_trustworthy_a_positive_prediction_is():
    # 陽性と予測した 4 件のうち 3 件が当たり
    matrix = ConfusionMatrix(true_positives=3, false_positives=1, false_negatives=2, true_negatives=4)
    assert precision(matrix) == pytest.approx(0.75)


def test_recall_measures_how_many_positives_were_found():
    # 実際の陽性 5 件のうち 3 件を拾えた
    matrix = ConfusionMatrix(true_positives=3, false_positives=1, false_negatives=2, true_negatives=4)
    assert recall(matrix) == pytest.approx(0.6)


def test_precision_and_recall_trade_off():
    # 全員を陽性と予測すると再現率は 1.0 だが適合率は下がる
    aggressive = confusion_matrix(SICK_LABELS, [1] * 1000)
    assert recall(aggressive) == pytest.approx(1.0)
    assert precision(aggressive) == pytest.approx(0.01)


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    matrix = ConfusionMatrix(true_positives=3, false_positives=1, false_negatives=2, true_negatives=4)
    # 2 * 0.75 * 0.6 / (0.75 + 0.6)
    assert f1_score(matrix) == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_f1_punishes_an_unbalanced_pair_more_than_the_arithmetic_mean():
    # 適合率 1.0、再現率 0.1 のモデル
    matrix = ConfusionMatrix(true_positives=1, false_positives=0, false_negatives=9, true_negatives=90)
    arithmetic_mean = (precision(matrix) + recall(matrix)) / 2
    assert f1_score(matrix) < arithmetic_mean


def test_f_beta_with_large_beta_favors_recall():
    # 適合率 0.75、再現率 0.6
    matrix = ConfusionMatrix(true_positives=3, false_positives=1, false_negatives=2, true_negatives=4)
    # beta=2 は再現率重視なので、再現率(0.6)寄りの値になる
    assert f_beta_score(matrix, beta=2.0) < f1_score(matrix)
    # beta=0.5 は適合率重視なので、適合率(0.75)寄りの値になる
    assert f_beta_score(matrix, beta=0.5) > f1_score(matrix)


def test_metrics_are_zero_when_undefined():
    empty = ConfusionMatrix(0, 0, 0, 0)
    assert accuracy(empty) == pytest.approx(0.0)
    assert precision(empty) == pytest.approx(0.0)
    assert recall(empty) == pytest.approx(0.0)
    assert f1_score(empty) == pytest.approx(0.0)


def test_predictions_at_threshold():
    probabilities = [0.9, 0.6, 0.4, 0.1]
    assert predictions_at_threshold(probabilities, 0.5) == [1, 1, 0, 0]
    assert predictions_at_threshold(probabilities, 0.95) == [0, 0, 0, 0]
    assert predictions_at_threshold(probabilities, 0.05) == [1, 1, 1, 1]


def test_lowering_the_threshold_increases_recall():
    labels = [1, 1, 0, 0]
    probabilities = [0.9, 0.4, 0.6, 0.1]
    strict = confusion_matrix(labels, predictions_at_threshold(probabilities, 0.8))
    loose = confusion_matrix(labels, predictions_at_threshold(probabilities, 0.3))
    assert recall(loose) > recall(strict)
    assert precision(loose) < precision(strict)


def test_roc_points_start_and_end_at_the_corners():
    labels = [1, 1, 0, 0]
    probabilities = [0.9, 0.6, 0.4, 0.1]
    points = roc_points(labels, probabilities)
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (1.0, 1.0)


def test_auc_is_one_for_a_perfect_ranking():
    labels = [1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.2, 0.1]
    assert auc(labels, probabilities) == pytest.approx(1.0)


def test_auc_is_a_half_for_a_useless_ranking():
    # 陽性が最上位と最下位に 1 つずつ。順序として何の情報も持たない並び
    labels = [1, 0, 0, 1]
    probabilities = [0.8, 0.6, 0.4, 0.2]
    assert auc(labels, probabilities) == pytest.approx(0.5)


def test_auc_counts_correctly_ordered_pairs():
    # 陽性 2 件と陰性 2 件の組 4 通りのうち、3 通りで陽性が上位
    labels = [1, 0, 1, 0]
    probabilities = [0.8, 0.6, 0.4, 0.2]
    assert auc(labels, probabilities) == pytest.approx(0.75)


def test_auc_is_zero_for_a_perfectly_inverted_ranking():
    labels = [0, 0, 1, 1]
    probabilities = [0.9, 0.8, 0.2, 0.1]
    assert auc(labels, probabilities) == pytest.approx(0.0)


def test_auc_ignores_the_threshold():
    # AUC は閾値に依存しない。確率の順位だけで決まる
    labels = [1, 1, 0, 0]
    scaled = [0.99, 0.98, 0.02, 0.01]
    compressed = [0.55, 0.54, 0.46, 0.45]
    assert auc(labels, scaled) == pytest.approx(auc(labels, compressed))
