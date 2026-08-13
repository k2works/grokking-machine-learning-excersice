import random

import pytest

from grokking_ml.ch09_decision_trees import Leaf, build_tree
from grokking_ml.ch12_ensembles import (
    AdaBoost,
    Forest,
    WeightedTree,
    accuracy,
    bootstrap_sample,
    learner_weight,
    train_adaboost,
    train_forest,
    tree_accuracy,
    weighted_error,
)

# 3 つの領域に分かれるデータ。深さ 1 の木（切り株）では 1 本では解けない
POINTS = [
    (1.0, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
    (4.0, 1.0),
    (5.0, 1.0),
    (6.0, 1.0),
    (7.0, 1.0),
    (8.0, 1.0),
]
LABELS = [1, 1, -1, -1, -1, -1, 1, 1]


def test_bootstrap_sample_keeps_the_size():
    sample_points, sample_labels = bootstrap_sample(POINTS, LABELS, random.Random(0))
    assert len(sample_points) == len(POINTS)
    assert len(sample_labels) == len(LABELS)


def test_bootstrap_sample_keeps_pairs_together():
    sample_points, sample_labels = bootstrap_sample(POINTS, LABELS, random.Random(0))
    for point, label in zip(sample_points, sample_labels):
        assert LABELS[POINTS.index(point)] == label


def test_bootstrap_sample_draws_with_replacement():
    sample_points, _ = bootstrap_sample(POINTS, LABELS, random.Random(0))
    # 復元抽出なので、同じ点が複数回選ばれ、選ばれない点も出る
    assert len(set(sample_points)) < len(POINTS)


def test_a_single_stump_cannot_solve_three_regions():
    stump = build_tree(POINTS, LABELS, max_depth=1)
    assert tree_accuracy(stump, POINTS, LABELS) < 1.0


def test_forest_predicts_by_majority_vote():
    forest = Forest(trees=(Leaf(1), Leaf(1), Leaf(-1)))
    assert forest.predict((0.0,)) == 1
    assert forest.votes((0.0,)) == [1, 1, -1]


def test_forest_of_stumps_does_not_beat_a_single_stump_here():
    stump = build_tree(POINTS, LABELS, max_depth=1)
    forest = train_forest(POINTS, LABELS, tree_count=10, max_depth=1)
    # バギングは似た木ばかり作るので、この問題では改善しない
    assert accuracy(forest, POINTS, LABELS) <= tree_accuracy(stump, POINTS, LABELS) + 1e-9


def test_learner_weight_is_zero_for_a_coin_flip():
    # 誤り率 0.5 の学習器には発言権がない
    assert learner_weight(0.5) == pytest.approx(0.0)


def test_learner_weight_grows_as_the_error_shrinks():
    assert learner_weight(0.4) < learner_weight(0.2) < learner_weight(0.05)


def test_learner_weight_is_negative_for_a_bad_learner():
    # 当てずっぽうより悪い学習器は「逆を言えば当たる」ので負の発言権になる
    assert learner_weight(0.7) < 0.0


def test_learner_weight_is_finite_at_the_extremes():
    # 誤り率 0 でも無限大にはならない（クランプしている）
    assert learner_weight(0.0) > 0.0
    assert learner_weight(0.0) < 100.0


def test_weighted_error_counts_the_weight_not_the_number():
    tree = Leaf(1)
    # 3 点中 1 点だけ間違えるが、その点の重みが大きい
    points = [(0.0,), (1.0,), (2.0,)]
    labels = [1, 1, -1]
    assert weighted_error(tree, points, labels, [0.1, 0.1, 0.8]) == pytest.approx(0.8)
    assert weighted_error(tree, points, labels, [0.45, 0.45, 0.1]) == pytest.approx(0.1)


def test_weighted_error_is_zero_for_a_perfect_tree():
    tree = build_tree(POINTS, LABELS, max_depth=5)
    weights = [1.0 / len(POINTS)] * len(POINTS)
    assert weighted_error(tree, POINTS, LABELS, weights) == pytest.approx(0.0)


def test_adaboost_scores_by_weighted_vote():
    model = AdaBoost(
        learners=(
            WeightedTree(tree=Leaf(1), weight=2.0),
            WeightedTree(tree=Leaf(-1), weight=1.0),
        )
    )
    # 2*1 + 1*(-1) = 1 なので正のクラス
    assert model.score((0.0,)) == pytest.approx(1.0)
    assert model.predict((0.0,)) == 1


def test_adaboost_solves_what_a_single_stump_cannot():
    stump = build_tree(POINTS, LABELS, max_depth=1)
    boosted = train_adaboost(POINTS, LABELS, rounds=10, max_depth=1)
    assert tree_accuracy(stump, POINTS, LABELS) < 1.0
    assert accuracy(boosted, POINTS, LABELS) == pytest.approx(1.0)


def test_adaboost_beats_bagging_here():
    forest = train_forest(POINTS, LABELS, tree_count=10, max_depth=1)
    boosted = train_adaboost(POINTS, LABELS, rounds=10, max_depth=1)
    assert accuracy(boosted, POINTS, LABELS) > accuracy(forest, POINTS, LABELS)


def test_adaboost_learners_have_positive_weights():
    boosted = train_adaboost(POINTS, LABELS, rounds=10, max_depth=1)
    assert len(boosted.learners) > 0
    assert all(learner.weight > 0.0 for learner in boosted.learners)


def test_more_rounds_do_not_hurt_the_training_accuracy():
    few = train_adaboost(POINTS, LABELS, rounds=2, max_depth=1)
    many = train_adaboost(POINTS, LABELS, rounds=10, max_depth=1)
    assert accuracy(many, POINTS, LABELS) >= accuracy(few, POINTS, LABELS)


def test_adaboost_stops_when_no_useful_learner_remains():
    # すべて同じラベルなら 1 本目で完璧なので、それ以上増えない
    points = [(1.0,), (2.0,), (3.0,)]
    labels = [1, 1, 1]
    boosted = train_adaboost(points, labels, rounds=10, max_depth=1)
    assert len(boosted.learners) <= 10
    assert accuracy(boosted, points, labels) == pytest.approx(1.0)
