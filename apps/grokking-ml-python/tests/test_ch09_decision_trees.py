import math

import pytest

from grokking_ml.ch09_decision_trees import (
    Leaf,
    Node,
    Split,
    accuracy,
    apply_split,
    best_split,
    build_tree,
    candidate_splits,
    depth,
    entropy,
    gini_impurity,
    information_gain,
    leaf_count,
    majority_label,
    weighted_impurity,
)

# 原著と同じアプリ推薦データ
# 特徴量 = (性別: 0=女性 1=男性, 年齢)、ラベル = 1 が推薦
POINTS = [
    (1.0, 15.0),
    (0.0, 25.0),
    (0.0, 32.0),
    (1.0, 35.0),
    (0.0, 12.0),
    (1.0, 14.0),
    (1.0, 55.0),
    (0.0, 40.0),
]
LABELS = [1, 0, 0, 0, 1, 1, 0, 0]


def test_gini_of_a_pure_set_is_zero():
    assert gini_impurity([1, 1, 1]) == pytest.approx(0.0)
    assert gini_impurity([0, 0]) == pytest.approx(0.0)


def test_gini_of_a_balanced_split_is_a_half():
    assert gini_impurity([0, 1]) == pytest.approx(0.5)
    assert gini_impurity([0, 0, 1, 1]) == pytest.approx(0.5)


def test_gini_of_an_empty_set_is_zero():
    assert gini_impurity([]) == pytest.approx(0.0)


def test_entropy_of_a_pure_set_is_zero():
    assert entropy([1, 1, 1]) == pytest.approx(0.0)


def test_entropy_of_a_balanced_split_is_one_bit():
    assert entropy([0, 1]) == pytest.approx(1.0)


def test_entropy_of_four_equal_classes_is_two_bits():
    assert entropy([0, 1, 2, 3]) == pytest.approx(2.0)


def test_entropy_matches_the_definition():
    # 3 つが 1、1 つが 0 のとき -(0.75*log2(0.75) + 0.25*log2(0.25))
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert entropy([1, 1, 1, 0]) == pytest.approx(expected)


def test_gini_and_entropy_agree_on_ordering():
    # どちらの指標でも「純粋 < 偏り < 均等」の順になる
    pure, skewed, balanced = [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]
    assert gini_impurity(pure) < gini_impurity(skewed) < gini_impurity(balanced)
    assert entropy(pure) < entropy(skewed) < entropy(balanced)


def test_split_sends_smaller_values_left():
    split = Split(feature=1, threshold=20.0)
    assert split.matches((0.0, 15.0)) is True
    assert split.matches((0.0, 25.0)) is False


def test_apply_split_partitions_without_loss():
    split = Split(feature=1, threshold=20.0)
    left_points, left_labels, right_points, right_labels = apply_split(POINTS, LABELS, split)
    assert len(left_points) + len(right_points) == len(POINTS)
    assert len(left_labels) + len(right_labels) == len(LABELS)
    assert all(point[1] < 20.0 for point in left_points)
    assert all(point[1] >= 20.0 for point in right_points)


def test_weighted_impurity_favors_pure_children():
    # 完全に分かれる分割は不純度 0
    assert weighted_impurity([1, 1], [0, 0]) == pytest.approx(0.0)
    # まったく分かれない分割は元のまま
    assert weighted_impurity([1, 0], [1, 0]) == pytest.approx(0.5)


def test_information_gain_is_zero_for_a_useless_split():
    labels = [1, 1, 0, 0]
    assert information_gain(labels, [1, 0], [1, 0]) == pytest.approx(0.0)


def test_information_gain_is_maximal_for_a_perfect_split():
    labels = [1, 1, 0, 0]
    assert information_gain(labels, [1, 1], [0, 0]) == pytest.approx(0.5)


def test_candidate_splits_use_midpoints():
    points = [(0.0, 10.0), (0.0, 20.0), (1.0, 30.0)]
    splits = candidate_splits(points)
    thresholds_of_feature_1 = sorted(s.threshold for s in splits if s.feature == 1)
    assert thresholds_of_feature_1 == [15.0, 25.0]
    # 特徴量 0 は 0.0 と 1.0 の 2 値なので中点は 1 つ
    assert [s.threshold for s in splits if s.feature == 0] == [0.5]


def test_best_split_finds_the_age_boundary():
    found = best_split(POINTS, LABELS)
    assert found is not None
    split, gain = found
    # 年齢（特徴量 1）の若年層で分かれる
    assert split.feature == 1
    assert 15.0 < split.threshold < 25.0
    assert gain > 0.0


def test_best_split_returns_none_when_nothing_improves():
    # すべて同じラベルなら、どう分けても利得がない
    assert best_split([(1.0,), (2.0,), (3.0,)], [1, 1, 1]) is None


def test_majority_label_breaks_ties_with_the_smaller_label():
    assert majority_label([1, 1, 0]) == 1
    assert majority_label([0, 1]) == 0


def test_build_tree_fits_the_training_data():
    tree = build_tree(POINTS, LABELS)
    assert accuracy(tree, POINTS, LABELS) == pytest.approx(1.0)


def test_build_tree_stops_at_max_depth():
    shallow = build_tree(POINTS, LABELS, max_depth=1)
    assert depth(shallow) == 1
    assert leaf_count(shallow) == 2


def test_a_pure_dataset_becomes_a_single_leaf():
    tree = build_tree([(1.0,), (2.0,)], [1, 1])
    assert isinstance(tree, Leaf)
    assert tree.label == 1
    assert depth(tree) == 0


def test_deeper_trees_are_not_shallower():
    shallow = build_tree(POINTS, LABELS, max_depth=1)
    deep = build_tree(POINTS, LABELS, max_depth=5)
    assert depth(deep) >= depth(shallow)
    assert accuracy(deep, POINTS, LABELS) >= accuracy(shallow, POINTS, LABELS)


def test_min_samples_limits_the_tree():
    big_leaf = build_tree(POINTS, LABELS, min_samples=len(LABELS))
    assert isinstance(big_leaf, Leaf)


def test_entropy_and_gini_produce_the_same_tree_here():
    by_gini = build_tree(POINTS, LABELS, impurity=gini_impurity)
    by_entropy = build_tree(POINTS, LABELS, impurity=entropy)
    assert by_gini == by_entropy


def test_tree_structure_is_inspectable():
    tree = build_tree(POINTS, LABELS, max_depth=1)
    assert isinstance(tree, Node)
    assert isinstance(tree.left, Leaf)
    assert isinstance(tree.right, Leaf)
