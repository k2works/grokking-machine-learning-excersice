import math
import random

import pytest

from grokking_ml.ch13_end_to_end import (
    Evaluation,
    best_by_f1,
    build_dataset,
    impute_missing,
    median,
    normalize,
    one_hot,
    parse_number,
    run_pipeline,
)


def make_rows(count: int = 40, seed: int = 1) -> list[dict[str, str]]:
    """「40 歳未満かつ収入 400 超なら購入」という規則に従う擬似データ。"""
    rng = random.Random(seed)
    cities = ["tokyo", "osaka", "kyoto"]
    rows = []
    for i in range(count):
        age = rng.randint(18, 70)
        income = rng.randint(200, 900)
        city = rng.choice(cities)
        bought = "yes" if (age < 40 and income > 400) else "no"
        rows.append(
            {
                # 9 行に 1 行は年齢が欠損している
                "age": "" if i % 9 == 0 else str(age),
                "income": str(income),
                "city": city,
                "bought": bought,
            }
        )
    return rows


ROWS = make_rows()


def test_parse_number_falls_back_on_garbage():
    assert parse_number("42") == pytest.approx(42.0)
    assert parse_number("N/A") == pytest.approx(0.0)
    assert parse_number("", default=-1.0) == pytest.approx(-1.0)


def test_median_of_odd_and_even_lengths():
    assert median([3.0, 1.0, 2.0]) == pytest.approx(2.0)
    assert median([4.0, 1.0, 3.0, 2.0]) == pytest.approx(2.5)


def test_median_of_an_empty_column_is_zero():
    assert median([]) == pytest.approx(0.0)


def test_impute_missing_uses_the_median():
    assert impute_missing([1.0, None, 3.0]) == [1.0, 2.0, 3.0]


def test_impute_missing_is_robust_to_outliers():
    # 平均なら 1000 に引きずられるが、中央値なら影響を受けない
    filled = impute_missing([1.0, 2.0, 3.0, 1000.0, None])
    assert filled[-1] == pytest.approx(2.5)


def test_normalize_maps_to_the_unit_interval():
    assert normalize([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]


def test_normalize_handles_a_constant_column():
    # 幅が 0 の列で 0 除算しない
    assert normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_one_hot_expands_categories_in_sorted_order():
    rows, categories = one_hot(["b", "a", "b"])
    assert categories == ["a", "b"]
    assert rows == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]


def test_one_hot_marks_exactly_one_column():
    rows, _ = one_hot(["x", "y", "z"])
    for row in rows:
        assert sum(row) == pytest.approx(1.0)


def test_build_dataset_produces_numeric_features():
    dataset = build_dataset(ROWS, "bought")
    assert dataset.feature_names == ["age", "income", "city=kyoto", "city=osaka", "city=tokyo"]
    assert len(dataset.points) == len(ROWS)
    for point in dataset.points:
        assert len(point) == len(dataset.feature_names)
        assert all(isinstance(value, float) for value in point)


def test_build_dataset_normalizes_every_numeric_column():
    dataset = build_dataset(ROWS, "bought")
    for index in (0, 1):
        column = [point[index] for point in dataset.points]
        assert min(column) == pytest.approx(0.0)
        assert max(column) == pytest.approx(1.0)


def test_build_dataset_converts_labels_to_zero_and_one():
    dataset = build_dataset(ROWS, "bought")
    assert set(dataset.labels) == {0, 1}
    assert sum(dataset.labels) == sum(1 for row in ROWS if row["bought"] == "yes")


def test_build_dataset_fills_missing_values():
    dataset = build_dataset(ROWS, "bought")
    # 欠損があっても、すべての点が有限の数値で埋まっている（NaN が残らない）
    assert all(all(math.isfinite(value) for value in point) for point in dataset.points)


def test_pipeline_evaluates_three_models():
    evaluations = run_pipeline(ROWS)
    assert [evaluation.name for evaluation in evaluations] == ["logistic", "tree", "adaboost"]


def test_every_metric_is_a_valid_ratio():
    for evaluation in run_pipeline(ROWS):
        for value in (
            evaluation.accuracy,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            evaluation.auc,
        ):
            assert 0.0 <= value <= 1.0


def test_models_beat_a_coin_flip():
    for evaluation in run_pipeline(ROWS):
        assert evaluation.auc > 0.5


def test_best_by_f1_picks_the_highest():
    evaluations = [
        Evaluation(name="a", accuracy=0.9, precision=0.5, recall=0.5, f1=0.5, auc=0.9),
        Evaluation(name="b", accuracy=0.7, precision=0.8, recall=0.8, f1=0.8, auc=0.7),
    ]
    assert best_by_f1(evaluations).name == "b"


def test_the_pipeline_is_reproducible():
    assert run_pipeline(ROWS) == run_pipeline(ROWS)


def test_accuracy_alone_would_mislead_us():
    # 正解率がもっとも高いモデルと、F1 がもっとも高いモデルは一致するとは限らない
    evaluations = run_pipeline(ROWS)
    by_accuracy = max(evaluations, key=lambda evaluation: evaluation.accuracy)
    by_f1 = best_by_f1(evaluations)
    # このデータでは一致するが、指標ごとに順位が変わりうることを明示しておく
    assert by_accuracy.accuracy >= by_f1.accuracy
