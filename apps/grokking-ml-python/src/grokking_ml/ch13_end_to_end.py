"""第 13 章: エンドツーエンドの実例。

生データの前処理から、複数モデルの比較、評価までを 1 本のパイプラインに
つなぐ。これまでの章で作った部品を組み合わせるだけで実現できる。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from grokking_ml.ch04_regularization import train_test_split
from grokking_ml.ch06_logistic_regression import LogisticClassifier, logistic_regression
from grokking_ml.ch07_metrics import (
    ConfusionMatrix,
    auc,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)
from grokking_ml.ch07_metrics import accuracy as matrix_accuracy
from grokking_ml.ch09_decision_trees import Tree, build_tree
from grokking_ml.ch12_ensembles import AdaBoost, train_adaboost

Row = dict[str, str]
Point = list[float]


@dataclass(frozen=True)
class Dataset:
    """前処理を終えた特徴量とラベル。"""

    points: list[Point]
    labels: list[int]
    feature_names: list[str]


def parse_number(text: str, default: float = 0.0) -> float:
    """数値に見えない値は既定値に落とす。欠損への最初の砦。"""
    try:
        return float(text)
    except ValueError:
        return default


def median(values: Sequence[float]) -> float:
    """中央値。欠損の穴埋めに使う。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def impute_missing(column: Sequence[float | None]) -> list[float]:
    """欠損を中央値で埋める。平均より外れ値に強い。"""
    known = [value for value in column if value is not None]
    filler = median(known)
    return [filler if value is None else value for value in column]


def normalize(column: Sequence[float]) -> list[float]:
    """最小 0・最大 1 に揃える。値の幅が違う特徴量を対等に扱うため。"""
    low = min(column)
    high = max(column)
    if high == low:
        return [0.0 for _ in column]
    return [(value - low) / (high - low) for value in column]


def one_hot(column: Sequence[str]) -> tuple[list[list[float]], list[str]]:
    """カテゴリ列を 0/1 の列に展開する。"""
    categories = sorted(set(column))
    rows = [[1.0 if value == category else 0.0 for category in categories] for value in column]
    return rows, list(categories)


def build_dataset(rows: Sequence[Row], label_column: str) -> Dataset:
    """生の行データを、数値の特徴量ベクトルとラベルに変換する。"""
    labels = [1 if row[label_column] == "yes" else 0 for row in rows]
    numeric_columns = ["age", "income"]
    categorical_columns = ["city"]

    columns: list[list[float]] = []
    names: list[str] = []
    for name in numeric_columns:
        raw = [None if row[name] == "" else parse_number(row[name]) for row in rows]
        columns.append(normalize(impute_missing(raw)))
        names.append(name)
    for name in categorical_columns:
        expanded, categories = one_hot([row[name] for row in rows])
        for index, category in enumerate(categories):
            columns.append([row[index] for row in expanded])
            names.append(f"{name}={category}")

    points = [[column[i] for column in columns] for i in range(len(rows))]
    return Dataset(points=points, labels=labels, feature_names=names)


@dataclass(frozen=True)
class Evaluation:
    """1 つのモデルの評価結果。"""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def evaluate(
    name: str,
    predict: Callable[[Point], int],
    probability: Callable[[Point], float],
    points: Sequence[Point],
    labels: Sequence[int],
) -> Evaluation:
    """予測関数と確率関数から、第 7 章の指標をまとめて算出する。"""
    matrix: ConfusionMatrix = confusion_matrix(labels, [predict(point) for point in points])
    return Evaluation(
        name=name,
        accuracy=matrix_accuracy(matrix),
        precision=precision(matrix),
        recall=recall(matrix),
        f1=f1_score(matrix),
        auc=auc(labels, [probability(point) for point in points]),
    )


def evaluate_logistic(
    model: LogisticClassifier,
    points: Sequence[Point],
    labels: Sequence[int],
) -> Evaluation:
    """第 6 章のロジスティック回帰を評価する。"""
    return evaluate(
        "logistic", model.predict, model.predict_probability, points, labels
    )


def evaluate_tree(tree: Tree, points: Sequence[Point], labels: Sequence[int]) -> Evaluation:
    """第 9 章の決定木を評価する。"""
    return evaluate(
        "tree", tree.predict, lambda point: float(tree.predict(point)), points, labels
    )


def evaluate_adaboost(
    model: AdaBoost,
    points: Sequence[Point],
    labels: Sequence[int],
) -> Evaluation:
    """第 12 章の AdaBoost を評価する。ラベルは +1 / -1 なので 0 / 1 に戻す。"""
    return evaluate(
        "adaboost",
        lambda point: 1 if model.predict(point) == 1 else 0,
        model.score,
        points,
        labels,
    )


def run_pipeline(rows: Sequence[Row], label_column: str = "bought") -> list[Evaluation]:
    """前処理 → 分割 → 3 モデルの学習 → 評価までを一気に通す。"""
    dataset = build_dataset(rows, label_column)
    train_x, train_y, test_x, test_y = train_test_split(
        dataset.points, dataset.labels, test_ratio=0.3, seed=0
    )

    logistic, _ = logistic_regression(train_x, train_y, learning_rate=0.5, epochs=2000, seed=0)
    tree = build_tree(train_x, train_y, max_depth=3)
    boosted = train_adaboost(
        train_x, [1 if label == 1 else -1 for label in train_y], rounds=5, max_depth=1
    )

    return [
        evaluate_logistic(logistic, test_x, test_y),
        evaluate_tree(tree, test_x, test_y),
        evaluate_adaboost(boosted, test_x, test_y),
    ]


def best_by_f1(evaluations: Sequence[Evaluation]) -> Evaluation:
    """F1 スコアがもっとも高いモデルを選ぶ。"""
    return max(evaluations, key=lambda evaluation: evaluation.f1)
