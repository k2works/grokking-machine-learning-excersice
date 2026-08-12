"""第 7 章: 分類モデルの評価指標。

正解率だけでは分類モデルの良し悪しを測れない。混同行列を土台に、
適合率・再現率・F1 スコア・ROC 曲線下面積（AUC）を実装する。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class ConfusionMatrix:
    """混同行列。すべての指標の土台になる 4 つの数。"""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )


def confusion_matrix(labels: Sequence[int], predictions: Sequence[int]) -> ConfusionMatrix:
    """正解ラベルと予測から混同行列を作る。"""
    counts = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    for label, prediction in zip(labels, predictions):
        counts[(label, prediction)] += 1
    return ConfusionMatrix(
        true_positives=counts[(1, 1)],
        false_positives=counts[(0, 1)],
        false_negatives=counts[(1, 0)],
        true_negatives=counts[(0, 0)],
    )


def accuracy(matrix: ConfusionMatrix) -> float:
    """正解率。全体のうち正しく当てた割合。"""
    if matrix.total == 0:
        return 0.0
    return (matrix.true_positives + matrix.true_negatives) / matrix.total


def precision(matrix: ConfusionMatrix) -> float:
    """適合率。陽性と予測したもののうち、本当に陽性だった割合。"""
    predicted_positive = matrix.true_positives + matrix.false_positives
    if predicted_positive == 0:
        return 0.0
    return matrix.true_positives / predicted_positive


def recall(matrix: ConfusionMatrix) -> float:
    """再現率。本当に陽性のもののうち、拾えた割合。"""
    actual_positive = matrix.true_positives + matrix.false_negatives
    if actual_positive == 0:
        return 0.0
    return matrix.true_positives / actual_positive


def f_beta_score(matrix: ConfusionMatrix, beta: float = 1.0) -> float:
    """F ベータスコア。beta が大きいほど再現率を重視する。"""
    p = precision(matrix)
    r = recall(matrix)
    if p == 0.0 and r == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * p * r / (beta_squared * p + r)


def f1_score(matrix: ConfusionMatrix) -> float:
    """F1 スコア。適合率と再現率の調和平均。"""
    return f_beta_score(matrix, beta=1.0)


def predictions_at_threshold(probabilities: Sequence[float], threshold: float) -> list[int]:
    """確率と閾値から 0 / 1 の予測を作る。"""
    return [1 if probability >= threshold else 0 for probability in probabilities]


def roc_points(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[tuple[float, float]]:
    """ROC 曲線の点列。閾値を動かしたときの (偽陽性率, 真陽性率) を返す。"""
    thresholds = sorted({*probabilities, 0.0, 1.0 + 1e-9}, reverse=True)
    points: list[tuple[float, float]] = []
    for threshold in thresholds:
        matrix = confusion_matrix(labels, predictions_at_threshold(probabilities, threshold))
        actual_negative = matrix.false_positives + matrix.true_negatives
        false_positive_rate = (
            matrix.false_positives / actual_negative if actual_negative > 0 else 0.0
        )
        points.append((false_positive_rate, recall(matrix)))
    return sorted(points)


def auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """ROC 曲線下の面積。台形則で積分する。"""
    points = roc_points(labels, probabilities)
    area = 0.0
    for (x1, y1), (x2, y2) in pairwise(points):
        area += (x2 - x1) * (y1 + y2) / 2
    return area
