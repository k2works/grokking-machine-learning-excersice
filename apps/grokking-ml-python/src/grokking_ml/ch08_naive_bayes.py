"""第 8 章: ナイーブベイズ。

「単語が独立に出現する」という（実際には正しくない）仮定を置くことで、
ベイズの定理による分類を単なる掛け算に単純化する。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


def tokenize(text: str) -> list[str]:
    """文章を小文字の単語列に分解する。"""
    return text.lower().split()


@dataclass(frozen=True)
class NaiveBayesClassifier:
    """ナイーブベイズ分類器。単語ごとの出現回数からスパム確率を求める。"""

    spam_word_counts: Counter[str] = field(default_factory=Counter)
    ham_word_counts: Counter[str] = field(default_factory=Counter)
    spam_documents: int = 0
    ham_documents: int = 0

    @property
    def total_documents(self) -> int:
        return self.spam_documents + self.ham_documents

    @property
    def vocabulary(self) -> set[str]:
        return set(self.spam_word_counts) | set(self.ham_word_counts)


def train(documents: Sequence[str], labels: Sequence[int]) -> NaiveBayesClassifier:
    """文書とラベル（1 がスパム）から分類器を学習する。"""
    spam_word_counts: Counter[str] = Counter()
    ham_word_counts: Counter[str] = Counter()
    spam_documents = 0
    ham_documents = 0
    for document, label in zip(documents, labels):
        # 同じ単語が何度出ても 1 文書につき 1 回だけ数える（ベルヌーイ型）
        words = set(tokenize(document))
        if label == 1:
            spam_word_counts.update(words)
            spam_documents += 1
        else:
            ham_word_counts.update(words)
            ham_documents += 1
    return NaiveBayesClassifier(
        spam_word_counts=spam_word_counts,
        ham_word_counts=ham_word_counts,
        spam_documents=spam_documents,
        ham_documents=ham_documents,
    )


def word_spam_probability(
    model: NaiveBayesClassifier,
    word: str,
    smoothing: float = 1.0,
) -> float:
    """その単語を含む文書がスパムである確率。ラプラス平滑化つき。"""
    spam = model.spam_word_counts[word] + smoothing
    ham = model.ham_word_counts[word] + smoothing
    return spam / (spam + ham)


def prior_spam_probability(model: NaiveBayesClassifier) -> float:
    """事前確率。何も見ないときのスパム率。"""
    if model.total_documents == 0:
        return 0.0
    return model.spam_documents / model.total_documents


def predict_probability(
    model: NaiveBayesClassifier,
    document: str,
    smoothing: float = 1.0,
) -> float:
    """文書がスパムである確率。対数空間で計算する。"""
    if model.total_documents == 0:
        return 0.0
    words = set(tokenize(document))
    log_spam = math.log(_safe(prior_spam_probability(model)))
    log_ham = math.log(_safe(1.0 - prior_spam_probability(model)))
    for word in words:
        if word not in model.vocabulary:
            # 学習時に見ていない単語は何も語らないので無視する
            continue
        spam_given_word = word_spam_probability(model, word, smoothing)
        log_spam += math.log(_safe(spam_given_word))
        log_ham += math.log(_safe(1.0 - spam_given_word))
    # log の差から確率へ戻す（シグモイドと同じ形）
    return 1.0 / (1.0 + math.exp(min(max(log_ham - log_spam, -700.0), 700.0)))


def _safe(probability: float) -> float:
    """log(0) を避けるためにごくわずかに内側へ丸める。"""
    epsilon = 1e-15
    return min(max(probability, epsilon), 1.0 - epsilon)


def predict(
    model: NaiveBayesClassifier,
    document: str,
    threshold: float = 0.5,
    smoothing: float = 1.0,
) -> int:
    """閾値による 0 / 1 の分類。"""
    return 1 if predict_probability(model, document, smoothing) >= threshold else 0


def accuracy(
    model: NaiveBayesClassifier,
    documents: Iterable[str],
    labels: Iterable[int],
) -> float:
    """正解率。"""
    pairs = list(zip(documents, labels))
    correct = sum(1 for document, label in pairs if predict(model, document) == label)
    return correct / len(pairs)
