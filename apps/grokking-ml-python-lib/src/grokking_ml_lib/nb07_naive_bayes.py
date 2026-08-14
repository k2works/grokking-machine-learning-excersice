"""原著ノートブック #07 `Chapter_08_Naive_Bayes/Coding_naive_Bayes.ipynb`。

メール 5728 通からスパム判定器をナイーブベイズで作る。
**この回だけ scikit-learn を使わず、pandas と辞書だけで書かれている。**
だからライブラリの差が出にくく、3 言語とも原著の数値をそのまま再現できる。

原著は最後の確率計算で `np.compat.long` を使っており、これが浮動小数点数を
**整数に切り捨てる**。切り捨てないと原著の出力と 8 桁目から食い違うので、
その挙動もそのまま写してある。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from grokking_ml_lib.datasets import dataset_path


@dataclass
class WordCounts:
    """ある単語が、スパムとハムそれぞれ何通に現れたか。

    原著は 1 から数え始める。ラプラス平滑化にあたり、
    「一度も見ていない側」の確率が 0 になるのを防ぐ。
    """

    spam: int = 1
    ham: int = 1


@dataclass
class Corpus:
    """学習に使ったメール全体の統計。"""

    total: int
    spam: int

    @property
    def ham(self) -> int:
        return self.total - self.spam

    @property
    def spam_probability(self) -> float:
        """事前確率。何も情報が無いときにスパムと判断する確率。"""
        return self.spam / self.total


@dataclass
class NaiveBayesModel:
    """単語ごとの出現数と、コーパス全体の統計。"""

    words: dict[str, WordCounts]
    corpus: Corpus


def load_emails() -> pd.DataFrame:
    """メールのデータセットを読み込む。5728 通、うち 1368 通がスパム。

    このファイルは約 8.5 MB あるためリポジトリには含めていない。
    未取得なら `dataset_path` が原著リポジトリから取得する。
    """
    return pd.read_csv(dataset_path("emails.csv"))


def process_email(text: str) -> list[str]:
    """メール本文を、重複を除いた小文字の単語リストにする。

    原著は `list(set(text.lower().split()))`。**同じ単語が何度出ても 1 回** と
    数えるのがナイーブベイズのこの実装の前提である。
    """
    return list(set(text.lower().split()))


def train(emails: pd.DataFrame) -> NaiveBayesModel:
    """単語ごとに、スパム・ハムそれぞれの出現通数を数える。"""
    words: dict[str, WordCounts] = {}
    for _, email in emails.iterrows():
        for word in process_email(email["text"]):
            counts = words.setdefault(word, WordCounts())
            if email["spam"]:
                counts.spam += 1
            else:
                counts.ham += 1

    return NaiveBayesModel(
        words, Corpus(total=len(emails), spam=int(emails["spam"].sum()))
    )


def predict_bayes(model: NaiveBayesModel, word: str) -> float:
    """単語 1 つだけを見たときの、スパムである確率。

    ベイズの定理そのものではなく、単純に「その単語を含むメールのうち
    スパムの割合」を返す。原著もそう書いている。
    """
    counts = model.words[word.lower()]
    return counts.spam / (counts.spam + counts.ham)


def predict_naive_bayes(model: NaiveBayesModel, email: str) -> float:
    """メール全体を見たときの、スパムである確率。

    「単語の出現が互いに独立」と仮定して、単語ごとの尤度比を掛け合わせる。
    語彙にない単語は無視するので、知らない単語ばかりのメールは事前確率に落ちる。

    原著は最後に `np.compat.long(...)` を通しており、これは Python の `int` と
    同じで **小数点以下を切り捨てる**。切り捨てないと 8 桁目から数値が変わるので、
    そのまま写してある。`np.compat` は NumPy 2.0 で削除されたため、
    ここでは `int()` を直接使う。
    """
    corpus = model.corpus
    spams = [1.0]
    hams = [1.0]
    for word in set(email.lower().split()):
        counts = model.words.get(word)
        if counts is not None:
            spams.append(counts.spam / corpus.spam * corpus.total)
            hams.append(counts.ham / corpus.ham * corpus.total)

    product_spams = _product(spams) * corpus.spam
    product_hams = _product(hams) * corpus.ham
    # 原著の np.compat.long と同じ切り捨て
    truncated_spams = int(product_spams)
    truncated_hams = int(product_hams)
    return truncated_spams / (truncated_spams + truncated_hams)


def _product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result
