"""原著ノートブック #07 の再現テスト。

この回は scikit-learn を使わないので、**原著の数値をすべて再現できる**。
`np.compat.long` による切り捨てまで写しているので、8 桁目まで一致する。

`emails.csv` は約 8.5 MB あり、リポジトリに含めていない。
テストからは自動ダウンロードせず、未取得ならスキップする。
"""

import pandas as pd
import pytest

from grokking_ml_lib.datasets import datasets_dir
from grokking_ml_lib.nb07_naive_bayes import (
    WordCounts,
    load_emails,
    predict_bayes,
    predict_naive_bayes,
    process_email,
    train,
)

#: 数え方を確かめるための小さなコーパス
TINY_EMAILS = pd.DataFrame(
    {
        "text": [
            "win lottery now",
            "lottery lottery lottery",
            "meeting at noon",
            "lunch meeting",
        ],
        "spam": [1, 1, 0, 0],
    }
)


def _emails_available() -> bool:
    return (datasets_dir() / "emails.csv").exists()


emails_required = pytest.mark.skipif(
    not _emails_available(),
    reason="emails.csv（約 8.5 MB）が未取得です。datasets の README を参照してください",
)


@pytest.fixture(scope="module")
def emails():
    return load_emails()


@pytest.fixture(scope="module")
def model(emails):
    return train(emails)


def test_メールは小文字の単語集合になる() -> None:
    # 原著は list(set(text.lower().split()))。同じ単語は 1 回しか数えない
    assert sorted(process_email("Win WIN the lottery")) == ["lottery", "the", "win"]


def test_出現数は1から数え始める() -> None:
    # ラプラス平滑化。一度も見ていない側の確率が 0 にならないようにする
    counts = WordCounts()

    assert counts.spam == 1
    assert counts.ham == 1


def test_小さなコーパスで出現通数を数える() -> None:
    trained = train(TINY_EMAILS)

    # lottery はスパム 2 通に出る。1 から数え始めるので spam は 3
    assert trained.words["lottery"].spam == 3
    assert trained.words["lottery"].ham == 1
    # meeting はハム 2 通に出る
    assert trained.words["meeting"].spam == 1
    assert trained.words["meeting"].ham == 3


def test_同じ単語が何度出ても1通と数える() -> None:
    # 2 通目は "lottery lottery lottery" だが、1 通ぶんしか数えない
    trained = train(TINY_EMAILS)

    assert trained.words["lottery"].spam == 3


def test_語彙にない単語は事前確率を返す() -> None:
    trained = train(TINY_EMAILS)

    # スパム 2 通 / 全 4 通
    assert predict_naive_bayes(trained, "zzzz") == pytest.approx(0.5)


@emails_required
def test_データセットは原著と同じ規模(model) -> None:
    # 原著の出力
    #   Number of emails: 5728
    #   Number of spam emails: 1368
    #   Probability of spam: 0.2388268156424581
    assert model.corpus.total == 5728
    assert model.corpus.spam == 1368
    assert model.corpus.ham == 5728 - 1368
    assert model.corpus.spam_probability == pytest.approx(0.2388268156424581)


@emails_required
def test_lotteryとsaleの出現数は原著と同じ(model) -> None:
    # 原著の出力
    #   model['lottery'] -> {'spam': 9, 'ham': 1}
    #   model['sale']    -> {'spam': 39, 'ham': 42}
    assert model.words["lottery"] == WordCounts(spam=9, ham=1)
    assert model.words["sale"] == WordCounts(spam=39, ham=42)


@emails_required
def test_単語1つの予測は原著と同じ(model) -> None:
    # 原著の出力
    #   predict_bayes('lottery') -> 0.9
    #   predict_bayes('sale')    -> 0.48148148148148145
    assert predict_bayes(model, "lottery") == pytest.approx(0.9)
    assert predict_bayes(model, "sale") == pytest.approx(0.48148148148148145)


@emails_required
@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("lottery sale", 0.9638144992048691),
        ("Hi mom how are you", 0.12554358867164464),
        ("meet me at the lobby of the hotel at nine am", 6.964603508395961e-05),
        ("enter the lottery to win three million dollars", 0.9995234218677428),
        ("buy cheap lottery easy money now", 0.999973472265966),
        ("Grokking Machine Learning by Luis Serrano", 0.4197107645488719),
        ("asdfgh", 0.2388268156424581),
    ],
)
def test_メール全体の予測は原著と同じ(model, email: str, expected: float) -> None:
    # 原著のセル出力をそのまま期待値にしている
    assert predict_naive_bayes(model, email) == pytest.approx(expected)


@emails_required
def test_知らない単語を足しても結果は変わらない(model) -> None:
    # 原著は同じ文に意味のない語を足しても同じ値になることを示している
    #   'Hi mom how are you' と
    #   'Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd' が同じ 0.12554358867164464
    assert predict_naive_bayes(model, "Hi mom how are you") == pytest.approx(
        predict_naive_bayes(model, "Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd")
    )


@emails_required
def test_切り捨てを外すと原著と食い違う(model) -> None:
    # 原著の np.compat.long は Python の int と同じで小数点以下を切り捨てる。
    # 切り捨てないと 'lottery sale' が 0.9638144470140118 になり、
    # 原著の 0.9638144992048691 と 8 桁目から分かれる
    corpus = model.corpus
    spams, hams = [1.0], [1.0]
    for word in ["lottery", "sale"]:
        counts = model.words[word]
        spams.append(counts.spam / corpus.spam * corpus.total)
        hams.append(counts.ham / corpus.ham * corpus.total)

    product_spams = spams[1] * spams[2] * corpus.spam
    product_hams = hams[1] * hams[2] * corpus.ham
    without_truncation = product_spams / (product_spams + product_hams)

    assert without_truncation == pytest.approx(0.9638144470140118, rel=1e-12)
    # 差は 8 桁目なので、既定の許容誤差（相対 1e-6）では見分けられない
    assert without_truncation != pytest.approx(0.9638144992048691, rel=1e-12)
    assert predict_naive_bayes(model, "lottery sale") == pytest.approx(
        0.9638144992048691, rel=1e-12
    )
