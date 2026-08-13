import pytest

from grokking_ml.ch08_naive_bayes import (
    accuracy,
    predict,
    predict_probability,
    prior_spam_probability,
    tokenize,
    train,
    word_spam_probability,
)

# 原著と同じ「lottery / sale / winning」を含むスパム判定
DOCUMENTS = [
    "lottery sale",
    "lottery winning",
    "winning lottery sale",
    "sale today",
    "meeting tomorrow",
    "project meeting",
    "lunch meeting today",
    "project deadline",
]
LABELS = [1, 1, 1, 0, 0, 0, 0, 0]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Lottery WINNING today") == ["lottery", "winning", "today"]


def test_train_counts_documents_per_class():
    model = train(DOCUMENTS, LABELS)
    assert model.spam_documents == 3
    assert model.ham_documents == 5
    assert model.total_documents == 8


def test_train_counts_each_word_once_per_document():
    # 同じ単語が 2 回出ても 1 文書として 1 回だけ数える
    model = train(["spam spam spam"], [1])
    assert model.spam_word_counts["spam"] == 1


def test_prior_is_the_share_of_spam_documents():
    model = train(DOCUMENTS, LABELS)
    assert prior_spam_probability(model) == pytest.approx(3 / 8)


def test_word_spam_probability_uses_laplace_smoothing():
    model = train(DOCUMENTS, LABELS)
    # lottery はスパム 3 件、ハム 0 件。平滑化なしなら 1.0 になってしまう
    assert model.spam_word_counts["lottery"] == 3
    assert model.ham_word_counts["lottery"] == 0
    # (3+1) / ((3+1) + (0+1)) = 0.8
    assert word_spam_probability(model, "lottery") == pytest.approx(0.8)


def test_smoothing_keeps_probabilities_away_from_the_extremes():
    model = train(DOCUMENTS, LABELS)
    for word in model.vocabulary:
        probability = word_spam_probability(model, word)
        assert 0.0 < probability < 1.0


def test_unknown_word_is_neutral_without_smoothing_bias():
    model = train(DOCUMENTS, LABELS)
    # 未知語は両クラス 0 件なので、平滑化後は五分五分になる
    assert word_spam_probability(model, "unseen") == pytest.approx(0.5)


def test_unknown_words_do_not_change_the_prediction():
    model = train(DOCUMENTS, LABELS)
    known = predict_probability(model, "lottery")
    with_unknown = predict_probability(model, "lottery zzzz qqqq")
    assert with_unknown == pytest.approx(known)


def test_spam_words_raise_the_probability():
    model = train(DOCUMENTS, LABELS)
    assert predict_probability(model, "lottery winning") > 0.5
    assert predict_probability(model, "project deadline") < 0.5


def test_stacking_spam_words_increases_confidence():
    model = train(DOCUMENTS, LABELS)
    one_word = predict_probability(model, "lottery")
    two_words = predict_probability(model, "lottery winning")
    assert two_words > one_word


def test_probability_stays_inside_the_unit_interval():
    model = train(DOCUMENTS, LABELS)
    for document in DOCUMENTS + ["", "lottery lottery lottery winning sale"]:
        assert 0.0 <= predict_probability(model, document) <= 1.0


def test_empty_document_returns_the_prior():
    model = train(DOCUMENTS, LABELS)
    assert predict_probability(model, "") == pytest.approx(prior_spam_probability(model))


def test_predict_uses_the_threshold():
    model = train(DOCUMENTS, LABELS)
    assert predict(model, "lottery winning") == 1
    assert predict(model, "lottery winning", threshold=0.99) == 0


def test_classifier_separates_the_training_set():
    model = train(DOCUMENTS, LABELS)
    assert accuracy(model, DOCUMENTS, LABELS) == pytest.approx(1.0)


def test_untrained_model_returns_zero():
    model = train([], [])
    assert prior_spam_probability(model) == pytest.approx(0.0)
    assert predict_probability(model, "lottery") == pytest.approx(0.0)
