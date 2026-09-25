import pytest

from rageval.metrics import (
    assertion_failures,
    context_precision,
    context_recall,
    faithfulness,
    get_similarity,
    is_refusal,
    token_f1,
)
from rageval.metrics.retrieval import fact_in_text

FACT = "The API rate limit is 600 requests per minute"
GOOD = "Our API rate limit is 600 requests per minute per organization."
NOISE = "Invoices are issued on the first day of each month."


class TestAnswerMetrics:
    def test_token_f1_identical_and_disjoint(self):
        assert token_f1("600 requests per minute", "600 requests per minute") == 1.0
        assert token_f1("blue whale", "600 requests") == 0.0

    def test_token_f1_ignores_case_and_stopwords(self):
        assert token_f1("The limit is 600", "limit 600") == 1.0

    def test_assertions_alternatives_and_forbidden(self):
        assert assertion_failures("It returns HTTP 429.", must_include=["429|too many requests"]) == []
        assert assertion_failures("Retry later.", must_include=["429|too many requests"]) == [
            "missing: '429|too many requests'"
        ]
        assert assertion_failures("Data is kept 30 days", must_not_include=["30 days"]) == ["forbidden: '30 days'"]

    def test_assertions_are_accent_and_case_insensitive(self):
        assert assertion_failures("Hébergé à FRANKFURT", must_include=["frankfurt", "heberge"]) == []

    @pytest.mark.parametrize("answer", ["I don't know based on the docs.", "I do NOT know.", "I don’t know"])
    def test_refusal_detected(self, answer):
        assert is_refusal(answer)

    def test_regular_answer_is_not_refusal(self):
        assert not is_refusal("The Team plan costs 29 euros.")


class TestSimilarity:
    def test_lexical_bounds(self):
        sim = get_similarity("lexical")
        assert sim("same text here", "same text here") == pytest.approx(1.0)
        assert sim("abc", "") == 0.0

    def test_lexical_is_robust_to_inflection(self):
        sim = get_similarity("lexical")
        assert sim("paid annually", "annual payment") > sim("paid annually", "monthly invoice")

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            get_similarity("magic")


class TestRetrievalMetrics:
    def test_fact_matching_tolerates_rewording(self):
        assert fact_in_text(FACT, GOOD)
        assert not fact_in_text(FACT, "The rate limit is 900 requests per minute")  # wrong number

    def test_context_recall(self):
        facts = [FACT, "Invoices are issued on the first day of each month"]
        assert context_recall(facts, [GOOD]) == 0.5
        assert context_recall(facts, [GOOD, NOISE]) == 1.0
        assert context_recall(facts, []) == 0.0

    def test_context_recall_requires_facts(self):
        with pytest.raises(ValueError):
            context_recall([], [GOOD])

    def test_context_precision_is_rank_aware(self):
        assert context_precision([FACT], [GOOD, NOISE]) == 1.0
        assert context_precision([FACT], [NOISE, GOOD]) == 0.5
        assert context_precision([FACT], [NOISE]) == 0.0
        assert context_precision([FACT], []) == 0.0

    def test_faithfulness_flags_unsupported_tokens(self):
        assert faithfulness("The limit is 600 requests per minute", [GOOD]) == 1.0
        assert faithfulness("The limit is 900 requests per minute", [GOOD]) < 1.0
        assert faithfulness("", [GOOD]) == 0.0
