"""Tests of the demo app's hybrid retrieval (skipped when the embeddings extra is absent)."""

import pytest

from app.rag_app import REFUSAL, RAGApp

pytest.importorskip("model2vec")

PINNED_BM25 = "tests/fixtures/app_config.toml"


@pytest.fixture(scope="module")
def hybrid():
    return RAGApp("app/config.toml", retrieval_mode="hybrid")


@pytest.fixture(scope="module")
def bm25():
    return RAGApp(PINNED_BM25)


def test_embeddings_fix_vocabulary_mismatch(hybrid, bm25):
    question = "What discount do annual payments get?"  # the KB says "annually"
    assert bm25.answer(question)["answer"] == REFUSAL
    assert "15 percent" in hybrid.answer(question)["answer"]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How are customers billed if they choose yearly payment?", "15 percent"),
        ("How long is a trashed file recoverable?", "30 days"),
        ("What is the monthly price for Starter?", "12 euros"),
    ],
)
def test_paraphrases_are_grounded_semantically(hybrid, question, expected):
    assert expected in hybrid.answer(question)["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "What is the API rate limit for the free tier?",  # 0.72 sentence similarity to the rate-limit chunk
        "Do you offer a student discount?",
        "How much is the Premium plan?",
        "Do you support Kubernetes?",
    ],
)
def test_on_topic_but_unanswerable_questions_are_still_refused(hybrid, question):
    assert hybrid.answer(question)["answer"] == REFUSAL


def test_rrf_returns_top_k_distinct_chunks(hybrid):
    contexts = hybrid.answer("How are webhooks signed?")["contexts"]
    assert len(contexts) == hybrid.top_k
    assert len({c["source"] for c in contexts}) == hybrid.top_k
    assert contexts[0]["source"].startswith("api.md")


def test_unknown_retrieval_mode_is_rejected():
    with pytest.raises(ValueError, match=r"retrieval\.mode"):
        RAGApp(PINNED_BM25, retrieval_mode="magic")
