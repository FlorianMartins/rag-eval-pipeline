import json

import pytest

from rageval.dataset import DatasetError, TestCase, load_dataset
from rageval.evaluator import EvalSettings, aggregate, run_evaluation
from rageval.metrics import get_similarity
from rageval.systems import SystemResponse

SETTINGS = EvalSettings(similarity=get_similarity("lexical"))
CASES = (
    TestCase(
        id="rate-limit",
        question="What is the API rate limit?",
        ground_truth="The API rate limit is 600 requests per minute.",
        expected_context=("The API rate limit is 600 requests per minute",),
        must_include=("600",),
    ),
    TestCase(id="oos", question="Who is the CEO?", answerable=False),
)


class ScriptedSystem:
    """Test double: returns canned responses, or raises for unknown questions."""

    def __init__(self, answers):
        self.answers = answers

    def query(self, question):
        return SystemResponse.from_payload(self.answers[question])

    def describe(self):
        return {"type": "scripted"}


def test_perfect_system_scores_perfectly():
    system = ScriptedSystem(
        {
            "What is the API rate limit?": {
                "answer": "The API rate limit is 600 requests per minute.",
                "contexts": [{"text": "The API rate limit is 600 requests per minute per organization."}],
            },
            "Who is the CEO?": {"answer": "I don't know.", "contexts": []},
        }
    )
    results = run_evaluation(system, CASES, SETTINGS)
    metrics = aggregate(results)
    assert [r.correct for r in results] == [True, True]
    assert metrics["accuracy"] == 1.0
    assert metrics["context_recall"] == 1.0
    assert metrics["faithfulness"] == 1.0


def test_hallucination_on_out_of_scope_question_fails():
    system = ScriptedSystem(
        {
            "What is the API rate limit?": {"answer": "600 requests per minute", "contexts": []},
            "Who is the CEO?": {"answer": "The CEO is Jane Doe.", "contexts": []},
        }
    )
    results = run_evaluation(system, CASES, SETTINGS)
    assert not results[1].correct
    assert "expected a refusal" in results[1].failures[0]
    assert aggregate(results)["refusal_accuracy"] == 0.0


def test_system_errors_are_captured_not_raised():
    results = run_evaluation(ScriptedSystem({}), CASES, SETTINGS)
    assert all(r.error and not r.correct for r in results)
    assert aggregate(results)["error_rate"] == 1.0


def test_results_keep_dataset_order_with_concurrency():
    system = ScriptedSystem({c.question: {"answer": "I don't know"} for c in CASES})
    assert [r.id for r in run_evaluation(system, CASES, SETTINGS, concurrency=8)] == ["rate-limit", "oos"]


def test_shipped_dataset_is_valid():
    ds = load_dataset("evals/dataset.json")
    assert len(ds.cases) >= 20
    assert any(not c.answerable for c in ds.cases)


@pytest.mark.parametrize(
    "cases, message",
    [
        ([], "no test cases"),
        ([{"id": "a", "question": "q?"}], "ground_truth"),
        ([{"id": "a", "question": "q?", "answerable": False}] * 2, "duplicate"),
    ],
)
def test_invalid_datasets_are_rejected(tmp_path, cases, message):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"cases": cases}))
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)
