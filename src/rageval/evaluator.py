"""Evaluation runner: queries the system under test, times each call, scores each case,
and aggregates the results into the metrics the quality gate consumes."""

from __future__ import annotations

import statistics
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from rageval.dataset import TestCase
from rageval.metrics import (
    assertion_failures,
    context_precision,
    context_recall,
    faithfulness,
    is_refusal,
    token_f1,
)
from rageval.metrics.similarity import SimilarityFn
from rageval.systems import System, SystemResponse


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    tags: list[str]
    answer: str | None = None
    ground_truth: str | None = None
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    correct: bool = False
    refused: bool = False
    answer_similarity: float | None = None
    token_f1: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    faithfulness: float | None = None
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalSettings:
    similarity: SimilarityFn
    min_case_similarity: float = 0.5
    refusal_markers: tuple[str, ...] | None = None
    fact_coverage: float = 0.8


def score_case(case: TestCase, response: SystemResponse, latency_ms: float, settings: EvalSettings) -> CaseResult:
    answer = response.answer
    contexts = [c.text for c in response.contexts]
    refused = is_refusal(answer, settings.refusal_markers) if settings.refusal_markers else is_refusal(answer)
    result = CaseResult(
        id=case.id,
        question=case.question,
        answerable=case.answerable,
        tags=list(case.tags),
        answer=answer,
        ground_truth=case.ground_truth,
        sources=[c.source for c in response.contexts],
        latency_ms=round(latency_ms, 2),
        refused=refused,
    )
    result.failures = assertion_failures(answer, case.must_include, case.must_not_include)

    if not case.answerable:
        # Out-of-scope question: the only correct behaviour is to decline instead of hallucinating.
        if not refused:
            result.failures.insert(0, "expected a refusal: the answer is not in the knowledge base")
        result.correct = not result.failures
        return result

    assert case.ground_truth is not None  # guaranteed by dataset validation
    result.answer_similarity = settings.similarity(answer, case.ground_truth)
    result.token_f1 = round(token_f1(answer, case.ground_truth), 4)
    result.context_recall = round(context_recall(case.expected_context, contexts, settings.fact_coverage), 4)
    result.context_precision = round(context_precision(case.expected_context, contexts, settings.fact_coverage), 4)
    if refused:
        result.failures.insert(0, "refused an answerable question")
    else:
        result.faithfulness = round(faithfulness(answer, contexts), 4)
    if result.answer_similarity < settings.min_case_similarity:
        result.failures.append(f"answer similarity {result.answer_similarity:.2f} < {settings.min_case_similarity:.2f}")
    result.correct = not result.failures
    return result


def _run_one(system: System, case: TestCase, settings: EvalSettings) -> CaseResult:
    start = time.perf_counter()
    try:
        response = system.query(case.question)
    except Exception as exc:  # noqa: BLE001 - one failing request must not abort the whole suite
        latency = (time.perf_counter() - start) * 1000
        return CaseResult(
            id=case.id,
            question=case.question,
            answerable=case.answerable,
            tags=list(case.tags),
            ground_truth=case.ground_truth,
            latency_ms=round(latency, 2),
            error=f"{type(exc).__name__}: {exc}",
            failures=["system error", *traceback.format_exception_only(exc)[-1:]],
        )
    latency = (time.perf_counter() - start) * 1000
    return score_case(case, response, latency, settings)


def run_evaluation(
    system: System,
    cases: list[TestCase] | tuple[TestCase, ...],
    settings: EvalSettings,
    concurrency: int = 4,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run every case (concurrently) and return results in dataset order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_run_one, system, case, settings) for case in cases]
        results = []
        for fut in futures:
            res = fut.result()
            if on_result:
                on_result(res)
            results.append(res)
    return results


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — simple and exact on small samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered) + 0.5 - 1e-9))
    return round(ordered[min(rank, len(ordered)) - 1], 2)


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    latencies = [r.latency_ms for r in results if r.error is None]
    n = len(results)
    return {
        "total_cases": n,
        "passed_cases": sum(r.correct for r in results),
        "accuracy": round(sum(r.correct for r in results) / n, 4) if n else 0.0,
        "answerable_accuracy": round(sum(r.correct for r in answerable) / len(answerable), 4) if answerable else None,
        "refusal_accuracy": round(sum(r.correct for r in unanswerable) / len(unanswerable), 4)
        if unanswerable
        else None,
        "answer_similarity": _mean([r.answer_similarity for r in answerable]),
        "token_f1": _mean([r.token_f1 for r in answerable]),
        "context_recall": _mean([r.context_recall for r in answerable]),
        "context_precision": _mean([r.context_precision for r in answerable]),
        "faithfulness": _mean([r.faithfulness for r in answerable]),
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_max_ms": round(max(latencies), 2) if latencies else None,
        "error_rate": round(sum(r.error is not None for r in results) / n, 4) if n else 0.0,
    }


def aggregate_by_tag(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    tags = sorted({t for r in results for t in r.tags})
    out = {}
    for tag in tags:
        subset = [r for r in results if tag in r.tags]
        out[tag] = {
            "cases": len(subset),
            "accuracy": round(sum(r.correct for r in subset) / len(subset), 4),
            "context_recall": _mean([r.context_recall for r in subset]),
        }
    return out
