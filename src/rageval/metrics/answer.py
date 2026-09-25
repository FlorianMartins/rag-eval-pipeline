"""Answer-level metrics: lexical overlap, deterministic assertions, refusal detection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from rageval.metrics.text import contains, tokens

DEFAULT_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "not in the provided",
    "not in the available",
    "cannot find",
    "no information",
)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between a prediction and a reference answer."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_refusal(answer: str, markers: Iterable[str] = DEFAULT_REFUSAL_MARKERS) -> bool:
    return any(contains(answer, m) for m in markers)


def assertion_failures(
    answer: str,
    must_include: Iterable[str] = (),
    must_not_include: Iterable[str] = (),
) -> list[str]:
    """Deterministic checks — the "unit tests" of an LLM answer.

    ``must_include`` items may be alternatives separated by ``|`` (e.g. ``"429|too many requests"``).
    Returns a human-readable list of failed assertions (empty list = all passed).
    """
    failures = []
    for item in must_include:
        options = [o.strip() for o in item.split("|") if o.strip()]
        if not any(contains(answer, o) for o in options):
            failures.append(f"missing: {item!r}")
    for item in must_not_include:
        if contains(answer, item):
            failures.append(f"forbidden: {item!r}")
    return failures
