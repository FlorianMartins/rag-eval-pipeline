"""Retrieval metrics, in the spirit of RAGAS but deterministic and dependency-free.

A *fact* is a short statement from the ground-truth context (``expected_context``).
A fact is considered present in a chunk when at least ``coverage`` of its content
tokens appear in that chunk — tolerant to rewording of stopwords and punctuation,
strict on the tokens that carry the information (numbers, names, units).
"""

from __future__ import annotations

from collections.abc import Sequence

from rageval.metrics.text import tokens

DEFAULT_COVERAGE = 0.8


def fact_in_text(fact: str, text: str, coverage: float = DEFAULT_COVERAGE) -> bool:
    fact_toks = set(tokens(fact))
    if not fact_toks:
        return False
    return len(fact_toks & set(tokens(text))) / len(fact_toks) >= coverage


def context_recall(facts: Sequence[str], contexts: Sequence[str], coverage: float = DEFAULT_COVERAGE) -> float:
    """Share of ground-truth facts found in *at least one* retrieved chunk.

    Answers: "did the retriever bring back everything needed to answer?"
    """
    if not facts:
        raise ValueError("context_recall needs at least one expected fact")
    found = sum(any(fact_in_text(f, c, coverage) for c in contexts) for f in facts)
    return found / len(facts)


def context_precision(facts: Sequence[str], contexts: Sequence[str], coverage: float = DEFAULT_COVERAGE) -> float:
    """Rank-aware precision (average precision over relevant chunks).

    A chunk is relevant if it contains at least one ground-truth fact. Relevant chunks
    ranked first score 1.0; relevant chunks buried under noise are penalised.
    Answers: "is the signal at the top of what we hand the LLM?"
    """
    if not contexts:
        return 0.0
    relevant = [any(fact_in_text(f, c, coverage) for f in facts) for c in contexts]
    hits, score = 0, 0.0
    for k, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            score += hits / k
    return score / hits if hits else 0.0


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Groundedness proxy: share of the answer's content tokens that appear in the retrieved context.

    A low score flags tokens the model produced without support — the cheap,
    deterministic first line of hallucination detection. Pair it with an
    LLM-as-judge for semantic groundedness when the budget allows.
    """
    answer_toks = tokens(answer)
    if not answer_toks:
        return 0.0
    context_toks = set(tokens(" ".join(contexts)))
    return sum(t in context_toks for t in answer_toks) / len(answer_toks)
