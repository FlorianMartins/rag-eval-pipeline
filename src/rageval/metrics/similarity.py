"""Answer similarity backends.

* ``lexical`` (default): zero-dependency, deterministic. Averages a word-level cosine
  and a character-trigram cosine, which makes it robust to inflections
  ("annual" / "annually") while staying fast enough for every CI run.
* ``embedding``: cosine similarity of sentence embeddings (``pip install .[semantic]``).
  Captures paraphrases; recommended for nightly runs or LLM-generated answers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from functools import lru_cache

from rageval.metrics.text import normalize, tokens

SimilarityFn = Callable[[str, str], float]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def _trigrams(text: str) -> Counter[str]:
    t = f"  {normalize(text)}  "
    return Counter(t[i : i + 3] for i in range(len(t) - 2))


def lexical_similarity(a: str, b: str) -> float:
    word = _cosine(Counter(tokens(a)), Counter(tokens(b)))
    char = _cosine(_trigrams(a), _trigrams(b))
    return round((word + char) / 2, 4)


@lru_cache(maxsize=1)
def _embedding_model(name: str):  # pragma: no cover - optional heavy dependency
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The 'embedding' similarity backend needs sentence-transformers: pip install '.[semantic]'"
        ) from exc
    return SentenceTransformer(name)


def embedding_similarity_factory(model: str = "sentence-transformers/all-MiniLM-L6-v2") -> SimilarityFn:
    def _sim(a: str, b: str) -> float:  # pragma: no cover - optional heavy dependency
        emb = _embedding_model(model).encode([a, b], normalize_embeddings=True)
        return round(float((emb[0] * emb[1]).sum()), 4)

    return _sim


def get_similarity(backend: str = "lexical", **kwargs: str) -> SimilarityFn:
    if backend == "lexical":
        return lexical_similarity
    if backend == "embedding":
        return embedding_similarity_factory(**kwargs)
    raise ValueError(f"unknown similarity backend {backend!r} (expected 'lexical' or 'embedding')")
