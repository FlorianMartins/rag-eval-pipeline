"""Dense retrieval with static embeddings (model2vec).

Two uses of the same model:

* **retrieval** — sentence embeddings rank chunks by meaning, fused with BM25;
* **grounding** — word embeddings decide whether a question word is covered by the answer
  chunk under another form ("yearly" ~ "annually", "trashed" ~ "trash"). That is what fixes
  vocabulary mismatches *without* answering questions that are merely on-topic.

Why model2vec rather than a transformer encoder: static embeddings give a meaningful vector
for a single word, need only numpy (no torch), weigh ~30 MB and encode in microseconds.
On this knowledge base they also separated answerable from out-of-scope questions better
than bge-small (0.47 vs 0.40 against 0.66 vs 0.65 at sentence level).

Install with ``pip install '.[embeddings]'``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _require_deps():
    try:
        import numpy
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel
    except ImportError as exc:
        raise RuntimeError("retrieval.mode = 'hybrid' needs the embedding extra: pip install '.[embeddings]'") from exc
    return numpy, snapshot_download, StaticModel


class DenseIndex:
    """Cosine-similarity index over chunk embeddings, built once at start-up."""

    def __init__(self, texts: Sequence[str], model: str, revision: str) -> None:
        np, snapshot_download, static_model = _require_deps()
        self._np = np
        # Pinning the revision makes the eval reproducible: a silently re-trained model
        # upstream must show up as a diff in config.toml, not as an unexplained score change.
        path = snapshot_download(repo_id=model, revision=revision)
        self.model = static_model.from_pretrained(path, force_download=False)
        self.matrix = self._encode(texts)
        self._term_cache: dict[str, np.ndarray] = {}

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._np.asarray(self.model.encode(list(texts)), dtype=self._np.float32)
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / self._np.maximum(norms, 1e-12)

    def similarities(self, query: str) -> list[float]:
        """Cosine similarity between the query and every chunk, in index order."""
        return (self.matrix @ self._encode([query])[0]).tolist()

    def _term(self, term: str) -> np.ndarray:
        if term not in self._term_cache:
            self._term_cache[term] = self._encode([term])[0]
        return self._term_cache[term]

    def term_similarity(self, term: str, candidates: Iterable[str]) -> float:
        """Highest cosine similarity between a word and any of the candidate words."""
        return max((float(self._term(term) @ self._term(c)) for c in candidates), default=0.0)
