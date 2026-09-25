"""A minimal RAG assistant used as the *system under test*.

It is intentionally small and dependency-free so the evaluation pipeline can run
offline in CI, but it exposes the same contract as a production RAG service:
``answer(question) -> {"answer": str, "contexts": [{"text", "source", "score"}]}``.
"""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFUSAL = "I don't know based on the available documentation."

_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    of on or our the their there this to was what when where which who why will
    with you your my me we after before every each any all per much many
    compared versus vs if they too happen happens long often get used use choose
    offer there""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        # Light stemming: "plans" -> "plan", "backups" -> "backup".
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    tokens: tuple[str, ...]


class BM25Index:
    """Okapi BM25 over paragraph-level chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.avgdl = sum(len(c.tokens) for c in chunks) / max(len(chunks), 1)
        df: Counter[str] = Counter()
        for c in chunks:
            df.update(set(c.tokens))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Terms never seen in the corpus are the most specific ones of all.
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)

    def weight(self, term: str) -> float:
        return self.idf.get(term, self.max_idf)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        q = _tokenize(query)
        scored = []
        for c in self.chunks:
            tf = Counter(c.tokens)
            dl = len(c.tokens)
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[t] * num / den
            if score > 0:
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def load_chunks(directory: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
        for i, para in enumerate(p for p in paragraphs if p and not p.startswith("#")):
            chunks.append(Chunk(text=para, source=f"{path.name}#{i}", tokens=tuple(_tokenize(para))))
    return chunks


class RAGApp:
    def __init__(self, config_path: str | Path = "app/config.toml", **overrides: Any) -> None:
        # Paths in the config (knowledge base, prompt) are relative to the working directory,
        # like every other path the CLI takes. Not relative to this file: once the package is
        # installed (Docker image), __file__ points into site-packages.
        self.root = Path.cwd()
        with open(config_path, "rb") as fh:
            cfg = tomllib.load(fh)
        self.top_k = int(overrides.get("top_k", cfg["retrieval"]["top_k"]))
        self.min_score = float(overrides.get("min_score", cfg["retrieval"]["min_score"]))
        self.min_coverage = float(overrides.get("min_coverage", cfg["retrieval"]["min_coverage"]))
        self.retrieval_mode = overrides.get("retrieval_mode", cfg["retrieval"].get("mode", "bm25"))
        self.mode = overrides.get("mode", cfg["generation"]["mode"])
        self.temperature = float(cfg["generation"].get("temperature", 0.0))
        self.prompt_template = (self.root / cfg["generation"]["prompt_template"]).read_text(encoding="utf-8")
        self.index = BM25Index(load_chunks(self.root / cfg["retrieval"]["knowledge_base"]))
        self.dense = None
        if self.retrieval_mode == "hybrid":
            from app.embeddings import DenseIndex

            hybrid = cfg["retrieval"]["hybrid"]
            self.embedding_model = hybrid["embedding_model"]
            self.rrf_k = int(overrides.get("rrf_k", hybrid.get("rrf_k", 60)))
            self.term_similarity = float(overrides.get("term_similarity", hybrid["term_similarity"]))
            self.dense = DenseIndex(
                [c.text for c in self.index.chunks], model=self.embedding_model, revision=hybrid["embedding_revision"]
            )
        elif self.retrieval_mode != "bm25":
            raise ValueError(f"unknown retrieval.mode {self.retrieval_mode!r} (expected 'bm25' or 'hybrid')")

    def describe(self) -> dict[str, Any]:
        info = {
            "retrieval": self.retrieval_mode,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "min_coverage": self.min_coverage,
            "mode": self.mode,
        }
        if self.dense is not None:
            info |= {
                "embedding_model": self.embedding_model,
                "rrf_k": self.rrf_k,
                "term_similarity": self.term_similarity,
            }
        return info

    # -- public contract -------------------------------------------------
    def answer(self, question: str) -> dict[str, Any]:
        hits = self._retrieve(question)
        contexts = [{"text": c.text, "source": c.source, "score": round(s, 4)} for c, s in hits]
        if not hits:
            return {"answer": REFUSAL, "contexts": contexts}
        selected = self._select_chunks(question, [c for c, _ in hits])
        lexical_signal = self.dense is not None or hits[0][1] >= self.min_score
        if not lexical_signal or self._coverage(question, selected) < self.min_coverage:
            # The documents talk about *something else*: refusing beats hallucinating.
            return {"answer": REFUSAL, "contexts": contexts}
        if self.mode == "openai_compatible":
            return {"answer": self._generate_llm(question, contexts), "contexts": contexts}
        return {"answer": " ".join(c.text for c in selected), "contexts": contexts}

    # -- retrieval -----------------------------------------------------------
    def _retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        bm25 = self.index.search(question, len(self.index.chunks))
        if self.dense is None:
            return bm25[: self.top_k]
        sims = self.dense.similarities(question)
        dense = sorted(zip(self.index.chunks, sims, strict=True), key=lambda x: x[1], reverse=True)
        # Reciprocal Rank Fusion: robust to the two retrievers having incomparable score scales.
        fused: dict[str, float] = {}
        for ranking in (bm25, dense):
            for rank, (chunk, _) in enumerate(ranking, start=1):
                fused[chunk.source] = fused.get(chunk.source, 0.0) + 1.0 / (self.rrf_k + rank)
        by_source = {c.source: c for c in self.index.chunks}
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [(by_source[src], score) for src, score in top]

    def _coverage(self, question: str, chunks: list[Chunk]) -> float:
        """Share of the question's IDF weight grounded in the chunks chosen as the answer.

        A word is grounded if it appears in the chunks or, in hybrid mode, if a chunk word is
        close enough in embedding space ("yearly" ~ "annually"). Words that are merely
        *on-topic* stay ungrounded ("free tier", "student"), so on-topic questions the
        documents do not answer are still refused. Unknown words weigh the most.
        """
        q = set(_tokenize(question))
        total = sum(self.index.weight(t) for t in q) or 1.0
        chunk_terms = {t for c in chunks for t in c.tokens}
        covered = 0.0
        for term in q:
            if term in chunk_terms or (
                self.dense is not None and self.dense.term_similarity(term, chunk_terms) >= self.term_similarity
            ):
                covered += self.index.weight(term)
        return covered / total

    # -- generators --------------------------------------------------------
    def _select_chunks(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        """Extractive generation: greedily pick the retrieved sentences that cover the question.

        A second sentence is added only if it covers a significant, still-uncovered part of
        the question (multi-part questions such as "Starter *and* Team prices").
        """
        q = set(_tokenize(question))
        total = sum(self.index.weight(t) for t in q) or 1.0
        covered: set[str] = set()
        selected: list[Chunk] = []
        candidates = list(chunks)
        while candidates and len(selected) < 3:
            gains = [
                (sum(self.index.weight(t) for t in (q & set(c.tokens)) - covered) - 0.01 * rank, c)
                for rank, c in enumerate(candidates)
            ]
            gain, chunk = max(gains, key=lambda g: g[0])
            if selected and gain < 0.10 * total:
                break
            selected.append(chunk)
            covered |= q & set(chunk.tokens)
            candidates.remove(chunk)
        return selected

    def _generate_llm(self, question: str, contexts: list[dict[str, Any]]) -> str:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if urllib.parse.urlparse(base_url).scheme not in ("http", "https"):
            raise ValueError(f"LLM_BASE_URL must be http(s), got {base_url!r}")
        prompt = self.prompt_template.format(context="\n".join(f"- {c['text']}" for c in contexts), question=question)
        body = json.dumps(
            {
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(  # noqa: S310 - scheme validated above
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - scheme validated above
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"].strip()
