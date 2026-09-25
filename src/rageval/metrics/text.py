"""Text normalisation shared by every metric, so that they all agree on what a "token" is."""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    of on or our the their there this to was what when where which who why will
    with you your my me we they them that these those than then so if into""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def normalize(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    return [t for t in toks if not (drop_stopwords and t in _STOPWORDS)]


def contains(haystack: str, needle: str) -> bool:
    """Case/accent-insensitive substring match on normalised text."""
    return normalize(needle) in normalize(haystack)
