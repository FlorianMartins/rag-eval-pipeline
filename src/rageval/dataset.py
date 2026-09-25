"""Test-case schema and loader. Validation fails loudly: a broken dataset must never
silently turn into a green quality gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # not a pytest test class, despite the name

    id: str
    question: str
    answerable: bool = True
    ground_truth: str | None = None
    expected_context: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TestCase:
        cid = raw.get("id") or "<missing id>"
        for key in ("id", "question"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise DatasetError(f"case {cid}: '{key}' must be a non-empty string")
        answerable = bool(raw.get("answerable", True))
        assertions = raw.get("assertions", {})
        case = cls(
            id=raw["id"],
            question=raw["question"],
            answerable=answerable,
            ground_truth=raw.get("ground_truth"),
            expected_context=tuple(raw.get("expected_context", ())),
            must_include=tuple(assertions.get("must_include", ())),
            must_not_include=tuple(assertions.get("must_not_include", ())),
            tags=tuple(raw.get("tags", ())),
        )
        if answerable and not case.ground_truth:
            raise DatasetError(f"case {cid}: answerable cases need a 'ground_truth'")
        if answerable and not case.expected_context:
            raise DatasetError(f"case {cid}: answerable cases need an 'expected_context' list")
        return case


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    cases: tuple[TestCase, ...]
    sha256: str


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON ({exc})") from exc
    cases = tuple(TestCase.from_dict(c) for c in raw.get("cases", []))
    if not cases:
        raise DatasetError(f"{path}: no test cases")
    ids = [c.id for c in cases]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise DatasetError(f"{path}: duplicate case ids {dupes}")
    return Dataset(
        name=raw.get("name", path.stem),
        version=str(raw.get("version", "unversioned")),
        cases=cases,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
