"""The contract every system under test must satisfy.

Keeping this contract tiny is what makes the harness reusable: a LangChain chain,
a LlamaIndex query engine, an agent behind an HTTP API or a Bedrock/Vertex endpoint
only need a ~20-line adapter to be evaluated by the same pipeline.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Context:
    text: str
    source: str = ""
    score: float | None = None


@dataclass(frozen=True)
class SystemResponse:
    answer: str
    contexts: tuple[Context, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SystemResponse:
        if not isinstance(payload.get("answer"), str):
            raise ValueError("system response must contain a string 'answer'")
        contexts = tuple(
            Context(text=c["text"], source=c.get("source", ""), score=c.get("score"))
            if isinstance(c, dict)
            else Context(text=str(c))
            for c in payload.get("contexts", [])
        )
        return cls(answer=payload["answer"], contexts=contexts, metadata=payload.get("metadata", {}))


@runtime_checkable
class System(Protocol):
    def query(self, question: str) -> SystemResponse: ...

    def describe(self) -> dict[str, Any]: ...


BUILTIN_SYSTEMS = {
    "local": "rageval.systems.local:LocalRAGSystem",
    "http": "rageval.systems.http:HTTPSystem",
}


def load_system(spec: str, options: dict[str, str] | None = None) -> System:
    """Instantiate a system from a builtin name (``local``, ``http``) or ``module:Class``."""
    target = BUILTIN_SYSTEMS.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown system {spec!r}: use one of {sorted(BUILTIN_SYSTEMS)} or 'module:Class'")
    module_name, class_name = target.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    system = cls(**(options or {}))
    if not isinstance(system, System):
        raise TypeError(f"{target} does not implement query() and describe()")
    return system
