"""In-process adapter for the demo RAG application in ``app/``."""

from __future__ import annotations

from typing import Any

from rageval.systems.base import SystemResponse


class LocalRAGSystem:
    def __init__(self, config: str = "app/config.toml", **overrides: str) -> None:
        from app.rag_app import RAGApp  # imported lazily: the harness does not depend on the app

        self.app = RAGApp(config, **overrides)

    def query(self, question: str) -> SystemResponse:
        return SystemResponse.from_payload(self.app.answer(question))

    def describe(self) -> dict[str, Any]:
        return {"type": "local", **self.app.describe()}
