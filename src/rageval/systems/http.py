"""Adapter for any system exposed over HTTP (staging deployment, preview environment...).

Expected contract::

    POST <url>  {"question": "..."}
    200         {"answer": "...", "contexts": [{"text": "...", "source": "..."}]}
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from rageval.systems.base import SystemResponse


class HTTPSystem:
    def __init__(
        self,
        url: str | None = None,
        timeout: str = "30",
        token_env: str = "RAG_API_TOKEN",  # noqa: S107 - name of an env var, not a secret
    ) -> None:
        self.url = url or os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/answer")
        if urllib.parse.urlparse(self.url).scheme not in ("http", "https"):
            # urllib would happily open file:// or ftp:// URLs.
            raise ValueError(f"system url must be http(s), got {self.url!r}")
        self.timeout = float(timeout)
        self.token = os.environ.get(token_env, "")

    def query(self, question: str) -> SystemResponse:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
            self.url, data=json.dumps({"question": question}).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - scheme validated in __init__
            return SystemResponse.from_payload(json.load(resp))

    def describe(self) -> dict[str, Any]:
        return {"type": "http", "url": self.url}
