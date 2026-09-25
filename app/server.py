"""Expose the RAG app over HTTP so the same eval suite can target a running service.

python -m app.server --port 8000
curl -s localhost:8000/answer -d '{"question": "What is the API rate limit?"}'
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.rag_app import RAGApp


def make_handler(app: RAGApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/answer":
                return self._send(404, {"error": "not found"})
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 64_000)
                question = json.loads(self.rfile.read(length))["question"]
                if not isinstance(question, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": 'expected JSON body {"question": str}'})
            self._send(200, app.answer(question))

        def log_message(self, *args: object) -> None:  # keep CI logs readable
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="app/config.toml")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(RAGApp(args.config)))
    print(f"RAG app listening on http://{args.host}:{args.port}/answer")
    server.serve_forever()


if __name__ == "__main__":
    main()
