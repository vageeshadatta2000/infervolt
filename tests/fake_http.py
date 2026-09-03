"""An in-process stand-in for a vLLM OpenAI-compatible server.

Real sockets, real HTTP, real threads: the adapter's readiness poll, its metrics sampler
and the quality guard all talk to this the same way they talk to a GPU box, so the tests
exercise the actual urllib paths instead of a mocked-out client.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType


class FakeServer:
    """Serves ``/health``, ``/metrics`` and ``/v1/chat/completions``.

    ``metrics`` is walked one entry per scrape and then held at the last one, so a test
    can hand it two recorded snapshots and get a window with a real delta in it.
    ``unhealthy_polls`` makes ``/health`` answer 503 that many times first, the way vLLM
    does while it is still loading weights.
    """

    def __init__(
        self,
        metrics: Sequence[str] = (),
        unhealthy_polls: int = 0,
        completion: Callable[[str], str] | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.unhealthy_polls = unhealthy_polls
        self.completion = completion or (lambda prompt: f"echo: {prompt}")
        self.scrapes = 0
        self.health_polls = 0
        self.chat_requests: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> FakeServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> FakeServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _next_metrics(self) -> str:
        with self._lock:
            if not self.metrics:
                return ""
            text = self.metrics[min(self.scrapes, len(self.metrics) - 1)]
            self.scrapes += 1
            return text

    def _next_health(self) -> int:
        with self._lock:
            self.health_polls += 1
            return 503 if self.health_polls <= self.unhealthy_polls else 200

    def _chat(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self.chat_requests.append(payload)
        messages = payload.get("messages")
        prompt = ""
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                prompt = str(last.get("content", ""))
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": payload.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.completion(prompt)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: object) -> None:
                """Silence the default stderr access log; pytest has enough to read."""

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
                if self.path == "/health":
                    self._send(outer._next_health(), b"", "text/plain")
                elif self.path == "/metrics":
                    self._send(200, outer._next_metrics().encode(), "text/plain; version=0.0.4")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if self.path != "/v1/chat/completions":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._send(400, b"bad json", "text/plain")
                    return
                body = json.dumps(outer._chat(payload)).encode()
                self._send(200, body, "application/json")

        return Handler
