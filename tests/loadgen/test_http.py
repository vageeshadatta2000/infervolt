"""The HTTP load generator against an in-process OpenAI-compatible SSE server.

No engine, no model, no network beyond loopback: the fake server streams a configurable
number of content chunks with a configurable delay, so timings are asserted as counts
and orderings rather than as absolute latencies. The whole module stays well under a
second of wall clock.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from infervolt.core.types import LoadSpec, TokenDist, Workload
from infervolt.loadgen.base import LoadGenerator
from infervolt.loadgen.http import HttpLoadGenerator, request_plan
from infervolt.loadgen.prompts import DEFAULT_WORDS_PER_TOKEN


@dataclass
class FakeConfig:
    """What the fake server should do, and what it saw while doing it."""

    tokens: int = 4
    delay_s: float = 0.0
    status: int = 200
    usage_tokens: int | None = None
    tokens_per_word: int = 2
    tokenize_shape: str = "list"
    tokenize_status: int = 200
    fail_first: int = 0

    lock: threading.Lock = field(default_factory=threading.Lock)
    prompts: list[str] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0
    started: int = 0
    tokenize_calls: int = 0

    def enter(self, body: dict[str, Any]) -> bool:
        """Record a chat request; return True when this one should be failed."""
        with self.lock:
            self.bodies.append(body)
            self.prompts.append(body["messages"][0]["content"])
            self.started += 1
            fail = self.started <= self.fail_first
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        return fail

    def leave(self) -> None:
        with self.lock:
            self.in_flight -= 1


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 with no Content-Length: the client streams until the connection closes,
    # which is all an SSE consumer needs and avoids hand-rolling chunked encoding.
    protocol_version = "HTTP/1.0"

    @property
    def cfg(self) -> FakeConfig:
        assert isinstance(self.server, _Server)
        return self.server.cfg

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/tokenize"):
            self._tokenize(body)
        else:
            self._chat(body)

    def _tokenize(self, body: dict[str, Any]) -> None:
        cfg = self.cfg
        with cfg.lock:
            cfg.tokenize_calls += 1
        if cfg.tokenize_status != 200:
            self._json(cfg.tokenize_status, {"error": "no"})
            return
        n = len(str(body.get("content", "")).split()) * cfg.tokens_per_word
        shapes: dict[str, dict[str, Any]] = {
            "list": {"tokens": list(range(n))},
            "count": {"count": n},
            "int": {"tokens": n},
            "junk": {"nothing": "useful"},
        }
        self._json(200, shapes[cfg.tokenize_shape])

    def _chat(self, body: dict[str, Any]) -> None:
        cfg = self.cfg
        fail = cfg.enter(body)
        try:
            if cfg.status != 200 or fail:
                self._json(cfg.status if cfg.status != 200 else 500, {"error": "boom"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(cfg.tokens):
                time.sleep(cfg.delay_s)
                self.event({"choices": [{"delta": {"content": f"w{i} "}}]})
            if cfg.usage_tokens is not None:
                self.event(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": cfg.usage_tokens},
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up (timeout test); nothing to report
        finally:
            cfg.leave()

    def event(self, obj: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cfg: FakeConfig) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.cfg = cfg


@contextmanager
def serve(cfg: FakeConfig) -> Iterator[str]:
    server = _Server(cfg)
    # A short poll interval: the default 0.5 s is paid on every shutdown, which across a
    # module of small tests costs more than all the requests put together.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.daemon = True
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def cfg() -> FakeConfig:
    return FakeConfig()


@pytest.fixture
def base_url(cfg: FakeConfig) -> Iterator[str]:
    with serve(cfg) as url:
        yield url


def wl(**kw: Any) -> Workload:
    base: dict[str, Any] = {
        "name": "t",
        "isl": TokenDist(p50=32, p99=64),
        "osl": TokenDist(p50=8, p99=16),
        "load": LoadSpec(concurrency=[2]),
    }
    base.update(kw)
    return Workload(**base)


def test_it_is_a_load_generator() -> None:
    assert isinstance(HttpLoadGenerator("http://x"), LoadGenerator)


# ---------------------------------------------------------------- streaming


def test_ttft_and_one_interval_per_chunk_after_the_first(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 5
    cfg.delay_s = 0.005
    res = HttpLoadGenerator(base_url, model="m").run(wl(), 2, num_requests=4, seed=1)

    assert len(res.requests) == 4
    assert all(r.ok for r in res.requests)
    for r in res.requests:
        assert r.output_tokens == 5
        assert len(r.itl_s) == 4
        assert r.ttft_s > 0
        assert all(x > 0 for x in r.itl_s)
    assert res.concurrency == 2
    assert res.health.error_rate == 0.0


def test_request_payload_carries_the_openai_streaming_contract(
    base_url: str, cfg: FakeConfig
) -> None:
    lg = HttpLoadGenerator(base_url, model="qwen", extra_body={"top_p": 0.4, "seed": 11})
    lg.run(wl(osl=TokenDist(p50=13, p99=20)), 1, num_requests=1, seed=1)

    body = cfg.bodies[0]
    assert body["model"] == "qwen"
    assert body["stream"] is True
    assert body["max_tokens"] == 13
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.4
    assert body["seed"] == 11
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_extra_body_can_override_the_defaults(base_url: str, cfg: FakeConfig) -> None:
    HttpLoadGenerator(base_url, extra_body={"temperature": 0.0}).run(wl(), 1, 1, 1)
    assert cfg.bodies[0]["temperature"] == 0.0


def test_headers_are_sent(base_url: str) -> None:
    lg = HttpLoadGenerator(base_url, headers={"Authorization": "Bearer x"})
    assert lg.run(wl(), 1, 1, 1).requests[0].ok


def test_usage_completion_tokens_beat_the_chunk_count(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 3
    cfg.usage_tokens = 42
    res = HttpLoadGenerator(base_url).run(wl(), 1, num_requests=2, seed=1)
    assert [r.output_tokens for r in res.requests] == [42, 42]
    # The intervals are still one per streamed chunk after the first.
    assert [len(r.itl_s) for r in res.requests] == [2, 2]


def test_duration_spans_first_send_to_last_completion(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 3
    cfg.delay_s = 0.01
    t0 = time.perf_counter()
    res = HttpLoadGenerator(base_url).run(wl(), 2, num_requests=4, seed=1)
    elapsed = time.perf_counter() - t0

    assert res.duration_s > 0
    assert res.duration_s <= elapsed
    # Two rounds of three 10 ms tokens is the floor; the ceiling is the wall clock above.
    assert res.duration_s >= 0.04


def test_empty_run_still_has_a_positive_duration(base_url: str) -> None:
    res = HttpLoadGenerator(base_url).run(wl(), 2, num_requests=0, seed=1)
    assert res.requests == []
    assert res.duration_s >= 1e-3


# ---------------------------------------------------------------- closed loop


def test_in_flight_never_exceeds_concurrency(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 3
    cfg.delay_s = 0.01
    res = HttpLoadGenerator(base_url).run(wl(), 4, num_requests=16, seed=1)

    assert len(res.requests) == 16
    assert cfg.started == 16
    assert cfg.max_in_flight <= 4
    assert cfg.max_in_flight > 1  # the loop really is concurrent


def test_every_request_is_sent_exactly_once(base_url: str, cfg: FakeConfig) -> None:
    HttpLoadGenerator(base_url).run(wl(), 8, num_requests=9, seed=1)
    assert cfg.started == 9


def test_concurrency_must_be_positive(base_url: str) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        HttpLoadGenerator(base_url).run(wl(), 0, num_requests=1, seed=1)


# ---------------------------------------------------------------- failures


def test_http_errors_produce_failed_records(base_url: str, cfg: FakeConfig) -> None:
    cfg.status = 503
    res = HttpLoadGenerator(base_url).run(wl(), 2, num_requests=4, seed=1)

    assert len(res.requests) == 4
    assert not any(r.ok for r in res.requests)
    assert all(r.output_tokens == 0 for r in res.requests)
    assert res.health.error_rate == 1.0
    assert res.duration_s > 0


def test_partial_failure_shows_up_as_a_partial_error_rate(base_url: str, cfg: FakeConfig) -> None:
    cfg.fail_first = 2
    res = HttpLoadGenerator(base_url).run(wl(), 1, num_requests=4, seed=1)
    assert sum(1 for r in res.requests if r.ok) == 2
    assert res.health.error_rate == pytest.approx(0.5)


def test_timeout_produces_a_failed_record_not_an_exception(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 3
    cfg.delay_s = 0.3
    res = HttpLoadGenerator(base_url, timeout_s=0.1).run(wl(), 1, num_requests=1, seed=1)
    assert len(res.requests) == 1
    assert not res.requests[0].ok


def test_unreachable_server_fails_every_request() -> None:
    # Port 1 on loopback: connection refused, immediately.
    res = HttpLoadGenerator("http://127.0.0.1:1", timeout_s=1.0).run(wl(), 2, 2, 1)
    assert [r.ok for r in res.requests] == [False, False]
    assert res.health.error_rate == 1.0


def test_malformed_sse_payload_is_an_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, cfg: FakeConfig
) -> None:
    def bad_event(self: _Handler, obj: dict[str, Any]) -> None:
        self.wfile.write(b"data: {not json\n\n")
        self.wfile.flush()

    monkeypatch.setattr(_Handler, "event", bad_event)
    with serve(cfg) as base_url:
        res = HttpLoadGenerator(base_url).run(wl(), 1, num_requests=1, seed=1)
    assert not res.requests[0].ok


def test_a_stream_with_no_content_is_not_a_served_request(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 0
    res = HttpLoadGenerator(base_url).run(wl(), 1, num_requests=2, seed=1)
    assert not any(r.ok for r in res.requests)


# ---------------------------------------------------------------- prompts


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def test_prompts_are_deterministic_across_runs(base_url: str, cfg: FakeConfig) -> None:
    lg = HttpLoadGenerator(base_url)
    lg.run(wl(), 4, num_requests=6, seed=5)
    first = sorted(cfg.prompts)
    cfg.prompts.clear()
    lg.run(wl(), 1, num_requests=6, seed=5)
    assert sorted(cfg.prompts) == first

    cfg.prompts.clear()
    lg.run(wl(), 1, num_requests=6, seed=6)
    assert sorted(cfg.prompts) != first


def test_prefix_share_gives_every_request_the_same_opening(base_url: str, cfg: FakeConfig) -> None:
    workload = wl(isl=TokenDist(p50=200, p99=400), prefix_share=0.5)
    HttpLoadGenerator(base_url).run(workload, 2, num_requests=6, seed=3)

    prefix_words = round(200 * 0.5 * DEFAULT_WORDS_PER_TOKEN)
    assert len({first_words(p, prefix_words) for p in cfg.prompts}) == 1
    # And the shared part really is a prefix, not the whole prompt.
    assert len(set(cfg.prompts)) == 6


def test_no_prefix_share_means_no_shared_opening(base_url: str, cfg: FakeConfig) -> None:
    HttpLoadGenerator(base_url).run(wl(isl=TokenDist(p50=200, p99=200)), 2, 6, 3)
    assert len({first_words(p, 5) for p in cfg.prompts}) > 1


def test_isl_mix_is_eighty_percent_p50_and_the_rest_in_the_tail() -> None:
    workload = wl(isl=TokenDist(p50=100, p99=400))
    lengths = [request_plan(workload, i, 9)[0] for i in range(400)]
    at_p50 = sum(1 for x in lengths if x == 100)
    assert 0.7 <= at_p50 / len(lengths) <= 0.9
    tail = [x for x in lengths if x != 100]
    assert tail and all(100 < x <= 400 for x in tail)


def test_isl_plan_is_deterministic_and_index_dependent() -> None:
    workload = wl(isl=TokenDist(p50=100, p99=400))
    assert request_plan(workload, 3, 9) == request_plan(workload, 3, 9)
    assert request_plan(workload, 3, 9) != request_plan(workload, 4, 9)


def test_degenerate_isl_distribution_never_samples_a_tail() -> None:
    workload = wl(isl=TokenDist(p50=100, p99=100))
    assert {request_plan(workload, i, 9)[0] for i in range(50)} == {100}


def test_prompt_length_follows_the_workload_isl(base_url: str, cfg: FakeConfig) -> None:
    HttpLoadGenerator(base_url).run(wl(isl=TokenDist(p50=120, p99=120)), 1, 3, 1)
    assert [len(p.split()) for p in cfg.prompts] == [round(120 * DEFAULT_WORDS_PER_TOKEN)] * 3


# ---------------------------------------------------------------- calibration


@pytest.mark.parametrize("shape", ["list", "count", "int"])
def test_tokenize_endpoint_shapes_all_calibrate(shape: str) -> None:
    cfg = FakeConfig(tokenize_shape=shape, tokens_per_word=2)
    with serve(cfg) as base_url:
        lg = HttpLoadGenerator(base_url, tokenize_url=f"{base_url}/tokenize")
        lg.run(wl(isl=TokenDist(p50=120, p99=120)), 1, num_requests=2, seed=1)
    # Two tokens per word means 0.5 words per token, not the 0.75 default.
    assert [len(p.split()) for p in cfg.prompts] == [60, 60]


def test_tokenize_is_relative_to_base_url_and_called_once_per_run(
    base_url: str, cfg: FakeConfig
) -> None:
    lg = HttpLoadGenerator(base_url, tokenize_url="/tokenize")
    lg.run(wl(isl=TokenDist(p50=120, p99=120)), 4, num_requests=8, seed=1)
    assert cfg.tokenize_calls == 1
    assert [len(p.split()) for p in cfg.prompts] == [60] * 8


def test_a_broken_tokenizer_falls_back_to_the_default_ratio(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokenize_status = 500
    lg = HttpLoadGenerator(base_url, tokenize_url=f"{base_url}/tokenize")
    lg.run(wl(isl=TokenDist(p50=120, p99=120)), 1, num_requests=1, seed=1)
    assert len(cfg.prompts[0].split()) == round(120 * DEFAULT_WORDS_PER_TOKEN)


def test_an_unparseable_tokenizer_response_falls_back(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokenize_shape = "junk"
    lg = HttpLoadGenerator(base_url, tokenize_url=f"{base_url}/tokenize")
    lg.run(wl(isl=TokenDist(p50=120, p99=120)), 1, num_requests=1, seed=1)
    assert len(cfg.prompts[0].split()) == round(120 * DEFAULT_WORDS_PER_TOKEN)


def test_an_unreachable_tokenizer_falls_back(base_url: str, cfg: FakeConfig) -> None:
    lg = HttpLoadGenerator(base_url, tokenize_url="http://127.0.0.1:1/tokenize", timeout_s=1.0)
    lg.run(wl(isl=TokenDist(p50=120, p99=120)), 1, num_requests=1, seed=1)
    assert len(cfg.prompts[0].split()) == round(120 * DEFAULT_WORDS_PER_TOKEN)


# ---------------------------------------------------------------- client health


def test_client_health_is_populated(base_url: str, cfg: FakeConfig) -> None:
    cfg.tokens = 4
    cfg.delay_s = 0.005
    res = HttpLoadGenerator(base_url).run(wl(), 2, num_requests=8, seed=1)
    assert 0.0 <= res.health.worker_cpu <= 1.0
    assert res.health.loop_lag_p99_ms >= 0.0
    assert res.health.error_rate == 0.0
