"""Closed-loop load generator for any OpenAI-compatible streaming endpoint.

This is the generator every real engine uses: vLLM, llama.cpp's server, SGLang, TGI and
anything else that speaks ``POST /v1/chat/completions`` with ``stream: true``. It runs
*in the same process as the engine* on a rented box, which is the whole point -- TTFT
and ITL measured across the public internet would be measuring the internet.

Design notes worth knowing before changing anything here:

**Closed loop, not open loop.** ``concurrency`` worker threads share a start counter and
each sends its next request only once its previous one has finished, so the number in
flight is exactly ``concurrency`` until the tail. That is what makes a load point a
point: an open-loop generator at a fixed arrival rate would queue without bound past
saturation and measure the queue rather than the server.

**Threads, not asyncio.** The work per request is one blocking stream read; threads keep
the timestamping code straight-line and let ``ClientHealth`` be measured with
``time.process_time``. The heartbeat below is what catches the case where the *client*
is the bottleneck.

**Token accounting.** ``ttft_s`` is the time to the first non-empty content delta and
``itl_s`` holds one interval per content delta *after* it, so a stream of ``n`` chunks
yields ``n - 1`` intervals against ``output_tokens == n``. That deliberately differs
from :class:`~infervolt.core.types.RequestRecord`'s simulated convention (where the
counts match): a real server tells us when tokens arrived, and inventing an interval for
the first one would double-count the prefill that ``ttft_s`` already reports. Chunk
count is a proxy for token count -- a chunk is usually one token, but not always -- so
``usage.completion_tokens`` wins whenever the server sends it.

**Failures are data.** A timeout, a 500, a refused connection or an unparseable stream
produces a record with ``ok=False`` rather than an exception: a config that makes the
server fall over is a measurement, and the search has to see it as one.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from typing import Any

import httpx
import numpy as np

from infervolt.core.types import ClientHealth, LoadResult, RequestRecord, Workload
from infervolt.loadgen.prompts import (
    DEFAULT_WORDS_PER_TOKEN,
    build_prompt,
    calibrate_words_per_token,
    shared_prefix,
)

TAIL_SHARE = 0.2
"""Fraction of requests drawn from the tail of the ISL distribution.

A workload states a p50 and a p99 and nothing in between, so the generator sends four
requests in five at the p50 and the fifth uniformly in ``(p50, p99]``. Sending every
request at the p50 would hide prefill spikes; sampling a full distribution we do not
have would be inventing one.
"""

HEARTBEAT_S = 0.010
"""How often the client checks whether it is itself stalling. See :class:`_Heartbeat`."""

MIN_DURATION_S = 1e-3
"""Floor on the reported duration: every rate in ``compute_metrics`` divides by it."""


def request_plan(workload: Workload, index: int, seed: int) -> tuple[int, int]:
    """The ``(input_tokens, prompt_seed)`` for one request of a run.

    Pure and index-addressed rather than drawn from a shared stream, so the request a
    worker picks up does not depend on which worker picked it up: the same
    ``(workload, index, seed)`` always describes the same request, whatever the thread
    scheduling did.
    """
    rng = random.Random(f"{seed}:{index}")
    p50, p99 = workload.isl.p50, workload.isl.p99
    tail = rng.random() < TAIL_SHARE
    tokens = rng.randint(p50 + 1, p99) if tail and p99 > p50 else p50
    return tokens, rng.getrandbits(32)


def _delta_content(obj: object) -> str:
    """The text of one SSE chunk, or ``""`` for a chunk that carries none.

    Every step is guarded because the payload is whatever the server sent: role-only
    opening chunks, finish-reason closing chunks and usage-only chunks are all normal,
    and none of them are tokens.
    """
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _usage_tokens(obj: object) -> int | None:
    """``usage.completion_tokens`` from a chunk that carries it, else ``None``."""
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    n = usage.get("completion_tokens")
    if isinstance(n, bool) or not isinstance(n, int):
        return None
    return n


def _token_count(payload: object) -> int:
    """Read a token count out of a tokenizer endpoint's reply.

    Three shapes are accepted: llama.cpp's ``{"tokens": [ids...]}``, and the
    ``{"count": n}`` / ``{"tokens": n}`` variants other servers use. Anything else
    raises, and the caller falls back to the default ratio.
    """
    if isinstance(payload, dict):
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            return tokens
        count = payload.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    raise ValueError(f"unrecognised tokenize response: {payload!r}")


class _Heartbeat:
    """Measures how late a fixed short sleep actually wakes.

    A load generator that is CPU-starved reports the *client's* latency as the server's.
    The GIL, an oversubscribed box or a swapping process all show up the same way: a
    thread that asked for 10 ms gets 60 ms. The p99 of that overshoot is the signal the
    ``client_artifact`` diagnosis rule reads, so it has to be sampled while the load is
    running, not before or after.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lags: list[float] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._stop.wait(HEARTBEAT_S)
            over = time.perf_counter() - t0 - HEARTBEAT_S
            self._lags.append(max(over, 0.0) * 1000)

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def p99_ms(self) -> float:
        # Read after __exit__ has joined the thread, so no lock is needed.
        return float(np.percentile(self._lags, 99)) if self._lags else 0.0


class HttpLoadGenerator:
    """Drives an OpenAI-compatible server and returns per-request timings.

    ``extra_body`` is merged into the request payload last, so it can both add knobs the
    server understands (``top_p``, ``ignore_eos``, ``top_k``) and override the defaults
    this class sets. ``tokenize_url`` may be absolute or a path relative to ``base_url``.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
        headers: dict[str, str] | None = None,
        tokenize_url: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self.headers = dict(headers or {})
        self.tokenize_url = tokenize_url

    # ------------------------------------------------------------ public API

    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if num_requests <= 0:
            return LoadResult(concurrency=concurrency, duration_s=MIN_DURATION_S, requests=[])

        # One connection per worker, plus headroom: httpx's default pool of 100 would
        # silently serialise a 128-way load point into 100 sockets and call it the
        # server's queueing.
        limits = httpx.Limits(
            max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
        )
        with httpx.Client(timeout=self.timeout_s, headers=self.headers, limits=limits) as client:
            wpt = self._words_per_token(client)
            prefix_tokens = int(workload.isl.p50 * workload.prefix_share)
            prefix = (
                shared_prefix(prefix_tokens, seed, words_per_token=wpt) if prefix_tokens > 0 else ""
            )
            cpu0, wall0 = time.process_time(), time.perf_counter()
            with _Heartbeat() as heartbeat:
                records, first_send, last_done = self._drive(
                    client, workload, concurrency, num_requests, seed, prefix, wpt
                )
            cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0

        duration = max(last_done - first_send, MIN_DURATION_S)
        failures = sum(1 for r in records if not r.ok)
        return LoadResult(
            concurrency=concurrency,
            duration_s=duration,
            requests=records,
            health=ClientHealth(
                worker_cpu=self._worker_cpu(cpu_s, wall_s, concurrency),
                loop_lag_p99_ms=heartbeat.p99_ms(),
                error_rate=failures / len(records) if records else 0.0,
            ),
        )

    # ------------------------------------------------------------ the loop

    def _drive(
        self,
        client: httpx.Client,
        workload: Workload,
        concurrency: int,
        num_requests: int,
        seed: int,
        prefix: str,
        wpt: float,
    ) -> tuple[list[RequestRecord], float, float]:
        """Run the closed loop; return records in request order plus the run's span."""
        lock = threading.Lock()
        done: dict[int, RequestRecord] = {}
        started = 0
        first_send = math.inf
        last_done = 0.0

        def worker() -> None:
            nonlocal started, first_send, last_done
            while True:
                with lock:
                    if started >= num_requests:
                        return
                    index = started
                    started += 1
                tokens, prompt_seed = request_plan(workload, index, seed)
                prompt = build_prompt(tokens, prompt_seed, prefix=prefix, words_per_token=wpt)
                sent = time.perf_counter()
                with lock:
                    first_send = min(first_send, sent)
                record = self._one(client, prompt, workload.osl.p50, sent)
                with lock:
                    last_done = max(last_done, time.perf_counter())
                    done[index] = record

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [done[i] for i in sorted(done)], first_send, last_done

    def _one(
        self, client: httpx.Client, prompt: str, max_tokens: int, sent: float
    ) -> RequestRecord:
        """One streamed completion, timestamped chunk by chunk."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            **self.extra_body,
        }
        ttft = 0.0
        itls: list[float] = []
        chunks = 0
        usage: int | None = None
        ok = True
        try:
            with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()  # drain, so the connection can be reused
                    return RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
                previous = sent
                for raw in resp.iter_lines():
                    line = raw.strip()
                    if not line.startswith("data:"):
                        continue  # SSE comments, event: lines and blank separators
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    now = time.perf_counter()
                    if (reported := _usage_tokens(obj)) is not None:
                        usage = reported
                    if not _delta_content(obj):
                        continue
                    chunks += 1
                    if chunks == 1:
                        ttft = now - sent
                    else:
                        itls.append(now - previous)
                    previous = now
        except (httpx.HTTPError, ValueError):
            # ValueError covers json.JSONDecodeError; httpx.HTTPError covers timeouts,
            # refused connections and mid-stream transport failures alike.
            ok = False
        # A stream that carried no content is not a served request, whatever its status:
        # counting it as a success would let a server that answers instantly with
        # nothing look like the fastest config in the search.
        if chunks == 0:
            ok = False
        return RequestRecord(
            ttft_s=ttft,
            itl_s=itls,
            output_tokens=usage if usage is not None else chunks,
            ok=ok,
        )

    # ------------------------------------------------------------ helpers

    def _words_per_token(self, client: httpx.Client) -> float:
        """Calibrate against the server's tokenizer once per run, or fall back.

        A wrong ratio does not break the run, it just means the workload's stated ISL is
        not the ISL that was sent -- so a tokenizer that is missing, slow or answering in
        an unknown shape is a fallback, never an error.
        """
        if not self.tokenize_url:
            return DEFAULT_WORDS_PER_TOKEN
        url = (
            self.tokenize_url
            if "://" in self.tokenize_url
            else f"{self.base_url}/{self.tokenize_url.lstrip('/')}"
        )

        def count(text: str) -> int:
            resp = client.post(url, json={"content": text})
            resp.raise_for_status()
            return _token_count(resp.json())

        try:
            return calibrate_words_per_token(count)
        except (httpx.HTTPError, ValueError):
            return DEFAULT_WORDS_PER_TOKEN

    @staticmethod
    def _worker_cpu(cpu_s: float, wall_s: float, concurrency: int) -> float:
        """Fraction of the client's available CPU the load phase burned.

        The denominator is the number of cores the run could actually have used -- there
        is no point dividing a 4-thread load by 64 cores -- so a value near 1.0 means the
        generator was saturated and its timings describe the client, not the server.
        """
        cores = max(1, min(concurrency, os.cpu_count() or 1))
        return cpu_s / (wall_s * cores) if wall_s > 0 else 0.0
