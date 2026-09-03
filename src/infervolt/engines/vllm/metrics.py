"""Prometheus scraping for vLLM: text parsing, metric-name resolution, window aggregation.

vLLM renames metrics between releases (``gpu_cache_usage_perc`` became
``kv_cache_usage_perc`` in V1, the prefix-cache counters lost their ``gpu_`` prefix and
gained ``_total``), so nothing here hard-codes a name. Each *role* the adapter needs is a
regex, resolved once against the names the server actually exposes; the resolved mapping
is recorded so a run says which counters produced its numbers.

Everything is computed over a *window* -- the snapshots taken while one load point ran.
Gauges are summarised across the window (p95, mean, max), counters are differenced across
it, and histograms are differenced bucket-by-bucket before a quantile is read, so a
percentile describes the requests served during that load point rather than every request
since the server booted.
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- text parsing

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?[ \t]+(?P<value>\S+)$"
)
_LE_RE = re.compile(r'\ble\s*=\s*"([^"]*)"')

GpuProbe = Callable[[], tuple[float, float] | None]
"""Returns (gpu utilization %, memory utilization %) averaged over devices, or None."""


def _to_float(token: str) -> float | None:
    """Prometheus value token to a float, or None for the ones no aggregate can use."""
    try:
        value = float(token)
    except ValueError:
        return None
    return None if math.isnan(value) else value


@dataclass(frozen=True)
class MetricSnapshot:
    """One ``/metrics`` scrape, summed over label sets.

    vLLM labels every series with ``model_name`` and ``engine``; a single server has one
    of each, and summing is the right reduction for the multi-engine case anyway
    (counters add, and the gauges we read -- queue depths, cache occupancy -- are
    per-engine quantities whose total is what the scheduler is actually carrying).
    """

    ts: float
    values: dict[str, float] = field(default_factory=dict)
    buckets: dict[str, dict[float, float]] = field(default_factory=dict)
    """family name -> upper bound -> cumulative count. ``inf`` is the ``+Inf`` bucket."""

    def families(self) -> set[str]:
        """Every metric family present, with the ``_sum``/``_count`` suffixes removed."""
        names = set(self.buckets)
        for name in self.values:
            for suffix in ("_sum", "_count"):
                if name.endswith(suffix):
                    names.add(name[: -len(suffix)])
                    break
            else:
                names.add(name)
        return names


def parse_metrics(text: str, ts: float | None = None) -> MetricSnapshot:
    """Parse Prometheus exposition text. Unparseable lines are skipped, never raised on.

    A scrape is telemetry, not input we control: one malformed line in a release we have
    not seen must cost us that line, not the whole observation.
    """
    values: dict[str, float] = {}
    buckets: dict[str, dict[float, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if m is None:
            continue
        value = _to_float(m["value"])
        if value is None:
            continue
        name = m["name"]
        if name.endswith("_bucket"):
            le_match = _LE_RE.search(m["labels"] or "")
            if le_match is None:
                continue
            bound = _to_float(le_match[1])
            if bound is None:
                continue
            family = buckets.setdefault(name[: -len("_bucket")], {})
            family[bound] = family.get(bound, 0.0) + value
        else:
            values[name] = values.get(name, 0.0) + value
    return MetricSnapshot(ts=time.monotonic() if ts is None else ts, values=values, buckets=buckets)


# ---------------------------------------------------------------- name resolution

METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "kv_usage": re.compile(r"^vllm:(?:kv_cache|gpu_cache)_usage_perc$"),
    "num_waiting": re.compile(r"^vllm:num_requests_waiting$"),
    "num_running": re.compile(r"^vllm:num_requests_running$"),
    "preemptions": re.compile(r"^vllm:num_preemptions(?:_total)?$"),
    "queue_time": re.compile(r"^vllm:(?:request_)?queue_time_seconds$"),
    "prefill_time": re.compile(r"^vllm:request_prefill_time_seconds$"),
    "inference_time": re.compile(r"^vllm:request_inference_time_seconds$"),
    "prefix_hits": re.compile(r"^vllm:(?:gpu_)?prefix_cache_hits(?:_total)?$"),
    "prefix_queries": re.compile(r"^vllm:(?:gpu_)?prefix_cache_queries(?:_total)?$"),
}
"""Role -> pattern matching the family name that fills it, across vLLM releases."""


def resolve_names(present: Iterable[str]) -> dict[str, str]:
    """Map each role to the family that serves it. Roles the server does not expose are absent.

    Candidates are matched in sorted order so a server exposing both the old and the new
    spelling of a metric resolves the same way on every run.
    """
    names = sorted(present)
    return {
        role: match
        for role, pattern in METRIC_PATTERNS.items()
        if (match := next((n for n in names if pattern.match(n)), None)) is not None
    }


# ---------------------------------------------------------------- aggregation


def series(snaps: Iterable[MetricSnapshot], name: str | None) -> list[float]:
    """Every value of ``name`` across the window, skipping snapshots that lacked it."""
    if name is None:
        return []
    return [s.values[name] for s in snaps if name in s.values]


def percentile(values: list[float], q: float) -> float | None:
    """Linearly interpolated quantile of a gauge series, or None when it is empty."""
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0)) if values else None


def counter_delta(snaps: list[MetricSnapshot], name: str | None) -> float | None:
    """Increase of a monotonic counter across the window, or None if it was never seen.

    A negative difference means the server restarted its counters mid-window; the honest
    answer is then "no rate measured", not a negative one.
    """
    vals = series(snaps, name)
    if len(vals) < 2:
        return None
    delta = vals[-1] - vals[0]
    return delta if delta >= 0 else None


def window_seconds(snaps: list[MetricSnapshot]) -> float:
    return snaps[-1].ts - snaps[0].ts if len(snaps) >= 2 else 0.0


def counter_rate(snaps: list[MetricSnapshot], name: str | None) -> float | None:
    """Per-second rate of a counter over the window."""
    delta = counter_delta(snaps, name)
    span = window_seconds(snaps)
    return delta / span if delta is not None and span > 0 else None


def bucket_delta(snaps: list[MetricSnapshot], family: str | None) -> dict[float, float]:
    """Cumulative buckets observed during the window: last snapshot minus first.

    Buckets present only in one of the two snapshots are read as zero in the other, which
    is what a histogram that gained its first observation mid-window actually did.
    """
    if family is None:
        return {}
    seen = [s.buckets[family] for s in snaps if family in s.buckets]
    if len(seen) < 2:
        return dict(seen[0]) if seen else {}
    first, last = seen[0], seen[-1]
    out = {le: last[le] - first.get(le, 0.0) for le in last}
    return {le: v for le, v in out.items() if v >= 0}


def histogram_quantile(buckets: dict[float, float], q: float) -> float | None:
    """Quantile of a cumulative histogram, interpolated inside the bucket that contains it.

    The same construction Prometheus' ``histogram_quantile`` uses: find the bucket whose
    cumulative count first reaches the target rank and interpolate linearly between its
    lower and upper bound. A rank landing in the ``+Inf`` bucket returns the largest
    finite bound -- the observation is somewhere above it and the histogram cannot say
    where -- and an empty histogram returns None rather than 0.0, because "nothing was
    measured" and "it was zero" must not read the same downstream.
    """
    items = sorted(buckets.items())
    if not items:
        return None
    total = items[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound, cum in items:
        if cum >= rank:
            if math.isinf(bound):
                return prev_bound
            if cum <= prev_count:
                return bound
            return prev_bound + (rank - prev_count) / (cum - prev_count) * (bound - prev_bound)
        prev_bound, prev_count = bound, cum
    return prev_bound


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def aggregate(snaps: list[MetricSnapshot], resolved: dict[str, str]) -> dict[str, float]:
    """Canonical engine metrics for one load window. Keys the server did not expose are absent.

    Absence is load-bearing: :mod:`infervolt.diagnose.rules` treats a missing key as "not
    reported" and refuses to fire a condition on it, so filling a gap with 0.0 here would
    manufacture evidence.
    """
    if not snaps:
        return {}
    get = resolved.get
    out: dict[str, float | None] = {
        "kv_usage_p95": percentile(series(snaps, get("kv_usage")), 0.95),
        "num_waiting": _mean(series(snaps, get("num_waiting"))),
        "num_running": _max(series(snaps, get("num_running"))),
        "preemptions_per_s": counter_rate(snaps, get("preemptions")),
        "queue_time_p90_s": histogram_quantile(bucket_delta(snaps, get("queue_time")), 0.90),
        "prefill_time_p50_s": histogram_quantile(bucket_delta(snaps, get("prefill_time")), 0.50),
        "prefill_share": _ratio(
            counter_delta(snaps, _sum_of(get("prefill_time"))),
            counter_delta(snaps, _sum_of(get("inference_time"))),
        ),
        "prefix_hit_rate": _ratio(
            counter_delta(snaps, get("prefix_hits")),
            counter_delta(snaps, get("prefix_queries")),
        ),
    }
    return {k: float(v) for k, v in out.items() if v is not None}


def _sum_of(family: str | None) -> str | None:
    return None if family is None else f"{family}_sum"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


# ---------------------------------------------------------------- sampling


def http_get(url: str, timeout_s: float = 5.0) -> tuple[int, str] | None:
    """GET ``url``. Returns (status, body), or None when the server could not be reached.

    An HTTP error status is a *reply* -- ``/health`` answers 503 while the engine is
    still loading weights -- so it comes back as a status, while a refused connection or
    a timeout comes back as None.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - our own URL
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return int(e.code), e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


class NvmlGpuProbe:
    """GPU utilization from NVML, falling back to ``nvidia-smi``, then giving up for good.

    ``pynvml`` is imported lazily and is not a dependency: infervolt has to import on a
    laptop with no CUDA at all. Whichever backend answers first is remembered, and once
    both have failed the probe stays disabled rather than paying a failed subprocess
    every 0.5 s for the rest of the run.
    """

    def __init__(self) -> None:
        self._backend: str | None = None
        self._nvml_handles: list[object] = []

    def __call__(self) -> tuple[float, float] | None:
        if self._backend is None:
            self._backend = "nvml" if self._init_nvml() else "smi"
        if self._backend == "nvml":
            sample = self._read_nvml()
            if sample is not None:
                return sample
            self._backend = "smi"
        if self._backend == "smi":
            sample = self._read_smi()
            if sample is not None:
                return sample
            self._backend = "none"
        return None

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:  # noqa: BLE001 - any NVML failure means "use the other backend"
            self._nvml_handles = []
            return False
        return bool(self._nvml_handles)

    def _read_nvml(self) -> tuple[float, float] | None:
        try:
            import pynvml

            rates = [pynvml.nvmlDeviceGetUtilizationRates(h) for h in self._nvml_handles]
        except Exception:  # noqa: BLE001 - a card that stopped answering is not fatal
            return None
        if not rates:
            return None
        return (
            sum(float(r.gpu) for r in rates) / len(rates),
            sum(float(r.memory) for r in rates) / len(rates),
        )

    @staticmethod
    def _read_smi() -> tuple[float, float] | None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        gpu: list[float] = []
        mem: list[float] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            a, b = _to_float(parts[0]), _to_float(parts[1])
            if a is not None and b is not None:
                gpu.append(a)
                mem.append(b)
        if not gpu:
            return None
        return sum(gpu) / len(gpu), sum(mem) / len(mem)


@dataclass
class Window:
    """What one load point's sampling produced."""

    snapshots: list[MetricSnapshot] = field(default_factory=list)
    gpu: list[tuple[float, float]] = field(default_factory=list)


class MetricsSampler:
    """Polls ``/metrics`` and the GPU on a background thread for the duration of a load point.

    Started when the load generator is handed out and stopped when the trial scrapes, so
    the window is exactly the interval the requests were in flight. Both endpoints are
    injectable: the tests drive a real in-process HTTP server for the metrics and a stub
    for the GPU, because no CI machine has a card.
    """

    def __init__(
        self,
        metrics_url: str,
        interval_s: float = 0.5,
        fetch: Callable[[], str | None] | None = None,
        gpu_probe: GpuProbe | None = None,
    ) -> None:
        self.metrics_url = metrics_url
        self.interval_s = interval_s
        self._fetch = fetch or self._default_fetch
        self._gpu_probe: GpuProbe = gpu_probe if gpu_probe is not None else NvmlGpuProbe()
        self._window = Window()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _default_fetch(self) -> str | None:
        got = http_get(self.metrics_url)
        return got[1] if got is not None and got[0] == 200 else None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        """Begin a fresh window. Restarting an already-running sampler discards the old one."""
        self.stop()
        self._window = Window()
        self._stop = threading.Event()
        thread = threading.Thread(target=self._loop, name="vllm-metrics", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> Window:
        """Halt sampling and return the window, taking one last sample to close it.

        The closing sample matters for short load points: every counter and histogram
        here is read as a difference, so a window with a single snapshot in it can only
        report the gauges.
        """
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(timeout=5.0)
            self._thread = None
            self._sample()
        with self._lock:
            return self._window

    def _loop(self) -> None:
        while True:
            self._sample()
            if self._stop.wait(self.interval_s):
                return

    def _sample(self) -> None:
        text = self._fetch()
        snap = parse_metrics(text) if text is not None else None
        gpu = self._gpu_probe()
        with self._lock:
            if snap is not None:
                self._window.snapshots.append(snap)
            if gpu is not None:
                self._window.gpu.append(gpu)


def gpu_stats(window: Window) -> dict[str, float]:
    """Mean GPU and memory utilization over the window, as fractions in [0, 1]."""
    if not window.gpu:
        return {}
    n = len(window.gpu)
    return {
        "sm_active": sum(g for g, _ in window.gpu) / n / 100.0,
        "dram_active": sum(m for _, m in window.gpu) / n / 100.0,
    }
