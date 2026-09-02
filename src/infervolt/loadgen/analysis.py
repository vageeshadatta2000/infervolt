"""Turn per-request records into the Metrics the rules and the objective consume.

Percentile units differ between the two layers this module bridges: ``_pct`` and numpy
take a percentile in 0-100, while :attr:`SLO.percentile` is a fraction in 0-1. Every
crossing multiplies by 100 -- keep that conversion at the call site, not in ``_pct``.
"""

from __future__ import annotations

import math

import numpy as np

from infervolt.core.types import SLO, HardwareProfile, LoadResult, Metrics, RequestRecord


def _pct(values: list[float], p: float) -> float:
    """Percentile of ``values``. ``p`` is in 0-100 (numpy's convention), not 0-1."""
    return float(np.percentile(values, p)) if values else 0.0


def request_meets_slo(r: RequestRecord, slo: SLO) -> bool:
    """True when the request did productive work *and* met every configured target.

    Being productive -- succeeding and emitting at least one output token -- is a
    precondition, not a target: goodput measures served work, so a request that failed
    or returned nothing must never count, however fast it was.
    """
    if not r.ok or r.output_tokens <= 0:
        return False
    if slo.ttft_ms is not None and r.ttft_s * 1000 > slo.ttft_ms:
        return False
    if (
        slo.itl_ms is not None
        and r.itl_s
        and _pct(r.itl_s, slo.percentile * 100) * 1000 > slo.itl_ms
    ):
        return False
    return not (slo.e2e_ms is not None and r.e2e_s * 1000 > slo.e2e_ms)


def compute_metrics(lr: LoadResult, slo: SLO, hw: HardwareProfile) -> Metrics:
    """Summarise one load point.

    Latency percentiles are taken over successful requests only; rates divide by
    ``lr.duration_s`` (validated positive). ``usd_per_m_tokens`` prices the node --
    ``hw.usd_per_hour`` is per GPU, so it is multiplied by ``hw.count`` -- against
    *output* tokens alone, and is infinite when no output tokens were produced.
    """
    ok = [r for r in lr.requests if r.ok]
    total = len(lr.requests)
    good = sum(1 for r in ok if request_meets_slo(r, slo))
    dur = lr.duration_s
    ttft = [r.ttft_s * 1000 for r in ok]
    itl = [x * 1000 for r in ok for x in r.itl_s]
    e2e = [r.e2e_s * 1000 for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)
    output_tps = out_tokens / dur
    usd_per_hour = hw.usd_per_hour * hw.count
    usd_per_m = (usd_per_hour / 3600 / output_tps * 1e6) if output_tps > 0 else math.inf
    return Metrics(
        ttft_p50_ms=_pct(ttft, 50),
        ttft_p90_ms=_pct(ttft, 90),
        ttft_p99_ms=_pct(ttft, 99),
        itl_p50_ms=_pct(itl, 50),
        itl_p90_ms=_pct(itl, 90),
        itl_p99_ms=_pct(itl, 99),
        e2e_p50_ms=_pct(e2e, 50),
        e2e_p90_ms=_pct(e2e, 90),
        output_tps=output_tps,
        req_per_s=len(ok) / dur,
        goodput_rps=good / dur,
        goodput_frac=(good / total) if total else 0.0,
        error_rate=((total - len(ok)) / total) if total else 0.0,
        tokens_per_s_per_gpu=output_tps / max(hw.count, 1),
        usd_per_m_tokens=usd_per_m,
    )
