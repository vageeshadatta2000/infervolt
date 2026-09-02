"""Turn per-request records into the Metrics the rules and the objective consume."""

from __future__ import annotations

import numpy as np

from infervolt.core.types import SLO, HardwareProfile, LoadResult, Metrics, RequestRecord


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def request_meets_slo(r: RequestRecord, slo: SLO) -> bool:
    if not r.ok:
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
    ok = [r for r in lr.requests if r.ok]
    total = len(lr.requests)
    good = sum(1 for r in ok if request_meets_slo(r, slo))
    dur = max(lr.duration_s, 1e-9)
    ttft = [r.ttft_s * 1000 for r in ok]
    itl = [x * 1000 for r in ok for x in r.itl_s]
    e2e = [r.e2e_s * 1000 for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)
    output_tps = out_tokens / dur
    usd_per_hour = hw.usd_per_hour * hw.count
    usd_per_m = (usd_per_hour / 3600 / output_tps * 1e6) if output_tps > 0 else 0.0
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
