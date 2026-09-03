import math

import numpy as np
import pytest
from pydantic import ValidationError

from infervolt.core.types import SLO, HardwareProfile, LoadResult, RequestRecord
from infervolt.hardware.profiles import get_profile
from infervolt.loadgen.analysis import compute_metrics, request_meets_slo


def _fast() -> RequestRecord:
    return RequestRecord(ttft_s=0.1, itl_s=[0.01] * 10, output_tokens=10)


def _lr() -> LoadResult:
    fast = _fast()
    slow = RequestRecord(ttft_s=1.0, itl_s=[0.05] * 10, output_tokens=10)
    bad = RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
    return LoadResult(concurrency=2, duration_s=10.0, requests=[fast, fast, slow, bad])


def _two_gpu_profile() -> HardwareProfile:
    return HardwareProfile(
        name="pair",
        gpu="test",
        count=2,
        mem_gb=80,
        hbm_bw_gbs=2039,
        peak_tflops=312,
        compute_capability=8.0,
        usd_per_hour=1.5,
    )


def test_goodput_counts_only_requests_meeting_every_slo() -> None:
    m = compute_metrics(_lr(), SLO(ttft_ms=500, itl_ms=30), get_profile("a100-80"))
    assert m.goodput_rps == pytest.approx(0.2)  # 2 fast requests / 10 s
    assert m.goodput_frac == pytest.approx(0.5)  # 2 of 4 submitted
    assert m.req_per_s == pytest.approx(0.3)
    assert m.error_rate == pytest.approx(0.25)
    assert m.output_tps == pytest.approx(3.0)


def test_no_slo_means_every_ok_request_is_good() -> None:
    m = compute_metrics(_lr(), SLO(), get_profile("a100-80"))
    assert m.goodput_rps == pytest.approx(m.req_per_s)


def test_cost_per_million_tokens_uses_hourly_price() -> None:
    hw = get_profile("a100-80")  # 1.5 $/h
    m = compute_metrics(_lr(), SLO(), hw)
    assert m.usd_per_m_tokens == pytest.approx(1.5 / 3600 / 3.0 * 1e6)


def test_request_with_no_output_tokens_is_never_good() -> None:
    empty = RequestRecord(ttft_s=0.001, itl_s=[], output_tokens=0, ok=True)
    assert request_meets_slo(empty, SLO()) is False
    assert request_meets_slo(empty, SLO(ttft_ms=500, itl_ms=30, e2e_ms=2000)) is False

    lr = LoadResult(concurrency=1, duration_s=10.0, requests=[empty, _fast()])
    m = compute_metrics(lr, SLO(), get_profile("a100-80"))
    assert m.goodput_rps == pytest.approx(0.1)  # only the productive request counts
    assert m.req_per_s == pytest.approx(0.2)  # but it did not error


def test_e2e_p50_converts_seconds_to_milliseconds() -> None:
    m = compute_metrics(_lr(), SLO(), get_profile("a100-80"))
    # fast: 0.1 s ttft + 10 x 0.01 s itl = 0.2 s; median of [200, 200, 1500] ms.
    assert m.e2e_p50_ms == 200.0


def test_ttft_p90_matches_numpy_percentile() -> None:
    m = compute_metrics(_lr(), SLO(), get_profile("a100-80"))
    expected = float(np.percentile([100.0, 100.0, 1000.0], 90))
    assert m.ttft_p90_ms == pytest.approx(expected)
    assert expected == pytest.approx(820.0)  # 100 + 0.8 * 900


def test_per_gpu_throughput_and_cost_divide_by_gpu_count() -> None:
    hw = _two_gpu_profile()
    m = compute_metrics(_lr(), SLO(), hw)
    assert m.output_tps == pytest.approx(3.0)
    assert m.tokens_per_s_per_gpu == pytest.approx(1.5)
    # usd_per_hour is per GPU, so a 2-GPU node costs 3.0 $/h in total.
    assert m.usd_per_m_tokens == pytest.approx(2 * 1.5 / 3600 / 3.0 * 1e6)


def test_empty_load_result_is_all_zeros_with_infinite_cost() -> None:
    empty = LoadResult(concurrency=1, duration_s=1.0, requests=[])
    m = compute_metrics(empty, SLO(), _two_gpu_profile())
    assert m.output_tps == 0.0
    assert m.req_per_s == 0.0
    assert m.goodput_rps == 0.0
    assert m.goodput_frac == 0.0
    assert m.error_rate == 0.0
    assert m.tokens_per_s_per_gpu == 0.0
    assert m.ttft_p50_ms == 0.0
    assert m.e2e_p90_ms == 0.0
    assert m.usd_per_m_tokens == math.inf


def test_all_errors_load_result() -> None:
    bad = RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
    lr = LoadResult(concurrency=2, duration_s=5.0, requests=[bad, bad, bad])
    m = compute_metrics(lr, SLO(), get_profile("a100-80"))
    assert m.error_rate == 1.0
    assert m.goodput_rps == 0.0
    assert m.goodput_frac == 0.0
    assert m.output_tps == 0.0
    assert m.usd_per_m_tokens == math.inf


def test_load_result_requires_positive_duration() -> None:
    with pytest.raises(ValidationError):
        LoadResult(concurrency=1, duration_s=-1.0, requests=[])
    with pytest.raises(ValidationError):
        LoadResult(concurrency=1, duration_s=0.0, requests=[])
