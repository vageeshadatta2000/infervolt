import pytest

from infervolt.core.types import SLO, LoadResult, RequestRecord
from infervolt.hardware.profiles import get_profile
from infervolt.loadgen.analysis import compute_metrics


def _lr() -> LoadResult:
    fast = RequestRecord(ttft_s=0.1, itl_s=[0.01] * 10, output_tokens=10)
    slow = RequestRecord(ttft_s=1.0, itl_s=[0.05] * 10, output_tokens=10)
    bad = RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
    return LoadResult(concurrency=2, duration_s=10.0, requests=[fast, fast, slow, bad])


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
