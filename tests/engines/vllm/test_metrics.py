"""Prometheus parsing, name resolution across vLLM releases, and window aggregation."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from infervolt.engines.vllm import metrics as m
from tests.fake_http import FakeServer

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def snap_a() -> m.MetricSnapshot:
    return m.parse_metrics(fixture("metrics_v0.11.txt"), ts=100.0)


@pytest.fixture(scope="module")
def snap_b() -> m.MetricSnapshot:
    return m.parse_metrics(fixture("metrics_v0.11_later.txt"), ts=102.0)


@pytest.fixture(scope="module")
def snap_legacy() -> m.MetricSnapshot:
    return m.parse_metrics(fixture("metrics_v0.9_gpu_cache.txt"), ts=100.0)


def test_parses_gauges_counters_and_buckets(snap_a: m.MetricSnapshot) -> None:
    assert snap_a.values["vllm:num_requests_running"] == 48.0
    assert snap_a.values["vllm:kv_cache_usage_perc"] == 0.62
    assert snap_a.values["vllm:request_queue_time_seconds_count"] == 1080.0
    queue = snap_a.buckets["vllm:request_queue_time_seconds"]
    assert queue[0.3] == 600.0
    assert queue[float("inf")] == 1080.0


def test_families_strip_the_histogram_suffixes(snap_a: m.MetricSnapshot) -> None:
    families = snap_a.families()
    assert "vllm:request_queue_time_seconds" in families
    assert "vllm:request_queue_time_seconds_sum" not in families
    assert "vllm:num_preemptions_total" in families


def test_label_sets_are_summed() -> None:
    text = (
        'vllm:num_requests_running{model_name="a",engine="0"} 3.0\n'
        'vllm:num_requests_running{model_name="a",engine="1"} 4.0\n'
    )
    assert m.parse_metrics(text).values["vllm:num_requests_running"] == 7.0


def test_unparseable_lines_are_skipped_not_raised() -> None:
    text = (
        "# HELP vllm:num_requests_running docs\n"
        "this is not prometheus\n"
        'vllm:num_requests_running{engine="0"} NaN\n'
        'vllm:num_requests_waiting{engine="0"} 5.0\n'
    )
    snap = m.parse_metrics(text)
    assert snap.values == {"vllm:num_requests_waiting": 5.0}


def test_resolves_v1_names(snap_a: m.MetricSnapshot) -> None:
    resolved = m.resolve_names(snap_a.families())
    assert resolved["kv_usage"] == "vllm:kv_cache_usage_perc"
    assert resolved["prefix_hits"] == "vllm:prefix_cache_hits_total"
    assert resolved["prefix_queries"] == "vllm:prefix_cache_queries_total"
    assert resolved["preemptions"] == "vllm:num_preemptions_total"
    assert resolved["queue_time"] == "vllm:request_queue_time_seconds"


def test_resolves_the_older_gpu_cache_spelling(snap_legacy: m.MetricSnapshot) -> None:
    resolved = m.resolve_names(snap_legacy.families())
    assert resolved["kv_usage"] == "vllm:gpu_cache_usage_perc"
    assert resolved["prefix_hits"] == "vllm:gpu_prefix_cache_hits"
    assert resolved["prefix_queries"] == "vllm:gpu_prefix_cache_queries"


def test_roles_the_server_does_not_expose_stay_unresolved() -> None:
    assert m.resolve_names(["vllm:num_requests_running"]) == {
        "num_running": "vllm:num_requests_running"
    }


def test_histogram_quantile_interpolates_inside_the_bucket() -> None:
    # 200 observations; p90 lands at rank 180, between le=0.5 (170) and le=0.8 (190).
    buckets = {0.3: 120.0, 0.5: 170.0, 0.8: 190.0, 1.0: 195.0, float("inf"): 200.0}
    assert m.histogram_quantile(buckets, 0.90) == pytest.approx(0.65)


def test_histogram_quantile_on_an_empty_histogram_is_none() -> None:
    assert m.histogram_quantile({}, 0.5) is None
    assert m.histogram_quantile({0.3: 0.0, float("inf"): 0.0}, 0.5) is None


def test_histogram_quantile_in_the_inf_bucket_returns_the_largest_finite_bound() -> None:
    assert m.histogram_quantile({1.0: 5.0, float("inf"): 100.0}, 0.99) == 1.0


def test_bucket_delta_is_the_windows_own_observations(
    snap_a: m.MetricSnapshot, snap_b: m.MetricSnapshot
) -> None:
    delta = m.bucket_delta([snap_a, snap_b], "vllm:request_queue_time_seconds")
    assert delta[0.3] == 120.0
    assert delta[0.5] == 170.0
    assert delta[float("inf")] == 200.0


def test_aggregate_reduces_a_window_to_the_canonical_keys(
    snap_a: m.MetricSnapshot, snap_b: m.MetricSnapshot
) -> None:
    snaps = [snap_a, snap_b]
    out = m.aggregate(snaps, m.resolve_names(snap_b.families()))
    # p95 of the two-point gauge series [0.62, 0.88], linearly interpolated.
    assert out["kv_usage_p95"] == pytest.approx(0.867)
    assert out["num_waiting"] == pytest.approx(10.0)
    assert out["num_running"] == pytest.approx(64.0)
    # 3 preemptions over the 2-second window.
    assert out["preemptions_per_s"] == pytest.approx(1.5)
    assert out["queue_time_p90_s"] == pytest.approx(0.65)
    assert out["prefill_time_p50_s"] == pytest.approx(0.4)
    # 90 s of prefill out of 300 s of inference during the window.
    assert out["prefill_share"] == pytest.approx(0.3)
    # 6200 cached tokens out of 10000 queried during the window.
    assert out["prefix_hit_rate"] == pytest.approx(0.62)


def test_a_one_snapshot_window_reports_gauges_but_no_rates(snap_a: m.MetricSnapshot) -> None:
    out = m.aggregate([snap_a], m.resolve_names(snap_a.families()))
    assert out["kv_usage_p95"] == pytest.approx(0.62)
    assert "preemptions_per_s" not in out
    assert "prefill_share" not in out


def test_metrics_the_server_never_reported_are_absent_not_zero() -> None:
    snap = m.parse_metrics('vllm:num_requests_running{engine="0"} 4.0')
    out = m.aggregate([snap, snap], m.resolve_names(snap.families()))
    assert out == {"num_running": 4.0}


def test_a_counter_that_went_backwards_reports_no_rate() -> None:
    """A restarted engine resets its counters; a negative rate would be a lie."""
    a = m.parse_metrics('vllm:num_preemptions_total{engine="0"} 40.0', ts=0.0)
    b = m.parse_metrics('vllm:num_preemptions_total{engine="0"} 2.0', ts=2.0)
    assert m.counter_rate([a, b], "vllm:num_preemptions_total") is None


def test_aggregate_of_an_empty_window_is_empty() -> None:
    assert m.aggregate([], {"kv_usage": "vllm:kv_cache_usage_perc"}) == {}


def test_sampler_collects_a_window_from_a_live_server() -> None:
    texts = [fixture("metrics_v0.11.txt"), fixture("metrics_v0.11_later.txt")]
    with FakeServer(metrics=texts) as server:
        sampler = m.MetricsSampler(
            f"{server.url}/metrics", interval_s=0.01, gpu_probe=lambda: (50.0, 30.0)
        )
        sampler.start()
        deadline = time.monotonic() + 5.0
        while server.scrapes < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        window = sampler.stop()
    assert not sampler.running
    assert len(window.snapshots) >= 2
    out = m.aggregate(window.snapshots, m.resolve_names(window.snapshots[-1].families()))
    assert out["num_running"] == 64.0
    assert out["prefix_hit_rate"] == pytest.approx(0.62)
    assert m.gpu_stats(window) == {"sm_active": 0.5, "dram_active": 0.3}


def test_sampler_survives_a_server_that_is_not_answering() -> None:
    sampler = m.MetricsSampler(
        "http://127.0.0.1:1/metrics", interval_s=0.01, gpu_probe=lambda: None
    )
    sampler.start()
    window = sampler.stop()
    assert window.snapshots == [] and window.gpu == []


def test_gpu_stats_averages_and_scales_to_fractions() -> None:
    window = m.Window(gpu=[(50.0, 30.0), (70.0, 50.0)])
    assert m.gpu_stats(window) == {"sm_active": 0.6, "dram_active": 0.4}


def test_gpu_stats_without_a_probe_is_empty() -> None:
    assert m.gpu_stats(m.Window()) == {}


def test_probe_falls_back_to_nvidia_smi_and_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 0, "37, 21\n43, 25\n", "")
        raise OSError("nvidia-smi: not found")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    probe = m.NvmlGpuProbe()
    monkeypatch.setattr(probe, "_init_nvml", lambda: False)

    assert probe() == pytest.approx((40.0, 23.0))
    assert calls[0][0] == "nvidia-smi"
    assert probe() is None
    # Once both backends have failed the probe stops paying for a subprocess per sample.
    assert probe() is None
    assert len(calls) == 2
