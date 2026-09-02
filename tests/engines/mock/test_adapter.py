import pytest

from infervolt.core.types import EngineConfig
from infervolt.engines.base import LaunchError
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.engines.registry import get_adapter


def test_scenarios_cover_four_bottlenecks() -> None:
    assert {s.expected for s in SCENARIOS.values()} == {
        "kv_capacity",
        "decode_bandwidth",
        "prefill_compute",
        "scheduler_cpu",
    }


def test_launch_load_scrape_stop() -> None:
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    assert adapter.ready(handle, 1.0)
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency=16, num_requests=32, seed=1)
    assert len(lr.requests) == 32 and lr.duration_s > 0
    assert all(len(r.itl_s) == ctx.workload.osl.p50 for r in lr.requests)
    snap = adapter.scrape(handle)
    assert snap["preemptions_per_s"] > 0 and snap["kv_dtype_bytes"] == 2
    gpu = adapter.gpu_stats(handle)
    assert 0 <= gpu["sm_active"] <= 1
    assert adapter.stop(handle).code == 0


def test_launch_oom_raises_launch_error() -> None:
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(
        engine="mock", knobs={**adapter.knob_space(ctx).defaults(), "max_model_len": 32768}
    )
    with pytest.raises(LaunchError) as ei:
        adapter.launch(cfg, ctx)
    assert adapter.classify_crash(ei.value.exit) == "oom"


def test_validate_rejects_fp8_quant_on_ampere() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")  # a100, cc 8.0
    cfg = EngineConfig(
        engine="mock", knobs={**adapter.knob_space(ctx).defaults(), "quantization": "fp8"}
    )
    assert adapter.validate(cfg, ctx)


def test_validate_rejects_values_outside_the_knob_space() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")
    defaults = adapter.knob_space(ctx).defaults()
    assert adapter.validate(EngineConfig(engine="mock", knobs=defaults), ctx) == []
    cfg = EngineConfig(
        engine="mock",
        knobs={**defaults, "kv_cache_dtype": "int4", "enable_prefix_caching": "maybe"},
    )
    errs = adapter.validate(cfg, ctx)
    assert any("kv_cache_dtype" in e and "'fp8'" in e for e in errs)
    assert any("enable_prefix_caching" in e and "'true'" in e for e in errs)


def test_deterministic_with_seed() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    h = adapter.launch(cfg, ctx)
    a = adapter.loadgen(h, ctx).run(ctx.workload, 4, 8, seed=3)
    b = adapter.loadgen(h, ctx).run(ctx.workload, 4, 8, seed=3)
    assert a == b


def test_registry_returns_mock() -> None:
    assert get_adapter("mock").name == "mock"
