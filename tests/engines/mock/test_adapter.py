import statistics

import pytest

from infervolt.core.types import EngineConfig, KnobValue
from infervolt.engines.base import LaunchError
from infervolt.engines.mock.adapter import MIN_RUN_FACTOR, MockAdapter, _state
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.engines.registry import get_adapter
from infervolt.loadgen.analysis import compute_metrics


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


def test_duration_reflects_closed_loop_throughput() -> None:
    """A closed loop retires ``running`` requests every ``lifetime_s``, not every lifetime+wait.

    Queue wait is already the time the *other* ``waiting`` requests spend outside the
    server, so charging it again to the completion rate double-counts it. The invariant
    that pins this down is Little's law: at steady state each of the ``concurrency``
    in-flight requests spends ``queue_wait_s + lifetime_s`` in the system, so the
    completion rate times that residence time must come back to ``concurrency``.
    """
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    p = _state(handle).pm.point(64)
    assert p.running > 0 and p.waiting > 0  # the case where the two formulas differ
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency=64, num_requests=32, seed=1)
    m = compute_metrics(lr, ctx.slo, ctx.hw)
    assert m.req_per_s * (p.lifetime_s + p.queue_wait_s) == pytest.approx(64, rel=0.05)


def test_itl_samples_have_the_modelled_mean() -> None:
    """Sampled ITLs must average to ``itl_mean_s``: the spikes come out of the baseline."""
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    p = _state(handle).pm.point(64)
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency=64, num_requests=32, seed=5)
    samples = [x for r in lr.requests for x in r.itl_s]
    assert statistics.fmean(samples) == pytest.approx(p.itl_mean_s, rel=0.03)


def test_quantization_choices_are_gated_on_compute_capability() -> None:
    """A card that cannot do fp8 GEMMs is not offered fp8 quantization at all.

    ``validate`` rejects it either way, but a choice that can only ever be rejected costs
    the search a trial to learn what the hardware profile already said -- and widens the
    space the novelty filter measures distances across.
    """
    adapter = MockAdapter()
    ampere = make_context("decode", run_dir="/tmp/x")  # a100-80, compute capability 8.0
    assert ampere.hw.compute_capability < 8.9
    assert adapter.knob_space(ampere).get("quantization").choices == ["none"]
    hopper = make_context("prefill", run_dir="/tmp/x")  # h100-80, compute capability 9.0
    assert adapter.knob_space(hopper).get("quantization").choices == ["none", "fp8"]


def test_validate_rejects_fp8_quant_on_ampere() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")  # a100, cc 8.0
    cfg = EngineConfig(
        engine="mock", knobs={**adapter.knob_space(ctx).defaults(), "quantization": "fp8"}
    )
    errs = adapter.validate(cfg, ctx)
    assert any("compute capability" in e for e in errs)


def test_validate_reports_bad_numeric_knobs_without_raising() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")
    d = adapter.knob_space(ctx).defaults()

    def errs_for(**overrides: KnobValue) -> list[str]:
        return adapter.validate(EngineConfig(engine="mock", knobs={**d, **overrides}), ctx)

    # A non-numeric max_model_len is reported, not raised, by the categorical check alone.
    bad_len = errs_for(max_model_len="big")
    assert any("max_model_len" in e for e in bad_len)
    assert not any("shorter than workload" in e for e in bad_len)
    assert any("gpu_memory_utilization" in e for e in errs_for(gpu_memory_utilization=1.5))
    assert any("max_num_seqs" in e for e in errs_for(max_num_seqs=-5))
    assert any("max_num_batched_tokens" in e for e in errs_for(max_num_batched_tokens="lots"))


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


def test_max_model_len_choices_start_at_the_workload_aware_default() -> None:
    """The search must not be offered a context shorter than the workload it is serving.

    ``validate`` would reject such a config anyway, so leaving 4096 in ``choices`` only
    buys wasted trials -- and a novelty filter that thinks the space is bigger than the
    part of it that can ever launch.
    """
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")  # chat-4k-512: needs 6000 + 512 tokens
    knob = adapter.knob_space(ctx).get("max_model_len")
    assert knob.default == 8192
    assert knob.choices == [8192, 16384, 32768]
    assert all(int(c) >= ctx.workload.isl.p99 + ctx.workload.osl.p50 for c in knob.choices)


def test_duration_varies_between_seeds_but_only_by_run_noise() -> None:
    """Two runs of the same config differ, the way two runs on a real card differ.

    A duration computed purely from the model is identical every time, so a verify with
    a zero-variance delta would accept any config that is even a hair better -- the CI
    would collapse onto the mean. A percent of run-to-run noise is what makes the
    statistics do work.
    """
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    a = adapter.loadgen(handle, ctx).run(ctx.workload, 64, 16, seed=1)
    b = adapter.loadgen(handle, ctx).run(ctx.workload, 64, 16, seed=2)
    assert a.duration_s != b.duration_s
    assert a.duration_s == pytest.approx(b.duration_s, rel=0.05)


def test_duration_stays_positive_under_every_run_noise_draw() -> None:
    """Every rate in ``compute_metrics`` divides by ``duration_s``; the floor guarantees it."""
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    gen = adapter.loadgen(handle, ctx)
    durations = [gen.run(ctx.workload, 64, 8, seed=s).duration_s for s in range(200)]
    assert all(d > 0 for d in durations)
    assert min(durations) >= MIN_RUN_FACTOR * 0.9 * statistics.fmean(durations)
