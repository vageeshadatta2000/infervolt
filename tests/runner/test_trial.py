from infervolt.core.types import (
    Candidate,
    ClientHealth,
    EngineConfig,
    KnobValue,
    LoadResult,
    Metrics,
    Observation,
    RequestRecord,
    RunContext,
    Trial,
)
from infervolt.engines.base import ExitInfo, ServerHandle
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.loadgen.base import LoadGenerator
from infervolt.runner.trial import (
    SWEEP_ERROR_MAX,
    run_candidate,
    run_load_point,
    run_sweep,
    summarize,
)


def _trial(knobs: dict[str, KnobValue], idx: int = 0) -> Trial:
    cand = Candidate(
        id=f"c{idx}", config=EngineConfig(engine="mock", knobs=knobs), origin="baseline"
    )
    return Trial(id=f"t{idx}", run_id="r", index=idx, candidate=cand)


def test_sweep_stops_after_goodput_collapses() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    handle.config = cfg
    obs, cost_s = run_sweep(adapter, handle, ctx, [1, 4, 16, 64, 128], num_requests=16)
    assert obs[0].load_point == 1 and len(obs) < 5
    assert all(o.valid for o in obs) and cost_s > 0
    assert obs[-1].engine["num_waiting"] > 0


def test_sweep_observations_carry_the_launched_config() -> None:
    adapter, ctx = MockAdapter(), make_context("decode", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    handle.config = cfg
    obs, _ = run_sweep(adapter, handle, ctx, [1, 4], num_requests=8)
    assert all(o.config == cfg for o in obs)


def test_run_candidate_ok_sets_objective_and_best_load_point() -> None:
    adapter, ctx = MockAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(
        adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1, 4, 16, 64], num_requests=16
    )
    assert t.status == "ok" and t.result is not None
    assert t.result.objective == max(o.metrics.goodput_rps for o in t.result.observations)
    assert t.result.best_load_point in {o.load_point for o in t.result.observations}
    assert t.cost_usd > 0 and t.started is not None and t.ended is not None


def test_run_candidate_oom_is_infeasible() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), "max_model_len": 32768}
    t = run_candidate(adapter, _trial(knobs), ctx, [1, 4], num_requests=8)
    assert t.status == "infeasible_oom" and t.crash_kind == "oom" and "KV cache" in t.log_tail


def test_run_candidate_static_rejection_never_launches() -> None:
    adapter, ctx = MockAdapter(), make_context("decode", run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), "quantization": "fp8"}
    t = run_candidate(adapter, _trial(knobs), ctx, [1], num_requests=8)
    assert t.status == "rejected" and "compute capability" in t.log_tail


class _ExplodingAdapter(MockAdapter):
    """An adapter that breaks its contract: launch raises something other than LaunchError."""

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        raise RuntimeError("boom")


def test_run_candidate_reports_a_contract_breaking_launch_as_a_crash() -> None:
    adapter, ctx = _ExplodingAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1], num_requests=8)
    assert t.status == "crash" and t.crash_kind == "startup"
    assert t.log_tail == "RuntimeError: boom"
    assert t.ended is not None and t.result is None


class _StopFailsAdapter(MockAdapter):
    """An adapter whose teardown itself raises. The trial still has to end."""

    def stop(self, handle: ServerHandle) -> ExitInfo:
        raise OSError("kill: no such process")


class _NeverReadyAdapter(MockAdapter):
    def __init__(self) -> None:
        self.stopped = False

    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        return False

    def stop(self, handle: ServerHandle) -> ExitInfo:
        self.stopped = True
        return ExitInfo(code=0)


class _LoadgenExplodesAdapter(MockAdapter):
    """An adapter that dies mid-sweep with an exception rather than an exit code."""

    def __init__(self) -> None:
        self.stopped = False

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        raise RuntimeError("the load generator caught fire")

    def stop(self, handle: ServerHandle) -> ExitInfo:
        self.stopped = True
        return ExitInfo(code=0)


class _DiesDuringSweepAdapter(MockAdapter):
    """The sweep completes, but the server is found dead when it is stopped."""

    def stop(self, handle: ServerHandle) -> ExitInfo:
        return ExitInfo(code=1, log_tail="Segmentation fault")


def test_run_candidate_reports_a_loadgen_explosion_as_a_crash() -> None:
    adapter, ctx = _LoadgenExplodesAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1], num_requests=8)
    assert t.status == "crash" and t.crash_kind == "runtime"
    assert t.log_tail == "RuntimeError: the load generator caught fire"
    assert t.ended is not None and t.result is None
    assert adapter.stopped, "the server must be stopped even when the sweep blows up"


def test_run_candidate_survives_a_stop_that_raises() -> None:
    adapter, ctx = _StopFailsAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(
        adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1, 4], num_requests=8
    )
    assert t.status in {"ok", "crash"} and t.status != "running"
    assert t.ended is not None


def test_run_candidate_that_never_becomes_ready_times_out_and_stops_the_server() -> None:
    adapter, ctx = _NeverReadyAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1], num_requests=8)
    assert t.status == "timeout" and t.crash_kind == "timeout"
    assert t.ended is not None and adapter.stopped


def test_run_candidate_bills_a_sweep_whose_server_died_at_the_end() -> None:
    """The GPU-hours were spent whatever the exit code says, so the trial still costs."""
    adapter, ctx = _DiesDuringSweepAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(
        adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1, 4], num_requests=8
    )
    assert t.status == "crash" and t.crash_kind == "runtime"
    assert "Segmentation fault" in t.log_tail
    assert t.cost_usd > 0


# ---------------------------------------------------------------- load point validity


def _load_result(concurrency: int, n_ok: int, n_bad: int, health: ClientHealth) -> LoadResult:
    reqs = [RequestRecord(ttft_s=0.05, itl_s=[0.01] * 8, output_tokens=8) for _ in range(n_ok)]
    reqs += [RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False) for _ in range(n_bad)]
    return LoadResult(concurrency=concurrency, duration_s=1.0, requests=reqs, health=health)


class _FixedLoadGen:
    def __init__(self, lr: LoadResult) -> None:
        self.lr = lr

    def run(self, workload: object, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        return self.lr


class _FixedLoadAdapter(MockAdapter):
    """MockAdapter with the simulator's load generator replaced by a canned result."""

    def __init__(self, lr: LoadResult) -> None:
        self.lr = lr

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        return _FixedLoadGen(self.lr)


def _handle(adapter: MockAdapter, ctx: RunContext) -> ServerHandle:
    cfg = EngineConfig(engine="mock", knobs=MockAdapter().knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    handle.config = cfg
    return handle


def test_load_point_is_invalid_when_no_request_succeeded() -> None:
    ctx = make_context("decode", run_dir="/tmp/x")
    adapter = _FixedLoadAdapter(_load_result(4, n_ok=0, n_bad=8, health=ClientHealth()))
    obs, _ = run_load_point(adapter, _handle(adapter, ctx), ctx, 4, num_requests=8)
    assert not obs.valid and "no successful requests" in obs.invalid_reason


def test_load_point_is_invalid_when_the_client_is_the_bottleneck() -> None:
    ctx = make_context("decode", run_dir="/tmp/x")
    adapter = _FixedLoadAdapter(
        _load_result(4, n_ok=8, n_bad=0, health=ClientHealth(worker_cpu=0.95))
    )
    obs, _ = run_load_point(adapter, _handle(adapter, ctx), ctx, 4, num_requests=8)
    assert not obs.valid and "client artifact" in obs.invalid_reason


def test_sweep_stops_once_the_server_sheds_requests() -> None:
    ctx = make_context("decode", run_dir="/tmp/x")
    n_bad = 4  # 4 of 20 failed: a 0.2 error rate, well over SWEEP_ERROR_MAX
    adapter = _FixedLoadAdapter(_load_result(1, n_ok=16, n_bad=n_bad, health=ClientHealth()))
    obs, _ = run_sweep(adapter, _handle(adapter, ctx), ctx, [1, 4, 16], num_requests=20)
    assert len(obs) == 1 and obs[0].valid
    assert obs[0].metrics.error_rate > SWEEP_ERROR_MAX


# ---------------------------------------------------------------- summarize


def _metrics(goodput_rps: float, goodput_frac: float = 1.0) -> Metrics:
    return Metrics(
        ttft_p50_ms=0.0,
        ttft_p90_ms=0.0,
        ttft_p99_ms=0.0,
        itl_p50_ms=0.0,
        itl_p90_ms=0.0,
        itl_p99_ms=0.0,
        e2e_p50_ms=0.0,
        e2e_p90_ms=0.0,
        output_tps=0.0,
        req_per_s=0.0,
        goodput_rps=goodput_rps,
        goodput_frac=goodput_frac,
        error_rate=0.0,
        tokens_per_s_per_gpu=0.0,
        usd_per_m_tokens=0.0,
    )


def _observation(load_point: int, goodput_rps: float, valid: bool = True) -> Observation:
    return Observation(
        load_point=load_point,
        config=EngineConfig(engine="mock"),
        metrics=_metrics(goodput_rps),
        valid=valid,
        invalid_reason="" if valid else "client artifact",
    )


def test_summarize_with_no_valid_observation_is_infeasible() -> None:
    ctx = make_context("decode", run_dir="/tmp/x")
    obs = [_observation(8, 99.0, valid=False), _observation(16, 50.0, valid=False)]
    r = summarize(obs, ctx)
    assert not r.feasible and not r.slo_met
    assert r.objective == 0.0 and r.best_load_point == 8
    assert r.observations == obs  # the failed points are still evidence for diagnosis


def test_summarize_of_nothing_at_all_is_infeasible() -> None:
    r = summarize([], make_context("decode", run_dir="/tmp/x"))
    assert not r.feasible and r.objective == 0.0 and r.best_load_point == 0


def test_objective_ignores_invalid_observations() -> None:
    """An invalid point's goodput is an artifact of the client, not a score to beat."""
    ctx = make_context("decode", run_dir="/tmp/x")
    obs = [_observation(4, 10.0), _observation(16, 99.0, valid=False)]
    r = summarize(obs, ctx)
    assert r.feasible and r.objective == 10.0 and r.best_load_point == 4
