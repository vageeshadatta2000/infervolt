from infervolt.core.types import Candidate, EngineConfig, KnobValue, RunContext, Trial
from infervolt.engines.base import ServerHandle
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.runner.trial import run_candidate, run_sweep


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
