import pytest

from infervolt.core.types import (
    Candidate,
    EngineConfig,
    LoadSpec,
    Observation,
    RunContext,
    Trial,
)
from infervolt.diagnose.rules import evaluate_rules
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.runner.trial import run_candidate


def _with_sweep(ctx: RunContext, concurrency: list[int]) -> RunContext:
    """The same context, but asking the sweep for a different list of load points."""
    return ctx.model_copy(
        update={
            "workload": ctx.workload.model_copy(update={"load": LoadSpec(concurrency=concurrency)})
        }
    )


def _baseline_obs(
    name: str, ctx: RunContext | None = None
) -> tuple[list[Observation], EngineConfig]:
    adapter = MockAdapter()
    ctx = ctx if ctx is not None else make_context(name, run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), **SCENARIOS[name].baseline}
    cfg = EngineConfig(engine="mock", knobs=knobs)
    t = Trial(
        id="t0",
        run_id="r",
        index=0,
        candidate=Candidate(id="c0", config=cfg, origin="baseline"),
    )
    t = run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)
    assert t.result is not None
    return t.result.observations, cfg


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_top_finding_matches_injected_bottleneck(name: str) -> None:
    obs, cfg = _baseline_obs(name)
    ctx = make_context(name, run_dir="/tmp/x")
    findings = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert findings, "no findings"
    assert findings[0].bottleneck == SCENARIOS[name].expected, [
        (f.rule_id, f.score) for f in findings
    ]
    # Also asserts the winner is neither R0 nor R6: those are the two rules that carry no
    # subspaces, so a top finding with a non-empty ``subspaces`` cannot be either of them.
    assert findings[0].evidence and findings[0].subspaces


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_top_finding_clears_the_runner_up(name: str) -> None:
    """A diagnosis worth acting on is not a coin flip between two bottlenecks."""
    obs, cfg = _baseline_obs(name)
    ctx = make_context(name, run_dir="/tmp/x")
    fs = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    if len(fs) > 1:
        assert fs[0].score - fs[1].score >= 0.25, [(f.rule_id, f.score) for f in fs]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_saturated_runs_are_never_called_under_loaded(name: str) -> None:
    """Every scenario drives the server into a real bottleneck, so R0 must stay silent."""
    obs, cfg = _baseline_obs(name)
    ctx = make_context(name, run_dir="/tmp/x")
    fs = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert "under_loaded" not in [f.bottleneck for f in fs], [(f.rule_id, f.score) for f in fs]


def test_under_loaded_still_fires_on_a_healthy_short_sweep() -> None:
    """Gating R0 must not mute it on the case it exists for: a sweep that stopped too soon."""
    ctx = _with_sweep(make_context("decode", run_dir="/tmp/x"), [1, 4])
    obs, cfg = _baseline_obs("decode", ctx)
    fs = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert "under_loaded" in [f.bottleneck for f in fs], [(f.rule_id, f.score) for f in fs]


def test_absent_counters_never_manufacture_a_finding() -> None:
    """An engine that reports no counters must not make any condition fire."""
    obs, cfg = _baseline_obs("decode")
    for o in obs:
        o.engine, o.gpu = {}, {}
    ctx = make_context("decode", run_dir="/tmp/x")
    fs = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert "under_loaded" not in [f.bottleneck for f in fs]
    # Whatever survives can only be leaning on the loadgen/roofline half of a rule.
    assert all(f.score <= 0.5 for f in fs), [(f.rule_id, f.score) for f in fs]


def test_evidence_keys_name_the_load_point_they_came_from() -> None:
    """Evidence keys must be stamped with the load point measured, not a hard-coded one."""
    ctx = _with_sweep(make_context("sched", run_dir="/tmp/x"), [2, 8, 32])
    obs, cfg = _baseline_obs("sched", ctx)
    fs = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert fs, "no findings"
    assert 1 not in {o.load_point for o in obs}
    keys = [e.key for f in fs for e in f.evidence]
    assert not any("@c1" in k for k in keys), keys


def test_client_artifact_invalidates() -> None:
    obs, cfg = _baseline_obs("decode")
    obs[-1].valid, obs[-1].invalid_reason = False, "client artifact: cpu=0.95"
    ctx = make_context("decode", run_dir="/tmp/x")
    findings = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert findings[0].bottleneck == "client_artifact"


def test_empty_observations_yield_no_findings() -> None:
    ctx = make_context("decode", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=MockAdapter().knob_space(ctx).defaults())
    assert evaluate_rules([], ctx, cfg, MockAdapter().knob_space(ctx)) == []


def test_all_invalid_yields_only_client_artifact() -> None:
    obs, cfg = _baseline_obs("decode")
    for o in obs:
        o.valid, o.invalid_reason = False, "client artifact: cpu=0.99"
    ctx = make_context("decode", run_dir="/tmp/x")
    findings = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert [f.rule_id for f in findings] == ["R6"]
