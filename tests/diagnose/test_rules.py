import pytest

from infervolt.core.types import Candidate, EngineConfig, Observation, Trial
from infervolt.diagnose.rules import evaluate_rules
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.runner.trial import run_candidate


def _baseline_obs(name: str) -> tuple[list[Observation], EngineConfig]:
    adapter, ctx = MockAdapter(), make_context(name, run_dir="/tmp/x")
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
    assert findings[0].evidence and findings[0].subspaces


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
