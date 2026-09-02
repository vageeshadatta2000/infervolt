"""Planner behaviour that the happy-path integration run never exercises.

Everything here is about the loop *not* going to plan: a mistyped baseline knob, an LLM
that is unavailable or hostile, a budget that runs out before verification, and the two
terminal states that are not "done with a recipe".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml
from pydantic import BaseModel

from infervolt.agent.planner import INFEASIBLE_STATUSES, Planner, trials_to_target
from infervolt.config import Settings
from infervolt.core.types import (
    Budget,
    Candidate,
    Diagnosis,
    EngineConfig,
    OptimizeSpec,
    Result,
    RunContext,
    RunOutcome,
    Trial,
    TrialStatus,
)
from infervolt.diagnose.ranker import FALLBACK_CAVEAT
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.llm.base import PriorOut, SearchPlanOut
from infervolt.llm.fake import FakeLLMClient
from infervolt.recipes.schema import Recipe
from infervolt.store.ledger import Ledger
from infervolt.workloads.presets import parse_slo

T = TypeVar("T", bound=BaseModel)


def _spec(name: str = "kv", **over: Any) -> OptimizeSpec:
    s = SCENARIOS[name]
    fields: dict[str, Any] = {
        "engine": "mock",
        "model": s.model,
        "hardware": s.hardware,
        "workload": s.workload,
        "slo": parse_slo(s.slo),
        "budget": Budget(max_trials=4),
        "baseline": dict(s.baseline),
        "llm": "fake",
    }
    return OptimizeSpec(**{**fields, **over})


class _Run:
    """One completed loop: its outcome, its log lines, and how many trials it recorded."""

    def __init__(self, outcome: RunOutcome, log: list[str], trials: int) -> None:
        self.outcome = outcome
        self.log = "\n".join(log)
        self.trials = trials


def _run(spec: OptimizeSpec, tmp_path: Path, llm: Any = None) -> _Run:
    settings = Settings(home=tmp_path)
    lines: list[str] = []
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        outcome = Planner(spec, settings, llm or FakeLLMClient(), ledger, log=lines.append).run()
        trials = ledger.trials(outcome.run_id)
        # The ledger is the record anyone debugging the run will read, so it has to agree
        # with what the caller was told.
        assert ledger.get_run(outcome.run_id).state == outcome.state
    return _Run(outcome, lines, len(trials))


# ---------------------------------------------------------------- baseline validation


def test_a_mistyped_baseline_knob_fails_the_run_before_any_trial(tmp_path: Path) -> None:
    """A knob the engine never heard of would otherwise be measured as its default.

    The run would then report the override in the recipe while having tuned around the
    default -- a wrong answer, which is worse than no answer.
    """
    r = _run(_spec(baseline={"typo_knob": 99}), tmp_path)
    assert r.outcome.state == "failed"
    assert "unknown knob 'typo_knob'" in r.outcome.message
    assert "max_num_seqs" in r.outcome.message  # the known knobs are listed
    assert r.trials == 0
    assert r.outcome.recipe_path is None and r.outcome.report_path is None


def test_a_baseline_value_outside_a_categoricals_choices_fails_the_run(tmp_path: Path) -> None:
    r = _run(_spec(baseline={"kv_cache_dtype": "int3"}), tmp_path)
    assert r.outcome.state == "failed"
    assert "kv_cache_dtype='int3' is not one of" in r.outcome.message
    assert r.trials == 0


def test_a_valid_baseline_override_still_runs(tmp_path: Path) -> None:
    r = _run(_spec("sched"), tmp_path)  # the sched scenario overrides enforce_eager
    assert r.outcome.state == "done", r.outcome.message


# ---------------------------------------------------------------- LLM degradation


class _BrokenLLM:
    """A client that fails the way a missing API key does: not with ``LLMError``."""

    model_id = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls += 1
        raise RuntimeError("Could not resolve authentication method")


def test_a_client_that_raises_anything_still_produces_a_recipe(tmp_path: Path) -> None:
    llm = _BrokenLLM()
    r = _run(_spec(), tmp_path, llm=llm)
    assert r.outcome.state == "done", r.outcome.message
    assert r.outcome.accepted and r.outcome.recipe_path
    assert llm.calls > 0  # it really was asked, and really did fail every time
    assert r.outcome.diagnosis is not None
    assert r.outcome.diagnosis.caveats == [FALLBACK_CAVEAT]
    assert "LLM ranking unavailable" in FALLBACK_CAVEAT
    assert "llm error: RuntimeError" in r.log
    assert "searching the diagnosis sub-spaces without priors" in r.log
    assert "using the template narrative" in r.log


def test_the_caveats_reach_the_recipe_and_the_report(tmp_path: Path) -> None:
    r = _run(_spec(), tmp_path, llm=_BrokenLLM())
    assert r.outcome.recipe_path and r.outcome.report_path
    recipe = Recipe.model_validate(yaml.safe_load(Path(r.outcome.recipe_path).read_text()))
    assert recipe.infervolt.diagnosis.caveats == [FALLBACK_CAVEAT]
    report = Path(r.outcome.report_path).read_text()
    assert "Caveats from diagnosis" in report
    assert FALLBACK_CAVEAT in report


def test_a_recipe_without_caveats_prints_no_caveats_section(tmp_path: Path) -> None:
    """The fake client always has one, so this checks the template's own condition."""
    r = _run(_spec(), tmp_path)
    assert r.outcome.recipe_path and r.outcome.report_path
    recipe = Recipe.model_validate(yaml.safe_load(Path(r.outcome.recipe_path).read_text()))
    assert recipe.infervolt.diagnosis.caveats  # the fake client supplies one
    recipe.infervolt.diagnosis.caveats = []
    from infervolt.recipes.emit import render_report

    assert "Caveats from diagnosis" not in render_report(recipe)


def test_the_summary_table_labels_each_arms_load_point(tmp_path: Path) -> None:
    r = _run(_spec(), tmp_path)
    assert r.outcome.recipe_path and r.outcome.report_path
    recipe = Recipe.model_validate(yaml.safe_load(Path(r.outcome.recipe_path).read_text()))
    assert recipe.baseline.load_point is not None and recipe.result.load_point is not None
    header = next(
        line
        for line in Path(r.outcome.report_path).read_text().splitlines()
        if "| Metric |" in line
    )
    assert "(c=" in header
    assert f"Baseline (c={recipe.baseline.load_point})" in header
    assert f"Tuned (c={recipe.result.load_point})" in header


def test_evidence_values_are_rounded_to_four_significant_digits(tmp_path: Path) -> None:
    r = _run(_spec(), tmp_path)
    assert r.outcome.recipe_path
    recipe = Recipe.model_validate(yaml.safe_load(Path(r.outcome.recipe_path).read_text()))
    values = [e.value for f in recipe.infervolt.diagnosis.findings for e in f.evidence]
    assert values
    assert all(v == float(f"{v:.4g}") for v in values)


# ---------------------------------------------------------------- budget wiring


def test_a_spent_cost_budget_skips_verification(tmp_path: Path) -> None:
    """Verification is several more launches; an exhausted wallet does not buy them."""
    r = _run(_spec(budget=Budget(max_trials=4, max_usd=1e-9)), tmp_path)
    assert r.outcome.state == "done"
    assert not r.outcome.accepted
    assert r.outcome.message.startswith("stopped before verification: cost budget exhausted")
    assert r.outcome.recipe_path is None and r.outcome.report_path
    assert "verify:" not in r.log


def test_spending_the_trial_budget_does_not_skip_verification(tmp_path: Path) -> None:
    """The search using every trial it was given is success, not a reason to stop."""
    r = _run(_spec(budget=Budget(max_trials=4)), tmp_path)
    assert r.outcome.state == "done" and r.outcome.accepted, r.outcome.message
    assert "verify: ACCEPTED" in r.log


def test_a_timeout_is_counted_as_infeasible() -> None:
    """A trial that never finished is not a candidate that lost; it is one that failed."""
    assert set(INFEASIBLE_STATUSES) == {"infeasible_oom", "crash", "rejected", "timeout"}


# ---------------------------------------------------------------- plan filtering


class _HostileLLM:
    """Answers the planning call with knobs, values and sub-spaces that do not exist."""

    model_id = "hostile"

    def __init__(self, plan: SearchPlanOut) -> None:
        self.plan = plan
        self.fake = FakeLLMClient()

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        if schema is SearchPlanOut:
            return schema.model_validate(self.plan.model_dump())
        return self.fake.structured(system=system, user=user, schema=schema)


def _planner(tmp_path: Path, llm: Any, name: str = "decode") -> tuple[Planner, RunContext]:
    settings = Settings(home=tmp_path)
    ledger = Ledger(settings.ledger_path, settings.runs_dir)
    planner = Planner(_spec(name), settings, llm, ledger, adapter=MockAdapter(), log=lambda _: None)
    return planner, make_context(name, run_dir=str(tmp_path))


def _diagnosis(subspaces: list[str]) -> Diagnosis:
    return Diagnosis(
        primary="decode_bandwidth",
        ranked=[],
        rationale="r",
        confidence=0.5,
        subspaces=subspaces,
    )


def test_plan_filters_everything_the_model_invented(tmp_path: Path) -> None:
    """The model may propose anything; only what the engine actually offers survives.

    Five priors go in -- one with an invented knob alongside a real one, one whose only
    knob is off the categorical's menu, one the adapter rejects on this hardware, one
    wildly out of range, and one spare -- and what comes back is the filtered remainder.
    """
    plan = SearchPlanOut(
        subspaces=["decode", "not_a_subspace"],
        max_trials=9999,
        priors=[
            PriorOut(knobs={"warp_drive": 9, "kv_cache_dtype": "fp8"}, hypothesis="invented knob"),
            PriorOut(knobs={"kv_cache_dtype": "int3"}, hypothesis="off the menu"),
            PriorOut(knobs={"quantization": "fp8"}, hypothesis="needs cc >= 8.9"),
            PriorOut(knobs={"max_num_seqs": 10**9}, hypothesis="out of range"),
            PriorOut(knobs={"speculative": "ngram"}, hypothesis="fine"),
        ],
    )
    logged: list[str] = []
    planner, ctx = _planner(tmp_path, _HostileLLM(plan))
    planner.log = logged.append
    space = planner.adapter.knob_space(ctx)
    base = EngineConfig(engine="mock", knobs=space.defaults())
    try:
        out = planner._plan(ctx, space, _diagnosis(["decode"]), base)
    finally:
        planner.ledger.close()

    hypotheses = [p.hypothesis for p in out.priors]
    assert "invented knob" in hypotheses  # the real half of it survived
    assert "off the menu" not in hypotheses  # nothing left once the value was dropped
    assert "needs cc >= 8.9" not in hypotheses  # a100-80 is compute capability 8.0
    kept = next(p for p in out.priors if p.hypothesis == "invented knob")
    assert "warp_drive" not in kept.config.knobs
    assert kept.config.knobs["kv_cache_dtype"] == "fp8"
    clamped = next(p for p in out.priors if p.hypothesis == "out of range")
    assert clamped.config.knobs["max_num_seqs"] == space.get("max_num_seqs").high

    # An unknown sub-space is discarded, and the known one is kept as given.
    assert out.subspaces == ["decode"]
    # The model does not get to raise its own budget.
    assert out.max_trials == 4

    log = "\n".join(logged)
    assert "dropping unknown knobs ['warp_drive']" in log
    assert "dropping out-of-choices knobs" in log and "int3" in log
    assert "rejected statically" in log


def test_plan_falls_back_when_every_subspace_is_unknown(tmp_path: Path) -> None:
    plan = SearchPlanOut(subspaces=["nowhere", "also_nowhere"], max_trials=2)
    planner, ctx = _planner(tmp_path, _HostileLLM(plan))
    space = planner.adapter.knob_space(ctx)
    try:
        out = planner._plan(
            ctx,
            space,
            _diagnosis(["decode", "kv"]),
            EngineConfig(engine="mock", knobs=space.defaults()),
        )
    finally:
        planner.ledger.close()
    assert out.subspaces == ["decode", "kv"]
    assert out.priors == []


def test_plan_never_returns_fewer_than_one_trial(tmp_path: Path) -> None:
    planner, ctx = _planner(tmp_path, _HostileLLM(SearchPlanOut(subspaces=[], max_trials=0)))
    space = planner.adapter.knob_space(ctx)
    try:
        out = planner._plan(
            ctx, space, _diagnosis(["decode"]), EngineConfig(engine="mock", knobs=space.defaults())
        )
    finally:
        planner.ledger.close()
    assert out.max_trials == 1


# ---------------------------------------------------------------- terminal states


def _baseline_trial(ctx: RunContext) -> Trial:
    cfg = EngineConfig(engine="mock", knobs=MockAdapter().knob_space(ctx).defaults())
    return Trial(
        id="t0",
        run_id=ctx.run_id,
        index=0,
        candidate=Candidate(id="c0", config=cfg, origin="baseline"),
    )


def test_finish_without_change_writes_a_report_and_ends_the_run_done(tmp_path: Path) -> None:
    planner, _ = _planner(tmp_path, FakeLLMClient())
    run_id = planner.ledger.create_run(planner.spec)
    ctx = make_context("decode", run_dir=str(planner.ledger.run_dir(run_id)), run_id=run_id)
    try:
        outcome = planner._finish_without_change(
            run_id, ctx, _baseline_trial(ctx), _diagnosis(["decode"]), "nothing to change"
        )
        assert planner.ledger.get_run(run_id).state == "done"
    finally:
        planner.ledger.close()
    assert outcome.state == "done" and not outcome.accepted
    assert outcome.message == "nothing to change"
    assert outcome.recipe_path is None and outcome.baseline_trial_id == "t0"
    body = Path(outcome.report_path or "").read_text()
    assert "no change recommended" in body
    assert "nothing to change" in body and "decode_bandwidth" in body


def test_fail_marks_the_run_failed_and_carries_the_reason(tmp_path: Path) -> None:
    planner, _ = _planner(tmp_path, FakeLLMClient())
    run_id = planner.ledger.create_run(planner.spec)
    try:
        outcome = planner._fail(run_id, "baseline failed: crash")
        assert planner.ledger.get_run(run_id).state == "failed"
    finally:
        planner.ledger.close()
    assert outcome.state == "failed"
    assert outcome.message == "baseline failed: crash"
    assert outcome.accepted is False and outcome.diagnosis is None


# ---------------------------------------------------------------- trials_to_target


def _scored(idx: int, objective: float | None, status: TrialStatus = "ok") -> Trial:
    return Trial(
        id=f"t{idx}",
        run_id="r",
        index=idx,
        candidate=Candidate(id=f"c{idx}", config=EngineConfig(engine="mock"), origin="tpe"),
        status=status,
        result=None
        if objective is None
        else Result(
            objective=objective,
            best_load_point=8,
            observations=[],
            feasible=True,
            slo_met=True,
        ),
    )


def test_trials_to_target_counts_the_first_trial_within_five_percent() -> None:
    assert trials_to_target([_scored(0, 1.0), _scored(1, 1.9), _scored(2, 2.0)], 2.0) == 2


@pytest.mark.parametrize(
    "trials",
    [
        pytest.param([], id="no trials at all"),
        pytest.param([_scored(0, 1.0), _scored(1, 1.5)], id="nothing came close"),
        pytest.param([_scored(0, 5.0, status="crash")], id="a crash with a number attached"),
        pytest.param([_scored(0, None)], id="ok but unmeasured"),
    ],
)
def test_trials_to_target_is_none_when_nothing_measured_reached_it(trials: list[Trial]) -> None:
    """Reporting 0, or the trial count, would answer a question the run cannot answer."""
    assert trials_to_target(trials, 2.0) is None
