import time
from pathlib import Path

from infervolt.core.types import (
    Budget,
    Candidate,
    EngineConfig,
    OptimizeSpec,
    RunContext,
    SearchPlan,
    Trial,
)
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.runner.trial import run_candidate
from infervolt.search.optuna_search import run_search
from infervolt.store.ledger import Ledger


def _baseline(adapter: MockAdapter, ctx: RunContext) -> Trial:
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    t = Trial(
        id="t0",
        run_id=ctx.run_id,
        index=0,
        candidate=Candidate(id="c0", config=cfg, origin="baseline"),
    )
    return run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)


def _plan(base: Trial, max_trials: int = 8) -> SearchPlan:
    oom_prior = Candidate(
        id="p-oom",
        config=base.candidate.config.with_knobs(max_model_len=32768, gpu_memory_utilization=0.7),
        origin="llm_prior",
        hypothesis="will OOM",
    )
    good_prior = Candidate(
        id="p-fp8",
        config=base.candidate.config.with_knobs(kv_cache_dtype="fp8"),
        origin="llm_prior",
        hypothesis="fp8 kv",
    )
    return SearchPlan(subspaces=["kv"], priors=[oom_prior, good_prior], max_trials=max_trials)


def test_search_improves_kv_scenario_and_handles_oom(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        space = adapter.knob_space(ctx)
        plan = _plan(base)
        trials = run_search(adapter, ctx, space, plan, base, ledger, Budget(max_trials=8), seed=7)
        statuses = {t.status for t in trials}
        assert "infeasible_oom" in statuses and "ok" in statuses
        best = max(
            (t for t in trials if t.status == "ok" and t.result is not None),
            key=lambda t: t.result.objective if t.result else 0.0,
        )
        assert best.result is not None and base.result is not None
        assert best.result.objective > 1.2 * base.result.objective
        assert len(trials) <= 8
        assert len({t.candidate.config.key() for t in trials}) == len(trials)
        assert len(ledger.trials("r1")) == len(trials)


def test_search_attributes_priors_and_reports_every_trial(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        seen: list[Trial] = []
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base, max_trials=4),
            base,
            ledger,
            Budget(max_trials=4),
            seed=7,
            on_trial=seen.append,
        )
        assert [t.id for t in seen] == [t.id for t in trials]
        # Priors are enqueued ahead of anything TPE would pick, and a prior that survives
        # clamping unchanged is recognised on the way back out and carries its hypothesis.
        assert trials[0].candidate.origin == "llm_prior"
        assert trials[0].candidate.hypothesis == "will OOM"
        assert all(t.candidate.parent_id == base.candidate.id for t in trials)
        # Trials are indexed after whatever the ledger already holds.
        assert [t.index for t in trials] == [1, 2, 3, 4]


def test_search_stops_at_the_deadline(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base),
            base,
            ledger,
            Budget(max_trials=8),
            seed=7,
            deadline=time.time() - 1.0,
        )
        assert trials == []


def test_search_returns_nothing_when_the_subspace_is_empty(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        plan = SearchPlan(subspaces=["no-such-group"], priors=[], max_trials=8)
        assert (
            run_search(
                adapter,
                ctx,
                adapter.knob_space(ctx),
                plan,
                base,
                ledger,
                Budget(max_trials=8),
                seed=7,
            )
            == []
        )


def test_budget_caps_the_plan(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base, max_trials=8),
            base,
            ledger,
            Budget(max_trials=2),
            seed=7,
        )
        assert len(trials) == 2


def test_stage1_pruning_shortens_the_weakest_trials(tmp_path: Path) -> None:
    """Pruned trials are recorded, but only after enough stage-1 scores exist to rank them."""
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base, max_trials=12),
            base,
            ledger,
            Budget(max_trials=12),
            seed=3,
        )
        assert all(t.stage in (1, 2) for t in trials)
        assert all(t.stage == 2 for t in trials if t.status == "ok")
        assert all(t.stage == 1 for t in trials if t.status == "pruned")
