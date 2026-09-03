import statistics
import time
from pathlib import Path

import pytest

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
from infervolt.search.optuna_search import (
    MAX_REJECTS,
    MAX_SKIPS,
    MIN_STAGE1_BEFORE_PRUNE,
    STAGE1_REQUESTS,
    _stage1_points,
    run_search,
)
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
        # Trials are indexed after the highest index the ledger already holds. Nothing was
        # saved here, so the search starts at 0; see the saved-baseline test for the case
        # the planner actually produces.
        assert [t.index for t in trials] == [0, 1, 2, 3]


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


def test_priors_keep_their_attribution_when_clamping_moves_them(tmp_path: Path) -> None:
    """A prior is the planner's hypothesis whatever the bounds do to it on the way in.

    The first prior OOMs the KV cache, which drops the ``max_num_seqs`` ceiling to 255.
    The second prior asks for the default 256 and is clamped to 255 -- a different config
    key, and under key-on-the-clamped-value attribution it would come back anonymous.
    """
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        kv_oom = Candidate(
            id="p-kv",
            config=base.candidate.config.with_knobs(max_model_len=32768),
            origin="llm_prior",
            hypothesis="more context",
        )
        clamped = Candidate(
            id="p-fp8",
            config=base.candidate.config.with_knobs(kv_cache_dtype="fp8", max_num_seqs=256),
            origin="llm_prior",
            hypothesis="fp8 kv",
        )
        plan = SearchPlan(subspaces=["kv"], priors=[kv_oom, clamped], max_trials=2)
        trials = run_search(
            adapter, ctx, adapter.knob_space(ctx), plan, base, ledger, Budget(max_trials=2), seed=7
        )
        assert trials[0].status == "infeasible_oom"
        assert trials[0].candidate.hypothesis == "more context"
        second = trials[1]
        assert second.candidate.config.knobs["max_num_seqs"] == 255  # the clamp did move it
        assert second.candidate.origin == "llm_prior"
        assert second.candidate.hypothesis == "fp8 kv"


def test_stage1_points_are_the_best_load_point_and_the_next_one_up() -> None:
    """Capacity wins are invisible below saturation, so stage 1 must straddle the knee."""
    assert _stage1_points([1, 4, 16, 64, 128], 16) == [16, 64]
    assert _stage1_points([1, 4, 16, 64, 128], 128) == [128]  # nothing higher on offer
    assert _stage1_points([1, 4, 16, 64, 128], 0) == [0]  # a baseline that served nothing


def test_stage1_measures_both_points(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        assert base.result is not None
        expected = _stage1_points(ctx.workload.load.concurrency, base.result.best_load_point)
        assert len(expected) == 2
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
        pruned = [t for t in trials if t.status == "pruned"]
        assert pruned, "seed 3 is expected to prune at least one trial"
        for t in pruned:
            assert t.result is not None
            assert [o.load_point for o in t.result.observations] == expected


def test_index_continues_after_a_saved_baseline_and_ids_stay_unique(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        ledger.save_trial(base)  # the planner saves the baseline before searching
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base, max_trials=4),
            base,
            ledger,
            Budget(max_trials=4),
            seed=7,
        )
        assert [t.id for t in trials] == ["t1", "t2", "t3", "t4"]
        assert len({t.id for t in ledger.trials("r1")}) == len(trials) + 1


def test_fixed_knobs_are_applied_and_never_resampled(tmp_path: Path) -> None:
    """``plan.fixed`` pins a knob for the whole search, inside the sub-space or out of it."""
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        plan = SearchPlan(
            subspaces=["kv"],
            # enforce_eager is a 'sched' knob (outside the sub-space); kv_cache_dtype is
            # a 'kv' knob the sampler would otherwise be free to overwrite.
            fixed={"enforce_eager": True, "kv_cache_dtype": "fp8"},
            priors=[],
            max_trials=5,
        )
        trials = run_search(
            adapter, ctx, adapter.knob_space(ctx), plan, base, ledger, Budget(max_trials=5), seed=7
        )
        assert trials
        assert all(t.candidate.config.knobs["enforce_eager"] is True for t in trials)
        assert all(t.candidate.config.knobs["kv_cache_dtype"] == "fp8" for t in trials)


def test_a_subspace_of_nothing_but_fixed_knobs_runs_no_trials(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        space = adapter.knob_space(ctx)
        plan = SearchPlan(
            subspaces=["kv"],
            fixed={k.name: k.default for k in space.subspace(["kv"]).knobs},
            priors=[],
            max_trials=5,
        )
        assert (
            run_search(adapter, ctx, space, plan, base, ledger, Budget(max_trials=5), seed=7) == []
        )


def test_search_gives_up_after_max_skips_novelty_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler that only ever re-proposes measured points has exhausted the region."""
    calls = 0

    def never_novel(*_args: object, **_kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr("infervolt.search.optuna_search.is_novel", never_novel)
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        trials = run_search(
            adapter,
            ctx,
            adapter.knob_space(ctx),
            _plan(base, max_trials=200),
            base,
            ledger,
            Budget(max_trials=200),
            seed=7,
        )
        assert trials == []
        assert calls == MAX_SKIPS


def test_search_gives_up_after_max_rejects_invalid_configs(tmp_path: Path) -> None:
    """Statically rejected configs cost no GPU time but are not free: cap them too."""

    class _RejectsEverything(MockAdapter):
        def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
            return ["nope"]

    ctx = make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(MockAdapter(), ctx)
        trials = run_search(
            _RejectsEverything(),
            ctx,
            MockAdapter().knob_space(ctx),
            _plan(base, max_trials=200),
            base,
            ledger,
            Budget(max_trials=200),
            seed=7,
        )
        assert len(trials) == MAX_REJECTS
        assert all(t.status == "rejected" for t in trials)


def test_pruned_trials_scored_below_the_running_median(tmp_path: Path) -> None:
    """Every pruned trial was worse at stage 1 than the median of the trials before it.

    The mock is deterministic in ``(ctx.seed, concurrency)``, so a trial's stage-1 score
    can be recovered after the fact by re-running its config at the stage-1 points -- the
    search does not have to carry the number around for the test's benefit.
    """
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
        base = _baseline(adapter, ctx)
        assert base.result is not None
        points = _stage1_points(ctx.workload.load.concurrency, base.result.best_load_point)
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
        assert any(t.status == "pruned" for t in trials)

        def stage1_score(t: Trial) -> float | None:
            replay = Trial(id=t.id, run_id=t.run_id, index=t.index, candidate=t.candidate, stage=1)
            replay = run_candidate(adapter, replay, ctx, points, STAGE1_REQUESTS)
            if replay.status != "ok" or replay.result is None or not replay.result.feasible:
                return None
            return replay.result.objective

        earlier: list[float] = []
        for t in trials:
            s1 = stage1_score(t)
            if s1 is None:
                continue  # crashed or was rejected: never part of the median
            if t.status == "pruned":
                assert len(earlier) >= MIN_STAGE1_BEFORE_PRUNE
                assert s1 < statistics.median(earlier)
            earlier.append(s1)
