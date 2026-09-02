from __future__ import annotations

import pytest

from infervolt.core.types import (
    Candidate,
    EngineConfig,
    KnobValue,
    QualityScore,
    RunContext,
    Trial,
)
from infervolt.engines.base import ExitInfo, LaunchError, ServerHandle
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.loadgen.base import LoadGenerator
from infervolt.runner.trial import run_candidate
from infervolt.verify.quality import (
    QUALITY_KNOBS,
    RECOVERY_MIN,
    MockQualityGuard,
    needs_quality_guard,
    recovery_threshold,
)
from infervolt.verify.verify import T_975, VerifyResult, paired_ci, verify


def _trial(adapter: MockAdapter, ctx: RunContext, knobs: dict[str, KnobValue], idx: int) -> Trial:
    cfg = EngineConfig(engine="mock", knobs={**adapter.knob_space(ctx).defaults(), **knobs})
    t = Trial(
        id=f"t{idx}",
        run_id="r",
        index=idx,
        candidate=Candidate(id=f"c{idx}", config=cfg, origin="tpe"),
    )
    return run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)


class _RecordingAdapter(MockAdapter):
    """Mock adapter that logs which config each launch saw, in order."""

    def __init__(self) -> None:
        self.launched: list[KnobValue] = []

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        self.launched.append(cfg.knobs["kv_cache_dtype"])
        return super().launch(cfg, ctx)


class _SeedRecordingAdapter(MockAdapter):
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        self.seeds.append(ctx.seed)
        return super().loadgen(handle, ctx)


class _LaunchFailsAdapter(MockAdapter):
    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        raise LaunchError(ExitInfo(code=1, log_tail="CUDA out of memory"))


class _ReadyExplodesAdapter(MockAdapter):
    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        raise RuntimeError("socket exploded")


class _NotReadyAdapter(MockAdapter):
    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        return False


class _StopExplodesAdapter(MockAdapter):
    def stop(self, handle: ServerHandle) -> ExitInfo:
        raise RuntimeError("teardown exploded")


class _LowRecoveryGuard:
    name = "low"

    def evaluate(self, cfg: EngineConfig, ctx: RunContext) -> QualityScore:
        return QualityScore(guard=self.name, tasks=["gsm8k"], recovery=0.5)


def test_paired_ci() -> None:
    lo, hi, mean = paired_ci([1.0, 1.2, 1.1])
    assert lo < mean < hi and lo > 0
    lo2, _, _ = paired_ci([0.1, -0.1, 0.05])
    assert lo2 < 0


def test_paired_ci_degenerate_samples() -> None:
    assert paired_ci([2.0]) == (2.0, 2.0, 2.0)
    assert paired_ci([]) == (0.0, 0.0, 0.0)
    assert paired_ci([1.5, 1.5, 1.5]) == (1.5, 1.5, 1.5)


def test_t_table_covers_two_through_ten_and_falls_back() -> None:
    assert set(T_975) == {2, 3, 4, 5, 6, 7, 8, 9, 10}
    # n=7..9 must not silently take the 2.0 fallback.
    assert T_975[7] == 2.447 and T_975[8] == 2.365 and T_975[9] == 2.306
    wide = paired_ci([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])  # n=7 -> t=2.447
    narrow = paired_ci([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])  # n=11 -> t=2.0
    assert wide[1] - wide[0] > 0 and narrow[1] - narrow[0] > 0


def test_verify_accepts_real_improvement_and_rejects_noise() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    better = _trial(adapter, ctx, {"kv_cache_dtype": "fp8"}, 1)
    same = _trial(adapter, ctx, {"enable_chunked_prefill": True}, 2)
    v = verify(adapter, ctx, base, better, MockQualityGuard(), repeats=3)
    assert v.accepted and v.ci_low > 0 and v.repeats == 3 and v.quality is not None
    assert v.quality.recovery >= 0.97
    assert v.improvement_pct > 0 and len(v.baseline_goodput) == 3
    v2 = verify(adapter, ctx, base, same, MockQualityGuard(), repeats=3)
    assert not v2.accepted


def test_verify_interleaves_baseline_and_candidate_with_distinct_seeds() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    base = _trial(MockAdapter(), ctx, {}, 0)
    better = _trial(MockAdapter(), ctx, {"kv_cache_dtype": "fp8"}, 1)

    rec = _RecordingAdapter()
    verify(rec, ctx, base, better, MockQualityGuard(), repeats=3)
    assert rec.launched == ["auto", "fp8", "auto", "fp8", "auto", "fp8"]

    seeds = _SeedRecordingAdapter()
    verify(seeds, ctx, base, better, MockQualityGuard(), repeats=3)
    # One seed per (repeat, arm); every repeat gets its own, and no arm reuses another's.
    assert len(set(seeds.seeds)) == 6
    assert ctx.seed not in seeds.seeds


def test_verify_survives_launch_ready_and_stop_failures() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    base = _trial(MockAdapter(), ctx, {}, 0)
    better = _trial(MockAdapter(), ctx, {"kv_cache_dtype": "fp8"}, 1)

    v = verify(_LaunchFailsAdapter(), ctx, base, better, MockQualityGuard(), repeats=3)
    assert not v.accepted and v.baseline_goodput == [0.0] * 3 and v.candidate_goodput == [0.0] * 3

    v2 = verify(_ReadyExplodesAdapter(), ctx, base, better, MockQualityGuard(), repeats=2)
    assert not v2.accepted and v2.candidate_goodput == [0.0] * 2
    assert any("socket exploded" in e for e in v2.errors)

    v3 = verify(_NotReadyAdapter(), ctx, base, better, MockQualityGuard(), repeats=2)
    assert not v3.accepted and v3.candidate_goodput == [0.0] * 2

    # A teardown that raises must not lose the measurement it was tearing down.
    v4 = verify(_StopExplodesAdapter(), ctx, base, better, MockQualityGuard(), repeats=3)
    assert v4.accepted and all(g > 0 for g in v4.candidate_goodput)


def test_verify_rejects_a_ci_separated_win_that_fails_the_quality_guard() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    better = _trial(adapter, ctx, {"kv_cache_dtype": "fp8"}, 1)
    v = verify(adapter, ctx, base, better, _LowRecoveryGuard(), repeats=3)
    assert v.ci_low > 0 and not v.accepted
    assert v.quality is not None and v.quality.recovery == 0.5
    assert "recovery" in v.reason


def test_verify_skips_the_quality_guard_when_no_numerics_knob_moved() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    # More KV headroom: a win with no numerics change, so the guard must not run.
    bigger = _trial(adapter, ctx, {"gpu_memory_utilization": 0.95}, 1)
    v = verify(adapter, ctx, base, bigger, _LowRecoveryGuard(), repeats=3)
    assert v.quality is None
    assert v.accepted == (v.ci_low > 0)


def test_verify_result_round_trips_and_rejects_bad_repeats() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    better = _trial(adapter, ctx, {"kv_cache_dtype": "fp8"}, 1)
    v = verify(adapter, ctx, base, better, MockQualityGuard(), repeats=2)
    assert VerifyResult.model_validate_json(v.model_dump_json()) == v
    assert better.result is not None and v.load_point == better.result.best_load_point
    with pytest.raises(ValueError, match="repeats"):
        verify(adapter, ctx, base, better, MockQualityGuard(), repeats=0)


def test_verify_requires_a_measured_candidate() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    unmeasured = Trial(
        id="t9", run_id="r", index=9, candidate=base.candidate.model_copy(update={"id": "c9"})
    )
    with pytest.raises(ValueError, match="result"):
        verify(adapter, ctx, base, unmeasured, MockQualityGuard(), repeats=2)


def test_needs_quality_guard_only_for_numerics_changing_knobs() -> None:
    assert needs_quality_guard({"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "fp8"})
    assert needs_quality_guard({"speculative": "none"}, {"speculative": "ngram"})
    assert not needs_quality_guard({"max_num_seqs": 256}, {"max_num_seqs": 128})


def test_needs_quality_guard_sees_knobs_appearing_or_disappearing() -> None:
    assert sorted(QUALITY_KNOBS) == ["kv_cache_dtype", "quantization", "speculative"]
    assert needs_quality_guard({}, {"quantization": "fp8"})
    assert needs_quality_guard({"quantization": "fp8"}, {})
    assert not needs_quality_guard({}, {})


def test_recovery_threshold_keys_on_quantization() -> None:
    def cfg(**knobs: KnobValue) -> EngineConfig:
        return EngineConfig(engine="mock", knobs=knobs)

    assert recovery_threshold(cfg(quantization="fp8")) == RECOVERY_MIN["fp8"]
    assert recovery_threshold(cfg(quantization="int4")) == RECOVERY_MIN["int4"]
    assert recovery_threshold(cfg(quantization="none")) == RECOVERY_MIN["default"]
    assert recovery_threshold(cfg()) == RECOVERY_MIN["default"]


def test_mock_quality_guard_reports_the_worst_recovery_of_the_config() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    g = MockQualityGuard()
    assert g.evaluate(EngineConfig(engine="mock"), ctx).recovery == 1.0
    assert (
        g.evaluate(EngineConfig(engine="mock", knobs={"kv_cache_dtype": "fp8"}), ctx).recovery
        > 0.99
    )
    both = g.evaluate(
        EngineConfig(engine="mock", knobs={"kv_cache_dtype": "fp8", "quantization": "fp8"}), ctx
    )
    assert both.recovery == 0.992 and both.guard == "mock" and both.tasks
