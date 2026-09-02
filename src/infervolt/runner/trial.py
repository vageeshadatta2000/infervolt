"""Execute one candidate: static validation, launch, sweep, scrape, stop, crash classification.

Every exit from :func:`run_candidate` is a ``Trial`` with a terminal status. Nothing
propagates: a candidate that OOMs, crashes, hangs or is statically rejected is a *result*
the search has to learn from, not an error the caller has to handle.
"""

from __future__ import annotations

import time

from infervolt.core.types import EngineConfig, Observation, Result, RunContext, Trial
from infervolt.engines.base import EngineAdapter, LaunchError, ServerHandle
from infervolt.loadgen.analysis import compute_metrics

# Above any of these the load generator, not the server, is the thing being measured.
CLIENT_CPU_MAX = 0.8
CLIENT_LAG_MAX_MS = 5.0
CLIENT_ERROR_MAX = 0.01
# Sweep stop rules: quit once goodput has fallen this far below the best seen, or once
# the server is shedding more than this share of requests.
SWEEP_COLLAPSE = 0.8
SWEEP_ERROR_MAX = 0.05


def run_load_point(
    adapter: EngineAdapter,
    handle: ServerHandle,
    ctx: RunContext,
    concurrency: int,
    num_requests: int,
) -> tuple[Observation, float]:
    """Drive one concurrency level and return the observation plus its wall-clock seconds."""
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency, num_requests, ctx.seed)
    obs = Observation(
        load_point=concurrency,
        config=handle.config or EngineConfig(engine=adapter.name),
        metrics=compute_metrics(lr, ctx.slo, ctx.hw),
        engine=adapter.scrape(handle),
        gpu=adapter.gpu_stats(handle),
    )
    h = lr.health
    if not any(r.ok for r in lr.requests):
        # No latency percentile means anything here, and the sweep must not read the
        # resulting zeros as a healthy point that simply scored badly.
        obs.valid, obs.invalid_reason = False, "no successful requests at this load point"
    elif (
        h.worker_cpu > CLIENT_CPU_MAX
        or h.loop_lag_p99_ms > CLIENT_LAG_MAX_MS
        or h.error_rate > CLIENT_ERROR_MAX
    ):
        obs.valid = False
        obs.invalid_reason = (
            f"client artifact: cpu={h.worker_cpu:.2f} "
            f"lag_p99={h.loop_lag_p99_ms:.1f}ms err={h.error_rate:.3f}"
        )
    return obs, lr.duration_s


def run_sweep(
    adapter: EngineAdapter,
    handle: ServerHandle,
    ctx: RunContext,
    concurrencies: list[int],
    num_requests: int,
) -> tuple[list[Observation], float]:
    """Increase concurrency until goodput collapses, errors rise, or the client is the bottleneck.

    Returns every observation taken (including the one that triggered the stop) and the
    total load seconds, which is what the trial is billed for.
    """
    obs: list[Observation] = []
    total_s = 0.0
    best = 0.0
    for c in concurrencies:
        o, dur = run_load_point(adapter, handle, ctx, c, num_requests)
        obs.append(o)
        total_s += dur
        if not o.valid or o.metrics.error_rate > SWEEP_ERROR_MAX:
            break
        best = max(best, o.metrics.goodput_rps)
        if len(obs) > 1 and o.metrics.goodput_rps < SWEEP_COLLAPSE * best:
            break
    return obs, total_s


def run_candidate(
    adapter: EngineAdapter,
    trial: Trial,
    ctx: RunContext,
    concurrencies: list[int],
    num_requests: int,
    ready_timeout_s: float = 900.0,
) -> Trial:
    """Take one candidate from config to a finished trial, in place."""
    cfg = trial.candidate.config
    trial.started = time.time()
    trial.status = "running"
    errs = adapter.validate(cfg, ctx)
    if errs:
        trial.status, trial.log_tail, trial.ended = "rejected", "; ".join(errs), time.time()
        return trial
    try:
        handle = adapter.launch(cfg, ctx)
    except LaunchError as e:
        kind = adapter.classify_crash(e.exit)
        trial.crash_kind = kind
        trial.log_tail = e.exit.log_tail[-2000:]
        trial.status = "infeasible_oom" if kind == "oom" else "crash"
        trial.ended = time.time()
        return trial
    except Exception as e:  # noqa: BLE001 - an adapter bug is still just a failed trial
        # Adapters are contracted to raise LaunchError. One that does not is misbehaving,
        # but taking the whole run down over it would lose every trial already completed.
        trial.crash_kind = "startup"
        trial.log_tail = f"{type(e).__name__}: {e}"
        trial.status = "crash"
        trial.ended = time.time()
        return trial
    handle.config = cfg
    try:
        if not adapter.ready(handle, ready_timeout_s):
            trial.status, trial.crash_kind = "timeout", "timeout"
            return trial
        obs, load_s = run_sweep(adapter, handle, ctx, concurrencies, num_requests)
    finally:
        exit_info = adapter.stop(handle)
        trial.ended = time.time()
    kind = adapter.classify_crash(exit_info)
    if kind != "none":
        # The server died during the sweep; the numbers it produced cannot be trusted.
        trial.crash_kind, trial.log_tail = kind, exit_info.log_tail[-2000:]
        trial.status = "infeasible_oom" if kind == "oom" else "crash"
        return trial
    trial.cost_usd = ctx.hw.usd_per_hour * ctx.hw.count / 3600.0 * load_s
    trial.result = summarize(obs, ctx)
    trial.status = "ok"
    return trial


def summarize(obs: list[Observation], ctx: RunContext) -> Result:
    """Pick the load point with the highest goodput and score the candidate by it.

    Invalid observations are excluded from the choice but kept in ``observations``: the
    diagnosis rules want to see the point where the sweep stopped and why.
    """
    valid = [o for o in obs if o.valid]
    if not valid:
        return Result(
            observations=obs,
            objective=0.0,
            feasible=True,
            slo_met=False,
            best_load_point=obs[0].load_point if obs else 0,
        )
    best = max(valid, key=lambda o: o.metrics.goodput_rps)
    return Result(
        observations=obs,
        objective=best.metrics.goodput_rps,
        feasible=True,
        slo_met=best.metrics.goodput_frac >= ctx.slo.goodput_target,
        best_load_point=best.load_point,
    )
