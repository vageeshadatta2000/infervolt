"""Interleaved baseline/candidate repeats with a paired-t confidence interval on goodput.

The search's best trial was measured once, against a baseline measured at a different
moment. Verify re-measures both, alternating arms within each repeat so that any drift
over the verification window -- a warming card, a noisy neighbour, a background job --
lands on both arms rather than on whichever ran second. The repeats are therefore
*paired*: the statistic is the per-repeat difference, and the claim is accepted only
when the 95% CI of that difference clears zero.
"""

from __future__ import annotations

import contextlib
import math
import statistics
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from infervolt.core.types import QualityScore, RunContext, Trial
from infervolt.engines.base import EngineAdapter, LaunchError
from infervolt.runner.trial import run_load_point
from infervolt.verify.quality import QualityGuard, needs_quality_guard, recovery_threshold

T_975 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}
"""Two-sided 95% critical values of Student's t, keyed by *sample size* n (df = n - 1).

The table stops at n = 10 because verification runs are short by construction -- three
repeats is the default, ten an extravagance. Any n outside the table falls back to
:data:`T_FALLBACK`; the table is only worth carrying at all because at the sizes we
actually use (n = 3 gives 4.303) the normal approximation would be far too narrow and
would accept noise as a win.
"""

T_FALLBACK = 2.228
"""The df = 10 critical value, used for every n outside :data:`T_975`.

Student's t shrinks monotonically towards 1.96 as df grows, so the value for the largest
df in the table is an upper bound for every n >= 11: the interval it produces is never
narrower than the correct one, and a verification that errs is meant to err towards
rejecting. The normal limit itself, 1.96, would be an under-estimate at every finite n.
"""

MIN_EFFECT_FRAC = 0.01
"""Smallest relative gain worth calling a win.

Separation from zero is a statement about confidence, not about size: with enough
repeats a reliably reproducible 0.1% clears the interval test and is still not worth
rewriting a production config for. This floor is what keeps "statistically significant"
from being mistaken for "significant".
"""

READY_TIMEOUT_S = 900.0
"""How long a re-measured arm gets to come up. Fifteen minutes covers a cold vLLM start."""

VERIFY_REQUESTS = 16


class VerifyResult(BaseModel):
    # improvement_pct is infinite against a baseline that served nothing, and plain JSON
    # has no spelling for that; "strings" emits "Infinity", which parses straight back.
    model_config = ConfigDict(ser_json_inf_nan="strings")

    accepted: bool
    repeats: int
    load_point: int
    baseline_goodput: list[float]
    candidate_goodput: list[float]
    delta_mean: float
    ci_low: float
    ci_high: float
    improvement_pct: float
    quality: QualityScore | None = None
    reason: str = ""
    errors: list[str] = Field(
        default_factory=list,
        description="One entry per repeat that failed to measure, tagged with its arm and "
        "index (``baseline[1]``, ``candidate[0]``) -- a launch that died, a server that "
        "never came up, an adapter that raised, or an observation the runner ruled "
        "invalid. Each of those scored 0.0, so any entry here is on its own grounds for "
        "rejection: an unmeasured arm is not an arm that lost.",
    )


def improvement_pct(mean: float, base_mean: float) -> float:
    """The mean delta as a percentage of the baseline, or infinity when there is no baseline.

    Verification drives both arms at the *candidate's* best load point, which the
    baseline may not reach at all: a config that OOMs there, or misses every deadline,
    scores a clean zero. There is no ratio to a zero, and reporting 0.0 would say "no
    change" about the one case where the change is total, so the answer is infinite and
    callers are expected to word it in absolute terms instead.
    """
    if base_mean <= 0:
        return math.inf if mean > 0 else 0.0
    return mean / base_mean * 100


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    """``(ci_low, ci_high, mean)`` for the 95% t interval around the mean difference.

    Fewer than two samples has no spread to estimate, so the interval collapses to the
    mean -- which ``verify`` then reads as "not separated from zero" unless the mean
    itself is positive. An empty list is treated as a zero mean rather than an error:
    every repeat having failed is a verdict, not a crash.
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(deltas)
    if n < 2:
        return mean, mean, mean
    sd = statistics.stdev(deltas)
    half = T_975.get(n, T_FALLBACK) * sd / math.sqrt(n)
    return mean - half, mean + half, mean


class _Point(NamedTuple):
    """One measured repeat: its goodput, whether it counted, and why if it did not."""

    goodput: float
    valid: bool
    error: str = ""


def _goodput_at(adapter: EngineAdapter, ctx: RunContext, trial: Trial, c: int, seed: int) -> _Point:
    """Measure one arm once, at load point ``c`` and this repeat's ``seed``.

    This never raises. A repeat that cannot be measured -- the config OOMs on launch, the
    server never becomes ready, the adapter throws, the runner rules the observation
    invalid -- scores 0.0 and says so. Anything else would let one bad repeat abort a
    verification whose other repeats were fine, and a verify that crashes is strictly
    worse than one that reports a rejection.
    """
    cfg = trial.candidate.config
    try:
        handle = adapter.launch(cfg, ctx)
    except LaunchError as e:
        return _Point(0.0, False, f"launch failed: {e.exit.log_tail[-200:]}")
    except Exception as e:  # noqa: BLE001 - an adapter bug is one dead repeat, not a dead run
        return _Point(0.0, False, f"launch raised {type(e).__name__}: {e}")
    handle.config = cfg
    try:
        if not adapter.ready(handle, READY_TIMEOUT_S):
            return _Point(0.0, False, "server never became ready")
        obs, _ = run_load_point(
            adapter, handle, ctx.model_copy(update={"seed": seed}), c, VERIFY_REQUESTS
        )
    except Exception as e:  # noqa: BLE001 - same: this repeat is lost, the rest are not
        return _Point(0.0, False, f"load point raised {type(e).__name__}: {e}")
    finally:
        # As in the runner: teardown is best-effort and must not mask the measurement it
        # was tearing down, nor turn a finished repeat into an exception.
        with contextlib.suppress(Exception):
            adapter.stop(handle)
    if not obs.valid:
        return _Point(0.0, False, obs.invalid_reason or "invalid observation")
    return _Point(obs.metrics.goodput_rps, True)


def verify(
    adapter: EngineAdapter,
    ctx: RunContext,
    baseline: Trial,
    candidate: Trial,
    guard: QualityGuard,
    repeats: int = 3,
) -> VerifyResult:
    """Re-measure both arms ``repeats`` times, interleaved, and decide whether to accept.

    Both arms are driven at the *candidate's* best load point: that is the operating
    point the recipe will claim, so it is the one the comparison has to be about.
    """
    if repeats < 2:
        # One repeat has no spread to estimate, so paired_ci collapses the interval onto
        # the mean and every positive delta -- noise included -- clears zero. A "verified"
        # win from a single pair of measurements is exactly what this function exists to
        # rule out.
        raise ValueError(f"repeats must be >= 2, got {repeats}")
    if candidate.result is None:
        raise ValueError("candidate has no result to verify; run it before verifying it")
    c = candidate.result.best_load_point
    b_points: list[_Point] = []
    c_points: list[_Point] = []
    for i in range(repeats):
        # B, C, B, C, ... -- alternating, and a fresh seed per repeat per arm so the
        # repeats are independent draws rather than the same draw measured twice.
        b_points.append(_goodput_at(adapter, ctx, baseline, c, ctx.seed + 100 + i))
        c_points.append(_goodput_at(adapter, ctx, candidate, c, ctx.seed + 200 + i))
    b_vals = [p.goodput for p in b_points]
    c_vals = [p.goodput for p in c_points]
    errors = [
        f"{arm}[{i}]: {p.error or 'invalid observation'}"
        for arm, points in (("baseline", b_points), ("candidate", c_points))
        for i, p in enumerate(points)
        if not p.valid or p.error
    ]
    deltas = [cv - bv for bv, cv in zip(b_vals, c_vals, strict=True)]
    lo, hi, mean = paired_ci(deltas)
    base_mean = statistics.fmean(b_vals) if b_vals else 0.0
    pct = improvement_pct(mean, base_mean)
    separated = lo > 0
    # No baseline to be a fraction of means the effect size cannot be relative; a
    # candidate serving anything at all where the baseline served nothing is as large an
    # effect as there is.
    material = base_mean <= 0 or mean >= MIN_EFFECT_FRAC * base_mean
    accepted = separated and material and not errors
    if errors:
        # A repeat that failed scored 0.0, which drags its arm's mean down and makes the
        # delta look bigger and better separated. The interval is measuring the failure,
        # not the config, so it cannot be allowed to carry the verdict.
        reason = f"repeat failed: {'; '.join(errors)}"
    elif not separated:
        reason = "improvement not distinguishable from noise"
    elif not material:
        reason = f"improvement {pct:.2f}% is below the {MIN_EFFECT_FRAC:.0%} minimum effect size"
    else:
        reason = "CI-separated improvement"
    quality: QualityScore | None = None
    if accepted and needs_quality_guard(
        baseline.candidate.config.knobs, candidate.candidate.config.knobs
    ):
        quality = guard.evaluate(candidate.candidate.config, ctx)
        if quality.recovery < recovery_threshold(candidate.candidate.config):
            accepted, reason = False, f"quality recovery {quality.recovery:.3f} below threshold"
    return VerifyResult(
        accepted=accepted,
        repeats=repeats,
        load_point=c,
        baseline_goodput=b_vals,
        candidate_goodput=c_vals,
        delta_mean=mean,
        ci_low=lo,
        ci_high=hi,
        improvement_pct=pct,
        quality=quality,
        reason=reason,
        errors=errors,
    )
