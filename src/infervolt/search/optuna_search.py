"""Optuna TPE search inside the planned sub-space.

Three things separate this from a plain ``study.optimize`` call. Priors from the planner
are enqueued so the LLM's hypotheses are tested first and attributed when they land. A
novelty filter refuses to spend a trial on a config the run has already measured. And a
crash is a *result*: an OOM tightens :class:`~infervolt.search.space.Bounds` so the
sampler stops proposing configs that cannot launch, rather than being retried.

Every candidate is measured twice: a cheap stage 1 over :func:`_stage1_points`, then --
only if it beats the median of the stage-1 scores so far -- a full sweep at stage 2. That
is ASHA's idea with a single rung, and it is what keeps a search of a dozen candidates
inside a trial budget meant for half that many.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Callable
from typing import cast

import optuna

from infervolt.core.types import (
    Budget,
    Candidate,
    Knob,
    KnobSpace,
    KnobValue,
    RunContext,
    SearchPlan,
    Trial,
)
from infervolt.engines.base import EngineAdapter
from infervolt.runner.trial import run_candidate
from infervolt.search.space import Bounds, clamp, is_novel
from infervolt.store.ledger import Ledger

WORST = -1.0
"""Objective reported for a trial that measured nothing.

Below every real goodput (which is non-negative), so TPE learns to avoid the region
without the sampler ever having to be told *why* the trial failed.
"""

MAX_SKIPS = 20
"""Consecutive-ish novelty rejections tolerated before giving up on the sub-space.

A sampler that keeps proposing points the run has already measured has exhausted the
region it believes in; more asks would only spend wall-clock.
"""

MAX_REJECTS = 20
"""Statically rejected configs tolerated before giving up on the sub-space.

A rejected config costs no GPU time, but it is not free either: it consumes a trial from
the budget and teaches TPE only :data:`WORST`. Twenty in a row means the sub-space the
planner chose does not fit this engine and hardware, which more asks will not fix.
"""

STAGE1_REQUESTS = 8
STAGE2_REQUESTS = 16
MIN_STAGE1_BEFORE_PRUNE = 3
"""Stage-1 scores needed before a median is worth pruning against."""


def _suggest(trial: optuna.Trial, k: Knob) -> KnobValue:
    """Ask Optuna for one value of ``k``, in the knob's own type."""
    if k.kind == "int":
        assert k.low is not None and k.high is not None  # guaranteed by the Knob validator
        return trial.suggest_int(k.name, int(k.low), int(k.high), log=k.log)
    if k.kind == "float":
        assert k.low is not None and k.high is not None  # guaranteed by the Knob validator
        if k.step is None:
            return trial.suggest_float(k.name, float(k.low), float(k.high), log=k.log)
        # Optuna walks a stepped grid as low + n*step in binary floating point, so the
        # fifth notch of a 0.05 step comes back as 0.8999999999999999. Numerically that
        # is 0.9, but it is not 0.9 in a config key or on an engine's command line, so
        # trim the dust before it reaches either.
        return float(
            f"{trial.suggest_float(k.name, float(k.low), float(k.high), step=float(k.step)):.12g}"
        )
    if k.kind == "bool":
        return bool(trial.suggest_categorical(k.name, [True, False]))
    # Optuna's categorical choices are None | bool | int | float | str, which is exactly
    # KnobValue plus None -- so the value coming back is a KnobValue, but the stub types
    # it as the wider union.
    return cast(KnobValue, trial.suggest_categorical(k.name, k.choices))


def _stage1_points(concurrency: list[int], best_load_point: int) -> list[int]:
    """The baseline's best load point, plus the next one up if the sweep offers one.

    A single point at the baseline's own knee is not enough to rank capacity candidates.
    Below saturation every config that launches serves the offered load at roughly the
    same rate, so a knob that buys *headroom* -- a bigger KV cache, more sequences in
    flight -- looks identical to the baseline there and gets pruned before the stage-2
    sweep that would have shown the win. The next point up is where the baseline is
    already past its knee and the extra capacity turns into goodput, which is exactly
    the difference stage 1 has to be able to see.

    Two points, not the whole sweep: stage 1 exists to be cheap, and the pair straddling
    the knee carries nearly all of the ranking signal the full sweep would.
    """
    points = [best_load_point]
    if best_load_point in concurrency:
        nxt = concurrency.index(best_load_point) + 1
        if nxt < len(concurrency):
            points.append(concurrency[nxt])
    return points


def run_search(
    adapter: EngineAdapter,
    ctx: RunContext,
    space: KnobSpace,
    plan: SearchPlan,
    baseline: Trial,
    ledger: Ledger,
    budget: Budget,
    seed: int,
    deadline: float | None = None,
    on_trial: Callable[[Trial], None] | None = None,
) -> list[Trial]:
    """Search ``plan.subspaces`` for a config that beats ``baseline``.

    Returns every trial run, in order, each already saved to the ledger. Nothing here
    raises on a bad candidate: a crash, an OOM or a rejected config all come back as
    trials with a terminal status, which is what the caller reports on.

    ``deadline`` is an absolute ``time.time()`` value; the loop stops before starting a
    trial it would cross.
    """
    assert baseline.result is not None
    sub = space.subspace(plan.subspaces)
    base_cfg = baseline.candidate.config.with_knobs(**plan.fixed)
    # A knob the plan pinned is not a knob the sampler gets to touch. Suggesting it and
    # then letting ``base_cfg`` win would leave TPE modelling a dimension that never
    # varies, and suggesting it and letting it win would break the pin outright.
    search_knobs = [k for k in sub.knobs if k.name not in plan.fixed]
    if not search_knobs:
        return []
    bounds = Bounds(sub)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=3),
    )
    # Priors go in ahead of anything TPE would pick, and are recognised on the way back
    # out by the *suggested* params so the trial can carry the planner's hypothesis.
    prior_hypotheses: dict[str, str] = {}
    for p in plan.priors:
        params = {k.name: p.config.knobs[k.name] for k in search_knobs if k.name in p.config.knobs}
        if params:
            study.enqueue_trial(params, skip_if_exists=True)
            prior_hypotheses[base_cfg.with_knobs(**params).key()] = p.hypothesis
    prior_trials = ledger.trials(ctx.run_id)
    seen: list[dict[str, KnobValue]] = [baseline.candidate.config.knobs]
    seen += [t.candidate.config.knobs for t in prior_trials]
    trials: list[Trial] = []
    stage1_scores: list[float] = []
    max_trials = min(plan.max_trials, budget.max_trials)
    skips = rejects = 0
    # Index off the highest the ledger holds, not the count: the baseline is index 0, so
    # counting would leave a gap and re-derive an id another trial may already own.
    index = max((t.index for t in prior_trials), default=-1)
    stage1_c = _stage1_points(ctx.workload.load.concurrency, baseline.result.best_load_point)
    stage2_c = ctx.workload.load.concurrency
    while (
        len(trials) < max_trials
        and skips < MAX_SKIPS
        and rejects < MAX_REJECTS
        and (deadline is None or time.time() < deadline)
    ):
        ot = study.ask()
        raw = {k.name: _suggest(ot, k) for k in search_knobs}
        # Attribution is keyed on what the sampler proposed, before ``clamp`` touches it:
        # a prior asking for one more sequence than an OOM has since left room for is
        # still the planner's hypothesis, and the trial has to say so.
        hypothesis = prior_hypotheses.get(base_cfg.with_knobs(**raw).key(), "")
        cfg = base_cfg.with_knobs(**clamp(raw, sub, bounds))
        if not is_novel(cfg.knobs, seen, sub):
            # FAIL rather than PRUNED. Optuna drops failed trials from TPE's observations
            # entirely, which is the truth here: nothing was measured. A pruned trial, by
            # contrast, ranks below every complete one, so reporting a duplicate as
            # PRUNED would teach the sampler to avoid the neighbourhood of a config we
            # already measured -- and the likeliest reason we measured it is that it was
            # good.
            study.tell(ot, state=optuna.trial.TrialState.FAIL)
            skips += 1
            continue
        seen.append(cfg.knobs)
        index += 1
        trial = Trial(
            id=f"t{index}",
            run_id=ctx.run_id,
            index=index,
            candidate=Candidate(
                id=f"c{uuid.uuid4().hex[:6]}",
                config=cfg,
                origin="llm_prior" if hypothesis else "tpe",
                hypothesis=hypothesis,
                parent_id=baseline.candidate.id,
            ),
            stage=1,
        )
        trial = run_candidate(adapter, trial, ctx, stage1_c, STAGE1_REQUESTS)
        s1 = _score(trial)
        if s1 is None:
            # Crashed, was rejected, or served nothing measurable. Either way it produced
            # no score to rank against, so it is not part of the pruning median.
            rejects += trial.status == "rejected"
            _tighten_if_oom(bounds, trial)
            study.tell(ot, WORST)
            _record(ledger, trial, trials, on_trial)
            continue
        if len(stage1_scores) >= MIN_STAGE1_BEFORE_PRUNE and s1 < statistics.median(stage1_scores):
            stage1_scores.append(s1)
            trial.status = "pruned"
            # The stage-1 number is real, just cheap: tell it to TPE rather than WORST,
            # which would teach the sampler that a merely-below-median region is fatal.
            study.tell(ot, s1)
            _record(ledger, trial, trials, on_trial)
            continue
        stage1_scores.append(s1)
        trial.stage = 2
        trial = run_candidate(adapter, trial, ctx, stage2_c, STAGE2_REQUESTS)
        s2 = _score(trial)
        _tighten_if_oom(bounds, trial)
        # TPE sees two scales at once: pruned trials reported at their stage-1 score over
        # two load points, promoted ones at their stage-2 score over the full sweep. The
        # mixing is deliberate -- a pruned trial's own number is still a better signal
        # than WORST -- and harmless in practice because stage 1 straddles the knee and
        # so tracks the sweep's ordering, but it does mean the sampler's objective is not
        # a single well-defined quantity. Worth revisiting if the two ever disagree.
        study.tell(ot, WORST if s2 is None else s2)
        _record(ledger, trial, trials, on_trial)
    return trials


def _score(trial: Trial) -> float | None:
    """The trial's objective, or ``None`` when it measured nothing usable.

    ``feasible`` is false when every observation in the sweep was invalid, which
    ``run_candidate`` still reports as status ``"ok"`` with an objective of 0.0. To the
    search that is indistinguishable from a crash and must not be ranked as a poor but
    working config.
    """
    if trial.status != "ok" or trial.result is None or not trial.result.feasible:
        return None
    return trial.result.objective


def _tighten_if_oom(bounds: Bounds, trial: Trial) -> None:
    """Move the bounds if this trial died out of memory, in whichever direction it says.

    The engine's own message distinguishes the two OOMs that point opposite ways -- too
    little memory reserved for weights, versus a KV cache too small for the context --
    so the log tail is passed along rather than thrown away.
    """
    if trial.crash_kind == "oom":
        bounds.tighten_for(trial.log_tail, trial.candidate.config.knobs)


def _record(
    ledger: Ledger, trial: Trial, trials: list[Trial], on_trial: Callable[[Trial], None] | None
) -> None:
    ledger.save_trial(trial)
    trials.append(trial)
    if on_trial is not None:
        on_trial(trial)
