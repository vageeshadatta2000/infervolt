"""The optimize loop as an explicit state machine with ledger checkpoints.

``prepare -> baseline -> diagnose -> plan -> search -> verify -> emit -> learn -> done``

Every transition is written to the ledger before the work it names, so a run that dies
leaves a row saying what it was doing. Two of the states can end the run early without
failing it: a diagnosis with nothing tunable in it, and a search or verification that
found no win. Both are legitimate answers -- "nothing to change here" is a result -- and
both produce a report rather than a recipe.
"""

from __future__ import annotations

import contextlib
import json
import math
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infervolt import __version__
from infervolt.agent.budget import BudgetTracker
from infervolt.config import Settings
from infervolt.core.types import (
    Candidate,
    Diagnosis,
    EngineConfig,
    Evidence,
    KnobSpace,
    KnobValue,
    OptimizeSpec,
    RunContext,
    RunOutcome,
    RunState,
    SearchPlan,
    Trial,
    TrialStatus,
)
from infervolt.diagnose.ranker import rank
from infervolt.diagnose.rules import evaluate_rules
from infervolt.engines.base import EngineAdapter
from infervolt.engines.registry import get_adapter
from infervolt.hardware.profiles import get_profile
from infervolt.llm.base import (
    SYSTEM_PROMPT,
    LLMClient,
    NarrativeOut,
    SearchPlanOut,
    prompts_sha,
    render_prompt,
)
from infervolt.models.catalog import get_model_info
from infervolt.recipes.emit import write_recipe, write_report
from infervolt.recipes.schema import (
    Recipe,
    RecipeDiagnosis,
    RecipeDist,
    RecipeEngine,
    RecipeFinding,
    RecipeHardware,
    RecipeInfervolt,
    RecipeMeasured,
    RecipeModel,
    RecipeProvenance,
    RecipeQuality,
    RecipeResult,
    RecipeSearch,
    RecipeServe,
    RecipeSLO,
    RecipeWorkload,
)
from infervolt.runner.trial import run_candidate
from infervolt.search.optuna_search import STAGE2_REQUESTS, run_search
from infervolt.search.space import Bounds, clamp
from infervolt.store.ledger import Ledger
from infervolt.verify.quality import MockQualityGuard, QualityGuard
from infervolt.verify.verify import VerifyResult, verify
from infervolt.workloads.presets import get_workload

REPORT_METRICS = [
    "goodput_rps",
    "goodput_frac",
    "req_per_s",
    "output_tps",
    "ttft_p90_ms",
    "itl_p90_ms",
    "usd_per_m_tokens",
]
"""The metrics a recipe reports before and after. Deliberately short: throughput under
the SLO, the raw rates behind it, the two latencies the SLO is written in, and cost."""

MAX_PRIORS = 4
"""Prior candidates accepted from the planning call. Enough for the model to express a
hypothesis and a couple of variants, few enough that TPE still gets most of the budget."""

TARGET_FRACTION = 0.95
"""Share of the final best objective that counts as "reached the target", for
``trials_to_target``."""

INFEASIBLE_STATUSES: tuple[TrialStatus, ...] = ("infeasible_oom", "crash", "rejected", "timeout")
"""Trial outcomes the recipe counts as "the config could not be measured".

A ``timeout`` belongs here with the OOMs and the crashes: a server that never came up, or
a load point that never finished, produced no objective, and counting it as a candidate
that merely lost would understate how much of the space this hardware refuses."""


class Planner:
    """Runs one optimization from spec to recipe."""

    def __init__(
        self,
        spec: OptimizeSpec,
        settings: Settings,
        llm: LLMClient,
        ledger: Ledger,
        adapter: EngineAdapter | None = None,
        guard: QualityGuard | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.spec = spec
        self.settings = settings
        self.llm = llm
        self.ledger = ledger
        self.log = log
        self.adapter = adapter or get_adapter(spec.engine)
        self.guard = guard or MockQualityGuard()

    # ---- entry point
    def run(self) -> RunOutcome:
        """Run the loop. Always returns; never raises."""
        run_id = self.ledger.create_run(self.spec)
        try:
            return self._run(run_id)
        except Exception as e:  # noqa: BLE001 - the run must always end in a terminal state
            # The ledger write is itself best-effort: a failing database must not replace
            # the exception that actually ended the run with one about bookkeeping.
            with contextlib.suppress(Exception):
                self.ledger.set_state(run_id, "failed")
            self.log(f"failed: {type(e).__name__}: {e}")
            return RunOutcome(run_id=run_id, state="failed", message=f"{type(e).__name__}: {e}")

    def _run(self, run_id: str) -> RunOutcome:
        ctx = self._prepare(run_id)
        space = self.adapter.knob_space(ctx)
        tracker = BudgetTracker(self.spec.budget)
        self.log(f"run: {run_id}")

        if errs := _baseline_errors(self.spec.baseline, space):
            return self._fail(run_id, f"invalid --baseline: {'; '.join(errs)}")

        self._state(run_id, "baseline")
        baseline = self._baseline(ctx, space)
        if baseline.status != "ok" or baseline.result is None:
            return self._fail(
                run_id, f"baseline failed: {baseline.status} {baseline.log_tail[:200]}"
            )
        self.log(
            f"baseline goodput {baseline.result.objective:.3f} rps "
            f"at c={baseline.result.best_load_point}"
        )

        self._state(run_id, "diagnose")
        findings = evaluate_rules(
            baseline.result.observations, ctx, baseline.candidate.config, space
        )
        diagnosis = rank(
            self.llm,
            findings,
            ctx,
            baseline.result.observations,
            baseline.candidate.config.knobs,
            log=self.log,
        )
        self.ledger.set_diagnosis(run_id, diagnosis.model_dump_json())
        self.log(
            f"primary bottleneck: {diagnosis.primary} (confidence {diagnosis.confidence:.2f}); "
            "findings: " + ", ".join(f"{f.rule_id}={f.score:.2f}" for f in diagnosis.ranked)
        )
        if diagnosis.primary in ("client_artifact", "under_loaded") or not diagnosis.subspaces:
            return self._finish_without_change(
                run_id, ctx, baseline, diagnosis, "no tunable bottleneck identified"
            )

        self._state(run_id, "plan")
        plan = self._plan(ctx, space, diagnosis, baseline.candidate.config)
        self.log(
            f"search plan: subspaces={plan.subspaces} priors={len(plan.priors)} "
            f"max_trials={plan.max_trials}"
        )

        self._state(run_id, "search")
        trials = run_search(
            self.adapter,
            ctx,
            space,
            plan,
            baseline,
            self.ledger,
            self.spec.budget,
            self.spec.seed,
            deadline=tracker.deadline,
            on_trial=lambda t: self._on_trial(t, tracker),
        )
        ok = [t for t in trials if t.status == "ok" and t.result is not None]
        if not ok:
            return self._finish_without_change(
                run_id, ctx, baseline, diagnosis, "no feasible candidate improved on baseline"
            )
        best = max(ok, key=_objective)
        if best.result is None or best.result.objective <= baseline.result.objective:
            return self._finish_without_change(
                run_id, ctx, baseline, diagnosis, "search found nothing better than baseline"
            )
        # Verification is several more launches -- the most expensive stage in the loop --
        # so the budget is re-checked here rather than only inside the search. The trial
        # counter is deliberately not consulted: a search that spent every trial did its
        # job, and refusing to verify its winner would throw the run away at the end.
        if why := tracker.exhausted(count_trials=False):
            return self._finish_without_change(
                run_id, ctx, baseline, diagnosis, f"stopped before verification: {why}"
            )

        self._state(run_id, "verify")
        v = verify(self.adapter, ctx, baseline, best, self.guard)
        self.log(
            f"verify: {'ACCEPTED' if v.accepted else 'rejected'} {_improvement_text(v)} {v.reason}"
        )
        if not v.accepted:
            return self._finish_without_change(
                run_id, ctx, baseline, diagnosis, f"verify rejected best trial: {v.reason}"
            )
        self.ledger.set_best(run_id, best.id)

        self._state(run_id, "emit")
        recipe = self._recipe(ctx, space, baseline, best, trials, diagnosis, plan, v)
        out_dir = Path(ctx.run_dir)
        recipe_path, report_path = write_recipe(recipe, out_dir), write_report(recipe, out_dir)
        self.ledger.set_recipe(run_id, str(recipe_path))
        self.log(f"recipe: {recipe_path}\nreport: {report_path}")

        # M4: this is where a run's insights are folded into cross-run memory.
        self._state(run_id, "learn")
        ttt = trials_to_target(trials, best.result.objective)
        self._state(run_id, "done")
        return RunOutcome(
            run_id=run_id,
            state="done",
            baseline_trial_id=baseline.id,
            best_trial_id=best.id,
            diagnosis=diagnosis,
            recipe_path=str(recipe_path),
            report_path=str(report_path),
            trials_to_target=ttt,
            improvement_pct=v.improvement_pct,
            accepted=True,
            message="ok",
        )

    # ---- states
    def _prepare(self, run_id: str) -> RunContext:
        if self.spec.hardware == "auto":
            raise ValueError("hardware auto-detection arrives in M2; pass --hardware <profile>")
        return RunContext(
            run_id=run_id,
            run_dir=str(self.ledger.run_dir(run_id)),
            hw=get_profile(self.spec.hardware),
            model=get_model_info(self.spec.model),
            workload=get_workload(self.spec.workload),
            slo=self.spec.slo,
            seed=self.spec.seed,
        )

    def _baseline(self, ctx: RunContext, space: KnobSpace) -> Trial:
        """Measure the config the user is running today, as a full sweep."""
        cfg = EngineConfig(
            engine=self.spec.engine, knobs={**space.defaults(), **self.spec.baseline}
        )
        trial = Trial(
            id="t0",
            run_id=ctx.run_id,
            index=0,
            candidate=Candidate(id="c0", config=cfg, origin="baseline"),
            stage=2,
        )
        trial = run_candidate(
            self.adapter, trial, ctx, ctx.workload.load.concurrency, STAGE2_REQUESTS
        )
        self.ledger.save_trial(trial)
        return trial

    def _plan(
        self, ctx: RunContext, space: KnobSpace, diagnosis: Diagnosis, base_cfg: EngineConfig
    ) -> SearchPlan:
        """Ask the model which sub-space to search and what to try first.

        Everything it proposes is filtered against what actually exists: sub-spaces it
        did not invent, knobs the engine offers, values inside the knob's own range, and
        finally the adapter's static validation on this hardware. A plan that survives
        none of that degrades to the diagnosis's own sub-spaces with no priors.
        """
        context: dict[str, Any] = {
            "diagnosis": diagnosis.model_dump(),
            "knob_space": [k.model_dump() for k in space.knobs],
            "current": base_cfg.knobs,
            "budget": self.spec.budget.model_dump(),
            "priors": [],
            "notes": [],
            "hardware": ctx.hw.model_dump(),
            "workload": ctx.workload.model_dump(),
        }
        try:
            out = self.llm.structured(
                system=SYSTEM_PROMPT, user=render_prompt("plan", context), schema=SearchPlanOut
            )
        except Exception as e:  # noqa: BLE001 - any client failure degrades to no priors
            self.log(
                f"plan: llm error: {type(e).__name__}: {e}; "
                "searching the diagnosis sub-spaces without priors"
            )
            out = SearchPlanOut(
                subspaces=diagnosis.subspaces, max_trials=self.spec.budget.max_trials
            )
        groups = set(space.groups())
        subspaces = [g for g in out.subspaces if g in groups] or diagnosis.subspaces
        names = set(space.names())
        bounds = Bounds(space)
        priors: list[Candidate] = []
        for i, p in enumerate(out.priors[:MAX_PRIORS]):
            unknown = set(p.knobs) - names
            if unknown:
                self.log(f"plan: dropping unknown knobs {sorted(unknown)} from prior {i}")
            knobs = clamp({k: v for k, v in p.knobs.items() if k in names}, space, bounds)
            off_menu = {k: v for k, v in knobs.items() if not _in_choices(space, k, v)}
            if off_menu:
                # Clamping cannot rescue a categorical: there is no nearest legal value to
                # move to, only a list the value is not on. Say so rather than dropping it
                # silently, so a plan that half survived does not read like one that fit.
                self.log(f"plan: dropping out-of-choices knobs {off_menu} from prior {i}")
            knobs = {k: v for k, v in knobs.items() if k not in off_menu}
            if not knobs:
                continue
            cfg = base_cfg.with_knobs(**knobs)
            if errs := self.adapter.validate(cfg, ctx):
                self.log(f"plan: prior {i} rejected statically: {errs}")
                continue
            priors.append(
                Candidate(
                    id=f"p{i}",
                    config=cfg,
                    origin="llm_prior",
                    hypothesis=p.hypothesis,
                    parent_id="c0",
                )
            )
        return SearchPlan(
            subspaces=subspaces,
            priors=priors,
            max_trials=max(1, min(out.max_trials, self.spec.budget.max_trials)),
            rationale=out.rationale,
        )

    def _on_trial(self, t: Trial, tracker: BudgetTracker) -> None:
        tracker.charge(t.cost_usd)
        obj = f"{t.result.objective:.3f}" if t.result else "-"
        self.log(
            f"trial {t.id} [{t.candidate.origin}] {t.status} stage={t.stage} "
            f"objective={obj} {t.candidate.hypothesis}"
        )

    def _recipe(
        self,
        ctx: RunContext,
        space: KnobSpace,
        baseline: Trial,
        best: Trial,
        trials: list[Trial],
        diagnosis: Diagnosis,
        plan: SearchPlan,
        v: VerifyResult,
    ) -> Recipe:
        assert baseline.result is not None and best.result is not None
        base_c = baseline.result.best_load_point
        b_obs = next(o for o in baseline.result.observations if o.load_point == base_c)
        c_obs = next(
            o for o in best.result.observations if o.load_point == best.result.best_load_point
        )
        b_metrics = {k: round(getattr(b_obs.metrics, k), 4) for k in REPORT_METRICS}
        c_metrics = {k: round(getattr(c_obs.metrics, k), 4) for k in REPORT_METRICS}
        winning = {
            k: val
            for k, val in best.candidate.config.knobs.items()
            if baseline.candidate.config.knobs.get(k) != val
        }
        narrative = self._narrative(diagnosis, v, winning, [t.id for t in trials])
        args, command = self.adapter.to_recipe_block(best.candidate.config, ctx)
        ver = self.adapter.version()
        w = ctx.workload
        return Recipe(
            model=RecipeModel(
                id=ctx.model.id, params_b=ctx.model.params_b, arch=ctx.model.arch, moe=ctx.model.moe
            ),
            hardware=RecipeHardware(
                gpu=ctx.hw.gpu,
                count=ctx.hw.count,
                topology=ctx.hw.interconnect,
                provider=ctx.hw.name,
            ),
            engine=RecipeEngine(
                name=ver.name, version=ver.version, image=ver.image_digest, commit=ver.commit
            ),
            workload=RecipeWorkload(
                name=w.name,
                isl=RecipeDist(p50=w.isl.p50, p99=w.isl.p99),
                osl=RecipeDist(p50=w.osl.p50, p99=w.osl.p99),
                prefix_share=w.prefix_share,
                load={"mode": "sweep", "concurrency": w.load.concurrency},
            ),
            slo=RecipeSLO(**ctx.slo.model_dump()),
            serve=RecipeServe(args=args, command=command),
            baseline=RecipeMeasured(
                serve_args=dict(baseline.candidate.config.knobs),
                metrics=b_metrics,
                load_point=base_c,
            ),
            result=RecipeResult(
                metrics=c_metrics,
                load_point=best.result.best_load_point,
                repeats=v.repeats,
                improvement={"goodput_rps": _improvement_text(v)},
                quality=RecipeQuality(**v.quality.model_dump()) if v.quality else None,
            ),
            infervolt=RecipeInfervolt(
                run_id=ctx.run_id,
                diagnosis=RecipeDiagnosis(
                    primary=diagnosis.primary,
                    confidence=diagnosis.confidence,
                    findings=[
                        RecipeFinding(
                            rule=f.rule_id, score=f.score, evidence=[_round(e) for e in f.evidence]
                        )
                        for f in diagnosis.ranked
                    ],
                    caveats=diagnosis.caveats,
                ),
                rationale=narrative.rationale,
                search=RecipeSearch(
                    trials=len(trials),
                    infeasible=sum(t.status in INFEASIBLE_STATUSES for t in trials),
                    subspace=[k.name for k in space.subspace(plan.subspaces).knobs],
                    optimizer="optuna-tpe",
                    seed=self.spec.seed,
                ),
                trials_to_target=trials_to_target(trials, best.result.objective),
                next_steps=narrative.next_steps,
                artifacts={"report": "report.md", "trials": "trials.jsonl"},
                provenance=RecipeProvenance(
                    tool_version=__version__,
                    llm=self.llm.model_id,
                    prompts_sha=prompts_sha(),
                    created=datetime.now(UTC).strftime("%Y-%m-%d"),
                ),
            ),
        )

    def _narrative(
        self,
        diagnosis: Diagnosis,
        v: VerifyResult,
        winning: dict[str, KnobValue],
        trial_ids: list[str],
    ) -> NarrativeOut:
        """Ask the model for the prose, from the numbers the verification actually compared.

        The summary table reports each arm at its *own* best load point, which is usually
        not the same concurrency; the difference between those two columns is not a
        measured delta and any percentage taken from it is fiction. Verify drove both arms
        at one load point, so its arms are what the narrative gets -- along with
        ``comparable``, because a baseline that served nothing there has no rate to be a
        percentage of.
        """
        context: dict[str, Any] = {
            "diagnosis": diagnosis.model_dump(),
            "baseline_metrics": {"goodput_rps": round(v.baseline_mean, 4)},
            "best_metrics": {
                "goodput_rps": round(
                    statistics.fmean(v.candidate_goodput) if v.candidate_goodput else 0.0, 4
                )
            },
            "load_point": v.load_point,
            "comparable": v.comparable,
            "winning_knobs": winning,
            "trial_ids": trial_ids,
        }
        try:
            return self.llm.structured(
                system=SYSTEM_PROMPT, user=render_prompt("emit", context), schema=NarrativeOut
            )
        except Exception as e:  # noqa: BLE001 - any client failure degrades to a template
            # The measurements are the recipe; the prose is not. Losing the model here
            # costs a sentence, not the run -- whatever the client failed with.
            self.log(f"emit: llm error: {type(e).__name__}: {e}; using the template narrative")
            return NarrativeOut(
                rationale=f"{diagnosis.rationale} Winning knobs: {json.dumps(winning)}.",
                next_steps=["Re-run diagnosis on the tuned config."],
            )

    def _finish_without_change(
        self, run_id: str, ctx: RunContext, baseline: Trial, diagnosis: Diagnosis, why: str
    ) -> RunOutcome:
        """End a run that measured and diagnosed but has nothing to recommend."""
        report = Path(ctx.run_dir) / "report.md"
        report.write_text(
            f"# infervolt run {run_id}: no change recommended\n\n{why}\n\n"
            f"Primary bottleneck: {diagnosis.primary} (confidence {diagnosis.confidence:.2f})\n\n"
            f"{diagnosis.rationale}\n"
        )
        self._state(run_id, "done")
        self.log(f"no recipe: {why}\nreport: {report}")
        return RunOutcome(
            run_id=run_id,
            state="done",
            baseline_trial_id=baseline.id,
            diagnosis=diagnosis,
            report_path=str(report),
            accepted=False,
            message=why,
        )

    def _fail(self, run_id: str, msg: str) -> RunOutcome:
        self._state(run_id, "failed")
        self.log(f"failed: {msg}")
        return RunOutcome(run_id=run_id, state="failed", message=msg)

    def _state(self, run_id: str, state: RunState) -> None:
        self.ledger.set_state(run_id, state)


def _round(e: Evidence) -> Evidence:
    """One evidence value at four significant digits.

    Rules compute in floating point, so a ratio lands as ``1.9999999999999998`` as often
    as ``2.0``; four significant digits is more precision than any of these numbers earn
    and stops the recipe from implying otherwise. Significant digits rather than decimal
    places because the values span ``0.0001234`` (a fraction) to ``123400`` (a token rate).
    """
    return e.model_copy(update={"value": float(f"{e.value:.4g}")})


def _baseline_errors(baseline: dict[str, KnobValue], space: KnobSpace) -> list[str]:
    """Everything wrong with ``--baseline`` overrides, checked before anything is launched.

    A typo'd knob name is silently harmless today -- it lands in the config dict, the
    adapter ignores it, and the run measures the default instead while reporting the
    override. That is worse than failing: the recipe then answers a question nobody asked.
    """
    errors: list[str] = []
    names = space.names()
    for name, value in baseline.items():
        if name not in names:
            errors.append(f"unknown knob {name!r}; known knobs: {', '.join(sorted(names))}")
            continue
        knob = space.get(name)
        if knob.kind == "cat" and value not in knob.choices:
            errors.append(f"{name}={value!r} is not one of {knob.choices!r}")
    return errors


def _improvement_text(v: VerifyResult) -> str:
    """The headline win, as a percentage when the baseline had a rate to compare against.

    ``VerifyResult.comparable`` is verify's own answer to "was there a baseline rate to be
    a percentage of", so it is taken rather than re-derived here; see its field docs for
    why an arm can score zero. ``improvement_pct`` is infinite in exactly that case and is
    checked too: "+inf%" is not a number to put in front of anyone.
    """
    ci = f"95% CI {v.ci_low:+.3f}..{v.ci_high:+.3f} rps at c={v.load_point}"
    if math.isinf(v.improvement_pct) or not v.comparable:
        return (
            f"{v.delta_mean:+.3f} rps, from a baseline that served nothing "
            f"at c={v.load_point} ({ci})"
        )
    return f"{v.improvement_pct:+.0f}% ({ci})"


def _objective(trial: Trial) -> float:
    """A trial's objective, and ``-inf`` for one that measured nothing.

    Only ever used as a ``max`` key over trials already filtered to ``status == "ok"``
    with a result, so the fallback is unreachable; it exists so the key function is total.
    """
    return trial.result.objective if trial.result is not None else float("-inf")


def _in_choices(space: KnobSpace, name: str, value: KnobValue) -> bool:
    k = space.get(name)
    return value in k.choices if k.kind == "cat" else True


def trials_to_target(trials: list[Trial], final_best: float) -> int | None:
    """How many trials it took to get within :data:`TARGET_FRACTION` of the final best.

    A cheap search-efficiency number for the recipe: the same win found in three trials
    instead of twelve is the difference between a technique that is worth running and one
    that is not.
    """
    target = TARGET_FRACTION * final_best
    for n, t in enumerate(trials, start=1):
        if t.status == "ok" and t.result is not None and t.result.objective >= target:
            return n
    return None
