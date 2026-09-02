"""Deterministic bottleneck rules.

Each rule scores a fraction of weighted sub-conditions in [0, 1] and attaches the
evidence it used. Findings below :data:`MIN_SCORE` are dropped. Sorting is by score,
then by ``BOTTLENECK_PRIORITY`` (capacity problems cap goodput before bandwidth does).

Rules read only the canonical engine keys, never engine-specific ones, so the same
rule set applies to any adapter that fills them in. Adapters are not required to fill
in *every* key, so a counter the engine never reported must never be the thing that
makes a condition fire: every threshold comparison against ``Observation.engine`` or
``Observation.gpu`` goes through :func:`_below` / :func:`_atleast`, which are False on
a missing key, and the few cross-key comparisons pick per-key defaults that land on the
same side. Comparisons against ``Observation.metrics`` and the roofline need no such
care -- those values are always computed, never scraped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from infervolt.core.types import (
    BOTTLENECK_PRIORITY,
    EngineConfig,
    Evidence,
    Finding,
    KnobSpace,
    Observation,
    RunContext,
)
from infervolt.hardware import roofline

MIN_SCORE = 0.3

# ``preemptions_per_s`` is a non-negative rate, so "any preemption at all" is a
# threshold like the others rather than a special case.
ANY_RATE = 1e-9


@dataclass
class RuleInput:
    """The observations one candidate produced, plus everything a rule needs to read them.

    ``top``, ``first`` and ``best_slo`` select over :attr:`valid` and therefore raise on a
    list with no valid observation. That is deliberate: they are only ever called from
    rules, and :func:`evaluate_rules` guards every rule but ``r6_client_artifact`` behind
    a non-empty :attr:`valid`, so a rule that reaches a selector always has one.
    """

    obs: list[Observation]
    ctx: RunContext
    cfg: EngineConfig
    space: KnobSpace

    @property
    def valid(self) -> list[Observation]:
        return [o for o in self.obs if o.valid]

    @property
    def top(self) -> Observation:
        """Highest load point that was measured cleanly."""
        return max(self.valid, key=lambda o: o.load_point)

    @property
    def first(self) -> Observation:
        """Lowest load point that was measured cleanly."""
        return min(self.valid, key=lambda o: o.load_point)

    @property
    def best_slo(self) -> Observation:
        """The load point the candidate is actually scored on."""
        return max(self.valid, key=lambda o: o.metrics.goodput_rps)


def _below(table: dict[str, float], key: str, threshold: float) -> bool:
    """``table[key] < threshold``, and False when the engine did not report ``key``."""
    return key in table and table[key] < threshold


def _atleast(table: dict[str, float], key: str, threshold: float) -> bool:
    """``table[key] >= threshold``, and False when the engine did not report ``key``."""
    return key in table and table[key] >= threshold


def _ev(o: Observation, key: str, source: str = "engine", unit: str = "") -> Evidence:
    """One scraped counter as evidence, stamped with the load point it was read at.

    A key the adapter never filled in is reported as such rather than as a measured
    zero -- "not reported" and "reported as 0.0" mean very different things to a reader.
    """
    table: dict[str, float] = o.engine if source == "engine" else o.gpu
    present = key in table
    return Evidence(
        source=source,
        key=f"{key}@c{o.load_point}",
        value=float(table[key]) if present else 0.0,
        unit=unit,
        note="" if present else "not reported",
    )


def _weight(conds: list[tuple[bool, float]]) -> float:
    """Unrounded sum of the weights whose condition fired."""
    return sum(w for ok, w in conds if ok)


def _score(conds: list[tuple[bool, float]]) -> float:
    return round(_weight(conds), 3)


def _subspaces(space: KnobSpace, *groups: str) -> list[str]:
    return [g for g in groups if g in space.groups()]


# ---------------------------------------------------------------- rules


def r0_under_loaded(x: RuleInput) -> Finding | None:
    """Nothing was wrong; the sweep simply never pushed hard enough to find out.

    This is a claim about the *experiment*, not the server, so it is gated rather than
    scored: any sign that the run actually hit a limit -- a queue, a missed SLO, a sweep
    that stopped before the last requested load point -- disqualifies it outright, and a
    missing ``num_waiting`` counts as a queue rather than as an empty one.
    """
    t = x.top
    e, g, m = t.engine, t.gpu, t.metrics
    requested = x.ctx.workload.load.concurrency
    if (
        not requested
        # Missing counter defaults to "there was a queue", which disqualifies.
        or e.get("num_waiting", 1.0) >= 0.5
        or m.goodput_frac < x.ctx.slo.goodput_target
        or t.load_point < max(requested)
    ):
        return None
    # Defaults are picked so an unreported counter never makes a condition fire: idle-GPU
    # tests default to "busy" (1.0), and the scheduler cap defaults to 0 so an engine that
    # never told us its ``max_num_seqs`` cannot be shown to be running below it.
    conds = [
        (_below(g, "sm_active", 0.3), 0.4),
        (_below(g, "dram_active", 0.3), 0.3),
        (e.get("num_running", 1.0) < 0.5 * e.get("max_num_seqs", 0.0), 0.3),
    ]
    return Finding(
        rule_id="R0",
        bottleneck="under_loaded",
        score=_score(conds),
        evidence=[
            _ev(t, "num_waiting"),
            _ev(t, "num_running"),
            _ev(t, "sm_active", "gpu"),
            _ev(t, "dram_active", "gpu"),
        ],
        subspaces=[],
        summary="Server is not saturated at the highest load point; extend the sweep.",
    )


def r1_kv_capacity(x: RuleInput) -> Finding | None:
    worst = max(x.valid, key=lambda o: o.engine.get("kv_usage_p95", 0.0))
    e = worst.engine
    conds = [
        (any(_atleast(o.engine, "preemptions_per_s", ANY_RATE) for o in x.valid), 0.5),
        (_atleast(e, "kv_usage_p95", 0.9) and _atleast(e, "num_waiting", 1.0), 0.3),
        # Queueing below the scheduler's own cap: seats are free but there is no KV for
        # them. An unreported cap defaults to 0, which no running count is below.
        (
            e.get("num_running", 0.0) < e.get("max_num_seqs", 0.0)
            and _atleast(e, "num_waiting", 1.0),
            0.2,
        ),
    ]
    return Finding(
        rule_id="R1",
        bottleneck="kv_capacity",
        score=_score(conds),
        evidence=[
            _ev(worst, "preemptions_per_s", unit="/s"),
            _ev(worst, "kv_usage_p95"),
            _ev(worst, "num_waiting"),
            _ev(worst, "num_running"),
        ],
        subspaces=_subspaces(x.space, "kv"),
        summary=(
            "KV cache is exhausted: requests queue or get preempted before max_num_seqs is reached."
        ),
    )


def r2_decode_bandwidth(x: RuleInput) -> Finding | None:
    o = x.best_slo
    n = max(1, int(o.engine.get("num_running", 1)))
    w = x.ctx.workload
    floor = roofline.decode_step_floor_s(
        x.ctx.hw, x.ctx.model, n, w.isl.p50 + w.osl.p50 // 2, int(o.engine.get("kv_dtype_bytes", 2))
    )
    ratio = (o.metrics.itl_p50_ms / 1000) / floor if floor > 0 else 99.0
    top, first = x.top, x.first
    c_ratio = top.load_point / max(first.load_point, 1)
    itl_ratio = top.metrics.itl_p50_ms / max(first.metrics.itl_p50_ms, 1e-6)
    # Memory-pipe dominance, not idleness: a decode-bound step can still keep the SMs
    # moderately busy, so what marks it is that the DRAM pipe is both hot *and* hotter
    # than the compute pipe. Defaults put an unreported counter on the losing side.
    dram = o.gpu.get("dram_active", 0.0)
    sm = o.gpu.get("sm_active", 1.0)
    conds = [
        (ratio <= 1.3, 0.5),
        (_atleast(o.gpu, "dram_active", 0.6) and dram > sm, 0.3),
        (c_ratio > 1 and itl_ratio < 0.5 * c_ratio, 0.2),
    ]
    score = _weight(conds)
    if any(_atleast(ob.engine, "kv_usage_p95", 0.9) for ob in x.valid):
        # A full KV cache explains the same symptoms more directly; defer to R1.
        score *= 0.7
    return Finding(
        rule_id="R2",
        bottleneck="decode_bandwidth",
        score=round(score, 3),
        evidence=[
            Evidence(
                source="roofline", key=f"itl_over_floor@c{o.load_point}", value=round(ratio, 3)
            ),
            _ev(o, "dram_active", "gpu"),
            _ev(o, "sm_active", "gpu"),
        ],
        subspaces=_subspaces(x.space, "decode"),
        summary=(
            "Decode runs at the HBM-bandwidth floor: fewer bytes per step "
            "(spec decode, FP8 KV, quantization) is the lever."
        ),
    )


def r3_prefill_compute(x: RuleInput) -> Finding | None:
    top, first = x.top, x.first
    c_ratio = top.load_point / max(first.load_point, 1)
    ttft_ratio = top.metrics.ttft_p90_ms / max(first.metrics.ttft_p90_ms, 1e-6)
    grows = c_ratio > 1 and ttft_ratio >= 0.5 * c_ratio
    conds = [
        (grows, 0.4),
        (_atleast(top.engine, "prefill_share", 0.5), 0.4),
        (_atleast(top.gpu, "sm_active", 0.7), 0.2),
    ]
    # The share/utilisation half of this rule can carry the finding on its own, so the
    # summary only claims TTFT growth when the condition that measures it actually fired.
    summary = "Prefill compute dominates: the GPU is busy on prompt tokens"
    if grows:
        summary += " and TTFT grows with concurrency"
    return Finding(
        rule_id="R3",
        bottleneck="prefill_compute",
        score=_score(conds),
        evidence=[
            Evidence(
                source="loadgen",
                key=f"ttft_p90_growth@c{top.load_point}",
                value=round(ttft_ratio, 3),
                note="" if grows else "did not fire",
            ),
            _ev(top, "prefill_share"),
            _ev(top, "sm_active", "gpu"),
        ],
        subspaces=_subspaces(x.space, "prefill"),
        summary=summary + ".",
    )


def r4_scheduler_cpu(x: RuleInput) -> Finding | None:
    low = [o for o in x.valid if o.load_point <= 8]
    if len(low) < 2:
        return None
    o1, o8 = low[0], low[-1]
    w = x.ctx.workload
    ctx_tokens = w.isl.p50 + w.osl.p50 // 2
    floor1 = roofline.decode_step_floor_s(
        x.ctx.hw, x.ctx.model, 1, ctx_tokens, int(o1.engine.get("kv_dtype_bytes", 2))
    )
    drift = abs(o8.metrics.itl_p50_ms - o1.metrics.itl_p50_ms) / max(o1.metrics.itl_p50_ms, 1e-6)
    flat = drift <= 0.15
    conds = [
        (flat, 0.4),
        (_below(o8.gpu, "sm_active", 0.4) and _below(o8.gpu, "dram_active", 0.4), 0.3),
        (o1.metrics.itl_p50_ms / 1000 > 2 * floor1, 0.3),
    ]
    return Finding(
        rule_id="R4",
        bottleneck="scheduler_cpu",
        score=_score(conds),
        evidence=[
            Evidence(
                source="loadgen",
                key=f"itl_p50_ms@c{o1.load_point}",
                value=round(o1.metrics.itl_p50_ms, 3),
                unit="ms",
            ),
            Evidence(
                source="loadgen",
                key=f"itl_p50_ms@c{o8.load_point}",
                value=round(o8.metrics.itl_p50_ms, 3),
                unit="ms",
            ),
            _ev(o8, "sm_active", "gpu"),
            _ev(o8, "dram_active", "gpu"),
        ],
        subspaces=_subspaces(x.space, "sched"),
        summary=(
            "Per-step overhead dominates: ITL is flat across low concurrency while the GPU idles."
        ),
    )


def r5_communication(x: RuleInput) -> Finding | None:
    tp = int(x.cfg.knobs.get("tensor_parallel_size", 1))
    # Its one condition is worth exactly MIN_SCORE, so R5 either scores 0.3 or is dropped:
    # it can tie another finding but never outrank one. That is the point -- the topology
    # is suggestive, and only a profile can promote it to a real diagnosis.
    conds = [(tp > 1 and x.ctx.hw.interconnect == "pcie", MIN_SCORE)]
    return Finding(
        rule_id="R5",
        bottleneck="communication",
        score=_score(conds),
        evidence=[Evidence(source="static", key="tensor_parallel_size", value=float(tp))],
        subspaces=_subspaces(x.space, "parallel"),
        summary=(
            "Tensor parallel over PCIe; all-reduce likely dominates "
            "(low confidence without a profile)."
        ),
    )


def r6_client_artifact(x: RuleInput) -> Finding | None:
    bad = [o for o in x.obs if not o.valid]
    if not bad:
        return None
    return Finding(
        rule_id="R6",
        bottleneck="client_artifact",
        score=1.0,
        evidence=[
            Evidence(
                source="loadgen", key=f"invalid@c{o.load_point}", value=1.0, note=o.invalid_reason
            )
            for o in bad
        ],
        subspaces=[],
        summary="Load generator was the bottleneck; measurements at those load points are invalid.",
    )


RULES: list[Callable[[RuleInput], Finding | None]] = [
    r6_client_artifact,
    r1_kv_capacity,
    r3_prefill_compute,
    r2_decode_bandwidth,
    r4_scheduler_cpu,
    r5_communication,
    r0_under_loaded,
]


def evaluate_rules(
    obs: list[Observation], ctx: RunContext, cfg: EngineConfig, space: KnobSpace
) -> list[Finding]:
    """Run every rule and return the findings worth showing, most convincing first.

    Rules other than R6 read at least one valid observation, so when the sweep produced
    nothing usable only R6 runs; with no observations at all there is nothing to say.
    """
    if not obs:
        return []
    x = RuleInput(obs=obs, ctx=ctx, cfg=cfg, space=space)
    has_valid = bool(x.valid)
    findings: list[Finding] = []
    for rule in RULES:
        if not has_valid and rule is not r6_client_artifact:
            continue
        f = rule(x)
        if f is not None and f.score >= MIN_SCORE:
            findings.append(f)
    findings.sort(key=lambda f: (-f.score, BOTTLENECK_PRIORITY[f.bottleneck]))
    return findings
