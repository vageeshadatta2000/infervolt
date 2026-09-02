"""LLM ranks and explains the rule findings, with the rule order as the fallback.

The rules decide *what fired*; the model only decides which of the things that fired
matters most and says why. That split is what keeps a hallucination cheap: a reply
naming a rule id the context never contained is discarded and the deterministic order
stands, so the worst an unavailable or confused model can do is cost the run its
narrative -- never its diagnosis.
"""

from __future__ import annotations

from typing import Any

from infervolt.core.types import Diagnosis, Finding, KnobValue, Observation, RunContext
from infervolt.llm.base import SYSTEM_PROMPT, DiagnosisOut, LLMClient, LLMError, render_prompt

ATTEMPTS = 2
"""Tries given to the model before falling back. A schema-valid but hallucinating reply
is usually a sampling accident, and a second draw is cheaper than losing the rationale."""

FALLBACK_CAVEAT = "LLM ranking unavailable or invalid; using rule order"


def rank(
    llm: LLMClient,
    findings: list[Finding],
    ctx: RunContext,
    obs: list[Observation],
    knobs: dict[str, KnobValue],
) -> Diagnosis:
    """Turn scored findings into a diagnosis, asking ``llm`` to rank and explain them."""
    if not findings:
        return Diagnosis(
            primary="under_loaded",
            ranked=[],
            rationale="No rule fired; the server was not saturated.",
            confidence=0.0,
            subspaces=[],
        )
    context: dict[str, Any] = {
        "findings": [f.model_dump() for f in findings],
        "workload": ctx.workload.model_dump(),
        "slo": ctx.slo.model_dump(),
        "hardware": ctx.hw.model_dump(),
        "model": ctx.model.model_dump(),
        "config": knobs,
        "metrics": [
            {
                "concurrency": o.load_point,
                "valid": o.valid,
                **o.metrics.model_dump(),
                "engine": o.engine,
                "gpu": o.gpu,
            }
            for o in obs
        ],
    }
    ids = {f.rule_id for f in findings}
    out: DiagnosisOut | None = None
    for _ in range(ATTEMPTS):
        try:
            cand = llm.structured(
                system=SYSTEM_PROMPT, user=render_prompt("rank", context), schema=DiagnosisOut
            )
        except LLMError:
            continue
        # The model may only rank rules it was shown. Anything else is a hallucination,
        # however confident, and the whole reply goes with it.
        if cand.primary_rule_id in ids and set(cand.ranked_rule_ids) <= ids:
            out = cand
            break
    by_id = {f.rule_id: f for f in findings}
    if out is None:
        top = findings[0]
        return Diagnosis(
            primary=top.bottleneck,
            ranked=findings,
            rationale=top.summary,
            confidence=top.score,
            subspaces=top.subspaces,
            caveats=[FALLBACK_CAVEAT],
        )
    # A partial ranking is honoured for the part it covers; findings the model left out
    # keep their rule order behind it rather than disappearing from the report. A repeated
    # id is listed once -- the report is a ranking, not a transcript of the reply.
    listed: list[str] = []
    for rule_id in out.ranked_rule_ids:
        if rule_id not in listed:
            listed.append(rule_id)
    ranked = [by_id[r] for r in listed]
    ranked += [f for f in findings if f.rule_id not in listed]
    primary = by_id[out.primary_rule_id]
    return Diagnosis(
        primary=primary.bottleneck,
        ranked=ranked,
        rationale=out.rationale,
        confidence=out.confidence,
        subspaces=primary.subspaces,
        caveats=out.caveats,
    )
