"""Deterministic stand-in for an LLM. Reads the <context> JSON and applies fixed heuristics.

Used in CI and as the offline default so the whole loop runs without any API key.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.core.types import KnobValue
from infervolt.llm.base import (
    DiagnosisOut,
    InsightOut,
    NarrativeOut,
    PriorOut,
    SearchPlanOut,
    extract_context,
)

T = TypeVar("T", bound=BaseModel)

PRIORS: dict[str, list[tuple[dict[str, KnobValue], str]]] = {
    "kv_capacity": [
        ({"kv_cache_dtype": "fp8"}, "FP8 KV halves bytes per token, doubling KV capacity"),
        ({"gpu_memory_utilization": 0.95}, "Give the KV cache more of the GPU memory"),
        ({"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.95}, "Both KV levers together"),
    ],
    "decode_bandwidth": [
        (
            {"speculative": "ngram"},
            "N-gram speculation amortizes weight reads over several tokens",
        ),
        ({"speculative": "eagle3"}, "EAGLE-3 draft head gives higher acceptance than n-gram"),
        ({"kv_cache_dtype": "fp8"}, "FP8 KV reduces bytes streamed per decode step"),
    ],
    "prefill_compute": [
        ({"quantization": "fp8"}, "FP8 GEMMs double prefill throughput on Hopper/Ada"),
        (
            {"max_num_batched_tokens": 8192},
            "Larger prefill chunks cut per-chunk scheduling overhead",
        ),
    ],
    "scheduler_cpu": [
        ({"enforce_eager": False}, "CUDA graphs remove per-step launch overhead"),
        (
            {"enforce_eager": False, "max_num_seqs": 64},
            "Graphs plus a smaller batch cap for lower scheduling cost",
        ),
    ],
}


class FakeLLMClient:
    model_id = "fake"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        ctx = extract_context(user)
        if schema is DiagnosisOut:
            return schema.model_validate(self._diagnose(ctx).model_dump())
        if schema is SearchPlanOut:
            return schema.model_validate(self._plan(ctx).model_dump())
        if schema is NarrativeOut:
            return schema.model_validate(self._narrate(ctx).model_dump())
        raise TypeError(f"FakeLLMClient cannot produce {schema.__name__}")

    def _diagnose(self, ctx: dict[str, Any]) -> DiagnosisOut:
        findings = ctx["findings"]
        top = findings[0]
        return DiagnosisOut(
            primary_rule_id=top["rule_id"],
            ranked_rule_ids=[f["rule_id"] for f in findings],
            rationale=f"Rule {top['rule_id']} ({top['bottleneck']}) has the highest evidence score "
            f"{top['score']}: {top['summary']}",
            confidence=min(0.95, 0.5 + float(top["score"]) / 2),
            caveats=["fake-llm: ranking follows rule scores"],
        )

    def _plan(self, ctx: dict[str, Any]) -> SearchPlanOut:
        primary = ctx["diagnosis"]["primary"]
        names = {k["name"] for k in ctx["knob_space"]}
        current = ctx.get("current", {})
        priors = []
        for knobs, hyp in PRIORS.get(primary, []):
            kept = {k: v for k, v in knobs.items() if k in names and current.get(k) != v}
            if kept:
                priors.append(PriorOut(knobs=kept, hypothesis=hyp))
        return SearchPlanOut(
            subspaces=list(ctx["diagnosis"]["subspaces"]),
            priors=priors[:4],
            max_trials=int(ctx["budget"]["max_trials"]),
            rationale=f"fake-llm: search the {primary} sub-space",
        )

    def _narrate(self, ctx: dict[str, Any]) -> NarrativeOut:
        d, base, best = ctx["diagnosis"], ctx["baseline_metrics"], ctx["best_metrics"]
        knobs = ", ".join(f"{k}={v}" for k, v in ctx["winning_knobs"].items())
        g0, g1 = float(base.get("goodput_rps", 0)), float(best.get("goodput_rps", 0))
        pct = (g1 - g0) / g0 * 100 if g0 else 0.0
        return NarrativeOut(
            rationale=f"Primary bottleneck {d['primary']}: {d.get('rationale', '')} "
            f"Changing {knobs} raised goodput from {g0:.3f} to {g1:.3f} rps ({pct:+.0f}%).",
            next_steps=[
                "Re-run diagnosis on the tuned config; the next bottleneck may differ.",
                "Validate on the real engine and hardware before deploying.",
            ],
            insights=[
                InsightOut(
                    text=f"For {d['primary']}, {knobs} helped.",
                    cites=list(ctx.get("trial_ids", [])),
                )
            ],
        )
