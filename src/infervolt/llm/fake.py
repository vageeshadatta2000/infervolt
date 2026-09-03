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
        """Turn the diagnosed bottleneck into prior candidates the search should try first.

        Filtering is about *relevance*, not feasibility: a prior survives here as long as
        it names knobs this engine offers and would actually change something. Whether the
        resulting config is legal on this hardware (fp8 quantization needs compute
        capability >= 8.9, say) is the planner's call -- it holds the RunContext and drops
        invalid priors before they reach the search. Duplicating that check here would put
        hardware rules in a client that only ever sees a JSON blob.
        """
        primary = ctx["diagnosis"]["primary"]
        names = {k["name"] for k in ctx["knob_space"]}
        current = ctx.get("current", {})
        priors: list[PriorOut] = []
        seen: set[tuple[tuple[str, KnobValue], ...]] = set()
        for knobs, hyp in PRIORS.get(primary, []):
            # A knob the engine does not offer is simply unknown here, and dropping it
            # leaves the rest of the hypothesis intact. A knob already at the proposed
            # value is different: the hypothesis is about *changing* it, so with that
            # change gone the remaining knobs no longer test what the sentence claims.
            if any(k in names and current.get(k) == v for k, v in knobs.items()):
                continue
            kept = {k: v for k, v in knobs.items() if k in names}
            key = tuple(sorted(kept.items(), key=lambda kv: kv[0]))
            if kept and key not in seen:
                seen.add(key)
                priors.append(PriorOut(knobs=kept, hypothesis=hyp))
        return SearchPlanOut(
            subspaces=list(ctx["diagnosis"]["subspaces"]),
            priors=priors[:4],
            max_trials=int(ctx["budget"]["max_trials"]),
            rationale=f"fake-llm: search the {primary} sub-space",
        )

    def _narrate(self, ctx: dict[str, Any]) -> NarrativeOut:
        """Write the recipe's rationale from the two verified arms.

        ``comparable`` is the caller's answer to "was there a baseline rate to be a
        percentage of": both arms are driven at the candidate's load point, and a baseline
        that OOMs or misses every deadline there scores a clean zero. There is no
        percentage of zero, so the gain is stated in absolute rps instead. Absent, it
        defaults to true -- a context that never mentions the question is one where the
        comparison is ordinary.
        """
        d, base, best = ctx["diagnosis"], ctx["baseline_metrics"], ctx["best_metrics"]
        knobs = ", ".join(f"{k}={v}" for k, v in ctx["winning_knobs"].items())
        g0, g1 = float(base.get("goodput_rps", 0)), float(best.get("goodput_rps", 0))
        if ctx.get("comparable", True):
            pct = (g1 - g0) / g0 * 100 if g0 else 0.0
            change = f"Changing {knobs} raised goodput from {g0:.3f} to {g1:.3f} rps ({pct:+.0f}%)."
        else:
            change = (
                f"Changing {knobs} raised goodput to {g1:.3f} rps, from a baseline that "
                f"served nothing at c={ctx.get('load_point')}."
            )
        return NarrativeOut(
            rationale=f"Primary bottleneck {d['primary']}: {d.get('rationale', '')} {change}",
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
