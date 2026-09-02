"""Quality guard: accuracy recovery check for knobs that change numerics.

A speed win bought by changing what the model computes is not a win until someone
checks the answers still hold. Only a few knobs can do that, so only they trigger an
eval -- see :data:`QUALITY_KNOBS`.
"""

from __future__ import annotations

from typing import Protocol

from infervolt.core.types import EngineConfig, KnobValue, QualityScore, RunContext

QUALITY_KNOBS = {"kv_cache_dtype", "quantization", "speculative"}
"""Knobs that change the numerics of generation, and so can move accuracy.

Everything else -- batch sizes, memory fractions, scheduling -- changes *when* work is
done, not what it computes, and needs no eval.
"""

RECOVERY_MIN = {"fp8": 0.99, "int4": 0.97, "default": 0.99}
"""Minimum share of the unquantized score a config must recover, by weight quantization."""


def needs_quality_guard(before: dict[str, KnobValue], after: dict[str, KnobValue]) -> bool:
    """True when the move touched a numerics-changing knob.

    Compared with ``.get`` on both sides so a knob that appears or disappears counts as
    a change, not as equal-by-absence.
    """
    return any(before.get(k) != after.get(k) for k in QUALITY_KNOBS)


class QualityGuard(Protocol):
    name: str

    def evaluate(self, cfg: EngineConfig, ctx: RunContext) -> QualityScore: ...


class MockQualityGuard:
    """Deterministic recovery numbers mirroring the Red Hat 500k-eval study."""

    name = "mock"

    def evaluate(self, cfg: EngineConfig, ctx: RunContext) -> QualityScore:
        rec = 1.0
        if cfg.knobs.get("kv_cache_dtype") == "fp8":
            rec = min(rec, 0.995)
        if cfg.knobs.get("quantization") == "fp8":
            rec = min(rec, 0.992)
        return QualityScore(guard=self.name, tasks=["gsm8k", "arc_challenge"], recovery=rec)


def recovery_threshold(cfg: EngineConfig) -> float:
    """The recovery floor for ``cfg``, keyed on its weight quantization.

    Lower-precision weights are allowed to lose more, because the speedup they buy is
    larger; anything else -- including an unquantized config whose KV cache went fp8 --
    is held to the default.
    """
    q = str(cfg.knobs.get("quantization", "none"))
    return RECOVERY_MIN.get(q, RECOVERY_MIN["default"])
