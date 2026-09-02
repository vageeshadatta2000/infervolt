"""Knob-space utilities: OOM-tightened bounds, clamping, novelty.

The sampler proposes points; these helpers decide which of them are worth spending a
trial on. :class:`Bounds` is the search's crash memory -- an OOM lowers the ceiling of
the knob that caused it, so the same too-large config cannot come back under a different
random draw -- and :func:`is_novel` keeps the search from re-measuring a config it has
already paid for.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from infervolt.core.types import Knob, KnobSpace, KnobValue

NOVELTY_EPS = 0.05
"""Two configs closer than this in normalised max-norm distance are the same config.

Normalisation puts every knob on [0, 1], so 0.05 is "within 5% of the range on *every*
knob": far enough apart to be worth a trial, close enough that a 0.005 nudge to
``gpu_memory_utilization`` is not.
"""


def _numeric_choices(k: Knob) -> list[float] | None:
    """The choices of a numeric categorical knob, ascending, or ``None`` if it is not one.

    Bools are excluded deliberately: ``True`` is numerically 1, but ordering a knob whose
    choices are ``[True, False]`` by value would invent a magnitude it does not have.
    """
    if k.kind != "cat" or not k.choices:
        return None
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in k.choices):
        return None
    return sorted(float(c) for c in k.choices)


BACKOFF: dict[str, Callable[[float], float]] = {
    "gpu_memory_utilization": lambda v: round(v - 0.05, 2),
    "max_model_len": lambda v: v / 2,
    "max_num_seqs": lambda v: v - 1,
}
"""How far each memory knob retreats from a value that just ran out of memory.

Deliberately coarse -- one utilisation notch, half the context, one fewer sequence --
because the goal is to leave the region that failed, not to bisect it. Knobs absent from
this table are not memory knobs and are never tightened.
"""


class Bounds:
    """Upper bounds per knob, lowered whenever a config runs out of memory.

    Only knobs with a numeric ceiling appear in ``high``; a categorical knob such as
    ``kv_cache_dtype`` has no direction to back off in and is left out entirely.
    """

    def __init__(self, space: KnobSpace) -> None:
        self.high: dict[str, float] = {}
        self.low: dict[str, float] = {}
        for k in space.knobs:
            if k.kind in ("int", "float") and k.low is not None and k.high is not None:
                self.low[k.name], self.high[k.name] = float(k.low), float(k.high)
            elif (nc := _numeric_choices(k)) is not None:
                self.low[k.name], self.high[k.name] = nc[0], nc[-1]

    def tighten_on_oom(self, knobs: dict[str, KnobValue]) -> None:
        """Back off the memory knobs of a config that just OOMed.

        ``min`` keeps each ceiling monotonically falling: an OOM at a value already above
        the current ceiling teaches nothing new.

        A back-off that would fall below the knob's own floor is *not* applied. An OOM
        blames every memory knob in the config at once, only one of which is usually
        guilty, so "this knob has no valid value left" is the wrong conclusion to draw
        from it -- and a ceiling under the floor would make every later config invalid
        rather than merely conservative, ending the search instead of steering it.
        """
        for name, value in knobs.items():
            back_off = BACKOFF.get(name)
            if back_off is None or name not in self.high or isinstance(value, (bool, str)):
                continue
            proposed = back_off(float(value))
            if proposed >= self.low[name]:
                self.high[name] = min(self.high[name], proposed)


def clamp(knobs: dict[str, KnobValue], space: KnobSpace, bounds: Bounds) -> dict[str, KnobValue]:
    """Pull every knob down to its current ceiling, preserving each knob's type.

    Categorical knobs snap to the largest *offered* choice at or below the ceiling, so a
    clamped ``max_model_len`` is still a value the engine accepts. Knobs the bounds do
    not track, and knobs absent from ``knobs``, pass through untouched.
    """
    out: dict[str, KnobValue] = dict(knobs)
    for k in space.knobs:
        if k.name not in out or k.name not in bounds.high:
            continue
        v = out[k.name]
        if isinstance(v, (bool, str)):
            continue
        hi = bounds.high[k.name]
        if k.kind == "int":
            out[k.name] = min(int(v), int(hi))
        elif k.kind == "float":
            out[k.name] = min(float(v), hi)
        else:
            nc = _numeric_choices(k) or []
            allowed = [c for c in nc if c <= hi]
            if not allowed:
                # Nothing the engine offers is under the ceiling, so there is no valid
                # value to snap to; leave the knob and let validation say so.
                continue
            capped = min(float(v), allowed[-1])
            out[k.name] = int(capped) if all(c.is_integer() for c in nc) else capped
    return out


def _normalize(k: Knob, v: KnobValue) -> float:
    """Map one knob value onto [0, 1] so distances are comparable across knobs.

    Log knobs are normalised in log space: ``max_num_seqs`` 8 and 16 are one octave of
    seven apart, not the 0.8% of the linear range they look like.
    """
    if k.kind == "bool":
        return 1.0 if v else 0.0
    if k.kind == "cat":
        # An off-space value has no position; 0.0 keeps it comparable without pretending
        # it sits anywhere in particular.
        return k.choices.index(v) / max(len(k.choices) - 1, 1) if v in k.choices else 0.0
    assert k.low is not None and k.high is not None  # guaranteed by the Knob validator
    lo, hi, x = float(k.low), float(k.high), float(v)
    if k.log and lo > 0 and hi > lo:
        return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (x - lo) / (hi - lo) if hi > lo else 0.0


def is_novel(
    knobs: dict[str, KnobValue],
    seen: list[dict[str, KnobValue]],
    space: KnobSpace,
    eps: float = NOVELTY_EPS,
) -> bool:
    """True when ``knobs`` differs from every config in ``seen`` on at least one knob.

    The max-norm is the right metric here rather than a Euclidean one: a single knob
    moved a long way is a genuinely different config, however many others stayed put.
    Knobs missing from either side are read as their default, which is what the engine
    would have used.
    """
    for other in seen:
        dist = max(
            (
                abs(
                    _normalize(k, knobs.get(k.name, k.default))
                    - _normalize(k, other.get(k.name, k.default))
                )
                for k in space.knobs
            ),
            default=1.0,
        )
        if dist < eps:
            return False
    return True
