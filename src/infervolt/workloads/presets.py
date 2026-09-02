"""Workload presets and SLO string parsing."""

from __future__ import annotations

from infervolt.core.types import SLO, LoadSpec, TokenDist, Workload

PRESETS: dict[str, Workload] = {
    "chat-4k-512": Workload(
        name="chat-4k-512",
        isl=TokenDist(p50=4096, p99=6000),
        osl=TokenDist(p50=512, p99=1024),
        prefix_share=0.1,
    ),
    "chat-1k-128": Workload(
        name="chat-1k-128",
        isl=TokenDist(p50=1024, p99=2048),
        osl=TokenDist(p50=128, p99=256),
        prefix_share=0.1,
    ),
    "chat-256-512": Workload(
        name="chat-256-512",
        isl=TokenDist(p50=256, p99=512),
        osl=TokenDist(p50=512, p99=768),
        prefix_share=0.0,
    ),
    "rag-16k-64": Workload(
        name="rag-16k-64",
        isl=TokenDist(p50=16384, p99=20000),
        osl=TokenDist(p50=64, p99=128),
        prefix_share=0.0,
        load=LoadSpec(concurrency=[1, 4, 16, 64]),
    ),
    "agentic-prefix-16k-512": Workload(
        name="agentic-prefix-16k-512",
        isl=TokenDist(p50=16384, p99=24000),
        osl=TokenDist(p50=512, p99=1024),
        prefix_share=0.6,
    ),
}


def get_workload(name: str) -> Workload:
    try:
        return PRESETS[name].model_copy(deep=True)
    except KeyError as e:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(PRESETS)}") from e


_UNITS = {"ms": 1.0, "s": 1000.0}
# Longest suffix first, so "500ms" matches "ms" and never the "s" inside it.
_SUFFIXES = sorted(_UNITS, key=len, reverse=True)

_DURATION_KEYS = {"ttft": "ttft_ms", "itl": "itl_ms", "e2e": "e2e_ms"}
_FRACTION_KEYS = {"p": "percentile", "g": "goodput_target"}


def _ms(text: str) -> float:
    """Convert a duration literal ('500ms', '2 s', '250') to milliseconds."""
    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * _UNITS[suffix]
    return float(text)


def parse_slo(text: str) -> SLO:
    """Parse 'ttft=500ms,itl=30ms,e2e=2s,p=0.9,g=0.9' into an SLO. Empty string -> no targets.

    Whitespace around keys and values is ignored. Durations must be non-negative;
    ``p`` and ``g`` are fractions in (0, 1], so a percentile written as ``p=95`` is
    rejected rather than silently accepted as an impossible target.
    """
    fields: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        raw_key, sep, raw_value = part.partition("=")
        key, value = raw_key.strip(), raw_value.strip()
        if key in _DURATION_KEYS:
            if not sep:
                raise ValueError(f"malformed SLO clause {part!r}")
            try:
                ms = _ms(value)
            except ValueError as e:
                raise ValueError(f"malformed SLO clause {part!r}") from e
            if ms < 0:
                raise ValueError(f"malformed SLO clause {part!r}: duration must be >= 0")
            fields[_DURATION_KEYS[key]] = ms
        elif key in _FRACTION_KEYS:
            if not sep:
                raise ValueError(f"malformed SLO clause {part!r}")
            try:
                frac = float(value)
            except ValueError as e:
                raise ValueError(f"malformed SLO clause {part!r}") from e
            if not 0 < frac <= 1:
                raise ValueError(
                    f"malformed SLO clause {part!r}: {key!r} is a fraction in (0, 1], got {frac!r}"
                )
            fields[_FRACTION_KEYS[key]] = frac
        else:
            raise ValueError(f"unknown SLO key {key!r}; use ttft, itl, e2e, p, g")
    return SLO(**fields)
