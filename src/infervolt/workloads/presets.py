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
        return PRESETS[name]
    except KeyError as e:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(PRESETS)}") from e


_UNITS = {"ms": 1.0, "s": 1000.0}


def _ms(text: str) -> float:
    for suffix, mult in _UNITS.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * mult
    return float(text)


def parse_slo(text: str) -> SLO:
    """Parse 'ttft=500ms,itl=30ms,e2e=2s,p=0.9,g=0.9' into an SLO. Empty string -> no targets."""
    slo = SLO()
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition("=")
        if key == "ttft":
            slo.ttft_ms = _ms(value)
        elif key == "itl":
            slo.itl_ms = _ms(value)
        elif key == "e2e":
            slo.e2e_ms = _ms(value)
        elif key == "p":
            slo.percentile = float(value)
        elif key == "g":
            slo.goodput_target = float(value)
        else:
            raise ValueError(f"unknown SLO key {key!r}; use ttft, itl, e2e, p, g")
    return slo
