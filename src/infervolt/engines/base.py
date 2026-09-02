"""Engine adapter contract. Every serving engine (mock, llama.cpp, vLLM, ...) implements this."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from infervolt.core.types import CrashKind, EngineConfig, KnobSpace, KnobValue, RunContext
from infervolt.loadgen.base import LoadGenerator


class EngineVersion(BaseModel):
    name: str
    version: str = ""
    commit: str = ""
    image_digest: str = ""


@dataclass
class ServerHandle:
    url: str
    state: Any = None


@dataclass
class ExitInfo:
    code: int
    log_tail: str = ""


class LaunchError(Exception):
    def __init__(self, exit: ExitInfo) -> None:
        super().__init__(exit.log_tail[-400:])
        self.exit = exit


OOM_PATTERNS = [
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"No available memory for the cache blocks",
    r"larger than the maximum number of tokens that can be stored in KV cache",
    r"Free memory on device .* less than desired",
    r"failed to allocate",
    r"ggml_metal.*alloc",
    r"kv_cache_init: failed",
]
_OOM_RE = re.compile("|".join(OOM_PATTERNS), re.IGNORECASE)


def classify_log(exit: ExitInfo) -> CrashKind:
    """Engine-agnostic crash classification from exit code and log tail."""
    if exit.code == 0:
        return "none"
    if exit.code == 124:
        return "timeout"
    if exit.code == -9 or _OOM_RE.search(exit.log_tail):
        return "oom"
    return "runtime"


class EngineAdapter(ABC):
    name: str = "abstract"

    @abstractmethod
    def version(self) -> EngineVersion: ...

    @abstractmethod
    def knob_space(self, ctx: RunContext) -> KnobSpace: ...

    @abstractmethod
    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """Static rejections. Non-empty means never launch this config."""

    @abstractmethod
    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        """Start the server. Raise LaunchError with the exit info if it dies during startup."""

    @abstractmethod
    def ready(self, handle: ServerHandle, timeout_s: float) -> bool: ...

    @abstractmethod
    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator: ...

    @abstractmethod
    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        """Canonical engine metrics since the last scrape: kv_usage_p95, num_waiting, num_running,
        preemptions_per_s, queue_time_p90_s, prefill_time_p50_s, prefill_share, prefix_hit_rate,
        max_num_seqs, kv_dtype_bytes."""

    @abstractmethod
    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        """sm_active and dram_active in [0, 1] over the last load window."""

    @abstractmethod
    def stop(self, handle: ServerHandle) -> ExitInfo: ...

    @abstractmethod
    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        """(serve args, reproduction command)."""

    def classify_crash(self, exit: ExitInfo) -> CrashKind:
        return classify_log(exit)
