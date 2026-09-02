"""Mock engine: no server, no GPU. Launch builds a PerfModel; the load generator samples from it."""

from __future__ import annotations

import numpy as np

from infervolt.core.types import (
    EngineConfig,
    Knob,
    KnobSpace,
    KnobValue,
    LoadResult,
    RequestRecord,
    RunContext,
    Workload,
)
from infervolt.engines.base import EngineAdapter, EngineVersion, ExitInfo, LaunchError, ServerHandle
from infervolt.engines.mock.model import DEFAULT_KNOBS, OomError, PerfModel, SimPoint, _as_bool
from infervolt.loadgen.base import LoadGenerator

NOISE = 0.03
MAX_MODEL_LEN_CHOICES: list[KnobValue] = [4096, 8192, 16384, 32768]


class MockState:
    def __init__(self, pm: PerfModel) -> None:
        self.pm = pm
        self.last: SimPoint | None = None


def _state(handle: ServerHandle) -> MockState:
    """Narrow ``ServerHandle.state``, which is deliberately ``Any`` in the base contract."""
    st = handle.state
    assert isinstance(st, MockState), f"handle was not produced by MockAdapter.launch: {st!r}"
    return st


class SimLoadGenerator:
    def __init__(self, state: MockState) -> None:
        self.state = state

    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        pm = self.state.pm
        p = pm.point(concurrency)
        self.state.last = p
        rng = np.random.default_rng(seed * 1000 + concurrency)
        osl = workload.osl.p50
        if p.running == 0:
            # Nothing was admitted, so nothing completed. duration_s still has to be
            # positive -- every rate in compute_metrics divides by it -- and one second
            # of a fully failed run is as good a stand-in as any.
            failed = [
                RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
                for _ in range(num_requests)
            ]
            return LoadResult(concurrency=concurrency, duration_s=1.0, requests=failed)
        base_itl = max(p.step_floor_s * 0.5, p.itl_mean_s - p.n_spikes * p.itl_spike_s / osl)
        reqs: list[RequestRecord] = []
        for _ in range(num_requests):
            ttft = p.ttft_s * (1 + NOISE * rng.standard_normal())
            itl = base_itl * (1 + NOISE * rng.standard_normal(osl))
            if p.n_spikes:
                idx = rng.choice(osl, size=p.n_spikes, replace=False)
                itl[idx] += p.itl_spike_s
            reqs.append(
                RequestRecord(
                    ttft_s=float(max(ttft, 1e-4)),
                    itl_s=[float(x) for x in itl],
                    output_tokens=osl,
                )
            )
        duration = num_requests * (p.lifetime_s + p.queue_wait_s) / p.running
        return LoadResult(concurrency=concurrency, duration_s=float(duration), requests=reqs)


class MockAdapter(EngineAdapter):
    name = "mock"

    def version(self) -> EngineVersion:
        return EngineVersion(name="mock", version="1.0", commit="sim")

    @staticmethod
    def _default_max_model_len(ctx: RunContext) -> KnobValue:
        """Shortest offered context that still covers the workload.

        The baseline config has to launch, and both ``validate`` (which rejects a
        max_model_len below the workload) and ``PerfModel.check_launch`` (which rejects
        one the KV cache cannot hold) get a say. Picking the smallest sufficient choice
        satisfies the first and gives the second the most headroom.
        """
        need = ctx.workload.isl.p99 + ctx.workload.osl.p50
        for choice in MAX_MODEL_LEN_CHOICES:
            if isinstance(choice, int) and choice >= need:
                return choice
        return MAX_MODEL_LEN_CHOICES[-1]

    def knob_space(self, ctx: RunContext) -> KnobSpace:
        d = DEFAULT_KNOBS
        return KnobSpace(
            knobs=[
                Knob(
                    name="max_num_seqs",
                    kind="int",
                    groups=["kv", "decode", "sched"],
                    default=d["max_num_seqs"],
                    low=8,
                    high=1024,
                    log=True,
                ),
                Knob(
                    name="max_num_batched_tokens",
                    kind="int",
                    groups=["prefill"],
                    default=d["max_num_batched_tokens"],
                    low=512,
                    high=16384,
                    log=True,
                ),
                Knob(
                    name="gpu_memory_utilization",
                    kind="float",
                    groups=["kv"],
                    default=d["gpu_memory_utilization"],
                    low=0.7,
                    high=0.95,
                    step=0.05,
                ),
                Knob(
                    name="max_model_len",
                    kind="cat",
                    groups=["kv"],
                    default=self._default_max_model_len(ctx),
                    choices=MAX_MODEL_LEN_CHOICES,
                ),
                Knob(
                    name="enable_prefix_caching",
                    kind="bool",
                    groups=["kv"],
                    default=d["enable_prefix_caching"],
                ),
                Knob(
                    name="enable_chunked_prefill",
                    kind="bool",
                    groups=["prefill"],
                    default=d["enable_chunked_prefill"],
                ),
                Knob(
                    name="kv_cache_dtype",
                    kind="cat",
                    groups=["kv", "decode"],
                    default=d["kv_cache_dtype"],
                    choices=["auto", "fp8"],
                ),
                Knob(
                    name="enforce_eager", kind="bool", groups=["sched"], default=d["enforce_eager"]
                ),
                Knob(
                    name="speculative",
                    kind="cat",
                    groups=["decode"],
                    default=d["speculative"],
                    choices=["none", "ngram", "eagle3"],
                ),
                Knob(
                    name="quantization",
                    kind="cat",
                    groups=["prefill", "decode"],
                    default=d["quantization"],
                    choices=["none", "fp8"],
                ),
            ]
        )

    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        errs: list[str] = []
        for knob in self.knob_space(ctx).knobs:
            if knob.name not in cfg.knobs:
                continue
            value = cfg.knobs[knob.name]
            if knob.kind == "cat" and value not in knob.choices:
                errs.append(f"{knob.name}={value!r} is not one of {knob.choices!r}")
            elif knob.kind == "bool":
                try:
                    _as_bool(value)
                except ValueError:
                    errs.append(
                        f"{knob.name}={value!r} is not a bool; "
                        f"choices: ['true', 'false', '1', '0', 'yes', 'no']"
                    )
        if cfg.knobs.get("quantization") == "fp8" and ctx.hw.compute_capability < 8.9:
            errs.append("fp8 quantization needs compute capability >= 8.9")
        if int(cfg.knobs.get("max_model_len", 32768)) < ctx.workload.isl.p99 + ctx.workload.osl.p50:
            errs.append("max_model_len shorter than workload p99 ISL + OSL")
        return errs

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        pm = PerfModel(ctx.hw, ctx.model, ctx.workload, cfg.knobs)
        try:
            pm.check_launch()
        except OomError as e:
            raise LaunchError(ExitInfo(code=1, log_tail=str(e))) from e
        return ServerHandle(url="mock://", state=MockState(pm))

    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        return True

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        return SimLoadGenerator(_state(handle))

    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        st = _state(handle)
        p = st.last
        if p is None:
            return {}
        w = st.pm.workload
        rate = p.running / p.lifetime_s if p.lifetime_s > 0 else 0.0
        hit = w.prefix_share if _as_bool(st.pm.knobs["enable_prefix_caching"]) else 0.0
        return {
            "kv_usage_p95": p.kv_usage,
            "num_waiting": float(p.waiting),
            "num_running": float(p.running),
            "preemptions_per_s": p.preempt_frac * rate,
            "queue_time_p90_s": p.queue_wait_s,
            "prefill_time_p50_s": p.ttft_s - p.queue_wait_s,
            "prefill_share": p.prefill_share,
            "prefix_hit_rate": hit,
            "max_num_seqs": float(st.pm.knobs["max_num_seqs"]),
            "kv_dtype_bytes": float(st.pm.kv_dtype_bytes),
        }

    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        p = _state(handle).last
        return {"sm_active": p.sm_active, "dram_active": p.dram_active} if p else {}

    def stop(self, handle: ServerHandle) -> ExitInfo:
        return ExitInfo(code=0)

    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(cfg.knobs.items()))
        return dict(cfg.knobs), f"mock-serve {ctx.model.id} {flags}"
