"""Analytic steady-state simulator of a continuous-batching LLM server.

It is deliberately simple: one closed-loop concurrency level -> admitted sequences, queueing,
prefill/decode interference, KV capacity, preemption, and roofline-derived step times. It is
not meant to be accurate in absolute terms, only to reproduce the *signatures* of each
bottleneck so the diagnosis rules and the search loop can be tested without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from infervolt.core.types import HardwareProfile, KnobValue, ModelInfo, Workload
from infervolt.hardware import roofline

DEFAULT_KNOBS: dict[str, KnobValue] = {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "speculative": "none",
    "quantization": "none",
    "tensor_parallel_size": 1,
}

SPEC_SPEEDUP = {"none": 1.0, "ngram": 1.25, "eagle3": 1.7}
EAGER_OVERHEAD_S = 0.004
GRAPH_OVERHEAD_S = 0.0008
PER_SEQ_OVERHEAD_S = 5e-6


_TRUE_WORDS = frozenset({"true", "1", "yes"})
_FALSE_WORDS = frozenset({"false", "0", "no"})


def _as_bool(v: KnobValue) -> bool:
    """Coerce a knob value to a bool, the way a CLI flag would be read.

    ``bool()`` is wrong here: every non-empty string is truthy, so ``bool("false")``
    is ``True`` and a knob set from YAML, JSON or a command line silently inverts.
    Accepts real bools, the ints 0 and 1, and the usual word spellings in any case;
    anything else raises rather than guessing.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        word = v.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"cannot read {v!r} as a bool; use true/false, 1/0 or yes/no")


class OomError(Exception):
    """Raised by check_launch when the config cannot fit."""


@dataclass
class SimPoint:
    """Steady state at one closed-loop concurrency level.

    Time fields are per request. ``lifetime_s`` is *service* time only -- the span from
    admission to the last token, excluding time spent waiting to be admitted -- so a
    request's end-to-end latency is ``queue_wait_s + lifetime_s``. ``ttft_s``, by
    contrast, is measured from arrival and so already includes ``queue_wait_s``.
    """

    concurrency: int
    running: int
    waiting: int
    queue_wait_s: float
    ttft_s: float
    prefill_s: float
    itl_mean_s: float
    itl_spike_s: float
    n_spikes: int
    lifetime_s: float
    step_floor_s: float
    kv_usage: float
    preempt_frac: float
    prefill_share: float
    sm_active: float
    dram_active: float


class PerfModel:
    def __init__(
        self, hw: HardwareProfile, model: ModelInfo, workload: Workload, knobs: dict[str, KnobValue]
    ) -> None:
        self.hw = hw
        self.workload = workload
        self.knobs = {**DEFAULT_KNOBS, **knobs}
        quant = str(self.knobs["quantization"])
        self.model = model.model_copy(update={"weight_bits": 8}) if quant == "fp8" else model
        self.peak_tflops = hw.peak_tflops * (2.0 if quant == "fp8" else 1.0)
        self.hw_eff = hw.model_copy(update={"peak_tflops": self.peak_tflops})
        self.kv_dtype_bytes = 1 if str(self.knobs["kv_cache_dtype"]) == "fp8" else 2
        self.util = float(self.knobs["gpu_memory_utilization"])
        self.capacity = roofline.kv_capacity_tokens(hw, self.model, self.util, self.kv_dtype_bytes)

    # ---- launch-time checks
    def check_launch(self) -> None:
        need = roofline.weight_bytes(self.model) + roofline.reserve_bytes()
        if need > roofline.mem_bytes(self.hw) * self.util:
            raise OomError(
                f"torch.OutOfMemoryError: CUDA out of memory. "
                f"Tried to allocate {need / 2**30:.1f} GiB"
            )
        max_len = int(self.knobs["max_model_len"])
        if self.capacity < max_len:
            raise OomError(
                f"ValueError: The model's max seq len ({max_len}) is larger than the maximum "
                f"number of tokens that can be stored in KV cache ({int(self.capacity)})."
            )

    # ---- steady state at one concurrency
    def _sched_overhead(self, n: int) -> float:
        base = EAGER_OVERHEAD_S if _as_bool(self.knobs["enforce_eager"]) else GRAPH_OVERHEAD_S
        return base + PER_SEQ_OVERHEAD_S * n

    def _spec_speedup(self) -> float:
        name = str(self.knobs["speculative"])
        try:
            return SPEC_SPEEDUP[name]
        except KeyError as e:
            raise ValueError(
                f"unknown speculative {name!r}; choices: {sorted(SPEC_SPEEDUP)}"
            ) from e

    def point(self, concurrency: int) -> SimPoint:
        w, m = self.workload, self.model
        isl, osl = w.isl.p50, w.osl.p50
        hit = w.prefix_share if _as_bool(self.knobs["enable_prefix_caching"]) else 0.0
        p_tokens = max(1, int(isl * (1 - hit)))
        max_seqs = int(self.knobs["max_num_seqs"])
        by_kv = int(self.capacity // isl) if self.capacity >= isl else 0
        n = max(0, min(concurrency, max_seqs, by_kv))
        if n == 0:
            # Nothing runs. Either the KV cache cannot hold even one request -- a saturated
            # server, which is what the diagnosis rules must see -- or there is simply
            # nothing to run (concurrency 0, max_num_seqs 0), which is an idle one.
            starved = by_kv == 0
            return SimPoint(
                concurrency=concurrency,
                running=0,
                waiting=max(0, concurrency),
                queue_wait_s=0.0,
                ttft_s=0.0,
                prefill_s=0.0,
                itl_mean_s=0.0,
                itl_spike_s=0.0,
                n_spikes=0,
                lifetime_s=0.0,
                step_floor_s=0.0,
                kv_usage=1.0 if starved else 0.0,
                preempt_frac=1.0 if starved else 0.0,
                prefill_share=0.0,
                sm_active=0.0,
                dram_active=0.0,
            )
        waiting = concurrency - n
        ctx = isl + osl // 2
        floor = roofline.decode_step_floor_s(self.hw_eff, m, n, ctx, self.kv_dtype_bytes)
        step = floor + self._sched_overhead(n)
        prefill = roofline.prefill_floor_s(self.hw_eff, m, p_tokens)
        per_tok = prefill / p_tokens
        if _as_bool(self.knobs["enable_chunked_prefill"]):
            chunk = min(p_tokens, int(self.knobs["max_num_batched_tokens"]))
            n_chunks = math.ceil(p_tokens / chunk)
            ttft_core = prefill + n_chunks * step
            spike = chunk * per_tok
        else:
            ttft_core, spike = prefill, prefill
        itl_mean = (step + (n - 1) * prefill / osl) / self._spec_speedup()
        need = n * (isl + osl)
        preempt_frac = max(0.0, (need - self.capacity) / self.capacity)
        lifetime_core = ttft_core + osl * itl_mean
        lifetime = lifetime_core * (1 + preempt_frac)
        queue_wait = (waiting / n) * lifetime if waiting > 0 else 0.0
        mem_part = (
            roofline.weight_bytes(m) + n * roofline.kv_bytes_per_token(m, self.kv_dtype_bytes) * ctx
        ) / (self.hw.hbm_bw_gbs * 1e9)
        comp_part = 2.0 * roofline.active_params(m) * n / (self.peak_tflops * 1e12)
        compute_time = n * prefill + osl * comp_part
        mem_time = osl * mem_part
        return SimPoint(
            concurrency=concurrency,
            running=n,
            waiting=waiting,
            queue_wait_s=queue_wait,
            ttft_s=queue_wait + ttft_core,
            prefill_s=prefill,
            itl_mean_s=itl_mean,
            itl_spike_s=spike,
            n_spikes=min(osl, n - 1),
            lifetime_s=lifetime,
            step_floor_s=floor,
            kv_usage=min(1.0, n * (isl + osl / 2) / self.capacity),
            preempt_frac=preempt_frac,
            prefill_share=min(1.0, n * prefill / lifetime_core),
            sm_active=min(1.0, compute_time / lifetime_core),
            dram_active=min(1.0, mem_time / lifetime_core),
        )
