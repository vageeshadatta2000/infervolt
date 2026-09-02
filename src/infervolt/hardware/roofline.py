"""Roofline estimates for LLM inference. All functions are pure and unit-tested.

Decode is memory-bound: each step streams every weight byte plus the KV cache of every
sequence in the batch through HBM. Prefill mixes a compute term (linear layers, quadratic
attention) with the same weight stream, and takes whichever dominates.

Two conventions hold throughout:

* **Memory is GiB, not GB.** ``HardwareProfile.mem_gb`` is what NVML and the engines
  report -- binary gibibytes -- so byte counts go through :func:`mem_bytes` and
  :func:`reserve_bytes` (``* 2**30``). Model and bandwidth figures stay decimal
  (``params_b * 1e9``, ``hbm_bw_gbs * 1e9``) because that is how they are specified.
* **Every figure is per GPU, i.e. per tensor-parallel shard.** Nothing here knows about
  TP: a caller modelling TP=4 must pass the per-shard model dimensions (divide layers'
  widths, KV heads and parameters itself) before calling.
"""

from __future__ import annotations

from infervolt.core.types import HardwareProfile, ModelInfo

ACTIVATION_RESERVE_GB = 2.0


def mem_bytes(hw: HardwareProfile) -> float:
    """Total device memory in bytes. ``mem_gb`` is GiB, as NVML reports it."""
    return hw.mem_gb * 2**30


def reserve_bytes() -> float:
    """Bytes held back for activations and fragmentation, i.e. not available for KV."""
    return ACTIVATION_RESERVE_GB * 2**30


def weight_bytes(m: ModelInfo) -> float:
    return m.params_b * 1e9 * m.weight_bits / 8


def active_params(m: ModelInfo) -> float:
    """Parameters touched per token: the MoE active count when declared, else all of them."""
    b = m.active_params_b if m.active_params_b is not None else m.params_b
    return b * 1e9


def kv_bytes_per_token(m: ModelInfo, kv_dtype_bytes: int = 2) -> float:
    if kv_dtype_bytes <= 0:
        raise ValueError(f"kv_dtype_bytes must be positive, got {kv_dtype_bytes!r}")
    return 2.0 * m.num_layers * m.num_kv_heads * m.head_dim * kv_dtype_bytes


def decode_step_floor_s(
    hw: HardwareProfile, m: ModelInfo, batch: int, ctx_tokens: int, kv_dtype_bytes: int = 2
) -> float:
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch!r}")
    streamed = weight_bytes(m) + batch * kv_bytes_per_token(m, kv_dtype_bytes) * ctx_tokens
    t_mem = streamed / (hw.hbm_bw_gbs * 1e9)
    t_compute = 2.0 * active_params(m) * batch / (hw.peak_tflops * 1e12)
    return max(t_mem, t_compute)


def prefill_floor_s(hw: HardwareProfile, m: ModelInfo, tokens: int) -> float:
    """Lower bound on the time to prefill ``tokens`` in one forward pass.

    Three terms, two of which race:

    * ``linear`` -- ``2 * active_params * tokens`` FLOPs: one multiply-add per active
      parameter per token through the dense/expert projections.
    * ``attn`` -- ``4 * tokens**2 * hidden * num_layers`` FLOPs: the quadratic
      score-and-weighted-sum pair (two matmuls, 2 FLOPs each) that the linear term
      ignores. Negligible at short context, dominant at long context.
    * the weight stream -- ``weight_bytes / hbm_bw``: even a one-token prefill must read
      every weight out of HBM once.

    The compute terms share the same SMs, so they add; the result is the larger of that
    sum and the weight stream, since the two overlap.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens!r}")
    linear = 2.0 * active_params(m) * tokens
    attn = 4.0 * tokens**2 * m.hidden * m.num_layers
    t_compute = (linear + attn) / (hw.peak_tflops * 1e12)
    t_mem = weight_bytes(m) / (hw.hbm_bw_gbs * 1e9)
    return max(t_compute, t_mem)


def kv_capacity_tokens(
    hw: HardwareProfile, m: ModelInfo, gpu_mem_util: float, kv_dtype_bytes: int = 2
) -> float:
    """KV-cache tokens that fit once weights and the activation reserve are subtracted."""
    if not 0 < gpu_mem_util <= 1:
        raise ValueError(f"gpu_mem_util must be in (0, 1], got {gpu_mem_util!r}")
    avail = mem_bytes(hw) * gpu_mem_util - weight_bytes(m) - reserve_bytes()
    return max(0.0, avail / kv_bytes_per_token(m, kv_dtype_bytes))
