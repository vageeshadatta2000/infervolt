"""Roofline estimates for LLM inference. All functions are pure and unit-tested.

Decode is memory-bound: each step streams every weight byte plus the KV cache of every
sequence in the batch through HBM. Prefill is compute-bound: 2 FLOPs per parameter per token.
"""

from __future__ import annotations

from infervolt.core.types import HardwareProfile, ModelInfo

ACTIVATION_RESERVE_GB = 2.0


def weight_bytes(m: ModelInfo) -> float:
    return m.params_b * 1e9 * m.weight_bits / 8


def active_params(m: ModelInfo) -> float:
    return (m.active_params_b or m.params_b) * 1e9


def kv_bytes_per_token(m: ModelInfo, kv_dtype_bytes: int = 2) -> float:
    return 2.0 * m.num_layers * m.num_kv_heads * m.head_dim * kv_dtype_bytes


def decode_step_floor_s(
    hw: HardwareProfile, m: ModelInfo, batch: int, ctx_tokens: int, kv_dtype_bytes: int = 2
) -> float:
    mem_bytes = weight_bytes(m) + batch * kv_bytes_per_token(m, kv_dtype_bytes) * ctx_tokens
    t_mem = mem_bytes / (hw.hbm_bw_gbs * 1e9)
    t_compute = 2.0 * active_params(m) * batch / (hw.peak_tflops * 1e12)
    return max(t_mem, t_compute)


def prefill_floor_s(hw: HardwareProfile, m: ModelInfo, tokens: int) -> float:
    return 2.0 * active_params(m) * tokens / (hw.peak_tflops * 1e12)


def kv_capacity_tokens(
    hw: HardwareProfile, m: ModelInfo, gpu_mem_util: float, kv_dtype_bytes: int = 2
) -> float:
    avail = hw.mem_gb * 1e9 * gpu_mem_util - weight_bytes(m) - ACTIVATION_RESERVE_GB * 1e9
    return max(0.0, avail / kv_bytes_per_token(m, kv_dtype_bytes))
