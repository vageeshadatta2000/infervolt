import pytest

from infervolt.core.types import ModelInfo
from infervolt.hardware import roofline
from infervolt.hardware.profiles import get_profile


def _qwen8b() -> ModelInfo:
    return ModelInfo(
        id="mock/qwen3-8b", params_b=8.2, num_layers=36, hidden=4096, num_kv_heads=8, head_dim=128
    )


def test_weight_and_kv_bytes() -> None:
    m = _qwen8b()
    assert roofline.weight_bytes(m) == pytest.approx(16.4e9)
    assert roofline.kv_bytes_per_token(m, kv_dtype_bytes=2) == 2 * 36 * 8 * 128 * 2


def test_decode_is_memory_bound_at_batch_one() -> None:
    hw, m = get_profile("a100-80"), _qwen8b()
    step = roofline.decode_step_floor_s(hw, m, batch=1, ctx_tokens=1024)
    mem_only = roofline.weight_bytes(m) / (hw.hbm_bw_gbs * 1e9)
    assert step == pytest.approx(mem_only, rel=0.05)


def test_kv_capacity_shrinks_with_lower_util_and_grows_with_fp8() -> None:
    hw, m = get_profile("rtx4090-24"), _qwen8b()
    fp16 = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.9, kv_dtype_bytes=2)
    fp8 = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.9, kv_dtype_bytes=1)
    low = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.8, kv_dtype_bytes=2)
    assert fp8 == pytest.approx(2 * fp16)
    assert low < fp16
    assert roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.5, kv_dtype_bytes=2) == 0.0


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("nope")
