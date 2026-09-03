from collections.abc import Callable

import pytest

from infervolt.core.types import ModelInfo
from infervolt.hardware import roofline
from infervolt.hardware.profiles import PROFILES, get_profile


def _qwen8b() -> ModelInfo:
    return ModelInfo(
        id="mock/qwen3-8b", params_b=8.2, num_layers=36, hidden=4096, num_kv_heads=8, head_dim=128
    )


def test_weight_and_kv_bytes() -> None:
    m = _qwen8b()
    assert roofline.weight_bytes(m) == pytest.approx(16.4e9)
    assert roofline.kv_bytes_per_token(m, kv_dtype_bytes=2) == 2 * 36 * 8 * 128 * 2


def test_memory_basis_is_gibibytes() -> None:
    hw = get_profile("a100-80")
    assert roofline.mem_bytes(hw) == 80 * 2**30
    assert roofline.reserve_bytes() == roofline.ACTIVATION_RESERVE_GIB * 2**30


def test_active_params_prefers_the_active_parameter_count() -> None:
    dense = _qwen8b()
    assert roofline.active_params(dense) == pytest.approx(8.2e9)
    sparse = dense.model_copy(update={"active_params_b": 0.5, "moe": True})
    assert roofline.active_params(sparse) == pytest.approx(0.5e9)


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


def test_kv_capacity_absolute_value_on_a_24gib_card() -> None:
    # 24 GiB * 0.9 - 16.4e9 weights - 2 GiB reserve = 4.65e9 bytes,
    # at 2 * 36 * 8 * 128 * 2 = 147_456 bytes per token -> ~31.5k tokens.
    tokens = roofline.kv_capacity_tokens(
        get_profile("rtx4090-24"), _qwen8b(), gpu_mem_util=0.9, kv_dtype_bytes=2
    )
    assert 30_000 < tokens < 33_000


def test_short_prefill_is_bound_by_streaming_the_weights() -> None:
    hw, m = get_profile("a100-80"), _qwen8b()
    weight_stream_s = roofline.weight_bytes(m) / (hw.hbm_bw_gbs * 1e9)
    assert roofline.prefill_floor_s(hw, m, tokens=64) == pytest.approx(weight_stream_s)


def test_long_prefill_pays_quadratic_attention_on_top_of_the_linear_term() -> None:
    hw, m = get_profile("a100-80"), _qwen8b()
    tokens = 16384
    linear_only_s = 2.0 * roofline.active_params(m) * tokens / (hw.peak_tflops * 1e12)
    # Causal attention adds tokens * hidden * layers / active_params = 16384 * 4096 * 36 / 8.2e9
    # = 0.2946 of the linear term, so the floor is ~1.295x the linear-only value.
    assert roofline.prefill_floor_s(hw, m, tokens) > 1.29 * linear_only_s


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda hw, m: roofline.kv_bytes_per_token(m, kv_dtype_bytes=0), id="kv_dtype_zero"
        ),
        pytest.param(
            lambda hw, m: roofline.decode_step_floor_s(
                hw, m, batch=1, ctx_tokens=8, kv_dtype_bytes=-1
            ),
            id="kv_dtype_negative",
        ),
        pytest.param(
            lambda hw, m: roofline.decode_step_floor_s(hw, m, batch=0, ctx_tokens=8),
            id="batch_zero",
        ),
        pytest.param(lambda hw, m: roofline.prefill_floor_s(hw, m, tokens=0), id="tokens_zero"),
        pytest.param(
            lambda hw, m: roofline.prefill_floor_s(hw, m, tokens=-5), id="tokens_negative"
        ),
        pytest.param(
            lambda hw, m: roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.0), id="util_zero"
        ),
        pytest.param(
            lambda hw, m: roofline.kv_capacity_tokens(hw, m, gpu_mem_util=1.5), id="util_above_one"
        ),
    ],
)
def test_invalid_inputs_raise_value_error(
    call: Callable[[object, ModelInfo], float],
) -> None:
    with pytest.raises(ValueError):
        call(get_profile("a100-80"), _qwen8b())


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError, match="unknown hardware profile"):
        get_profile("nope")


def test_get_profile_returns_a_copy_the_caller_cannot_use_to_mutate_the_registry() -> None:
    p = get_profile("a100-80")
    p.usd_per_hour = 999.0
    p.count = 8
    assert PROFILES["a100-80"].usd_per_hour == 1.5
    assert PROFILES["a100-80"].count == 1
    assert get_profile("a100-80").usd_per_hour == 1.5
