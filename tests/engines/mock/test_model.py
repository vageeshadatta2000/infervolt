import pytest

from infervolt.engines.mock.model import DEFAULT_KNOBS, OomError, PerfModel
from infervolt.hardware.profiles import get_profile
from infervolt.models.catalog import get_model_info
from infervolt.workloads.presets import get_workload


def _pm(hw: str, model: str, workload: str, **knobs: object) -> PerfModel:
    return PerfModel(
        get_profile(hw),
        get_model_info(model),
        get_workload(workload),
        {**DEFAULT_KNOBS, **knobs},
    )


def test_kv_limited_scenario_preempts_and_queues() -> None:
    pm = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512")
    p = pm.point(16)
    assert p.running < 16 and p.waiting > 0
    assert p.preempt_frac > 0 and p.kv_usage >= 0.9
    assert pm.point(4).preempt_frac == 0


def test_fp8_kv_doubles_admitted_sequences() -> None:
    a = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512").point(64).running
    b = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", kv_cache_dtype="fp8").point(64).running
    assert b >= 2 * a - 1


def test_decode_bound_scenario_is_dram_heavy() -> None:
    p = _pm("a100-80", "mock/qwen3-8b", "chat-256-512").point(64)
    assert p.dram_active > 0.6 and p.sm_active < 0.5
    assert p.itl_mean_s < 1.3 * p.step_floor_s


def test_prefill_bound_scenario_is_sm_heavy() -> None:
    p = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64").point(16)
    assert p.prefill_share > 0.5 and p.sm_active > 0.7


def test_scheduler_bound_scenario_has_flat_itl() -> None:
    pm = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager=True)
    p1, p8 = pm.point(1), pm.point(8)
    assert abs(p8.itl_mean_s - p1.itl_mean_s) / p1.itl_mean_s < 0.15
    assert p8.sm_active < 0.4 and p8.dram_active < 0.4
    fast = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager=False).point(8)
    assert fast.itl_mean_s < 0.5 * p8.itl_mean_s


def test_speculative_and_quantization_effects() -> None:
    base = _pm("h100-80", "mock/qwen3-8b", "chat-256-512").point(16)
    spec = _pm("h100-80", "mock/qwen3-8b", "chat-256-512", speculative="eagle3").point(16)
    quant = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64", quantization="fp8").point(4)
    plain = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64").point(4)
    assert spec.itl_mean_s < base.itl_mean_s
    assert quant.prefill_s < plain.prefill_s


def test_oom_at_launch() -> None:
    with pytest.raises(OomError, match="CUDA out of memory"):
        _pm("rtx4090-24", "mock/llama-70b", "chat-4k-512").check_launch()
    with pytest.raises(OomError, match="KV cache"):
        _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", max_model_len=32768).check_launch()
    _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", max_model_len=8192).check_launch()
