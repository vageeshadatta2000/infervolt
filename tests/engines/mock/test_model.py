import pytest

from infervolt.engines.base import ExitInfo, classify_log
from infervolt.engines.mock.model import DEFAULT_KNOBS, OomError, PerfModel, _as_bool
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


def test_as_bool_parses_strings_and_ints() -> None:
    for truthy in (True, 1, "true", "TRUE", "1", "yes", "Yes"):
        assert _as_bool(truthy) is True
    for falsey in (False, 0, "false", "False", "0", "no", "NO"):
        assert _as_bool(falsey) is False


def test_as_bool_rejects_unparseable() -> None:
    with pytest.raises(ValueError, match="maybe"):
        _as_bool("maybe")
    with pytest.raises(ValueError, match="2"):
        _as_bool(2)


def test_string_false_knob_is_not_truthy() -> None:
    """bool("false") is True; the knob must not be."""
    as_string = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager="false").point(8)
    as_bool = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager=False).point(8)
    assert as_string.itl_mean_s == as_bool.itl_mean_s

    off_str = _pm("a100-80", "mock/qwen3-8b", "chat-4k-512", enable_prefix_caching="false").point(1)
    off = _pm("a100-80", "mock/qwen3-8b", "chat-4k-512", enable_prefix_caching=False).point(1)
    on = _pm("a100-80", "mock/qwen3-8b", "chat-4k-512", enable_prefix_caching=True).point(1)
    assert off_str.prefill_s == off.prefill_s > on.prefill_s


def test_unparseable_bool_knob_raises() -> None:
    with pytest.raises(ValueError, match="maybe"):
        _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager="maybe").point(8)


def test_unknown_speculative_raises() -> None:
    with pytest.raises(ValueError, match="unknown speculative 'medusa'"):
        _pm("a100-80", "mock/qwen3-8b", "chat-256-512", speculative="medusa").point(8)


def test_zero_admission_reports_saturated_kv() -> None:
    """KV capacity below one request: the point is starved, not idle."""
    pm = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", gpu_memory_utilization=0.7)
    p = pm.point(16)
    assert p.running == 0 and p.waiting == 16
    assert p.kv_usage == 1.0 and p.preempt_frac == 1.0


def test_zero_concurrency_is_idle_not_saturated() -> None:
    p = _pm("a100-80", "mock/qwen3-8b", "chat-256-512").point(0)
    assert p.running == 0 and p.waiting == 0
    assert p.kv_usage == 0.0 and p.preempt_frac == 0.0


def test_oom_messages_classify_as_oom() -> None:
    """Both check_launch failures must be recognised by the engine-agnostic classifier."""
    for pm in (
        _pm("rtx4090-24", "mock/llama-70b", "chat-4k-512"),
        _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", max_model_len=32768),
    ):
        with pytest.raises(OomError) as ei:
            pm.check_launch()
        assert classify_log(ExitInfo(code=1, log_tail=str(ei.value))) == "oom"
