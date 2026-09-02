import pytest

from infervolt.models.catalog import MODELS, get_model_info
from infervolt.workloads.presets import PRESETS, get_workload, parse_slo


def test_model_catalog_has_mock_models() -> None:
    m = get_model_info("mock/qwen3-8b")
    assert m.params_b == 8.2 and m.num_kv_heads == 8


def test_workload_presets() -> None:
    w = get_workload("chat-4k-512")
    assert w.isl.p50 == 4096 and w.osl.p50 == 512
    assert get_workload("agentic-prefix-16k-512").prefix_share > 0.5


def test_get_model_info_returns_a_copy() -> None:
    m = get_model_info("mock/qwen3-8b")
    m.params_b = 1.0
    assert MODELS["mock/qwen3-8b"].params_b == 8.2
    assert get_model_info("mock/qwen3-8b").params_b == 8.2


def test_get_workload_returns_a_deep_copy() -> None:
    w = get_workload("chat-4k-512")
    w.isl.p50 = 1
    w.load.concurrency.append(999)
    assert PRESETS["chat-4k-512"].isl.p50 == 4096
    assert 999 not in PRESETS["chat-4k-512"].load.concurrency


def test_parse_slo() -> None:
    slo = parse_slo("ttft=500ms,itl=30ms")
    assert slo.ttft_ms == 500 and slo.itl_ms == 30 and slo.e2e_ms is None
    assert parse_slo("e2e=2s,p=0.95").e2e_ms == 2000
    assert parse_slo("g=0.85").goodput_target == 0.85
    assert parse_slo("").ttft_ms is None


def test_parse_slo_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        parse_slo("latency=1ms")
