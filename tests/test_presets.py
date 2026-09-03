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


def test_parse_slo_tolerates_whitespace_around_keys_and_values() -> None:
    slo = parse_slo(" ttft = 500ms , itl =  30 ms ,  p = 0.95 ")
    assert slo.ttft_ms == 500
    assert slo.itl_ms == 30
    assert slo.percentile == 0.95


def test_parse_slo_matches_the_longest_unit_suffix_first() -> None:
    # 'ms' must win over 's', or 500ms would be read as 500 000 ms.
    assert parse_slo("ttft=500ms").ttft_ms == 500
    assert parse_slo("ttft=500s").ttft_ms == 500_000


def test_parse_slo_rejects_a_clause_without_a_value() -> None:
    with pytest.raises(ValueError, match="malformed SLO clause 'ttft'"):
        parse_slo("ttft")
    with pytest.raises(ValueError, match="malformed SLO clause"):
        parse_slo("ttft=fast")


@pytest.mark.parametrize("text", ["p=95", "p=0", "p=-0.5", "p=1.5", "g=95", "g=0", "g=1.01"])
def test_parse_slo_rejects_out_of_range_fractions(text: str) -> None:
    with pytest.raises(ValueError):
        parse_slo(text)


@pytest.mark.parametrize("text", ["ttft=-1ms", "itl=-30ms", "e2e=-2s"])
def test_parse_slo_rejects_negative_durations(text: str) -> None:
    with pytest.raises(ValueError):
        parse_slo(text)


def test_parse_slo_accepts_the_boundary_values() -> None:
    assert parse_slo("p=1,g=1").percentile == 1.0
    assert parse_slo("ttft=0ms").ttft_ms == 0.0
