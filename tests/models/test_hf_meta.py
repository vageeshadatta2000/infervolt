"""Mapping a Hugging Face ``config.json`` to a ModelInfo. No network: the configs are
literals shaped like the real ones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from infervolt.models import hf_meta
from infervolt.models.hf_meta import (
    estimate_params_b,
    fetch_hf_config,
    model_info_from_hf_config,
)

QWEN3_8B: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}

LLAMA_70B: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}


def test_qwen3_8b_shaped_config() -> None:
    info = model_info_from_hf_config("Qwen/Qwen3-8B", QWEN3_8B)
    assert info.id == "Qwen/Qwen3-8B"
    assert info.arch == "qwen3"
    assert (info.num_layers, info.hidden) == (36, 4096)
    assert (info.num_kv_heads, info.head_dim) == (8, 128)
    assert info.max_pos == 40960
    assert info.weight_bits == 16
    assert info.moe is False
    assert info.active_params_b is None
    # 12 * 36 * 4096^2 is 7.2B against a real 8.2B: the documented order of accuracy.
    assert info.params_b == pytest.approx(7.25, abs=0.1)


def test_llama_70b_shaped_config() -> None:
    info = model_info_from_hf_config("meta-llama/Llama-3.1-70B", LLAMA_70B)
    assert (info.num_layers, info.hidden) == (80, 8192)
    assert info.num_kv_heads == 8
    # No head_dim key: hidden / heads.
    assert info.head_dim == 128
    assert info.max_pos == 131072
    assert info.params_b == pytest.approx(64.4, abs=0.5)


def test_head_dim_and_kv_heads_fall_back_to_the_attention_heads() -> None:
    config = {"num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 8}
    info = model_info_from_hf_config("x/y", config)
    assert info.num_kv_heads == 8
    assert info.head_dim == 32


def test_max_pos_defaults_when_the_config_omits_it() -> None:
    config = {"num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 8}
    assert model_info_from_hf_config("x/y", config).max_pos == hf_meta.DEFAULT_MAX_POS


@pytest.mark.parametrize(
    ("dtype", "bits"),
    [("bfloat16", 16), ("float16", 16), ("float32", 32), ("float8_e4m3fn", 8)],
)
def test_torch_dtype_maps_to_weight_bits(dtype: str, bits: int) -> None:
    config = {**QWEN3_8B, "torch_dtype": dtype}
    assert model_info_from_hf_config("x/y", config).weight_bits == bits


def test_the_newer_dtype_key_is_accepted_too() -> None:
    config = {k: v for k, v in QWEN3_8B.items() if k != "torch_dtype"} | {"dtype": "float32"}
    assert model_info_from_hf_config("x/y", config).weight_bits == 32


def test_an_unknown_dtype_assumes_sixteen_bit() -> None:
    config = {**QWEN3_8B, "torch_dtype": "something_new"}
    assert model_info_from_hf_config("x/y", config).weight_bits == 16


def test_a_quantization_config_beats_the_tensor_dtype() -> None:
    config = {**QWEN3_8B, "quantization_config": {"quant_method": "awq", "bits": 4}}
    assert model_info_from_hf_config("x/y", config).weight_bits == 4


def test_mixture_of_experts_fields() -> None:
    config = {
        **QWEN3_8B,
        "model_type": "qwen3_moe",
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }
    info = model_info_from_hf_config("Qwen/Qwen3-30B-A3B", config)
    assert info.moe is True
    assert info.active_params_b is not None
    # One part attention to two parts MLP: (1 + 2*8) / (1 + 2*128) of the total.
    assert info.active_params_b == pytest.approx(info.params_b * 17 / 257)
    assert info.params_b > estimate_params_b(36, 4096)


def test_mixtral_style_expert_keys_are_recognised() -> None:
    config = {**QWEN3_8B, "num_local_experts": 8, "num_experts_per_tok": 2}
    info = model_info_from_hf_config("x/y", config)
    assert info.moe is True
    assert info.active_params_b == pytest.approx(info.params_b * 5 / 17)


def test_a_declared_parameter_count_beats_the_estimate() -> None:
    config = {**QWEN3_8B, "num_parameters": 8_190_000_000}
    assert model_info_from_hf_config("x/y", config).params_b == pytest.approx(8.19)


def test_multimodal_configs_describe_their_text_tower() -> None:
    config = {
        "model_type": "gemma3",
        "vision_config": {"hidden_size": 1152},
        "text_config": QWEN3_8B,
    }
    info = model_info_from_hf_config("x/y", config)
    assert info.num_layers == 36
    assert info.hidden == 4096


def test_a_config_with_no_decoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not describe a decoder"):
        model_info_from_hf_config("x/y", {"model_type": "clip"})


def test_a_config_with_no_head_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="head geometry"):
        model_info_from_hf_config("x/y", {"num_hidden_layers": 2, "hidden_size": 256})


def test_fetch_reads_the_downloaded_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(QWEN3_8B), encoding="utf-8")
    seen: list[tuple[str, str]] = []

    def fake_download(repo_id: str, filename: str) -> str:
        seen.append((repo_id, filename))
        return str(path)

    monkeypatch.setattr(hf_meta, "hf_hub_download", fake_download)
    assert fetch_hf_config("Qwen/Qwen3-8B") == QWEN3_8B
    assert seen == [("Qwen/Qwen3-8B", "config.json")]


def test_fetch_rejects_a_config_that_is_not_an_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")
    monkeypatch.setattr(hf_meta, "hf_hub_download", lambda *_: str(path))
    with pytest.raises(ValueError, match="not a JSON object"):
        fetch_hf_config("x/y")
