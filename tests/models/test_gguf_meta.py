"""GGUF metadata reading, against files written by ``gguf.GGUFWriter`` in tmp.

The files are a few kilobytes of real GGUF -- header, KV metadata and tiny tensors --
so the reader is exercised end to end without anything being downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gguf import GGUFWriter, LlamaFileType

from infervolt.models.gguf_meta import (
    WEIGHT_BITS_BY_FILE_TYPE,
    read_gguf_model_info,
)


def write_gguf(
    path: Path,
    arch: str = "qwen3",
    *,
    tensors: dict[str, np.ndarray[Any, Any]] | None = None,
    **kv: Any,
) -> Path:
    """A minimal but genuine GGUF file. ``kv`` overrides any metadata default."""
    fields: dict[str, Any] = {
        "block_count": 4,
        "embedding_length": 64,
        "head_count": 4,
        "head_count_kv": 2,
        "key_length": 16,
        "context_length": 2048,
        "file_type": int(LlamaFileType.MOSTLY_Q4_K_M),
    }
    fields.update(kv)
    writer = GGUFWriter(path, arch)
    if (name := fields.pop("name", None)) is not None:
        writer.add_name(name)
    adders = {
        "block_count": writer.add_block_count,
        "embedding_length": writer.add_embedding_length,
        "head_count": writer.add_head_count,
        "head_count_kv": writer.add_head_count_kv,
        "key_length": writer.add_key_length,
        "context_length": writer.add_context_length,
        "file_type": writer.add_file_type,
        "expert_count": writer.add_expert_count,
        "expert_used_count": writer.add_expert_used_count,
    }
    for key, value in fields.items():
        if value is not None:
            adders[key](value)
    for tensor_name, array in (
        tensors or {"blk.0.attn_q.weight": np.zeros((8, 8), dtype=np.float32)}
    ).items():
        writer.add_tensor(tensor_name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_reads_the_architecture_and_geometry(tmp_path: Path) -> None:
    info = read_gguf_model_info(write_gguf(tmp_path / "tiny.gguf"))
    assert info.arch == "qwen3"
    assert info.num_layers == 4
    assert info.hidden == 64
    assert info.num_kv_heads == 2
    assert info.head_dim == 16
    assert info.max_pos == 2048
    assert info.moe is False
    assert info.active_params_b is None


def test_id_falls_back_to_the_filename_and_prefers_general_name(tmp_path: Path) -> None:
    assert read_gguf_model_info(write_gguf(tmp_path / "tiny.gguf")).id == "tiny"
    named = write_gguf(tmp_path / "other.gguf", name="Qwen3 8B")
    assert read_gguf_model_info(named).id == "Qwen3 8B"


def test_head_dim_falls_back_to_hidden_over_heads(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", key_length=None, embedding_length=64, head_count=8)
    assert read_gguf_model_info(path).head_dim == 8


def test_kv_heads_fall_back_to_the_attention_head_count(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", head_count_kv=None, head_count=4)
    assert read_gguf_model_info(path).num_kv_heads == 4


def test_weight_bits_come_from_the_file_type(tmp_path: Path) -> None:
    for ftype, bits in [(LlamaFileType.MOSTLY_Q4_K_M, 4), (LlamaFileType.MOSTLY_F16, 16)]:
        path = write_gguf(tmp_path / f"{int(ftype)}.gguf", file_type=int(ftype))
        assert read_gguf_model_info(path).weight_bits == bits


def test_an_unknown_file_type_assumes_sixteen_bit(tmp_path: Path) -> None:
    unknown = max(WEIGHT_BITS_BY_FILE_TYPE) + 50
    path = write_gguf(tmp_path / "m.gguf", file_type=unknown)
    assert read_gguf_model_info(path).weight_bits == 16


def test_params_come_from_the_tensor_index(tmp_path: Path) -> None:
    tensors = {
        "blk.0.attn_q.weight": np.zeros((1000, 1000), dtype=np.float32),
        "blk.0.attn_k.weight": np.zeros((500, 1000), dtype=np.float32),
    }
    info = read_gguf_model_info(write_gguf(tmp_path / "m.gguf", tensors=tensors))
    assert info.params_b == pytest.approx(1.5e6 / 1e9)


def test_moe_active_params_count_only_the_routed_expert_share(tmp_path: Path) -> None:
    tensors = {
        # 1M dense parameters and 8M spread across 8 experts, 2 of which run per token,
        # so the active count is 1M + 2M.
        "blk.0.attn_q.weight": np.zeros((1000, 1000), dtype=np.float32),
        "blk.0.ffn_down_exps.weight": np.zeros((8, 1000, 1000), dtype=np.float32),
    }
    path = write_gguf(tmp_path / "moe.gguf", tensors=tensors, expert_count=8, expert_used_count=2)
    info = read_gguf_model_info(path)
    assert info.moe is True
    assert info.params_b == pytest.approx(9e6 / 1e9)
    assert info.active_params_b == pytest.approx(3e6 / 1e9)


def test_a_single_expert_is_not_a_mixture(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", expert_count=1, expert_used_count=1)
    info = read_gguf_model_info(path)
    assert info.moe is False
    assert info.active_params_b is None


def test_a_file_without_decoder_geometry_is_rejected(tmp_path: Path) -> None:
    # What a projector or an adapter shard looks like: an arch key, and nothing under it.
    path = write_gguf(tmp_path / "m.gguf", arch="clip", block_count=0, embedding_length=0)
    with pytest.raises(ValueError, match="block_count"):
        read_gguf_model_info(path)
