"""How ``--model`` is resolved. Nothing is downloaded: the hub calls are monkeypatched."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from infervolt.models import catalog, hf_meta
from infervolt.models.catalog import MODELS, get_model_info
from tests.models.test_gguf_meta import write_gguf


def test_static_ids_come_from_the_table() -> None:
    assert get_model_info("mock/qwen3-8b").params_b == 8.2


def test_static_lookup_returns_a_copy() -> None:
    got = get_model_info("mock/qwen3-8b")
    got.params_b = 1.0
    assert MODELS["mock/qwen3-8b"].params_b == 8.2


def test_unknown_short_ids_still_name_the_catalog() -> None:
    with pytest.raises(KeyError, match="unknown model 'nope'"):
        get_model_info("nope")


def test_a_local_gguf_path_is_read_in_place(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "tiny.gguf")
    info = get_model_info(str(path))
    assert info.arch == "qwen3"
    assert info.num_layers == 4


def test_an_hf_gguf_id_downloads_then_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = write_gguf(tmp_path / "q4.gguf", tensors={"t": np.zeros((4, 4), dtype=np.float32)})
    seen: list[tuple[str, str]] = []

    def fake_download(repo_id: str, filename: str) -> str:
        seen.append((repo_id, filename))
        return str(path)

    monkeypatch.setattr(hf_meta, "download_hf_file", fake_download)
    info = get_model_info("hf:Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    assert seen == [("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")]
    assert info.arch == "qwen3"


def test_a_malformed_hf_gguf_id_is_rejected() -> None:
    with pytest.raises(KeyError, match="malformed GGUF id"):
        get_model_info("hf:justafile.gguf")


def test_a_repo_id_is_resolved_from_its_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config: dict[str, Any] = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
    }
    seen: list[str] = []

    def fake_fetch(model_id: str) -> dict[str, Any]:
        seen.append(model_id)
        return config

    monkeypatch.setattr(hf_meta, "fetch_hf_config", fake_fetch)
    info = get_model_info("Qwen/Qwen3-8B")
    assert seen == ["Qwen/Qwen3-8B"]
    assert info.id == "Qwen/Qwen3-8B"
    assert info.num_layers == 36


def test_catalog_entries_win_over_a_hub_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mock/qwen3-8b`` has a slash but is not a repo; the table answers first."""

    def explode(model_id: str) -> dict[str, Any]:
        raise AssertionError(f"should not have hit the hub for {model_id}")

    monkeypatch.setattr(hf_meta, "fetch_hf_config", explode)
    assert get_model_info("mock/qwen3-8b").arch == "qwen3"


def test_the_hf_prefix_is_checked_before_the_gguf_suffix() -> None:
    # Both rules match "hf:...gguf"; if the suffix won, this would be read as a path.
    assert catalog.HF_GGUF_PREFIX == "hf:"
    with pytest.raises(KeyError, match="malformed GGUF id"):
        get_model_info("hf:x.gguf")
