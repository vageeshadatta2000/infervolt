"""Resolve a ``--model`` argument to a :class:`ModelInfo`.

Four kinds of identifier are understood, checked in this order:

``hf:<repo>/<file.gguf>``
    A GGUF inside a Hugging Face repo: downloaded, then read as a GGUF.
``*.gguf``
    A GGUF already on this machine, read in place.
``mock/...`` and other catalog keys
    The static table below, which is what the simulator and the tests run on.
``<org>/<name>``
    Any other id containing a slash is a Hugging Face repo: its ``config.json`` is
    fetched (a few kilobytes, never the weights) and mapped.

The ordering matters in one place only: ``hf:`` is checked before the ``.gguf`` suffix,
because an ``hf:`` id usually ends in ``.gguf`` too.
"""

from __future__ import annotations

from infervolt.core.types import ModelInfo
from infervolt.models import hf_meta
from infervolt.models.gguf_meta import read_gguf_model_info

HF_GGUF_PREFIX = "hf:"

MODELS: dict[str, ModelInfo] = {
    "mock/qwen3-0.6b": ModelInfo(
        id="mock/qwen3-0.6b",
        arch="qwen3",
        params_b=0.6,
        num_layers=28,
        hidden=1024,
        num_kv_heads=8,
        head_dim=128,
        max_pos=40960,
    ),
    "mock/qwen3-8b": ModelInfo(
        id="mock/qwen3-8b",
        arch="qwen3",
        params_b=8.2,
        num_layers=36,
        hidden=4096,
        num_kv_heads=8,
        head_dim=128,
        max_pos=40960,
    ),
    "mock/llama-70b": ModelInfo(
        id="mock/llama-70b",
        arch="llama",
        params_b=70.0,
        num_layers=80,
        hidden=8192,
        num_kv_heads=8,
        head_dim=128,
        max_pos=131072,
    ),
}


def _from_hf_gguf(model_id: str) -> ModelInfo:
    """``hf:Org/Repo/path/to/file.gguf`` -> download that file, then read it."""
    spec = model_id[len(HF_GGUF_PREFIX) :]
    repo_id, _, filename = spec.rpartition("/")
    if not repo_id or not filename:
        raise KeyError(f"malformed GGUF id {model_id!r}; expected {HF_GGUF_PREFIX}<repo>/<file>")
    return read_gguf_model_info(hf_meta.download_hf_file(repo_id, filename))


def get_model_info(model_id: str) -> ModelInfo:
    if model_id.startswith(HF_GGUF_PREFIX):
        return _from_hf_gguf(model_id)
    if model_id.endswith(".gguf"):
        return read_gguf_model_info(model_id)
    if model_id in MODELS:
        # A private copy, so callers can adapt an entry without editing the registry.
        return MODELS[model_id].model_copy(deep=True)
    if "/" in model_id:
        return hf_meta.model_info_from_hf_config(model_id, hf_meta.fetch_hf_config(model_id))
    raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODELS)}")
