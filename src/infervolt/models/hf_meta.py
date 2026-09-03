"""Turn a Hugging Face ``config.json`` into a :class:`ModelInfo`.

``config.json`` is the only description of a model that every serving engine agrees on,
it is a few kilobytes, and it is what is already on the box next to the weights -- so a
run against ``Qwen/Qwen3-8B`` never needs a hand-written catalog entry.

The mapping is mostly one-to-one. The interesting part is what the file does *not* say:
no config records its own parameter count, and the KV-cache arithmetic downstream cares
about the total (weights against device memory) and, for a mixture of experts, the
active fraction (bandwidth per decoded token). Both are estimated here, deliberately
crudely and always documented as estimates -- see :func:`estimate_params_b`.
"""

from __future__ import annotations

import json
from typing import Any

from huggingface_hub import hf_hub_download

from infervolt.core.types import ModelInfo

WEIGHT_BITS_BY_DTYPE: dict[str, int] = {
    "float32": 32,
    "float": 32,
    "float16": 16,
    "half": 16,
    "bfloat16": 16,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "int8": 8,
    "uint8": 8,
    "int4": 4,
}

DEFAULT_WEIGHT_BITS = 16
DEFAULT_MAX_POS = 32768

_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_EXPERT_USED_KEYS = ("num_experts_per_tok", "num_experts_per_token")
_PARAM_COUNT_KEYS = ("num_parameters", "num_params", "total_params")

_ATTENTION_SHARE = 1.0
_MLP_SHARE = 2.0
"""How ``12 * layers * hidden^2`` splits between attention and the MLP.

A dense decoder layer is ``4 * hidden^2`` of attention projections and ``8 * hidden^2``
of MLP (three matrices against an intermediate width of roughly ``4 * hidden``), so the
classic estimate is one part attention to two parts MLP. Only the MLP is replicated
across experts, which is what makes the split worth naming.
"""


def _int(config: dict[str, Any], key: str, default: int = 0) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _first_int(config: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if (value := _int(config, key)) > 0:
            return value
    return 0


def estimate_params_b(layers: int, hidden: int, experts: int = 1) -> float:
    """The classic ``12 * layers * hidden^2`` transformer estimate, in billions.

    It is an approximation and a rough one: it ignores embedding and output matrices,
    assumes an MLP intermediate width of ``4 * hidden``, ignores the shrinkage grouped
    query attention gives the K and V projections, and takes no account of tied weights.
    For the models this tool targets it lands within roughly 10-20% of the truth, which
    is enough to size weights against device memory and no more than that. A real count
    from the config (or from a GGUF's tensor index) is always preferred.

    ``experts`` scales only the MLP share, since attention is not replicated.
    """
    per_layer = hidden * hidden * (_ATTENTION_SHARE + _MLP_SHARE * experts) * 4
    return layers * per_layer / 1e9


def weight_bits_from_config(config: dict[str, Any]) -> int:
    """Bits per stored weight: an explicit quantisation config beats the tensor dtype."""
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        bits = quant.get("bits")
        if isinstance(bits, int) and not isinstance(bits, bool) and bits > 0:
            return bits
    # ``torch_dtype`` was renamed ``dtype`` in recent transformers; both are in the wild.
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if isinstance(value, str) and value in WEIGHT_BITS_BY_DTYPE:
            return WEIGHT_BITS_BY_DTYPE[value]
    return DEFAULT_WEIGHT_BITS


def model_info_from_hf_config(model_id: str, config: dict[str, Any]) -> ModelInfo:
    """Describe ``model_id`` from its ``config.json`` contents.

    Raises ``ValueError`` when the config carries no decoder description at all, which
    is what a tokenizer-only or adapter-only repo looks like.
    """
    # Multimodal repos put the decoder under ``text_config`` and keep only the vision
    # tower at the top level; the language model is the thing being served.
    inner = config.get("text_config")
    if "num_hidden_layers" not in config and isinstance(inner, dict):
        config = inner

    layers = _int(config, "num_hidden_layers")
    hidden = _int(config, "hidden_size")
    if layers <= 0 or hidden <= 0:
        raise ValueError(
            f"{model_id}: config.json has no num_hidden_layers/hidden_size; "
            "it does not describe a decoder"
        )

    heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads", default=heads) or heads
    head_dim = _int(config, "head_dim") or (hidden // heads if heads else 0)
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"{model_id}: config.json has no usable attention head geometry")

    experts = _first_int(config, _EXPERT_COUNT_KEYS)
    used_experts = _first_int(config, _EXPERT_USED_KEYS)
    moe = experts > 1

    declared = _first_int(config, _PARAM_COUNT_KEYS)
    params_b = declared / 1e9 if declared else estimate_params_b(layers, hidden, max(experts, 1))
    active_b: float | None = None
    if moe and used_experts > 0:
        # Same split as the estimate, applied to whatever the total turned out to be, so
        # a declared total and an estimated one scale the same way.
        total_share = _ATTENTION_SHARE + _MLP_SHARE * experts
        active_share = _ATTENTION_SHARE + _MLP_SHARE * used_experts
        active_b = params_b * active_share / total_share

    return ModelInfo(
        id=model_id,
        arch=str(config.get("model_type") or ""),
        params_b=params_b,
        active_params_b=active_b,
        num_layers=layers,
        hidden=hidden,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        weight_bits=weight_bits_from_config(config),
        max_pos=_int(config, "max_position_embeddings", default=DEFAULT_MAX_POS) or DEFAULT_MAX_POS,
        moe=moe,
    )


def fetch_hf_config(model_id: str) -> dict[str, Any]:
    """Download and parse ``config.json`` for a Hugging Face repo id.

    Only the config is fetched -- never the weights -- so this is a few kilobytes even
    for a 70B model, and the hub cache makes a repeat call free.
    """
    path = hf_hub_download(model_id, "config.json")
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"{model_id}: config.json is not a JSON object")
    return config


def download_hf_file(repo_id: str, filename: str) -> str:
    """Fetch one file from a repo and return its local path (used for GGUF weights)."""
    return str(hf_hub_download(repo_id, filename))
