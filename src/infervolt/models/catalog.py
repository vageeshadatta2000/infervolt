"""Static model catalog. M3 adds resolution of real Hugging Face ids from config.json."""

from __future__ import annotations

from infervolt.core.types import ModelInfo

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


def get_model_info(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id].model_copy(deep=True)
    except KeyError as e:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODELS)}") from e
