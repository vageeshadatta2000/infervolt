"""Read a :class:`ModelInfo` out of a GGUF file's metadata.

A GGUF file already carries everything the roofline and the KV-cache arithmetic need --
layer count, hidden size, KV head count, head dim, context length and the quantisation
the weights were written in -- so a llama.cpp run never has to be told what it is
serving. Parameter counts come from the tensor index rather than from any metadata key,
because the count is the one number a quantised repack genuinely changes and the one no
GGUF writer is obliged to record.

Keys are namespaced by architecture (``qwen3.block_count``, ``llama.block_count``, ...),
which is why ``general.architecture`` is read first and everything else through it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gguf import GGUFReader

from infervolt.core.types import ModelInfo

WEIGHT_BITS_BY_FILE_TYPE: dict[int, int] = {
    0: 32,  # ALL_F32
    1: 16,  # MOSTLY_F16
    2: 4,  # MOSTLY_Q4_0
    3: 4,  # MOSTLY_Q4_1
    4: 4,  # MOSTLY_Q4_1_SOME_F16
    7: 8,  # MOSTLY_Q8_0
    8: 5,  # MOSTLY_Q5_0
    9: 5,  # MOSTLY_Q5_1
    10: 2,  # MOSTLY_Q2_K
    11: 3,  # MOSTLY_Q3_K_S
    12: 3,  # MOSTLY_Q3_K_M
    13: 3,  # MOSTLY_Q3_K_L
    14: 4,  # MOSTLY_Q4_K_S
    15: 4,  # MOSTLY_Q4_K_M
    16: 5,  # MOSTLY_Q5_K_S
    17: 5,  # MOSTLY_Q5_K_M
    18: 6,  # MOSTLY_Q6_K
    19: 2,  # MOSTLY_IQ2_XXS
    20: 2,  # MOSTLY_IQ2_XS
    21: 2,  # MOSTLY_Q2_K_S
    22: 3,  # MOSTLY_IQ3_XS
    23: 3,  # MOSTLY_IQ3_XXS
    24: 1,  # MOSTLY_IQ1_S
    25: 4,  # MOSTLY_IQ4_NL
    26: 3,  # MOSTLY_IQ3_S
    27: 3,  # MOSTLY_IQ3_M
    28: 2,  # MOSTLY_IQ2_S
    29: 2,  # MOSTLY_IQ2_M
    30: 4,  # MOSTLY_IQ4_XS
    31: 1,  # MOSTLY_IQ1_M
    32: 16,  # MOSTLY_BF16
    36: 8,  # MOSTLY_TQ1_0 / ternary family, rounded up to a byte
    37: 8,  # MOSTLY_TQ2_0
}
"""``general.file_type`` (llama.cpp's ``LlamaFileType``) to bits per weight.

Nominal bits, not the effective rate: the k-quants spend a little extra on per-block
scales, and a mixed file type such as ``Q4_K_M`` keeps some tensors at higher precision.
The number is used to size the weights against device memory, where a few percent under
is the right kind of wrong -- it is the *cache* estimate that must not be optimistic.
"""

DEFAULT_WEIGHT_BITS = 16
"""Assumed when the file records no ``general.file_type``."""

EXPERT_TENSOR_MARKER = "_exps"
"""Substring llama.cpp uses for fused expert tensors (``blk.0.ffn_down_exps.weight``)."""


def _scalar(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    return field.contents() if field is not None else None


def _int(reader: GGUFReader, key: str, default: int = 0) -> int:
    value = _scalar(reader, key)
    return int(value) if isinstance(value, (int, float)) else default


def read_gguf_model_info(path: str | Path) -> ModelInfo:
    """Describe the model in the GGUF file at ``path``.

    Raises ``ValueError`` when the file does not carry the architecture metadata a
    serving engine needs -- a projector or an adapter shard, say, rather than a model.
    """
    reader = GGUFReader(Path(path))
    arch = _scalar(reader, "general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{path}: no general.architecture; this is not a model GGUF")

    layers = _int(reader, f"{arch}.block_count")
    hidden = _int(reader, f"{arch}.embedding_length")
    if layers <= 0 or hidden <= 0:
        raise ValueError(f"{path}: missing {arch}.block_count / {arch}.embedding_length")

    heads = _int(reader, f"{arch}.attention.head_count")
    kv_heads = _int(reader, f"{arch}.attention.head_count_kv", default=heads) or heads
    head_dim = _int(reader, f"{arch}.attention.key_length") or (hidden // heads if heads else 0)
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"{path}: cannot determine KV head count or head dim for {arch}")

    file_type = _scalar(reader, "general.file_type")
    weight_bits = DEFAULT_WEIGHT_BITS
    if isinstance(file_type, (int, float)):
        weight_bits = WEIGHT_BITS_BY_FILE_TYPE.get(int(file_type), DEFAULT_WEIGHT_BITS)

    experts = _int(reader, f"{arch}.expert_count")
    used_experts = _int(reader, f"{arch}.expert_used_count")

    total = sum(int(t.n_elements) for t in reader.tensors)
    # Only the expert stacks are sparsely activated; attention, embeddings and any shared
    # FFN run for every token, so the active count is the dense part plus the routed
    # fraction of the experts. Named tensors make that split exact rather than assumed.
    expert_elements = sum(
        int(t.n_elements) for t in reader.tensors if EXPERT_TENSOR_MARKER in t.name
    )
    moe = experts > 1
    active_b: float | None = None
    if moe and used_experts > 0:
        active = (total - expert_elements) + expert_elements * used_experts / experts
        active_b = active / 1e9

    name = _scalar(reader, "general.name")
    return ModelInfo(
        id=name if isinstance(name, str) and name else Path(path).stem,
        arch=arch,
        params_b=total / 1e9,
        active_params_b=active_b,
        num_layers=layers,
        hidden=hidden,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        weight_bits=weight_bits,
        max_pos=_int(reader, f"{arch}.context_length", default=32768) or 32768,
        moe=moe,
    )
