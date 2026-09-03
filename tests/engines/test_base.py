import pytest

from infervolt.engines.base import ExitInfo, classify_log
from infervolt.engines.registry import get_adapter


@pytest.mark.parametrize(
    "code,log,kind",
    [
        (1, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "oom"),
        (1, "No available memory for the cache blocks", "oom"),
        (1, "is larger than the maximum number of tokens that can be stored in KV cache", "oom"),
        (-9, "", "oom"),
        (1, "ggml_metal_graph_compute: failed to allocate", "oom"),
        (1, "RuntimeError: something else", "runtime"),
        (0, "", "none"),
        (124, "", "timeout"),
    ],
)
def test_classify_log(code: int, log: str, kind: str) -> None:
    assert classify_log(ExitInfo(code=code, log_tail=log)) == kind


def test_registry_unknown_engine() -> None:
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")
