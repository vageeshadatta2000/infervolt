from infervolt.core.types import LoadResult, Workload
from infervolt.loadgen.base import LoadGenerator


class _Stub:
    def run(
        self, workload: Workload, concurrency: int, num_requests: int, seed: int
    ) -> LoadResult:  # pragma: no cover - never called
        return LoadResult(concurrency=concurrency, duration_s=1.0, requests=[])


class _NotAGenerator:
    pass


def test_load_generator_protocol_is_runtime_checkable() -> None:
    assert isinstance(_Stub(), LoadGenerator)
    assert not isinstance(_NotAGenerator(), LoadGenerator)
