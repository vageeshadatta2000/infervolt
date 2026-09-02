"""Load generator protocol.

M1 ships the mock simulator; M2 adds the multi-process HTTP generator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from infervolt.core.types import LoadResult, Workload


@runtime_checkable
class LoadGenerator(Protocol):
    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        """Drive `num_requests` requests at fixed closed-loop `concurrency` and return records."""
        ...
