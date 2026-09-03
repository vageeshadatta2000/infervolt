"""A provider and a session that record what the runner did to them.

No sockets, no subprocesses, no money: the fakes answer the two questions the runner's
tests are about -- in what order did it do things, and did it stop paying afterwards.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from infervolt.infra.base import Provider
from infervolt.infra.types import (
    Instance,
    InstanceSpec,
    LaunchMode,
    Offer,
    ProvisionError,
    RunResult,
    SshTarget,
)
from infervolt.remote.bootstrap import BOOTSTRAP_OK, WORK_MARKER
from infervolt.store.ledger import Ledger

RUN_ID = "2026-09-01T12-00-00Z-ab12"
WORK = "/workspace/infervolt"


class FakeSession:
    """Canned answers keyed off what the command obviously is."""

    def __init__(
        self,
        *,
        bootstrap_ok: bool = True,
        bootstrap_code: int = 0,
        optimize_code: int = 0,
        run_id: str | None = RUN_ID,
        work: str = WORK,
        block_optimize: bool = False,
    ) -> None:
        self.bootstrap_ok = bootstrap_ok
        self.bootstrap_code = bootstrap_code
        self.optimize_code = optimize_code
        self.run_id = run_id
        self.work = work
        self.block_optimize = block_optimize
        self.commands: list[str] = []
        self.gets: list[tuple[str, Path]] = []
        self.closed = False
        self.killed = threading.Event()

    # ---- Session
    def run(
        self,
        cmd: str,
        *,
        stream: Callable[[str], None] | None = None,
        timeout_s: float = 3600.0,
    ) -> RunResult:
        self.commands.append(cmd)
        code, lines = self._answer(cmd)
        for line in lines:
            if stream is not None:
                stream(line)
        return RunResult(code=code, stdout_tail="\n".join(lines))

    def _answer(self, cmd: str) -> tuple[int, list[str]]:
        if "pkill" in cmd:
            self.killed.set()
            return 0, []
        if BOOTSTRAP_OK in cmd:
            lines = [f"{WORK_MARKER}{self.work}", "Installed 1 package"]
            if self.bootstrap_ok:
                lines.append(BOOTSTRAP_OK)
            return self.bootstrap_code, lines
        if "optimize" in cmd:
            if self.block_optimize:
                # Stands in for a run that would outlast the budget: it ends only when
                # the cap's pkill arrives.
                self.killed.wait(timeout=10.0)
                return 143, ["baseline ..."]
            lines = ["baseline p95_ttft_ms=812"]
            if self.run_id is not None:
                lines.insert(0, f"run: {self.run_id}")
            return self.optimize_code, lines
        return 0, []

    def put(self, local: Path, remote: str) -> None:  # pragma: no cover - unused here
        raise AssertionError("the runner should not push files")

    def get(self, remote: str, local: Path) -> None:
        self.gets.append((remote, Path(local)))
        Path(local).mkdir(parents=True, exist_ok=True)
        (Path(local) / "recipe.yaml").write_text("engine: vllm\n")

    def tunnel(self, remote_port: int) -> int:  # pragma: no cover - unused here
        return remote_port

    def close(self) -> None:
        self.closed = True


class FakeProvider(Provider):
    """Provisions an instance out of a dict and never touches the network."""

    name: ClassVar[str] = "fake"
    launch_mode: ClassVar[LaunchMode] = "vm"

    def __init__(
        self,
        ledger: Ledger | None = None,
        *,
        session: FakeSession | None = None,
        usd_per_hour: float = 1.09,
        fail_provision: bool = False,
    ) -> None:
        super().__init__(ledger)
        self.session = session or FakeSession()
        self.usd_per_hour = usd_per_hour
        self.fail_provision = fail_provision
        self.provisioned: list[InstanceSpec] = []
        self.waited: list[str] = []
        self.terminated: list[str] = []

    def list_offers(self, gpu: str | None = None) -> list[Offer]:  # pragma: no cover
        return []

    def provision(self, spec: InstanceSpec) -> Instance:
        if self.fail_provision:
            raise ProvisionError("no capacity for a100xl")
        self.provisioned.append(spec)
        return self._record(
            Instance(
                provider=self.name,
                id="i-fake",
                gpu=spec.gpu,
                count=spec.count,
                usd_per_hour=self.usd_per_hour,
                status="starting",
            )
        )

    def refresh(self, inst: Instance) -> Instance:
        target = SshTarget(host="203.0.113.7", key_path="/dev/null")
        return inst.model_copy(update={"status": "running", "ssh": target})

    def wait_ready(self, inst: Instance, timeout_s: float = 900.0, **kwargs: object) -> Instance:
        self.waited.append(inst.id)
        return self.refresh(inst)

    def connect(self, inst: Instance) -> FakeSession:
        return self.session

    def terminate(self, inst: Instance) -> None:
        self.terminated.append(inst.id)
        self._record_terminated(inst)

    def cost_per_hour(self, inst: Instance) -> float:
        return self.usd_per_hour


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def provider(session: FakeSession) -> FakeProvider:
    return FakeProvider(session=session)
