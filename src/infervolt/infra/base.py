"""The provider contract: rent it, reach it, price it, give it back.

Five methods are provider-specific (`list_offers`, `provision`, `refresh`, `terminate`,
`cost_per_hour`); everything a run actually does with a box -- wait for it, open a
session, record it in the ledger -- is implemented once, here, on top of them.
"""

from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from infervolt.infra.ssh import Session, SshSession
from infervolt.infra.types import Instance, InstanceSpec, LaunchMode, Offer, ProvisionError

if TYPE_CHECKING:  # pragma: no cover - import cycle only exists for the type checker
    from infervolt.store.ledger import Ledger


def tcp_open(host: str, port: int, timeout_s: float = 5.0) -> bool:
    """Can we open a TCP connection right now?

    "The provider says RUNNING" and "sshd is accepting connections" are minutes apart on
    a fresh VM, and only the second one means the next command will work.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class Provider(ABC):
    """One place to rent compute from."""

    name: ClassVar[str] = ""
    launch_mode: ClassVar[LaunchMode] = "vm"
    supports_scp: ClassVar[bool] = True
    """False for providers whose only SSH path is a proxy that refuses file transfer.

    It decides whether artifacts can be pulled back over the same connection that ran the
    work, or whether the run has to ship them somewhere else first.
    """

    def __init__(self, ledger: Ledger | None = None) -> None:
        self.ledger = ledger

    # ---- provider-specific
    @abstractmethod
    def list_offers(self, gpu: str | None = None) -> list[Offer]:
        """Everything this provider will sell us right now, optionally filtered by GPU."""

    @abstractmethod
    def provision(self, spec: InstanceSpec) -> Instance:
        """Create an instance and record it. Returns before the box is reachable."""

    @abstractmethod
    def refresh(self, inst: Instance) -> Instance:
        """Re-read status and SSH details. Returns a new instance; does not mutate."""

    @abstractmethod
    def terminate(self, inst: Instance) -> None:
        """Destroy the instance and mark the ledger row. Idempotent."""

    @abstractmethod
    def cost_per_hour(self, inst: Instance) -> float: ...

    # ---- shared
    def wait_ready(
        self,
        inst: Instance,
        timeout_s: float = 900.0,
        *,
        poll_s: float = 5.0,
        log: Callable[[str], None] | None = None,
    ) -> Instance:
        """Poll until the box answers on its SSH port, or give up and say so.

        The timeout is a real failure, not a warning: a provider that has taken our money
        for fifteen minutes without producing a reachable machine is a provisioning
        error, and the caller's ``finally`` should terminate what it got.
        """
        deadline = time.monotonic() + timeout_s
        current = inst
        last = ""
        while True:
            current = self.refresh(current)
            if log is not None and current.status != last:
                log(f"{self.name}:{current.id} {current.status}")
                last = current.status
            if current.ssh is not None and tcp_open(current.ssh.host, current.ssh.port):
                return current
            if time.monotonic() >= deadline:
                raise ProvisionError(
                    f"{self.name} instance {current.id} not reachable after {timeout_s:.0f}s "
                    f"(status: {current.status})"
                )
            time.sleep(poll_s)

    def connect(self, inst: Instance) -> Session:
        if inst.ssh is None:
            raise ProvisionError(
                f"{self.name} instance {inst.id} has no ssh target yet; call wait_ready first"
            )
        return SshSession(inst.ssh)

    # ---- ledger bookkeeping, called by the concrete providers
    def _record(self, inst: Instance) -> Instance:
        if self.ledger is not None:
            self.ledger.save_instance(inst)
        return inst

    def _record_terminated(self, inst: Instance) -> None:
        if self.ledger is not None:
            self.ledger.mark_terminated(inst.id, provider=self.name)


__all__ = [
    "Instance",
    "InstanceSpec",
    "Offer",
    "Provider",
    "ProvisionError",
    "Session",
    "SshSession",
    "tcp_open",
]
