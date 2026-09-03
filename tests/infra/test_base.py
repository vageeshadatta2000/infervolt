"""The parts of the contract every provider inherits: wait_ready, connect, bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pytest

from infervolt.infra import base
from infervolt.infra.base import Provider
from infervolt.infra.registry import available_providers, get_provider
from infervolt.infra.ssh import SshSession
from infervolt.infra.types import (
    Instance,
    InstanceSpec,
    Offer,
    ProvisionError,
    SshTarget,
)
from infervolt.store.ledger import Ledger


class FakeProvider(Provider):
    """A provider whose instance becomes reachable after ``ready_after`` refreshes."""

    name = "fake"

    def __init__(self, ledger: Ledger | None = None, *, ready_after: int = 1) -> None:
        super().__init__(ledger)
        self.ready_after = ready_after
        self.refreshes = 0
        self.terminated: list[str] = []

    def list_offers(self, gpu: str | None = None) -> list[Offer]:
        return [Offer(provider="fake", gpu="FakeGPU", gpu_mem_gb=80, usd_per_hour=1.0, raw_id="f1")]

    def provision(self, spec: InstanceSpec) -> Instance:
        return self._record(
            Instance(provider="fake", id="i-1", gpu=spec.gpu, usd_per_hour=1.0, status="starting")
        )

    def refresh(self, inst: Instance) -> Instance:
        self.refreshes += 1
        if self.refreshes < self.ready_after:
            return inst.model_copy(update={"status": "starting"})
        target = SshTarget(host="203.0.113.9", port=22, user="ubuntu", key_path="/k/id")
        return inst.model_copy(update={"status": "running", "ssh": target})

    def terminate(self, inst: Instance) -> None:
        self.terminated.append(inst.id)
        self._record_terminated(inst)

    def cost_per_hour(self, inst: Instance) -> float:
        return inst.usd_per_hour


def test_wait_ready_polls_until_the_ssh_port_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base, "tcp_open", lambda host, port, timeout_s=5.0: True)
    provider = FakeProvider(ready_after=3)
    lines: list[str] = []
    ready = provider.wait_ready(
        Instance(provider="fake", id="i-1", gpu="a100"),
        timeout_s=10.0,
        poll_s=0.0,
        log=lines.append,
    )
    assert ready.status == "running" and ready.ssh is not None
    assert provider.refreshes == 3
    # Only transitions are logged, so a ten-minute wait is three lines, not a hundred.
    assert lines == ["fake:i-1 starting", "fake:i-1 running"]


def test_wait_ready_gives_up_when_the_port_never_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base, "tcp_open", lambda host, port, timeout_s=5.0: False)
    with pytest.raises(ProvisionError, match="not reachable"):
        FakeProvider().wait_ready(Instance(provider="fake", id="i-1", gpu="a100"), timeout_s=0.0)


def test_status_running_is_not_enough_without_a_reachable_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The provider says RUNNING minutes before sshd accepts anything.
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base, "tcp_open", lambda host, port, timeout_s=5.0: False)
    provider = FakeProvider(ready_after=1)
    with pytest.raises(ProvisionError):
        provider.wait_ready(Instance(provider="fake", id="i-1", gpu="a100"), timeout_s=0.0)


def test_connect_needs_an_ssh_target() -> None:
    provider = FakeProvider()
    with pytest.raises(ProvisionError, match="wait_ready"):
        provider.connect(Instance(provider="fake", id="i-1", gpu="a100"))
    target = SshTarget(host="h", key_path="/k/id")
    session = provider.connect(Instance(provider="fake", id="i-1", gpu="a100", ssh=target))
    assert isinstance(session, SshSession)


def test_tcp_open_is_false_for_a_port_nobody_is_listening_on() -> None:
    # 203.0.113.0/24 is reserved for documentation, so nothing can answer.
    assert base.tcp_open("203.0.113.1", 22, timeout_s=0.01) is False


def test_provision_and_terminate_move_the_ledger_row(tmp_path: Path) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider = FakeProvider(ledger)
        inst = provider.provision(InstanceSpec(gpu="a100"))
        assert [i.id for i in ledger.instances()] == ["i-1"]
        provider.terminate(inst)
        assert ledger.instances() == [] and provider.terminated == ["i-1"]


def test_a_provider_without_a_ledger_still_works() -> None:
    provider = FakeProvider()
    inst = provider.provision(InstanceSpec(gpu="a100"))
    provider.terminate(inst)
    assert provider.terminated == ["i-1"]


def test_the_shipped_providers_are_discoverable_through_entry_points() -> None:
    assert {"local", "thunder"} <= set(available_providers())
    assert get_provider("local").name == "local"
    with pytest.raises(KeyError, match="unknown provider"):
        get_provider("nowhere")
