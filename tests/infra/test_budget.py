from __future__ import annotations

import time

import pytest

from infervolt.infra import budget as budget_mod
from infervolt.infra.budget import SpendGuard, ensure_terminated
from infervolt.infra.types import Instance, InstanceSpec

from .test_base import FakeProvider


class FakeClock:
    """Time moves only when the test says so."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def wait_for(predicate: object, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.001)


def test_the_guard_fires_once_when_the_cap_is_reached() -> None:
    clock = FakeClock()
    calls: list[int] = []
    # $6/h for half an hour is $3, which is over a $2 cap.
    guard = SpendGuard(6.0, 2.0, lambda: calls.append(1), interval_s=0.001, clock=clock)
    with guard:
        assert not guard.fired
        clock.t = 1800.0
        wait_for(lambda: guard.fired)
    assert guard.fired and calls == [1]


def test_the_guard_stays_quiet_below_the_cap() -> None:
    clock = FakeClock()
    guard = SpendGuard(
        1.09, 10.0, lambda: pytest.fail("must not fire"), interval_s=0.001, clock=clock
    )
    with guard:
        clock.t = 60.0
        time.sleep(0.05)
    assert not guard.fired
    assert guard.spent_usd() == pytest.approx(1.09 / 60.0)


def test_a_zero_cap_means_no_cap_and_starts_no_thread() -> None:
    guard = SpendGuard(3.2, 0.0, lambda: pytest.fail("must not fire")).start()
    assert guard.watching is False and guard.exceeded is False
    guard.stop()


def test_a_free_instance_needs_no_watcher() -> None:
    guard = SpendGuard(0.0, 10.0, lambda: pytest.fail("must not fire")).start()
    assert guard.watching is False
    guard.stop()


def test_firing_twice_calls_the_callback_once() -> None:
    calls: list[int] = []
    guard = SpendGuard(1.0, 1.0, lambda: calls.append(1))
    guard._fire()
    guard._fire()
    assert calls == [1]


def test_ensure_terminated_runs_once_and_deregisters_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(budget_mod.atexit, "register", registered.append)
    monkeypatch.setattr(
        budget_mod.atexit,
        "unregister",
        lambda fn: registered.remove(fn) if fn in registered else None,
    )
    provider = FakeProvider()
    terminate = ensure_terminated(provider, Instance(provider="fake", id="i-1", gpu="a100"))
    assert registered == [terminate]
    terminate()
    terminate()
    assert provider.terminated == ["i-1"]
    # Having run, it must not run a second time at interpreter exit.
    assert registered == []


def test_ensure_terminated_swallows_a_provider_that_is_already_broken() -> None:
    class Broken(FakeProvider):
        def terminate(self, inst: Instance) -> None:
            raise RuntimeError("api down")

    terminate = ensure_terminated(Broken(), Instance(provider="fake", id="i-1", gpu="a100"))
    terminate()  # must not raise: nothing on this path is left to report an error to


def test_the_guard_can_be_wired_to_terminate_the_instance() -> None:
    """The wiring the remote runner will use: cap reached -> instance destroyed, once."""
    clock = FakeClock()
    provider = FakeProvider()
    inst = provider.provision(InstanceSpec(gpu="a100"))
    guard = SpendGuard(6.0, 1.0, ensure_terminated(provider, inst), interval_s=0.001, clock=clock)
    with guard:
        clock.t = 3600.0
        wait_for(lambda: bool(provider.terminated))
    assert provider.terminated == ["i-1"]
