"""The instances table: what stops a crashed controller from billing all night."""

from __future__ import annotations

from pathlib import Path

from infervolt.infra.types import Instance, SshTarget
from infervolt.store.ledger import Ledger


def _inst(provider: str = "thunder", id_: str = "42", **kw: object) -> Instance:
    return Instance(provider=provider, id=id_, gpu="a100xl", usd_per_hour=1.09, **kw)


def test_instances_round_trip_through_the_ledger(tmp_path: Path) -> None:
    inst = _inst(ssh=SshTarget(host="203.0.113.10", key_path="/k/thunder-42"), status="running")
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.save_instance(inst)
        (back,) = ledger.instances()
        assert back == inst


def test_a_second_process_can_see_and_terminate_what_the_first_one_rented(
    tmp_path: Path,
) -> None:
    db, runs = tmp_path / "l.sqlite", tmp_path / "runs"
    with Ledger(db, runs) as first:
        first.save_instance(_inst())
    with Ledger(db, runs) as second:
        assert [i.id for i in second.instances()] == ["42"]
        second.mark_terminated("42")
        assert second.instances() == []


def test_terminated_rows_survive_for_the_audit_but_leave_the_live_list(tmp_path: Path) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.save_instance(_inst())
        ledger.mark_terminated("42", provider="thunder")
        assert ledger.instances(active_only=True) == []
        assert [i.id for i in ledger.instances(active_only=False)] == ["42"]


def test_resaving_an_instance_updates_it_without_reopening_a_closed_row(tmp_path: Path) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.save_instance(_inst(status="provisioning"))
        ledger.save_instance(_inst(status="running"))
        assert [i.status for i in ledger.instances()] == ["running"]
        ledger.mark_terminated("42")
        # A refresh landing after teardown must not resurrect the row into 'infra gc'.
        ledger.save_instance(_inst(status="running"))
        assert ledger.instances() == []


def test_the_same_id_at_two_providers_is_two_instances(tmp_path: Path) -> None:
    # Thunder numbers instances from 1, so "42" is a plausible id at more than one place.
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.save_instance(_inst(provider="thunder"))
        ledger.save_instance(_inst(provider="runpod"))
        assert len(ledger.instances()) == 2
        assert [i.provider for i in ledger.instances(provider="runpod")] == ["runpod"]
        ledger.mark_terminated("42", provider="runpod")
        assert [i.provider for i in ledger.instances()] == ["thunder"]


def test_marking_an_unknown_instance_is_quiet(tmp_path: Path) -> None:
    # Termination is called from paths that must not raise, including atexit.
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        ledger.mark_terminated("nobody")
        assert ledger.instances(active_only=False) == []
