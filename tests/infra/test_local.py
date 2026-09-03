from pathlib import Path

import pytest

from infervolt.hardware.profiles import get_profile
from infervolt.infra import local
from infervolt.infra.local import LocalProvider
from infervolt.infra.ssh import LocalSession
from infervolt.infra.types import InstanceSpec
from infervolt.store.ledger import Ledger


def test_local_session_runs_a_command_and_streams_it() -> None:
    lines: list[str] = []
    result = LocalSession().run("echo hello && echo world", stream=lines.append)
    assert result.code == 0
    assert result.stdout_tail.splitlines() == ["hello", "world"]
    assert lines == ["hello", "world"]


def test_local_session_reports_a_failing_command() -> None:
    result = LocalSession().run("exit 3")
    assert result.code == 3


def test_local_session_put_and_get_copy_files(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("payload")
    session = LocalSession()
    session.put(src, str(tmp_path / "out" / "b.txt"))
    assert (tmp_path / "out" / "b.txt").read_text() == "payload"
    session.get(str(tmp_path / "out" / "b.txt"), tmp_path / "back" / "c.txt")
    assert (tmp_path / "back" / "c.txt").read_text() == "payload"


def test_local_session_put_copies_a_directory(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "recipe.yaml").write_text("knobs: {}")
    LocalSession().put(tmp_path / "run", str(tmp_path / "copy"))
    assert (tmp_path / "copy" / "recipe.yaml").read_text() == "knobs: {}"


def test_local_session_tunnel_is_the_port_itself() -> None:
    assert LocalSession().tunnel(8000) == 8000
    LocalSession().close()


def test_local_provider_provisions_a_free_instance_and_records_it(tmp_path: Path) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider = LocalProvider(ledger, profile_name="m3-8")
        inst = provider.provision(InstanceSpec(gpu="m3"))
        assert inst.id == "local" and inst.ssh is None and inst.usd_per_hour == 0.0
        assert provider.cost_per_hour(inst) == 0.0
        assert [i.id for i in ledger.instances()] == ["local"]

        ready = provider.wait_ready(inst, timeout_s=0.0)
        assert ready.status == "running"
        assert isinstance(provider.connect(ready), LocalSession)

        provider.terminate(inst)
        assert ledger.instances() == []
        assert [i.id for i in ledger.instances(active_only=False)] == ["local"]


def test_local_provider_offers_the_named_profile() -> None:
    offers = LocalProvider(profile_name="a100-80").list_offers()
    assert [(o.provider, o.gpu_mem_gb, o.raw_id) for o in offers] == [("local", 80.0, "a100-80")]
    assert LocalProvider(profile_name="a100-80").list_offers("h100") == []
    assert LocalProvider(profile_name="a100-80").list_offers("a100")[0].raw_id == "a100-80"


def test_local_provider_without_a_profile_offers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # When detection cannot say what this machine is, the honest answer is silence rather
    # than a guessed GPU that a search would then optimise against.
    monkeypatch.setattr(local, "_detected_profile", lambda: None)
    assert LocalProvider().list_offers() == []


def test_local_provider_uses_the_detected_profile_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local, "_detected_profile", lambda: get_profile("h100-80"))
    offers = LocalProvider().list_offers()
    assert [o.gpu for o in offers] == ["NVIDIA H100 80GB HBM3"]
