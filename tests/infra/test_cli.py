"""The infra command group, driven against a stub provider. Nothing here goes online."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from infervolt.cli import infra as infra_cli
from infervolt.cli.main import app
from infervolt.infra.base import Provider
from infervolt.infra.types import InfraError, Instance, InstanceSpec, Offer, SshTarget
from infervolt.store.ledger import Ledger

runner = CliRunner()


class StubProvider(Provider):
    name = "thunder"

    def __init__(self, ledger: Ledger | None = None, *, fail: bool = False) -> None:
        super().__init__(ledger)
        self.fail = fail
        self.terminated: list[str] = []

    def list_offers(self, gpu: str | None = None) -> list[Offer]:
        if self.fail:
            raise InfraError("no Thunder token: set TNR_API_TOKEN")
        offers = [
            Offer(
                provider="thunder",
                gpu="NVIDIA A100 (80GB)",
                gpu_mem_gb=80,
                count=1,
                usd_per_hour=1.09,
                raw_id="a100xl_x1",
            ),
            Offer(
                provider="thunder",
                gpu="NVIDIA H100",
                gpu_mem_gb=80,
                count=8,
                usd_per_hour=25.6,
                raw_id="h100_x8",
            ),
        ]
        return [o for o in offers if not gpu or gpu in o.raw_id]

    def provision(self, spec: InstanceSpec) -> Instance:  # pragma: no cover - never called
        raise AssertionError("the CLI must not create instances")

    def refresh(self, inst: Instance) -> Instance:
        if self.fail:
            raise InfraError("api down")
        return inst.model_copy(
            update={"status": "running", "ssh": SshTarget(host="203.0.113.10", key_path="/k/id")}
        )

    def terminate(self, inst: Instance) -> None:
        if self.fail:
            raise InfraError("api down")
        self.terminated.append(inst.id)
        self._record_terminated(inst)

    def cost_per_hour(self, inst: Instance) -> float:
        return inst.usd_per_hour


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubProvider:
    provider = StubProvider()

    def factory(name: str, ledger: Ledger | None = None, **kwargs: object) -> Provider:
        if name != "thunder":
            raise KeyError(f"unknown provider {name!r}")
        provider.ledger = ledger
        return provider

    monkeypatch.setattr(infra_cli, "get_provider", factory)
    return provider


def _seed(home: Path, *instances: Instance) -> None:
    with Ledger(home / "ledger.sqlite", home / "runs") as ledger:
        for inst in instances:
            ledger.save_instance(inst)


def test_offers_prints_a_priced_table(stub: StubProvider) -> None:
    result = runner.invoke(app, ["infra", "offers", "--provider", "thunder"])
    assert result.exit_code == 0
    assert "a100xl_x1" in result.stdout and "1.09" in result.stdout
    assert "NVIDIA A100 (80GB)" in result.stdout and "80G" in result.stdout
    assert "h100_x8" in result.stdout and "25.60" in result.stdout


def test_offers_can_be_filtered(stub: StubProvider) -> None:
    result = runner.invoke(app, ["infra", "offers", "--gpu", "h100"])
    assert result.exit_code == 0 and "a100xl_x1" not in result.stdout


def test_offers_says_so_when_the_filter_matches_nothing(stub: StubProvider) -> None:
    result = runner.invoke(app, ["infra", "offers", "--gpu", "mi300x"])
    assert result.exit_code == 0 and "no offers" in result.stdout


def test_offers_reports_a_missing_token_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infra_cli, "get_provider", lambda *a, **k: StubProvider(fail=True))
    result = runner.invoke(app, ["infra", "offers"])
    assert result.exit_code == 1 and "TNR_API_TOKEN" in result.output


def test_offers_rejects_an_unknown_provider(stub: StubProvider) -> None:
    result = runner.invoke(app, ["infra", "offers", "--provider", "nowhere"])
    assert result.exit_code != 0 and "nowhere" in result.output


def test_list_refreshes_ledger_rows(stub: StubProvider, tmp_path: Path) -> None:
    _seed(tmp_path, Instance(provider="thunder", id="42", gpu="a100xl", usd_per_hour=1.09))
    result = runner.invoke(app, ["infra", "list", "--home", str(tmp_path)])
    assert result.exit_code == 0
    assert "42" in result.stdout and "running" in result.stdout
    assert "203.0.113.10" in result.stdout


def test_list_shows_a_row_it_cannot_refresh_rather_than_hiding_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The row is how the user stops paying; an unreachable API must not remove it.
    monkeypatch.setattr(infra_cli, "get_provider", lambda *a, **k: StubProvider(fail=True))
    _seed(tmp_path, Instance(provider="thunder", id="42", gpu="a100xl", status="running"))
    result = runner.invoke(app, ["infra", "list", "--home", str(tmp_path)])
    assert result.exit_code == 0 and "unreachable" in result.stdout


def test_list_is_empty_when_nothing_is_rented(stub: StubProvider, tmp_path: Path) -> None:
    result = runner.invoke(app, ["infra", "list", "--home", str(tmp_path)])
    assert result.exit_code == 0 and "no active instances" in result.stdout


def test_terminate_asks_before_destroying_anything(stub: StubProvider, tmp_path: Path) -> None:
    _seed(tmp_path, Instance(provider="thunder", id="42", gpu="a100xl"))
    result = runner.invoke(
        app,
        ["infra", "terminate", "42", "--provider", "thunder", "--home", str(tmp_path)],
        input="n\n",
    )
    assert result.exit_code == 1 and stub.terminated == []


def test_terminate_with_yes_destroys_and_closes_the_row(stub: StubProvider, tmp_path: Path) -> None:
    _seed(tmp_path, Instance(provider="thunder", id="42", gpu="a100xl"))
    result = runner.invoke(
        app,
        ["infra", "terminate", "42", "--provider", "thunder", "--yes", "--home", str(tmp_path)],
    )
    assert result.exit_code == 0 and stub.terminated == ["42"]
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        assert ledger.instances() == []


def test_terminate_works_for_an_id_the_ledger_never_saw(stub: StubProvider, tmp_path: Path) -> None:
    # A box created before the ledger existed still has to be stoppable.
    result = runner.invoke(
        app,
        ["infra", "terminate", "99", "--provider", "thunder", "--yes", "--home", str(tmp_path)],
    )
    assert result.exit_code == 0 and stub.terminated == ["99"]


def test_gc_terminates_everything_the_ledger_still_lists(
    stub: StubProvider, tmp_path: Path
) -> None:
    _seed(
        tmp_path,
        Instance(provider="thunder", id="42", gpu="a100xl"),
        Instance(provider="thunder", id="43", gpu="h100", count=2),
    )
    result = runner.invoke(app, ["infra", "gc", "--yes", "--home", str(tmp_path)])
    assert result.exit_code == 0 and stub.terminated == ["42", "43"]
    assert "terminated thunder:42" in result.stdout and "h100 x2" in result.stdout
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        assert ledger.instances() == []


def test_gc_reports_a_failure_and_still_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(infra_cli, "get_provider", lambda *a, **k: StubProvider(fail=True))
    _seed(tmp_path, Instance(provider="thunder", id="42", gpu="a100xl"))
    result = runner.invoke(app, ["infra", "gc", "--yes", "--home", str(tmp_path)])
    assert result.exit_code == 1 and "FAILED thunder:42" in result.output


def test_gc_on_a_clean_ledger_does_nothing(stub: StubProvider, tmp_path: Path) -> None:
    result = runner.invoke(app, ["infra", "gc", "--yes", "--home", str(tmp_path)])
    assert result.exit_code == 0 and "nothing to collect" in result.stdout


def test_providers_lists_what_is_installed() -> None:
    result = runner.invoke(app, ["infra", "providers"])
    assert result.exit_code == 0
    assert "local" in result.stdout and "thunder" in result.stdout
