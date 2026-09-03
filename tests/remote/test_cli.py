"""``infervolt remote``, driven against a stub runner. Nothing here rents anything.

The command's whole job is argument handling: what reaches the box after ``--``, what the
instance spec says, and what the exit code is. The orchestration it wraps is tested in
``test_runner``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from infervolt.cli import remote as remote_cli
from infervolt.cli.main import app
from infervolt.infra.types import Instance, SshTarget
from infervolt.remote.runner import RemoteOutcome
from infervolt.store.ledger import Ledger
from tests.remote.conftest import FakeProvider

runner = CliRunner()


class StubRun:
    """Records the constructor arguments and returns a canned outcome."""

    calls: list[dict[str, Any]] = []
    outcome = RemoteOutcome(
        instance_id="i-1",
        ok=True,
        message="ok",
        cost_usd=1.2345,
        elapsed_s=61.0,
        run_id="2026-09-01T12-00-00Z-ab12",
        local_run_dir=Path("/tmp/runs/2026-09-01T12-00-00Z-ab12"),
    )

    def __init__(self, provider: object, spec: object, args: list[str], **kwargs: Any) -> None:
        StubRun.calls.append({"provider": provider, "spec": spec, "args": args, **kwargs})

    def run(self) -> RemoteOutcome:
        return StubRun.outcome


@pytest.fixture(autouse=True)
def stub(monkeypatch: pytest.MonkeyPatch) -> None:
    StubRun.calls = []
    StubRun.outcome = RemoteOutcome(
        instance_id="i-1",
        ok=True,
        message="ok",
        cost_usd=1.2345,
        elapsed_s=61.0,
        run_id="2026-09-01T12-00-00Z-ab12",
        local_run_dir=Path("/tmp/runs/2026-09-01T12-00-00Z-ab12"),
    )
    monkeypatch.setattr(remote_cli, "RemoteRun", StubRun)
    monkeypatch.setattr(remote_cli, "get_provider", lambda name, ledger=None: FakeProvider(ledger))


def invoke(tmp_path: Path, *args: str) -> Any:
    return runner.invoke(app, ["remote", "optimize", "--home", str(tmp_path), *args])


def test_everything_after_the_separator_goes_to_the_remote_optimize(tmp_path: Path) -> None:
    result = invoke(
        tmp_path,
        "--provider",
        "fake",
        "--gpu",
        "a100xl",
        "--max-usd",
        "3",
        "--",
        "--engine",
        "vllm",
        "--model",
        "Qwen/Qwen3-8B",
        "--slo",
        "ttft=500ms,itl=30ms",
        "--max-trials",
        "8",
    )
    assert result.exit_code == 0, result.output
    call = StubRun.calls[0]
    assert call["args"] == [
        "--engine",
        "vllm",
        "--model",
        "Qwen/Qwen3-8B",
        "--slo",
        "ttft=500ms,itl=30ms",
        "--max-trials",
        "8",
    ]
    assert call["max_usd"] == 3.0


def test_flags_the_remote_side_shares_with_this_one_are_not_stolen(tmp_path: Path) -> None:
    # --max-usd before -- is the cap; --max-usd after it is the remote run's own budget.
    result = invoke(tmp_path, "--provider", "fake", "--max-usd", "10", "--", "--max-usd", "2")
    assert result.exit_code == 0, result.output
    call = StubRun.calls[0]
    assert call["max_usd"] == 10.0 and call["args"] == ["--max-usd", "2"]


def test_the_instance_spec_comes_from_the_gpu_count_and_disk_options(tmp_path: Path) -> None:
    invoke(
        tmp_path,
        "--provider",
        "fake",
        "--gpu",
        "h100",
        "--count",
        "2",
        "--disk",
        "250",
        "--",
        "--engine",
        "vllm",
    )
    spec = StubRun.calls[0]["spec"]
    assert (spec.gpu, spec.count, spec.disk_gb, spec.image) == ("h100", 2, 250, None)


def test_docker_install_names_the_image_the_box_should_pull(tmp_path: Path) -> None:
    invoke(tmp_path, "--provider", "fake", "--engine-install", "docker", "--", "--engine", "vllm")
    call = StubRun.calls[0]
    assert call["engine_install"] == "docker"
    assert call["spec"].image.startswith("vllm/vllm-openai:v")


def test_keep_instance_and_ref_are_forwarded(tmp_path: Path) -> None:
    invoke(
        tmp_path,
        "--provider",
        "fake",
        "--keep",
        "--instance",
        "i-9",
        "--ref",
        "abc123",
        "--",
        "--engine",
        "vllm",
    )
    call = StubRun.calls[0]
    assert call["keep"] is True and call["instance_id"] == "i-9" and call["ref"] == "abc123"


def test_omitting_the_optimize_arguments_is_a_usage_error(tmp_path: Path) -> None:
    result = invoke(tmp_path, "--provider", "fake")
    assert result.exit_code != 0
    assert "optimize arguments after" in result.output


def test_an_unknown_engine_install_is_refused(tmp_path: Path) -> None:
    result = invoke(
        tmp_path, "--provider", "fake", "--engine-install", "conda", "--", "--engine", "vllm"
    )
    assert result.exit_code != 0
    assert "venv or docker" in result.output


def test_the_outcome_and_the_local_run_dir_are_printed(tmp_path: Path) -> None:
    result = invoke(tmp_path, "--provider", "fake", "--", "--engine", "vllm")
    assert "ok: ok" in result.output
    assert "$1.234" in result.output or "$1.235" in result.output
    assert "2026-09-01T12-00-00Z-ab12 -> /tmp/runs/2026-09-01T12-00-00Z-ab12" in result.output


def test_a_failed_run_exits_nonzero(tmp_path: Path) -> None:
    StubRun.outcome = RemoteOutcome(
        instance_id="i-1", ok=False, message="cost cap reached", cost_usd=10.0, elapsed_s=900.0
    )
    result = invoke(tmp_path, "--provider", "fake", "--", "--engine", "vllm")
    assert result.exit_code == 1
    assert "FAILED: cost cap reached" in result.output


def test_an_unknown_provider_is_a_parameter_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str, ledger: object = None) -> object:
        raise KeyError(f"unknown provider {name!r}")

    monkeypatch.setattr(remote_cli, "get_provider", missing)
    result = invoke(tmp_path, "--provider", "nowhere", "--", "--engine", "vllm")
    assert result.exit_code != 0 and "nowhere" in result.output


def test_remote_shell_prints_an_ssh_command_for_a_known_instance(tmp_path: Path) -> None:
    settings_home = tmp_path
    with Ledger(settings_home / "ledger.sqlite", settings_home / "runs") as ledger:
        ledger.save_instance(
            Instance(
                provider="fake",
                id="i-7",
                gpu="a100xl",
                ssh=SshTarget(host="203.0.113.7", port=2222, user="ubuntu", key_path="/k/id"),
            )
        )
    result = runner.invoke(
        app, ["remote", "shell", "--provider", "fake", "--instance", "i-7", "--home", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "ssh " in result.output and "ubuntu@203.0.113.7" in result.output


def test_remote_shell_says_so_when_the_ledger_has_no_such_instance(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["remote", "shell", "--provider", "fake", "--instance", "i-x", "--home", str(tmp_path)]
    )
    assert result.exit_code == 1
