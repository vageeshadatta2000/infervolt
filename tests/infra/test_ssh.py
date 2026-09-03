"""SshSession is tested by what it asks the OS to run. Nothing here opens a connection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from infervolt.infra import ssh as ssh_mod
from infervolt.infra.ssh import SshSession
from infervolt.infra.types import InfraError, SshTarget

TARGET = SshTarget(host="203.0.113.10", port=2222, user="ubuntu", key_path="/keys/id_ed25519")


class FakePopen:
    """Enough of subprocess.Popen for the streaming path: output, an exit code, a kill."""

    def __init__(self, lines: list[str], code: int = 0, *, hang: bool = False) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self._code = code
        self._hang = hang
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout or 0.0)
        return self._code

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture every argv the session builds, and answer with a successful command."""
    seen: list[list[str]] = []

    def fake_popen(argv: Any, **kwargs: Any) -> FakePopen:
        seen.append(argv if isinstance(argv, list) else [argv])
        return FakePopen(["ok"])

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return seen


def test_run_builds_an_ssh_command_with_key_port_and_hardening(calls: list[list[str]]) -> None:
    result = SshSession(TARGET).run("nvidia-smi")
    assert result.code == 0 and result.stdout_tail == "ok"
    argv = calls[0]
    assert argv[0] == "ssh"
    assert argv[-2:] == ["ubuntu@203.0.113.10", "nvidia-smi"]
    assert argv[argv.index("-i") + 1] == "/keys/id_ed25519"
    assert argv[argv.index("-p") + 1] == "2222"
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "ServerAliveInterval=15" in argv


def test_proxy_command_is_passed_through_when_the_provider_needs_one(
    calls: list[list[str]],
) -> None:
    target = TARGET.model_copy(update={"proxy_command": "cloudflared access ssh --hostname %h"})
    SshSession(target).run("true")
    assert "ProxyCommand=cloudflared access ssh --hostname %h" in calls[0]


def test_scp_spells_the_port_with_a_capital_p(calls: list[list[str]], tmp_path: Path) -> None:
    # -p on scp means "preserve times" and would silently leave the port at 22.
    session = SshSession(TARGET)
    session.put(tmp_path / "recipe.yaml", "/workspace/recipe.yaml")
    argv = calls[0]
    assert argv[0] == "scp" and "-p" not in argv
    assert argv[argv.index("-P") + 1] == "2222"
    assert argv[-2:] == [
        str(tmp_path / "recipe.yaml"),
        "ubuntu@203.0.113.10:/workspace/recipe.yaml",
    ]

    session.get("/workspace/run", tmp_path / "runs" / "run")
    assert calls[1][-2:] == ["ubuntu@203.0.113.10:/workspace/run", str(tmp_path / "runs" / "run")]
    assert (tmp_path / "runs").is_dir()


def test_a_failed_transfer_raises_rather_than_returning_a_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: FakePopen(["No such file"], code=1))
    with pytest.raises(InfraError, match="scp put"):
        SshSession(TARGET).put(tmp_path / "missing", "/workspace/missing")


def test_run_streams_each_line_as_it_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kw: FakePopen(["loading", "ready"], code=0)
    )
    seen: list[str] = []
    result = SshSession(TARGET).run("vllm serve", stream=seen.append)
    assert seen == ["loading", "ready"]
    assert result.stdout_tail == "loading\nready"


def test_only_the_tail_of_a_long_stream_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kw: FakePopen([str(i) for i in range(500)])
    )
    result = SshSession(TARGET, tail_lines=3).run("bootstrap")
    assert result.stdout_tail == "497\n498\n499"


def test_a_hung_command_is_killed_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakePopen([], hang=True)
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: proc)
    with pytest.raises(InfraError, match="timed out"):
        SshSession(TARGET).run("sleep 1000", timeout_s=0.01)
    assert proc.killed


def test_tunnel_forwards_a_free_local_port_and_close_tears_it_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTunnel:
        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    started: list[list[str]] = []
    tunnel = FakeTunnel()

    def fake_popen(argv: Any, **kwargs: Any) -> FakeTunnel:
        started.append(argv)
        return tunnel

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ssh_mod, "_free_port", lambda: 41234)

    session = SshSession(TARGET)
    assert session.tunnel(8000) == 41234
    argv = started[0]
    assert argv[0] == "ssh" and "-N" in argv
    assert argv[argv.index("-L") + 1] == "41234:localhost:8000"
    # The forward must be set up before the host argument, or ssh reads it as a command.
    assert argv.index("-L") < argv.index("ubuntu@203.0.113.10")

    session.close()
    assert tunnel.terminated
