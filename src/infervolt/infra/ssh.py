"""Command execution and file transfer, locally or over SSH.

The SSH implementation shells out to the system ``ssh``/``scp`` binaries rather than
using a library. That is a deliberate trade: every box we rent already speaks OpenSSH,
the user's own ``~/.ssh/config`` and agent keep working, ``ProxyCommand`` costs nothing,
and there is one less native dependency to build on a controller that may be a laptop.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from infervolt.infra.types import InfraError, RunResult, SshTarget

DEFAULT_TIMEOUT_S = 3600.0
TAIL_LINES = 200


class Session(Protocol):
    """Somewhere to run commands and move files -- this machine, or a rented one."""

    def run(
        self,
        cmd: str,
        *,
        stream: Callable[[str], None] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> RunResult: ...

    def put(self, local: Path, remote: str) -> None: ...

    def get(self, remote: str, local: Path) -> None: ...

    def tunnel(self, remote_port: int) -> int:
        """Make ``remote_port`` reachable on localhost; return the local port."""
        ...

    def close(self) -> None: ...


def _drain(
    proc: subprocess.Popen[str], stream: Callable[[str], None] | None, tail: deque[str]
) -> None:
    """Pump the child's combined output into ``tail``, and to ``stream`` as it arrives.

    Runs on its own thread so that a child which produces no output at all can still be
    killed on the timeout -- reading the pipe from the calling thread would block past
    any deadline we set.
    """
    if proc.stdout is None:  # pragma: no cover - always a pipe here
        return
    for line in proc.stdout:
        text = line.rstrip("\n")
        tail.append(text)
        if stream is not None:
            stream(text)


def _execute(
    argv: Sequence[str] | str,
    *,
    shell: bool,
    stream: Callable[[str], None] | None,
    timeout_s: float,
    what: str,
    tail_lines: int = TAIL_LINES,
) -> RunResult:
    proc = subprocess.Popen(
        argv,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=tail_lines)
    reader = threading.Thread(target=_drain, args=(proc, stream, tail), daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5.0)
        raise InfraError(f"{what} timed out after {timeout_s:.0f}s") from None
    reader.join(timeout=5.0)
    return RunResult(code=code, stdout_tail="\n".join(tail))


def _free_port() -> int:
    """Ask the kernel for a free local port and immediately give it back.

    Racy by construction -- something else may take the port before ``ssh -L`` binds it --
    but it is what every port-forwarding helper does, and the alternative (letting ssh
    choose) gives us no way to learn the number.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalSession:
    """Runs everything here, in this process's machine. No network, no keys."""

    def run(
        self,
        cmd: str,
        *,
        stream: Callable[[str], None] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> RunResult:
        return _execute(cmd, shell=True, stream=stream, timeout_s=timeout_s, what="local command")

    def put(self, local: Path, remote: str) -> None:
        self._copy(Path(local), Path(remote))

    def get(self, remote: str, local: Path) -> None:
        self._copy(Path(remote), Path(local))

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def tunnel(self, remote_port: int) -> int:
        """Already local: the port is the port."""
        return remote_port

    def close(self) -> None:
        return None


class SshSession:
    """A connection to one rented box, over the system ``ssh``.

    Not a persistent channel: each :meth:`run` is its own ``ssh`` invocation. Anything
    that must survive between commands belongs on the box (a file, a tmux session), not
    in this object -- which is also what makes the session safe to rebuild after the
    controller restarts.
    """

    def __init__(self, target: SshTarget, *, tail_lines: int = TAIL_LINES) -> None:
        self.target = target
        self.tail_lines = tail_lines
        self._tunnels: list[subprocess.Popen[bytes]] = []

    # ---- argv construction
    def _options(self) -> list[str]:
        """Options common to ssh and scp; the port flag differs, so it is not here.

        ``accept-new`` rather than ``no``: a box we just created has a host key we have
        never seen, but a key that *changed* under a host we are already talking to is
        worth failing on.
        """
        opts = [
            "-i",
            self.target.key_path,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=15",
        ]
        if self.target.proxy_command:
            opts += ["-o", f"ProxyCommand={self.target.proxy_command}"]
        return opts

    def ssh_argv(self, cmd: str | None = None, *, extra: Sequence[str] = ()) -> list[str]:
        argv = ["ssh", *self._options(), "-p", str(self.target.port), *extra]
        argv.append(f"{self.target.user}@{self.target.host}")
        if cmd is not None:
            argv.append(cmd)
        return argv

    def scp_argv(self, src: str, dst: str) -> list[str]:
        # scp spells the port -P; ssh spells it -p. Getting this wrong asks scp to
        # preserve mtimes and then connect to port 22.
        return ["scp", *self._options(), "-P", str(self.target.port), "-r", src, dst]

    def _remote(self, path: str) -> str:
        return f"{self.target.user}@{self.target.host}:{path}"

    # ---- Session
    def run(
        self,
        cmd: str,
        *,
        stream: Callable[[str], None] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> RunResult:
        return _execute(
            self.ssh_argv(cmd),
            shell=False,
            stream=stream,
            timeout_s=timeout_s,
            what=f"ssh {self.target.host}",
            tail_lines=self.tail_lines,
        )

    def put(self, local: Path, remote: str) -> None:
        self._transfer(str(local), self._remote(remote), f"scp put {remote}")

    def get(self, remote: str, local: Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._transfer(self._remote(remote), str(local), f"scp get {remote}")

    def _transfer(self, src: str, dst: str, what: str) -> None:
        result = _execute(
            self.scp_argv(src, dst),
            shell=False,
            stream=None,
            timeout_s=DEFAULT_TIMEOUT_S,
            what=what,
            tail_lines=self.tail_lines,
        )
        if result.code != 0:
            raise InfraError(f"{what} failed (exit {result.code}): {result.stdout_tail[-400:]}")

    def tunnel(self, remote_port: int) -> int:
        local_port = _free_port()
        argv = self.ssh_argv(extra=["-N", "-L", f"{local_port}:localhost:{remote_port}"])
        self._tunnels.append(subprocess.Popen(argv))
        return local_port

    def close(self) -> None:
        for proc in self._tunnels:
            proc.terminate()
        self._tunnels.clear()
