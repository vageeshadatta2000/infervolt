"""One remote optimize run, from an empty account to artifacts on the controller.

The whole file is arranged around a single invariant: **whatever happens, stop paying.**
Termination is reachable from three directions -- the ``finally`` of the run, the spend
guard's thread, and ``atexit`` -- and :func:`~infervolt.infra.budget.ensure_terminated`
makes all three idempotent so that only the first one costs an API call. Failures are
returned as a :class:`RemoteOutcome` rather than raised, because an exception escaping
here is an exception that has to travel past the teardown to be useful, and a caller that
sees a traceback instead of "cost cap reached, instance i-fake terminated" learns less.

The run itself is deliberately thin. Everything about *how to optimize* lives on the box,
in the same ``infervolt optimize`` a laptop runs; the controller only knows how to read
two lines out of its output -- where the box put its work directory, and which run id it
opened -- and where to put the artifacts afterwards.
"""

from __future__ import annotations

import atexit
import contextlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from infervolt.config import Settings
from infervolt.infra.base import Provider
from infervolt.infra.budget import SpendGuard, ensure_terminated
from infervolt.infra.ssh import Session
from infervolt.infra.types import InfraError, Instance, InstanceSpec
from infervolt.remote.bootstrap import (
    BOOTSTRAP_OK,
    WORK_MARKER,
    EngineInstall,
    bootstrap_script,
    controller_sha,
    kill_command,
    optimize_command,
    passthrough_env,
)
from infervolt.store.ledger import Ledger

DEFAULT_MAX_USD = 10.0
DEFAULT_WORK = "/workspace/infervolt"
CAP_MESSAGE = "cost cap reached"

RUN_ID_RE = re.compile(r"^run:\s*(\S+)\s*$")
"""``optimize`` prints ``run: <id>`` before it does anything else.

Parsing the log is not elegant, but it is the only channel that exists: the run id is
minted on the box, in the box's ledger, and the controller needs it to know which
directory to pull back."""


class RemoteError(InfraError):
    """The remote side of a run failed in a way that makes continuing pointless."""


class RemoteOutcome(BaseModel):
    """What the run cost, what it produced, and whether it worked."""

    instance_id: str
    ok: bool
    message: str
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    run_id: str | None = None
    local_run_dir: Path | None = None


class RemoteRun:
    """Provision, bootstrap, optimize, pull back, terminate.

    ``keep`` skips only the teardown, never the guard: an instance kept for the next trial
    is still an instance being billed, and the cap is what the user was promised.
    """

    def __init__(
        self,
        provider: Provider,
        spec: InstanceSpec,
        optimize_args: Sequence[str],
        *,
        max_usd: float = DEFAULT_MAX_USD,
        keep: bool = False,
        ref: str | None = None,
        engine_install: EngineInstall = "venv",
        instance_id: str | None = None,
        ledger: Ledger | None = None,
        settings: Settings | None = None,
        env: Mapping[str, str] | None = None,
        log: Callable[[str], None] = lambda line: None,
        guard_interval_s: float = 10.0,
        wait_timeout_s: float = 900.0,
        bootstrap_timeout_s: float = 3600.0,
        optimize_timeout_s: float = 24 * 3600.0,
    ) -> None:
        self.provider = provider
        self.spec = spec
        self.optimize_args = list(optimize_args)
        self.max_usd = max_usd
        self.keep = keep
        self.ref = ref
        self.engine_install = engine_install
        self.instance_id = instance_id
        self.ledger = ledger
        self.settings = settings or Settings()
        self.env = env
        self.log = log
        self.guard_interval_s = guard_interval_s
        self.wait_timeout_s = wait_timeout_s
        self.bootstrap_timeout_s = bootstrap_timeout_s
        self.optimize_timeout_s = optimize_timeout_s

        self._started = time.monotonic()
        self._instance: Instance | None = None
        self._session: Session | None = None
        self._guard: SpendGuard | None = None
        self._terminate: Callable[[], None] | None = None
        self._run_id: str | None = None
        self._local_run_dir: Path | None = None
        self._exceeded = False

    # ---- logging
    def _elapsed(self) -> float:
        return time.monotonic() - self._started

    def _cost(self) -> float:
        return self._guard.spent_usd() if self._guard is not None else 0.0

    def _say(self, phase: str, detail: str = "") -> None:
        """Every transition, with the two numbers that decide whether to intervene."""
        head = f"[{self._elapsed():6.1f}s ${self._cost():6.3f}] {phase}"
        self.log(f"{head} {detail}".rstrip())

    # ---- entry point
    def run(self) -> RemoteOutcome:
        self._started = time.monotonic()
        try:
            return self._orchestrate()
        except (InfraError, OSError) as e:
            self._say("failed", str(e))
            return self._outcome(ok=False, message=str(e))
        finally:
            self._teardown()

    def _outcome(self, *, ok: bool, message: str) -> RemoteOutcome:
        return RemoteOutcome(
            instance_id=self._instance.id if self._instance else "",
            ok=ok,
            message=message,
            cost_usd=self._cost(),
            elapsed_s=self._elapsed(),
            run_id=self._run_id,
            local_run_dir=self._local_run_dir,
        )

    # ---- the run
    def _orchestrate(self) -> RemoteOutcome:
        ref = self._resolve_ref()
        inst = self._acquire()
        self._instance = inst
        self._arm(inst)

        inst = self.provider.wait_ready(inst, timeout_s=self.wait_timeout_s, log=self.log)
        self._instance = inst
        self._say("ready", f"{inst.provider}:{inst.id} {inst.ssh.host if inst.ssh else '-'}")

        session = self.provider.connect(inst)
        self._session = session

        work = self._bootstrap(session, ref)
        code = self._optimize(session, work)
        if self._exceeded:
            return self._outcome(ok=False, message=CAP_MESSAGE)
        if self._run_id is not None:
            self._pull(session, work, self._run_id)
        if code != 0:
            return self._outcome(ok=False, message=f"remote optimize exited {code}")
        kept = " (instance kept)" if self.keep else ""
        return self._outcome(ok=True, message=f"ok{kept}")

    def _resolve_ref(self) -> str:
        if self.ref:
            return self.ref
        sha = controller_sha()
        if sha is None:
            raise RemoteError(
                "cannot tell which commit this controller is running; pass --ref <sha> "
                "so the box installs a reviewable version"
            )
        return sha

    def _acquire(self) -> Instance:
        """Rent a box, or pick up one the ledger says we are already renting."""
        if self.instance_id is None:
            inst = self.provider.provision(self.spec)
            self._say(
                "provision",
                f"{inst.provider}:{inst.id} {inst.gpu} x{inst.count} "
                f"${self.provider.cost_per_hour(inst):.2f}/h",
            )
            return inst
        inst = self._from_ledger(self.instance_id)
        self._say("provision", f"reusing {inst.provider}:{inst.id} {inst.gpu}")
        return inst

    def _from_ledger(self, instance_id: str) -> Instance:
        rows = self.ledger.instances(provider=self.provider.name) if self.ledger else []
        for inst in rows:
            if inst.id == instance_id:
                return inst
        raise RemoteError(
            f"no live {self.provider.name} instance {instance_id!r} in the ledger; "
            "see 'infervolt infra list'"
        )

    def _arm(self, inst: Instance) -> None:
        """Start the cap and register the teardown, before anything slow happens.

        Before ``wait_ready``, not after: a provider that takes fifteen minutes to hand
        over a machine is billing for those fifteen minutes.
        """
        self._terminate = ensure_terminated(self.provider, inst)
        if self.keep:
            # ensure_terminated always registers at exit; --keep means we hold on to the
            # callable for the cost cap but do not want interpreter shutdown to fire it.
            atexit.unregister(self._terminate)
        self._guard = SpendGuard(
            self.provider.cost_per_hour(inst),
            self.max_usd,
            self._on_exceed,
            interval_s=self.guard_interval_s,
        ).start()

    def _on_exceed(self) -> None:
        """Runs on the guard's thread while the main thread is blocked in ``session.run``.

        Killing the remote process is what unblocks it: closing the SSH connection from
        here would leave ``infervolt optimize`` running on a box we are about to destroy,
        and destroying it first would race the kill.
        """
        self._exceeded = True
        self._say("cap", f"{CAP_MESSAGE} (${self.max_usd:.2f}); stopping the run")
        session = self._session
        if session is not None:
            # The box may already be unreachable; terminate is the stop that matters.
            with contextlib.suppress(InfraError, OSError):
                session.run(kill_command(), timeout_s=60.0)
        if self._terminate is not None:
            self._terminate()

    # ---- steps
    def _bootstrap(self, session: Session, ref: str) -> str:
        self._say("bootstrap", f"ref {ref[:12]} engine-install {self.engine_install}")
        script = bootstrap_script(
            ref=ref,
            engine_install=self.engine_install,
            vllm_version=self.settings.vllm_version,
            image=self._image(),
            env=passthrough_env(self.env),
        )
        lines: list[str] = []
        result = session.run(
            script, stream=self._collect(lines), timeout_s=self.bootstrap_timeout_s
        )
        text = "\n".join(lines) if lines else result.stdout_tail
        # Both conditions: `set -e` should make them agree, but a marker printed by a
        # script that then failed, or an exit code lost in an ssh multiplexer, are each
        # enough reason not to start paying for trials.
        if result.code != 0 or BOOTSTRAP_OK not in text:
            raise RemoteError(
                f"remote bootstrap did not finish (exit {result.code}); "
                f"last output: {text[-400:]!r}"
            )
        return self._work_dir(lines)

    def _image(self) -> str | None:
        if self.engine_install != "docker":
            return None
        return self.spec.image or f"vllm/vllm-openai:v{self.settings.vllm_version}"

    @staticmethod
    def _work_dir(lines: Sequence[str]) -> str:
        for line in lines:
            if line.startswith(WORK_MARKER):
                return line[len(WORK_MARKER) :].strip()
        return DEFAULT_WORK

    def _optimize(self, session: Session, work: str) -> int:
        self._say(
            "optimize", f"{work}/venv/bin/infervolt optimize ({len(self.optimize_args)} args)"
        )
        lines: list[str] = []
        result = session.run(
            optimize_command(work, self.optimize_args),
            stream=self._collect(lines, watch_run_id=True),
            timeout_s=self.optimize_timeout_s,
        )
        if self._run_id is None:
            self._scan_for_run_id(lines or result.stdout_tail.splitlines())
        return result.code

    def _pull(self, session: Session, work: str, run_id: str) -> None:
        local = self.settings.runs_dir / run_id
        self._say("pull", f"{run_id} -> {local}")
        session.get(f"{work}/state/runs/{run_id}", local)
        self._local_run_dir = local

    # ---- output plumbing
    def _collect(self, sink: list[str], *, watch_run_id: bool = False) -> Callable[[str], None]:
        """Tee the remote stream: to the caller's log, and to a list we parse afterwards."""

        def handle(line: str) -> None:
            sink.append(line)
            self.log(line)
            if watch_run_id and self._run_id is None:
                self._scan_for_run_id([line])

        return handle

    def _scan_for_run_id(self, lines: Sequence[str]) -> None:
        for line in lines:
            match = RUN_ID_RE.match(line.strip())
            if match:
                self._run_id = match.group(1)
                return

    # ---- teardown
    def _teardown(self) -> None:
        if self._guard is not None:
            self._guard.stop()
        if self._session is not None:
            self._session.close()
        if self._terminate is None or self._instance is None:
            return
        if self.keep and not self._exceeded:
            self._say("keep", f"{self._instance.provider}:{self._instance.id} left running")
            return
        self._say("terminate", f"{self._instance.provider}:{self._instance.id}")
        self._terminate()
        self._say("done", f"${self._cost():.3f} spent")


__all__ = ["RemoteError", "RemoteOutcome", "RemoteRun"]
