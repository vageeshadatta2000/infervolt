"""The controller's kill switch: stop paying when the cap is reached.

Rented GPUs are billed by wall-clock time, so the only quantity that matters is elapsed
seconds times the hourly rate -- no metering call, no provider agreement, nothing that
can fail silently. The guard runs on the controller rather than on the box, because the
failure it exists for is "the box stopped answering and kept billing".
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import time
from collections.abc import Callable

from infervolt.infra.base import Provider
from infervolt.infra.types import Instance


class SpendGuard:
    """Fire ``on_exceed`` once, when ``usd_per_hour`` x elapsed reaches ``max_usd``.

    Deliberately a timer and not an accountant: it does not know what the provider will
    actually invoice, only that we agreed to stop at a number. ``max_usd <= 0`` means no
    cap, matching :class:`~infervolt.core.types.Budget`.
    """

    def __init__(
        self,
        usd_per_hour: float,
        max_usd: float,
        on_exceed: Callable[[], None],
        *,
        interval_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.usd_per_hour = usd_per_hour
        self.max_usd = max_usd
        self.on_exceed = on_exceed
        self.interval_s = interval_s
        self.clock = clock
        self.started_at = clock()
        self.fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def spent_usd(self) -> float:
        return max(0.0, self.clock() - self.started_at) * self.usd_per_hour / 3600.0

    @property
    def exceeded(self) -> bool:
        return self.max_usd > 0 and self.spent_usd() >= self.max_usd

    @property
    def watching(self) -> bool:
        """True while a thread is actually checking. False for an uncapped or free run."""
        return self._thread is not None

    def start(self) -> SpendGuard:
        if self.max_usd <= 0 or self.usd_per_hour <= 0:
            # Nothing to watch: an untracked or free instance would otherwise keep a
            # thread alive for the whole run and never have anything to say.
            return self
        self.started_at = self.clock()
        thread = threading.Thread(target=self._watch, name="infervolt-spend-guard", daemon=True)
        self._thread = thread
        thread.start()
        return self

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self.exceeded:
                self._fire()
                return
            self._stop.wait(self.interval_s)

    def _fire(self) -> None:
        with self._lock:
            if self.fired:
                return
            self.fired = True
        self.on_exceed()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5.0)
            self._thread = None

    def __enter__(self) -> SpendGuard:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def ensure_terminated(provider: Provider, inst: Instance) -> Callable[[], None]:
    """Register an idempotent teardown at exit, and return it so it can be run sooner.

    Idempotent because it is called from at least three places -- the ``finally`` of the
    run, the spend guard, and interpreter shutdown -- and only the first one should cost
    an API call. Termination failures are swallowed: by the time this runs there is
    usually nothing left that could report them, and ``infervolt infra gc`` is the
    backstop for a box the ledger still lists.
    """
    state = {"done": False}

    def terminate() -> None:
        if state["done"]:
            return
        state["done"] = True
        atexit.unregister(terminate)
        with contextlib.suppress(Exception):  # shutdown path; see the docstring
            provider.terminate(inst)

    atexit.register(terminate)
    return terminate
