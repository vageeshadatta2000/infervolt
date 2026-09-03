"""The machine you are already sitting at, behind the same interface as a rented one.

This is what makes ``infervolt optimize`` and ``infervolt remote optimize`` the same
code path: the local provider provisions nothing, costs nothing, and hands back a session
that runs subprocesses here.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, ClassVar

from infervolt.core.types import HardwareProfile
from infervolt.hardware.profiles import get_profile
from infervolt.infra.base import Provider
from infervolt.infra.ssh import LocalSession, Session
from infervolt.infra.types import Instance, InstanceSpec, LaunchMode, Offer

if TYPE_CHECKING:  # pragma: no cover
    from infervolt.store.ledger import Ledger

LOCAL_ID = "local"


def _detected_profile() -> HardwareProfile | None:
    """Ask ``hardware.detect`` what this machine is, if that module exists yet.

    Detection lands in a later work package; until then the local provider still has to
    answer ``list_offers``, and "I do not know what this box is" is a better answer than
    a guessed A100.
    """
    if importlib.util.find_spec("infervolt.hardware.detect") is None:
        return None
    module = importlib.import_module("infervolt.hardware.detect")
    detect = getattr(module, "detect_profile", None)
    if detect is None:
        return None
    profile = detect()
    return profile if isinstance(profile, HardwareProfile) else None


class LocalProvider(Provider):
    """No provisioning, no cost, no keys."""

    name: ClassVar[str] = "local"
    launch_mode: ClassVar[LaunchMode] = "vm"
    supports_scp: ClassVar[bool] = True

    def __init__(self, ledger: Ledger | None = None, *, profile_name: str | None = None) -> None:
        super().__init__(ledger)
        self.profile_name = profile_name

    def _profile(self) -> HardwareProfile | None:
        if self.profile_name is not None:
            return get_profile(self.profile_name)
        return _detected_profile()

    def list_offers(self, gpu: str | None = None) -> list[Offer]:
        profile = self._profile()
        if profile is None:
            return []
        offer = Offer(
            provider=self.name,
            gpu=profile.gpu,
            gpu_mem_gb=profile.mem_gb,
            count=profile.count,
            usd_per_hour=profile.usd_per_hour,
            raw_id=profile.name,
        )
        if gpu and gpu.lower() not in f"{offer.gpu} {offer.raw_id}".lower():
            return []
        return [offer]

    def provision(self, spec: InstanceSpec) -> Instance:
        """Hand back the machine we are on. The id is fixed, so the ledger holds one row.

        ``ssh`` stays ``None``: there is nothing to connect to, and :meth:`connect`
        returns a local session instead of trying.
        """
        profile = self._profile()
        return self._record(
            Instance(
                provider=self.name,
                id=LOCAL_ID,
                gpu=profile.gpu if profile else spec.gpu,
                count=profile.count if profile else spec.count,
                usd_per_hour=profile.usd_per_hour if profile else 0.0,
                ssh=None,
                launch_mode=self.launch_mode,
                status="running",
            )
        )

    def refresh(self, inst: Instance) -> Instance:
        return inst.model_copy(update={"status": "running"})

    def wait_ready(
        self,
        inst: Instance,
        timeout_s: float = 900.0,
        **kwargs: object,
    ) -> Instance:
        """Already here. Overridden because the base class waits for an SSH port."""
        return self.refresh(inst)

    def connect(self, inst: Instance) -> Session:
        return LocalSession()

    def terminate(self, inst: Instance) -> None:
        """Nothing to destroy; the ledger row is closed so ``infra gc`` stops listing it."""
        self._record_terminated(inst)

    def cost_per_hour(self, inst: Instance) -> float:
        """Zero: you already paid for this machine, and a run here should not pretend
        otherwise when it compares itself against a rented one."""
        return 0.0
