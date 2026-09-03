"""Data models for compute we rent (or already own).

This module imports nothing internal on purpose: the ledger persists :class:`Instance`
rows, and the providers that create them import the ledger.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

LaunchMode = Literal["vm", "container"]


class InfraError(Exception):
    """Anything a provider can fail at."""


class ProvisionError(InfraError):
    """The instance was not created, or never became reachable."""


class Offer(BaseModel):
    """One purchasable configuration, as the provider advertises it today.

    ``raw_id`` is the provider's own key for the offer (Thunder's ``a100xl_x1``, RunPod's
    gpu type id); it is what a later ``provision`` call is built from, so it must survive
    the trip through the ledger unchanged rather than being normalised away.
    """

    provider: str
    gpu: str
    gpu_mem_gb: float
    count: int = 1
    usd_per_hour: float
    region: str | None = None
    spot: bool = False
    raw_id: str


class InstanceSpec(BaseModel):
    """What we want. ``gpu`` is a family name (``a100``, ``h100``), not an offer id.

    Providers resolve the family to their own vocabulary, so the same spec can be sent to
    Thunder or RunPod. ``image`` is the container image on container providers and the OS
    template on VM providers; ``None`` means "the provider's default".
    """

    gpu: str
    count: int = 1
    disk_gb: int = 100
    image: str | None = None
    ports: list[int] = Field(default_factory=lambda: [22, 8000])
    env: dict[str, str] = Field(default_factory=dict)
    ssh_public_key: str | None = None
    name: str = "infervolt"


class SshTarget(BaseModel):
    host: str
    port: int = 22
    user: str = "ubuntu"
    key_path: str
    proxy_command: str | None = None


class Instance(BaseModel):
    """A rented box, in whatever state the provider last reported.

    ``ssh`` is ``None`` until the provider knows where the box is; ``raw`` carries the
    provider's own bookkeeping (uuid, key path, offer id) so a fresh process can pick up
    a ledger row and still terminate the thing.
    """

    provider: str
    id: str
    gpu: str
    count: int = 1
    usd_per_hour: float = 0.0
    created_at: float = Field(default_factory=time.time)
    ssh: SshTarget | None = None
    launch_mode: LaunchMode = "vm"
    status: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Exit code plus the tail of the combined stdout/stderr stream.

    Only the tail: a bootstrap that installs vLLM prints tens of megabytes, and the
    caller that wanted all of it passed a ``stream`` callback.
    """

    code: int
    stdout_tail: str = ""
