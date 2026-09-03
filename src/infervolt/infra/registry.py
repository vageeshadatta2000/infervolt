"""Provider discovery through the ``infervolt.providers`` entry-point group.

Same shape as the engine registry: a provider that lives in another distribution is
usable the moment it is installed, without infervolt importing it by name.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from infervolt.infra.base import Provider

if TYPE_CHECKING:  # pragma: no cover
    from infervolt.store.ledger import Ledger

GROUP = "infervolt.providers"


def available_providers() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=GROUP))


def get_provider(name: str, ledger: Ledger | None = None, **kwargs: Any) -> Provider:
    for ep in entry_points(group=GROUP):
        if ep.name == name:
            cls = ep.load()
            provider = cls(ledger, **kwargs)
            assert isinstance(provider, Provider)
            return provider
    raise KeyError(f"unknown provider {name!r}; available: {available_providers()}")
