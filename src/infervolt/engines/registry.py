"""Adapter discovery through the `infervolt.engines` entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points

from infervolt.engines.base import EngineAdapter


def available_engines() -> list[str]:
    return sorted(ep.name for ep in entry_points(group="infervolt.engines"))


def get_adapter(name: str) -> EngineAdapter:
    for ep in entry_points(group="infervolt.engines"):
        if ep.name == name:
            cls = ep.load()
            adapter = cls()
            assert isinstance(adapter, EngineAdapter)
            return adapter
    raise KeyError(f"unknown engine {name!r}; available: {available_engines()}")
