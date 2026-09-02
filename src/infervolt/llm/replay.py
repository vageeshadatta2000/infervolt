"""Cassette wrapper: record real LLM replies, replay them in CI without keys."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.llm.base import LLMClient, LLMError

T = TypeVar("T", bound=BaseModel)


class ReplayLLMClient:
    def __init__(self, path: Path, inner: LLMClient | None) -> None:
        self.path = path
        self.inner = inner
        self.model_id = f"replay({inner.model_id if inner else 'offline'})"
        self._data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _key(system: str, user: str, schema: type[BaseModel]) -> str:
        return hashlib.sha256(f"{schema.__name__}\n{system}\n{user}".encode()).hexdigest()

    @staticmethod
    def _payload(entry: Any) -> Any:
        """The recorded reply inside a cassette entry, in either supported shape.

        Entries are ``{"model_id": ..., "data": ...}`` so a cassette says which model
        produced each reply. The model is deliberately *not* part of the key: replay
        has to hit without an inner client, which is exactly when no model id is known.
        Older cassettes stored the bare dump, so those still load.
        """
        if isinstance(entry, dict) and set(entry) == {"model_id", "data"}:
            return entry["data"]
        return entry

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        key = self._key(system, user, schema)
        if key in self._data:
            return schema.model_validate(self._payload(self._data[key]))
        if self.inner is None:
            raise LLMError(f"no cassette entry for {schema.__name__} and no inner client")
        out = self.inner.structured(system=system, user=user, schema=schema)
        self._data[key] = {"model_id": self.inner.model_id, "data": out.model_dump(mode="json")}
        self._write()
        return out

    def _write(self) -> None:
        """Replace the cassette atomically: a crash mid-write must not truncate it.

        The temp file is a sibling so ``os.replace`` stays within one filesystem, where
        it is atomic.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)
