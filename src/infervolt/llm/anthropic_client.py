"""Anthropic reference client using structured outputs (messages.parse)."""

from __future__ import annotations

import importlib
from functools import cache
from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.llm.base import LLMError

T = TypeVar("T", bound=BaseModel)


@cache
def _provider_errors() -> tuple[type[BaseException], ...]:
    """Exception classes that mean "the provider failed", imported lazily.

    The Fake and replay paths must work with no SDK installed, so nothing is imported
    at module scope. Transport errors surface under whichever HTTP client the installed
    SDK is built on -- ``httpx`` historically, ``httpx2`` in current releases -- so both
    are tried and whatever is present contributes. Missing modules simply drop out; an
    empty tuple is a valid ``except`` target and catches nothing.
    """
    found: list[type[BaseException]] = []
    sources = (("anthropic", "APIError"), ("httpx", "HTTPError"), ("httpx2", "HTTPError"))
    for module, attr in sources:
        try:
            exc = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(exc, type) and issubclass(exc, BaseException):
            found.append(exc)
    return tuple(found)


class AnthropicClient:
    def __init__(self, model_id: str = "claude-opus-5", client: Any | None = None) -> None:
        self.model_id = model_id
        self._client: Any | None = client

    def _sdk(self) -> Any:
        """The SDK client, built on first use.

        Constructing it is what discovers a missing ``ANTHROPIC_API_KEY``, so it happens
        inside ``structured`` where that failure becomes an :class:`LLMError` the loop can
        degrade around -- rather than at construction, where it would kill a run that had
        every measurement it needed and only wanted prose.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        try:
            response = self._sdk().messages.parse(
                model=self.model_id,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except _provider_errors() as e:
            # Callers up the loop handle one failure type from every client. A bare
            # SDK/transport error leaking out would make each of them import the SDKs.
            raise LLMError(f"anthropic: {type(e).__name__}: {e}") from e
        except Exception as e:  # noqa: BLE001 - see below: every escape here is the SDK's
            # The tuple above is the documented set, but it does not cover everything the
            # SDK raises: a missing ``ANTHROPIC_API_KEY`` surfaces as a plain ``TypeError``
            # from the constructor at the first call, and auth/config mistakes generally
            # arrive as builtins. This frame only ever calls into the SDK, so anything
            # that escapes it is the provider failing, and a caller that degrades on
            # ``LLMError`` should degrade on a missing key too rather than die.
            raise LLMError(f"anthropic: {type(e).__name__}: {e}") from e
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("no parsed output")
        return schema.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
