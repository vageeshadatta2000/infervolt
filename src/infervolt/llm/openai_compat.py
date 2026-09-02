"""Client for any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, OpenAI, ...)."""

from __future__ import annotations

import importlib
import re
from functools import cache
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from infervolt.llm.base import LLMError

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@cache
def _provider_errors() -> tuple[type[BaseException], ...]:
    """Exception classes that mean "the provider failed", imported lazily.

    See ``anthropic_client._provider_errors``: nothing is imported at module scope so
    the Fake path never needs an SDK, and both ``httpx`` and ``httpx2`` are tried
    because which one carries transport errors depends on the installed SDK release.
    """
    found: list[type[BaseException]] = []
    for module, attr in (("openai", "APIError"), ("httpx", "HTTPError"), ("httpx2", "HTTPError")):
        try:
            exc = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(exc, type) and issubclass(exc, BaseException):
            found.append(exc)
    return tuple(found)


def _json_candidates(text: str) -> list[str]:
    """The substrings of a reply worth trying to parse, best guess first.

    Small local models routinely wrap the object in prose ("Here is the JSON:") or a
    code fence even when asked for JSON only. Stripping the fence handles the common
    case; the outermost brace pair rescues the rest without a second round trip.
    """
    candidates = [_FENCE.sub("", text.strip()).strip()]
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        candidates.append(text[lo : hi + 1])
    return candidates


class OpenAICompatClient:
    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key)  # api_key is a plain str
        self._client: Any = client

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        last_error = ""
        for _ in range(2):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id, messages=messages, response_format=fmt
                )
            except _provider_errors() as e:
                # A transport or API failure is not something a repair round trip can
                # fix, so it ends the loop instead of burning the retry.
                raise LLMError(f"openai: {type(e).__name__}: {e}") from e
            text = resp.choices[0].message.content or ""
            for candidate in _json_candidates(text):
                try:
                    return schema.model_validate_json(candidate)
                except (ValidationError, ValueError) as e:
                    last_error = str(e)
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"That was not valid {schema.__name__} JSON: {last_error}. "
                    "Reply with only the corrected JSON.",
                },
            ]
        raise LLMError(f"could not obtain valid {schema.__name__}: {last_error}")
