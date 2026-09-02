"""Anthropic reference client using structured outputs (messages.parse)."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.llm.base import LLMError

T = TypeVar("T", bound=BaseModel)


class AnthropicClient:
    def __init__(self, model_id: str = "claude-opus-5", client: Any | None = None) -> None:
        self.model_id = model_id
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client: Any = client

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        response = self._client.messages.parse(
            model=self.model_id,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("no parsed output")
        return schema.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
