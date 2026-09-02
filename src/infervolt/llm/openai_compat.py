"""Client for any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, OpenAI, ...)."""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from infervolt.llm.base import LLMError

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


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
            resp = self._client.chat.completions.create(
                model=self.model_id, messages=messages, response_format=fmt
            )
            text = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_FENCE.sub("", text.strip()))
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
