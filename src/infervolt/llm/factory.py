"""Build the configured `LLMClient`, optionally wrapped in a record/replay cassette."""

from __future__ import annotations

from infervolt.config import Settings
from infervolt.llm.anthropic_client import AnthropicClient
from infervolt.llm.base import LLMClient
from infervolt.llm.fake import FakeLLMClient
from infervolt.llm.openai_compat import OpenAICompatClient
from infervolt.llm.replay import ReplayLLMClient


def make_llm(name: str, settings: Settings) -> LLMClient:
    inner: LLMClient
    if name == "fake":
        inner = FakeLLMClient()
    elif name == "anthropic":
        inner = AnthropicClient(model_id=settings.anthropic_model)
    elif name == "openai":
        inner = OpenAICompatClient(
            model_id=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    else:
        raise KeyError(f"unknown llm {name!r}; use fake, anthropic, or openai")
    if settings.llm_cassette is not None:
        return ReplayLLMClient(settings.llm_cassette, inner=inner)
    return inner
