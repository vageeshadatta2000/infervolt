from pathlib import Path
from types import SimpleNamespace

import pytest

from infervolt.config import Settings
from infervolt.llm.anthropic_client import AnthropicClient
from infervolt.llm.base import DiagnosisOut, LLMError
from infervolt.llm.factory import make_llm
from infervolt.llm.fake import FakeLLMClient
from infervolt.llm.openai_compat import OpenAICompatClient
from infervolt.llm.replay import ReplayLLMClient

GOOD = DiagnosisOut(primary_rule_id="R1", ranked_rule_ids=["R1"], rationale="x", confidence=0.8)


class _AnthropicStub:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)
        self.stop_reason = stop_reason

    def _parse(self, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(kw)
        return SimpleNamespace(parsed_output=GOOD, stop_reason=self.stop_reason)


def test_anthropic_client_uses_messages_parse() -> None:
    stub = _AnthropicStub()
    client = AnthropicClient(model_id="claude-opus-5", client=stub)
    out = client.structured(system="s", user="u", schema=DiagnosisOut)
    assert out == GOOD
    call = stub.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is DiagnosisOut
    assert call["system"] == "s"


def test_anthropic_refusal_raises() -> None:
    with pytest.raises(LLMError):
        AnthropicClient(client=_AnthropicStub("refusal")).structured(
            system="s", user="u", schema=DiagnosisOut
        )


class _OpenAIStub:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(kw)
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_openai_compat_repairs_invalid_json_once() -> None:
    stub = _OpenAIStub(["not json", "```json\n" + GOOD.model_dump_json() + "\n```"])
    out = OpenAICompatClient(model_id="m", client=stub).structured(
        system="s", user="u", schema=DiagnosisOut
    )
    assert out == GOOD and len(stub.calls) == 2
    assert stub.calls[0]["response_format"]["type"] == "json_schema"


def test_openai_compat_gives_up_after_two_failures() -> None:
    with pytest.raises(LLMError):
        OpenAICompatClient(model_id="m", client=_OpenAIStub(["x", "y"])).structured(
            system="s", user="u", schema=DiagnosisOut
        )


def test_replay_records_then_replays(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    stub = _AnthropicStub()
    rec = ReplayLLMClient(path, inner=AnthropicClient(client=stub))
    assert rec.structured(system="s", user="u", schema=DiagnosisOut) == GOOD
    replay = ReplayLLMClient(path, inner=None)
    assert replay.structured(system="s", user="u", schema=DiagnosisOut) == GOOD
    with pytest.raises(LLMError):
        replay.structured(system="s", user="different", schema=DiagnosisOut)


def test_factory(tmp_path: Path) -> None:
    s = Settings(home=tmp_path)
    assert isinstance(make_llm("fake", s), FakeLLMClient)
    cassette = Settings(home=tmp_path, llm_cassette=tmp_path / "c.json")
    assert isinstance(make_llm("fake", cassette), ReplayLLMClient)
    with pytest.raises(KeyError):
        make_llm("nope", s)
