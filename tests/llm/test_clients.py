import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import openai
import pytest
from pydantic import SecretStr

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


class _AnthropicBoom(anthropic.APIError):
    """A real ``anthropic.APIError`` subclass, built without the SDK's constructor."""

    def __init__(self) -> None:
        Exception.__init__(self, "upstream exploded")


class _RaisingAnthropicStub:
    def __init__(self, exc: BaseException) -> None:
        self.messages = SimpleNamespace(parse=self._parse)
        self.exc = exc

    def _parse(self, **kw):  # type: ignore[no-untyped-def]
        raise self.exc


def test_anthropic_provider_error_becomes_llm_error() -> None:
    stub = _RaisingAnthropicStub(_AnthropicBoom())
    with pytest.raises(LLMError) as excinfo:
        AnthropicClient(client=stub).structured(system="s", user="u", schema=DiagnosisOut)
    assert "anthropic: _AnthropicBoom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, anthropic.APIError)


def test_anthropic_non_provider_error_also_becomes_llm_error() -> None:
    """Everything that escapes ``structured`` is the provider: the frame only calls the SDK.

    The documented ``APIError``/``HTTPError`` tuple does not cover the failure a user is
    most likely to hit -- no ``ANTHROPIC_API_KEY``, which the SDK reports as a plain
    ``TypeError`` when the client is first built -- and the loop degrades on ``LLMError``.
    A missing key must cost a run its narrative, not its measurements.
    """
    stub = _RaisingAnthropicStub(TypeError("Could not resolve authentication method"))
    with pytest.raises(LLMError) as excinfo:
        AnthropicClient(client=stub).structured(system="s", user="u", schema=DiagnosisOut)
    assert "anthropic: TypeError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_anthropic_missing_key_at_construction_becomes_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction is lazy, so a key discovered missing there is still an ``LLMError``.

    Building the SDK client in ``__init__`` would raise inside ``make_llm``, before the
    planner exists to degrade around it -- killing a run over prose it could have
    templated.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = AnthropicClient(model_id="claude-opus-5")  # no client passed: builds its own
    with pytest.raises(LLMError):
        client.structured(system="s", user="u", schema=DiagnosisOut)


def test_openai_missing_client_construction_error_becomes_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kw: (_ for _ in ()).throw(TypeError("no api_key"))
    )
    with pytest.raises(LLMError):
        OpenAICompatClient(model_id="m").structured(system="s", user="u", schema=DiagnosisOut)


def test_anthropic_missing_parsed_output_raises() -> None:
    class _NoneStub:
        messages = SimpleNamespace(
            parse=lambda **kw: SimpleNamespace(parsed_output=None, stop_reason="end_turn")
        )

    with pytest.raises(LLMError):
        AnthropicClient(client=_NoneStub()).structured(system="s", user="u", schema=DiagnosisOut)


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


class _OpenAIBoom(openai.APIError):
    """A real ``openai.APIError`` subclass, built without the SDK's constructor."""

    def __init__(self) -> None:
        Exception.__init__(self, "upstream exploded")


def test_openai_provider_error_becomes_llm_error() -> None:
    class _RaisingOpenAIStub:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):  # type: ignore[no-untyped-def]
            raise _OpenAIBoom()

    with pytest.raises(LLMError) as excinfo:
        OpenAICompatClient(model_id="m", client=_RaisingOpenAIStub()).structured(
            system="s", user="u", schema=DiagnosisOut
        )
    assert "openai: _OpenAIBoom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, openai.APIError)


def test_openai_compat_extracts_json_from_prose_on_first_call() -> None:
    stub = _OpenAIStub(["Here is the JSON:\n```json\n" + GOOD.model_dump_json() + "\n```"])
    out = OpenAICompatClient(model_id="m", client=stub).structured(
        system="s", user="u", schema=DiagnosisOut
    )
    assert out == GOOD and len(stub.calls) == 1


def test_replay_records_then_replays(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    stub = _AnthropicStub()
    rec = ReplayLLMClient(path, inner=AnthropicClient(client=stub))
    assert rec.structured(system="s", user="u", schema=DiagnosisOut) == GOOD
    replay = ReplayLLMClient(path, inner=None)
    assert replay.structured(system="s", user="u", schema=DiagnosisOut) == GOOD
    with pytest.raises(LLMError):
        replay.structured(system="s", user="different", schema=DiagnosisOut)
    entry = next(iter(json.loads(path.read_text()).values()))
    assert entry["model_id"] == "claude-opus-5"
    assert entry["data"]["primary_rule_id"] == "R1"
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_reads_legacy_cassettes_without_model_id(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    key = ReplayLLMClient._key("s", "u", DiagnosisOut)
    path.write_text(json.dumps({key: GOOD.model_dump(mode="json")}))
    replay = ReplayLLMClient(path, inner=None)
    assert replay.structured(system="s", user="u", schema=DiagnosisOut) == GOOD


def test_factory(tmp_path: Path) -> None:
    s = Settings(home=tmp_path)
    assert isinstance(make_llm("fake", s), FakeLLMClient)
    cassette = Settings(home=tmp_path, llm_cassette=tmp_path / "c.json")
    assert isinstance(make_llm("fake", cassette), ReplayLLMClient)
    with pytest.raises(KeyError):
        make_llm("nope", s)


def _capture_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Record the kwargs the lazy ``anthropic.Anthropic()`` construction is given."""
    seen: dict[str, object] = {}

    def build(**kw: object) -> _AnthropicStub:
        seen.update(kw)
        return _AnthropicStub()

    monkeypatch.setattr("anthropic.Anthropic", build)
    return seen


def test_factory_passes_the_configured_anthropic_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``INFERVOLT_ANTHROPIC_API_KEY`` has to reach the SDK, or setting it does nothing."""
    seen = _capture_sdk(monkeypatch)
    settings = Settings(home=tmp_path, anthropic_api_key=SecretStr("sk-test"))
    client = make_llm("anthropic", settings)
    assert isinstance(client, AnthropicClient)
    client.structured(system="s", user="u", schema=DiagnosisOut)
    assert seen == {"api_key": "sk-test"}


def test_factory_leaves_key_resolution_to_the_sdk_when_none_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset means unset: the SDK still reads ``ANTHROPIC_API_KEY`` or a stored profile.

    Passing ``api_key=""`` would be different -- it is a value, and it would override
    whatever the environment had with an empty credential. A blank counts as unset
    because .env.example ships the line with nothing after the ``=``.
    """
    for settings in (
        Settings(home=tmp_path),
        Settings(home=tmp_path, anthropic_api_key=SecretStr("")),
    ):
        seen = _capture_sdk(monkeypatch)
        client = make_llm("anthropic", settings)
        assert isinstance(client, AnthropicClient)
        client.structured(system="s", user="u", schema=DiagnosisOut)
        assert seen == {}
