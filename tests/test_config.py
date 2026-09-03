import os
from pathlib import Path

import pytest

from infervolt.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give every test a pristine environment.

    Settings reads ``.env`` relative to the working directory, so run from an empty
    tmp_path as well as dropping ambient INFERVOLT_* vars -- otherwise a developer's
    local .env silently changes what the defaults test sees.
    """
    for key in list(os.environ):
        if key.startswith("INFERVOLT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def test_home_env_override_moves_derived_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INFERVOLT_HOME", str(tmp_path / "iv"))
    s = Settings()
    assert s.home == tmp_path / "iv"
    assert s.runs_dir == tmp_path / "iv" / "runs"
    assert s.ledger_path == tmp_path / "iv" / "ledger.sqlite"


def test_api_key_is_secret_and_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERVOLT_OPENAI_API_KEY", "sk-x")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-x"
    assert "sk-x" not in repr(s)


def test_defaults() -> None:
    s = Settings()
    assert s.home == Path.home() / ".infervolt"
    assert s.anthropic_model == "claude-opus-5"
    assert s.openai_base_url == "http://localhost:8000/v1"
    assert s.openai_model == "default"
    assert s.openai_api_key.get_secret_value() == "EMPTY"
    assert s.llm_cassette is None
