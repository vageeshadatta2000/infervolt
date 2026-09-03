"""Process-wide settings. Every value can be overridden with INFERVOLT_<NAME> env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERVOLT_", env_file=".env", extra="ignore")

    home: Path = Field(default_factory=lambda: Path.home() / ".infervolt")
    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: SecretStr | None = None
    """Key for the Anthropic client, from ``INFERVOLT_ANTHROPIC_API_KEY``.

    ``None`` is not "no key": it means infervolt has none to hand over and the SDK should
    resolve one the way it always does -- ``ANTHROPIC_API_KEY`` or a stored profile. The
    prefixed name exists so a user who runs several tools against several accounts can
    point this one somewhere without moving the ambient variable."""
    openai_base_url: str = "http://localhost:8000/v1"
    openai_model: str = "default"
    openai_api_key: SecretStr = SecretStr("EMPTY")
    llm_cassette: Path | None = None

    thunder_api_token: SecretStr | None = None
    """Thunder Compute REST token, from ``INFERVOLT_THUNDER_API_TOKEN``.

    Optional, and not the only source: ``TNR_API_TOKEN`` (which the ``tnr`` CLI already
    uses) is checked first, and the CLI's own credential file last. See
    :func:`infervolt.infra.thunder.resolve_token`."""
    runpod_api_key: SecretStr | None = None
    prime_api_key: SecretStr | None = None

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ledger_path(self) -> Path:
        return self.home / "ledger.sqlite"

    @property
    def keys_dir(self) -> Path:
        """Where SSH keys for rented boxes live -- ours and the ones providers mint."""
        return self.home / "keys"


def get_settings() -> Settings:
    return Settings()
