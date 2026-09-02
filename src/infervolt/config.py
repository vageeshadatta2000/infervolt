"""Process-wide settings. Every value can be overridden with INFERVOLT_<NAME> env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERVOLT_", env_file=".env", extra="ignore")

    home: Path = Field(default_factory=lambda: Path.home() / ".infervolt")
    anthropic_model: str = "claude-opus-5"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_model: str = "default"
    openai_api_key: SecretStr = SecretStr("EMPTY")
    llm_cassette: Path | None = None

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ledger_path(self) -> Path:
        return self.home / "ledger.sqlite"


def get_settings() -> Settings:
    return Settings()
