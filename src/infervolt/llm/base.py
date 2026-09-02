"""LLM client contract, structured-output schemas, and prompt rendering.

Every prompt is instructions followed by a machine-readable <context> JSON block. Real models read
the whole thing; the Fake client only reads the JSON. The LLM never sees engine flags, only knob
names from the knob space it is shown, and every reply is validated against a pydantic schema.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any, Protocol, TypeVar

from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

from infervolt.core.types import KnobValue

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = (
    "You are infervolt, an LLM-inference performance engineer. You reason from measured evidence "
    "(load-generator metrics, engine counters, roofline estimates) and never invent flags: you may "
    "only reference rule ids and knob names that appear in the <context> block. Reply with JSON "
    "matching the requested schema and nothing else."
)


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    model_id: str

    def structured(self, *, system: str, user: str, schema: type[T]) -> T: ...


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiagnosisOut(_Strict):
    primary_rule_id: str
    ranked_rule_ids: list[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)


class PriorOut(_Strict):
    knobs: dict[str, KnobValue]
    hypothesis: str


class SearchPlanOut(_Strict):
    subspaces: list[str]
    priors: list[PriorOut] = Field(default_factory=list)
    max_trials: int
    rationale: str = ""


class InsightOut(_Strict):
    text: str
    cites: list[str] = Field(default_factory=list)


class NarrativeOut(_Strict):
    rationale: str
    next_steps: list[str] = Field(default_factory=list)
    insights: list[InsightOut] = Field(default_factory=list)


_env = Environment(
    loader=PackageLoader("infervolt.llm", "prompts"),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(name: str, context: dict[str, Any]) -> str:
    context_json = json.dumps(context, indent=1, sort_keys=True, default=str)
    return _env.get_template(f"{name}.j2").render(context_json=context_json)


def extract_context(user: str) -> dict[str, Any]:
    start, end = user.index("<context>") + len("<context>"), user.rindex("</context>")
    data: dict[str, Any] = json.loads(user[start:end])
    return data


def prompts_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(files("infervolt.llm.prompts").iterdir(), key=lambda p: p.name):
        if p.name.endswith(".j2"):
            h.update(p.read_bytes())
    return h.hexdigest()[:12]
