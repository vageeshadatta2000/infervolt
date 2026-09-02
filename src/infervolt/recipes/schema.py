"""Recipe document: engine-native serve args plus infervolt's evidence and provenance block.

Top-level sections (model, hardware, engine, workload, slo, serve, baseline, result) are
engine-agnostic; everything infervolt-specific lives under `infervolt:` so the document stays
readable by tools that only understand the plain serve block.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from infervolt.core.types import Bottleneck, Evidence, KnobValue


class RecipeModel(BaseModel):
    id: str
    revision: str = ""
    params_b: float
    arch: str = ""
    moe: bool = False


class RecipeHardware(BaseModel):
    gpu: str
    count: int = 1
    driver: str = ""
    topology: str = "single"
    provider: str = ""


class RecipeEngine(BaseModel):
    name: str
    version: str = ""
    image: str = ""
    commit: str = ""


class RecipeDist(BaseModel):
    p50: int
    p99: int


class RecipeWorkload(BaseModel):
    name: str
    isl: RecipeDist
    osl: RecipeDist
    prefix_share: float = 0.0
    load: dict[str, Any] = Field(default_factory=dict)


class RecipeSLO(BaseModel):
    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    percentile: float = 0.9
    goodput_target: float = 0.9


class RecipeServe(BaseModel):
    args: dict[str, KnobValue]
    env: dict[str, str] = Field(default_factory=dict)
    command: str = ""


LOAD_POINT_DESC = (
    "Concurrency the metrics were measured at. Baseline and tuned configs are each "
    "reported at their own best load point, which is usually not the same number: a "
    "config that holds more sequences serves its peak goodput further right. Without it "
    "the summary table reads as two measurements of one operating point."
)


class RecipeMeasured(BaseModel):
    serve_args: dict[str, KnobValue] = Field(default_factory=dict)
    metrics: dict[str, float]
    load_point: int | None = Field(default=None, description=LOAD_POINT_DESC)


class RecipeQuality(BaseModel):
    guard: str
    tasks: list[str]
    recovery: float


class RecipeResult(BaseModel):
    metrics: dict[str, float]
    load_point: int | None = Field(default=None, description=LOAD_POINT_DESC)
    repeats: int
    improvement: dict[str, str] = Field(default_factory=dict)
    quality: RecipeQuality | None = None


class RecipeFinding(BaseModel):
    rule: str
    score: float
    evidence: list[Evidence]


class RecipeDiagnosis(BaseModel):
    primary: Bottleneck
    confidence: float
    findings: list[RecipeFinding]
    caveats: list[str] = Field(
        default_factory=list,
        description="What the diagnosis is not sure of -- a measurement artifact, a load "
        "point that never saturated, a ranking that fell back to rule order because the "
        "model was unavailable. The reader of a recipe is entitled to the same doubts the "
        "run had.",
    )


class RecipeSearch(BaseModel):
    trials: int
    infeasible: int
    subspace: list[str]
    optimizer: str
    seed: int


class RecipeProvenance(BaseModel):
    tool_version: str
    llm: str
    prompts_sha: str = ""
    created: str


class RecipeInfervolt(BaseModel):
    run_id: str
    diagnosis: RecipeDiagnosis
    rationale: str
    search: RecipeSearch
    trials_to_target: int | None = None
    warm_start: dict[str, list[str]] = Field(default_factory=dict)
    next_steps: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    provenance: RecipeProvenance


class Recipe(BaseModel):
    schema_version: int = 1
    model: RecipeModel
    hardware: RecipeHardware
    engine: RecipeEngine
    workload: RecipeWorkload
    slo: RecipeSLO
    serve: RecipeServe
    baseline: RecipeMeasured
    result: RecipeResult
    infervolt: RecipeInfervolt


def recipe_json_schema() -> dict[str, Any]:
    return Recipe.model_json_schema()
