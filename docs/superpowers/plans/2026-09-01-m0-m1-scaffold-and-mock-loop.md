# infervolt M0+M1 Implementation Plan (scaffold + mock-engine loop end to end)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A published-quality Python repo where `infervolt optimize --engine mock ...` runs the full loop (baseline sweep → rule-based diagnosis → LLM ranking/planning → Optuna search with crash handling → interleaved verify → recipe.yaml + report.md) in under 60 s on a CPU-only GitHub runner, and correctly names the injected bottleneck in four mock scenarios.

**Architecture:** `src/infervolt` package. Deterministic rules produce evidence; an `LLMClient` (Fake for CI, Anthropic and OpenAI-compatible for real use) ranks findings, picks knob sub-spaces and priors, and writes the recipe narrative. Optuna TPE searches inside the sub-space; OOM tightens bounds. A roofline-based mock engine makes everything testable without a GPU. SQLite ledger + per-run artifact dir.

**Tech Stack:** Python 3.11+, uv, pydantic v2, typer, optuna, numpy, pyyaml, jinja2, anthropic, openai, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-01-infervolt-design.md`. Deliberate M1 simplifications vs the spec (all noted in ROADMAP.md in Task 18): interfaces are synchronous (async arrives with the HTTP load generator in M2); artifacts are JSONL not Parquet (Parquet + DuckDB arrive with `memory stats` in M4); `--resume` and warm-start memory are M4.

---

## File structure

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.gitignore`, `LICENSE`, `NOTICE`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `ROADMAP.md`, `.github/` | Repo hygiene, CI |
| `src/infervolt/__init__.py` | `__version__` |
| `src/infervolt/config.py` | `Settings` (pydantic-settings, `INFERVOLT_` prefix): home dir, LLM keys, default model ids |
| `src/infervolt/core/types.py` | Every shared pydantic model. Single import point. |
| `src/infervolt/hardware/profiles.py` | Named `HardwareProfile`s (a100-80, h100-80, rtx4090-24, l4-24, m3-8) |
| `src/infervolt/hardware/roofline.py` | Roofline math: weight bytes, KV bytes/token, decode step floor, prefill floor, KV capacity |
| `src/infervolt/models/catalog.py` | Named `ModelInfo`s for mock models |
| `src/infervolt/workloads/presets.py` | Workload presets + `parse_slo("ttft=500ms,itl=30ms")` |
| `src/infervolt/loadgen/base.py` | `LoadGenerator` protocol |
| `src/infervolt/loadgen/analysis.py` | `compute_metrics(LoadResult, SLO, HardwareProfile) -> Metrics` |
| `src/infervolt/engines/base.py` | `EngineAdapter` protocol, `ServerHandle`, `ExitInfo`, `LaunchError`, `classify_log` |
| `src/infervolt/engines/registry.py` | `get_adapter(name)` via entry points |
| `src/infervolt/engines/mock/model.py` | `PerfModel`: roofline-based analytic simulator |
| `src/infervolt/engines/mock/adapter.py` | `MockAdapter` + `SimLoadGenerator` + mock knob space |
| `src/infervolt/engines/mock/scenarios.py` | 4 injected-bottleneck scenarios used by tests and README |
| `src/infervolt/runner/trial.py` | `run_load_point`, `run_sweep`, `run_candidate` (launch → load → scrape → stop → classify) |
| `src/infervolt/diagnose/rules.py` | Rules R0–R6 → `Finding`s |
| `src/infervolt/diagnose/ranker.py` | LLM ranking → `Diagnosis` (validated, fallback to rule order) |
| `src/infervolt/llm/base.py` | `LLMClient` protocol + structured-output schemas |
| `src/infervolt/llm/prompts/{rank,plan,emit}.j2` | Prompt templates (instructions + JSON context block) |
| `src/infervolt/llm/fake.py` | `FakeLLMClient`: deterministic, parses the JSON context |
| `src/infervolt/llm/anthropic_client.py` | `AnthropicClient` via `messages.parse` |
| `src/infervolt/llm/openai_compat.py` | `OpenAICompatClient` via `response_format=json_schema` + repair |
| `src/infervolt/llm/replay.py` | `ReplayLLMClient` cassette wrapper |
| `src/infervolt/llm/factory.py` | `make_llm(name, settings)` |
| `src/infervolt/search/space.py` | `Bounds` (tightened on OOM), `clamp`, `config_key`, novelty |
| `src/infervolt/search/optuna_search.py` | `run_search(...)` Optuna TPE + priors + ASHA-style stage-1 prune |
| `src/infervolt/verify/quality.py` | `QualityGuard` protocol + `MockQualityGuard` |
| `src/infervolt/verify/verify.py` | Interleaved 3x verify with paired-t CI |
| `src/infervolt/store/ledger.py` | SQLite `Ledger` + run artifact dirs |
| `src/infervolt/recipes/schema.py` | `Recipe` pydantic model + JSON-schema export |
| `src/infervolt/recipes/emit.py` | `write_recipe`, `render_report` |
| `src/infervolt/recipes/templates/report.md.j2` | Report template |
| `src/infervolt/agent/budget.py` | `BudgetTracker` |
| `src/infervolt/agent/planner.py` | `Planner` state machine |
| `src/infervolt/cli/main.py` | Typer app: `optimize`, `recipe validate`, `report`, `--version` |
| `tests/` | Mirrors `src/` layout; `tests/integration/test_mock_loop.py` is the M1 acceptance test |

Conventions used in every task:
- Run commands from `/Users/vageeshadattaganapaneni/infervolt`.
- `uv run pytest -q` runs all tests; `uv run ruff check . && uv run ruff format --check . && uv run mypy src` is the lint gate.
- Commit after each task with a conventional-commit message. Do not use `git add -A`; add the files named in the task.

---

### Task 1: Project scaffold (pyproject, tooling, CLI skeleton)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`, `LICENSE`, `NOTICE`, `README.md`, `src/infervolt/__init__.py`, `src/infervolt/py.typed`, `src/infervolt/cli/__init__.py`, `src/infervolt/cli/main.py`, `src/infervolt/config.py`, `tests/__init__.py`, `tests/test_cli.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "infervolt"
version = "0.1.0.dev0"
description = "Self-learning agent that measures, diagnoses, and fixes LLM inference bottlenecks and emits verified recipes."
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11"
authors = [{ name = "infervolt contributors" }]
keywords = ["llm", "inference", "vllm", "llama.cpp", "optimization", "benchmark", "agent"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "typer>=0.12",
  "optuna>=3.6",
  "numpy>=1.26",
  "pyyaml>=6.0",
  "jinja2>=3.1",
  "anthropic>=0.40",
  "openai>=1.40",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "ruff>=0.5",
  "mypy>=1.10",
  "types-PyYAML",
  "pre-commit>=3.7",
]

[project.scripts]
infervolt = "infervolt.cli.main:app"

[project.entry-points."infervolt.engines"]
mock = "infervolt.engines.mock.adapter:MockAdapter"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/infervolt"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]
ignore = ["N818"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: end-to-end loop tests", "learning: memory learning tests"]
addopts = "-ra"
```

- [ ] **Step 2: Write `.gitignore`, `.pre-commit-config.yaml`, `LICENSE`, `NOTICE`**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
dist/
build/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.env
.infervolt/
runs/
*.sqlite
.DS_Store
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
```

`LICENSE`: the full Apache License 2.0 text. Fetch it verbatim:
```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

`NOTICE`:
```
infervolt
Copyright 2026 infervolt contributors

This product includes software developed by the infervolt contributors
(https://github.com/infervolt/infervolt).
```

- [ ] **Step 3: Write package init, config, CLI skeleton**

`src/infervolt/__init__.py`:
```python
"""infervolt: measure -> diagnose -> fix -> verify -> recipe -> remember."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("infervolt")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
```

`src/infervolt/py.typed`: empty file.

`src/infervolt/config.py`:
```python
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
```

`src/infervolt/cli/__init__.py`: empty.

`src/infervolt/cli/main.py`:
```python
"""infervolt command-line interface."""

from __future__ import annotations

import typer

from infervolt import __version__

app = typer.Typer(help="Measure, diagnose, fix, verify, and remember LLM inference optimizations.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"infervolt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """infervolt CLI."""


if __name__ == "__main__":  # pragma: no cover
    app()
```

- [ ] **Step 4: Write the failing CLI test**

`tests/__init__.py`: empty.

`tests/test_cli.py`:
```python
from typer.testing import CliRunner

from infervolt.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("infervolt ")
```

- [ ] **Step 5: Install and run the test**

```bash
uv sync --extra dev && uv run pytest tests/test_cli.py -q
```
Expected: `1 passed`.

- [ ] **Step 6: Lint gate and README stub**

`README.md` (stub; Task 18 completes it):
```markdown
# infervolt

Measure → diagnose the bottleneck → targeted fix → re-measure → verify → emit a recipe → remember.

infervolt is an open-source agent that optimizes inference for open-source LLMs on your hardware and
explains *why* the winning configuration wins. Status: pre-alpha, mock engine only.

```bash
uv sync --extra dev
uv run infervolt --version
```
```

Run: `uv run ruff check . && uv run ruff format . && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .pre-commit-config.yaml LICENSE NOTICE README.md src tests
git commit -m "chore: scaffold infervolt package, CLI, tooling"
```

---

### Task 2: Core types

**Files:**
- Create: `src/infervolt/core/__init__.py`, `src/infervolt/core/types.py`, `tests/core/__init__.py`, `tests/core/test_types.py`

- [ ] **Step 1: Write the failing tests**

`tests/core/__init__.py`: empty.

`tests/core/test_types.py`:
```python
from infervolt.core.types import EngineConfig, Knob, KnobSpace, RequestRecord


def _space() -> KnobSpace:
    return KnobSpace(
        knobs=[
            Knob(name="a", kind="int", groups=["kv"], default=8, low=1, high=64, log=True),
            Knob(name="b", kind="bool", groups=["sched"], default=False),
            Knob(name="c", kind="cat", groups=["kv", "decode"], default="auto", choices=["auto", "fp8"]),
        ]
    )


def test_subspace_filters_by_group_membership() -> None:
    sub = _space().subspace(["decode"])
    assert sub.names() == ["c"]
    assert _space().subspace(["kv"]).names() == ["a", "c"]


def test_defaults_and_with_knobs() -> None:
    cfg = EngineConfig(engine="mock", knobs=_space().defaults())
    assert cfg.knobs == {"a": 8, "b": False, "c": "auto"}
    new = cfg.with_knobs(c="fp8")
    assert new.knobs["c"] == "fp8" and cfg.knobs["c"] == "auto"


def test_config_key_is_order_independent() -> None:
    k1 = EngineConfig(engine="mock", knobs={"x": 1, "y": 2}).key()
    k2 = EngineConfig(engine="mock", knobs={"y": 2, "x": 1}).key()
    assert k1 == k2


def test_request_record_e2e() -> None:
    r = RequestRecord(ttft_s=0.5, itl_s=[0.01, 0.02], output_tokens=2)
    assert abs(r.e2e_s - 0.53) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'infervolt.core'`.

- [ ] **Step 3: Write `core/types.py`**

`src/infervolt/core/__init__.py`: empty.

`src/infervolt/core/types.py`:
```python
"""All shared data models.

Everything else imports from here; this module imports nothing internal.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

KnobValue = int | float | str | bool

# ---------------------------------------------------------------- workload / SLO / hardware / model


class TokenDist(BaseModel):
    p50: int
    p99: int


class LoadSpec(BaseModel):
    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 16, 64, 128])


class Workload(BaseModel):
    name: str
    isl: TokenDist
    osl: TokenDist
    prefix_share: float = 0.0
    num_requests: int = 64
    seed: int = 0
    load: LoadSpec = Field(default_factory=LoadSpec)


class SLO(BaseModel):
    """Latency targets plus how strictly they must hold.

    ``percentile`` applies to the per-request inter-token-latency distribution *only*:
    a request passes ``itl_ms`` when the ``percentile`` quantile of its own ITLs is
    under the target (0.9 means that request's p90 ITL). ``ttft_ms`` and ``e2e_ms``
    have no distribution to summarise -- each request has one of each -- so they are
    compared per request, directly. ``goodput_target`` is a different axis again: the
    fraction of requests that must meet the SLO for the run to count as "SLO met".
    """

    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    percentile: float = Field(default=0.9, gt=0, le=1)
    goodput_target: float = Field(default=0.9, gt=0, le=1)


class HardwareProfile(BaseModel):
    name: str
    gpu: str
    count: int = 1
    mem_gb: float = Field(description="Device memory in GiB, as reported by NVML.")
    hbm_bw_gbs: float
    peak_tflops: float
    compute_capability: float
    interconnect: Literal["single", "nvlink", "pcie"] = "single"
    usd_per_hour: float = Field(
        default=0.0,
        description="Rental price of a single GPU, in USD per hour; multiply by 'count' "
        "for the price of the whole node.",
    )


class ModelInfo(BaseModel):
    id: str
    arch: str = ""
    params_b: float
    active_params_b: float | None = None
    num_layers: int
    hidden: int
    num_kv_heads: int
    head_dim: int
    weight_bits: int = 16
    max_pos: int = 32768
    moe: bool = False


# ---------------------------------------------------------------- knobs / config


class Knob(BaseModel):
    name: str
    kind: Literal["int", "float", "cat", "bool"]
    groups: list[str]
    default: KnobValue
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    choices: list[KnobValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> Knob:
        if self.kind == "cat":
            if not self.choices:
                raise ValueError(f"knob {self.name!r}: kind 'cat' requires a non-empty 'choices'")
            if self.default not in self.choices:
                raise ValueError(
                    f"knob {self.name!r}: default {self.default!r} is not in choices "
                    f"{self.choices!r}"
                )
        elif self.kind in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"knob {self.name!r}: kind {self.kind!r} requires both 'low' and 'high'"
                )
            if self.low > self.high:
                raise ValueError(
                    f"knob {self.name!r}: low {self.low!r} must be <= high {self.high!r}"
                )
            if isinstance(self.default, (str, bool)):
                raise ValueError(
                    f"knob {self.name!r}: kind {self.kind!r} requires a numeric default, "
                    f"got {self.default!r}"
                )
            if not self.low <= self.default <= self.high:
                raise ValueError(
                    f"knob {self.name!r}: default {self.default!r} is outside "
                    f"[{self.low!r}, {self.high!r}]"
                )
        elif self.kind == "bool" and not isinstance(self.default, bool):
            raise ValueError(
                f"knob {self.name!r}: kind 'bool' requires a bool default, got {self.default!r}"
            )
        return self


class KnobSpace(BaseModel):
    knobs: list[Knob]

    def names(self) -> list[str]:
        return [k.name for k in self.knobs]

    def get(self, name: str) -> Knob:
        for k in self.knobs:
            if k.name == name:
                return k
        raise KeyError(name)

    def groups(self) -> list[str]:
        seen: list[str] = []
        for k in self.knobs:
            for g in k.groups:
                if g not in seen:
                    seen.append(g)
        return seen

    def subspace(self, groups: list[str]) -> KnobSpace:
        wanted = set(groups)
        return KnobSpace(knobs=[k for k in self.knobs if wanted & set(k.groups)])

    def defaults(self) -> dict[str, KnobValue]:
        return {k.name: k.default for k in self.knobs}


class EngineConfig(BaseModel):
    engine: str
    knobs: dict[str, KnobValue] = Field(default_factory=dict)

    def with_knobs(self, **updates: KnobValue) -> EngineConfig:
        return EngineConfig(engine=self.engine, knobs={**self.knobs, **updates})

    def key(self) -> str:
        return json.dumps({"engine": self.engine, "knobs": self.knobs}, sort_keys=True)


# ---------------------------------------------------------------- measurement


class RequestRecord(BaseModel):
    """One request's timing.

    Invariant: ``len(itl_s) == output_tokens`` -- one inter-token latency per generated
    token. ``ttft_s`` covers the first token (time from request start to its arrival),
    so ``e2e_s`` is ``ttft_s`` plus the whole of ``itl_s``.
    """

    ttft_s: float
    itl_s: list[float]
    output_tokens: int
    ok: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def e2e_s(self) -> float:
        return self.ttft_s + sum(self.itl_s)


class ClientHealth(BaseModel):
    worker_cpu: float = 0.0
    loop_lag_p99_ms: float = 0.0
    error_rate: float = 0.0


class LoadResult(BaseModel):
    concurrency: int
    duration_s: float = Field(
        gt=0, description="Wall-clock seconds the load phase ran; every rate divides by it."
    )
    requests: list[RequestRecord]
    health: ClientHealth = Field(default_factory=ClientHealth)


class Metrics(BaseModel):
    # ``usd_per_m_tokens`` is infinite for a run with no output tokens; the default JSON
    # serialiser turns inf into null, which fails to validate back, so encode it as a string.
    model_config = ConfigDict(ser_json_inf_nan="strings")

    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    e2e_p50_ms: float
    e2e_p90_ms: float
    output_tps: float
    req_per_s: float
    goodput_rps: float
    goodput_frac: float
    error_rate: float
    tokens_per_s_per_gpu: float
    usd_per_m_tokens: float = Field(
        description="USD per million *output* tokens (input tokens are not counted); "
        "infinite when the run produced no output tokens."
    )


class Evidence(BaseModel):
    source: str
    key: str
    value: float
    unit: str = ""
    note: str = ""


class Observation(BaseModel):
    load_point: int
    config: EngineConfig
    metrics: Metrics
    engine: dict[str, float] = Field(default_factory=dict)
    gpu: dict[str, float] = Field(default_factory=dict)
    valid: bool = True
    invalid_reason: str = ""


# ---------------------------------------------------------------- diagnosis

Bottleneck = Literal[
    "under_loaded",
    "kv_capacity",
    "decode_bandwidth",
    "prefill_compute",
    "scheduler_cpu",
    "communication",
    "client_artifact",
]

BOTTLENECK_PRIORITY: dict[Bottleneck, int] = {
    "client_artifact": 0,
    "kv_capacity": 1,
    "prefill_compute": 2,
    "decode_bandwidth": 3,
    "scheduler_cpu": 4,
    "communication": 5,
    "under_loaded": 6,
}


class Finding(BaseModel):
    rule_id: str
    bottleneck: Bottleneck
    score: float
    evidence: list[Evidence]
    subspaces: list[str]
    summary: str


class Diagnosis(BaseModel):
    primary: Bottleneck
    ranked: list[Finding]
    rationale: str
    confidence: float
    subspaces: list[str]
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- search / trials

Origin = Literal["baseline", "warm_start", "llm_prior", "tpe", "manual"]


class Candidate(BaseModel):
    id: str
    config: EngineConfig
    origin: Origin
    hypothesis: str = ""
    parent_id: str | None = None


class SearchPlan(BaseModel):
    subspaces: list[str]
    fixed: dict[str, KnobValue] = Field(default_factory=dict)
    priors: list[Candidate] = Field(default_factory=list)
    max_trials: int
    rationale: str = ""


class QualityScore(BaseModel):
    guard: str
    tasks: list[str]
    recovery: float


class Result(BaseModel):
    observations: list[Observation]
    objective: float
    feasible: bool
    slo_met: bool
    best_load_point: int
    quality: QualityScore | None = None


TrialStatus = Literal[
    "pending", "running", "ok", "infeasible_oom", "crash", "timeout", "pruned", "rejected"
]
CrashKind = Literal["none", "oom", "startup", "runtime", "timeout"]


class Trial(BaseModel):
    id: str
    run_id: str
    index: int
    candidate: Candidate
    status: TrialStatus = "pending"
    stage: int = 0
    result: Result | None = None
    crash_kind: CrashKind = "none"
    log_tail: str = ""
    cost_usd: float = 0.0
    started: float | None = None
    ended: float | None = None


class Budget(BaseModel):
    max_trials: int = 12
    max_wall_s: float = 3600.0
    max_usd: float = 0.0  # 0 means unlimited


class OptimizeSpec(BaseModel):
    engine: str
    model: str
    hardware: str = "auto"
    workload: str = "chat-4k-512"
    slo: SLO = Field(default_factory=SLO)
    budget: Budget = Field(default_factory=Budget)
    baseline: dict[str, KnobValue] = Field(default_factory=dict)
    seed: int = 7
    llm: str = "fake"
    run_id: str | None = None


class RunContext(BaseModel):
    run_id: str
    run_dir: str
    hw: HardwareProfile
    model: ModelInfo
    workload: Workload
    slo: SLO
    seed: int


RunState = Literal[
    "prepare", "baseline", "diagnose", "plan", "search", "verify", "emit", "learn", "done", "failed"
]


class RunOutcome(BaseModel):
    run_id: str
    state: RunState
    baseline_trial_id: str | None = None
    best_trial_id: str | None = None
    diagnosis: Diagnosis | None = None
    recipe_path: str | None = None
    report_path: str | None = None
    trials_to_target: int | None = None
    improvement_pct: float | None = None
    accepted: bool = False
    message: str = ""
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/core -q && uv run ruff check . && uv run mypy src`
Expected: `4 passed`, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/core tests/core
git commit -m "feat(core): shared pydantic types for workloads, knobs, trials, diagnosis"
```

---

### Task 3: Hardware profiles and roofline math

**Files:**
- Create: `src/infervolt/hardware/__init__.py`, `src/infervolt/hardware/profiles.py`, `src/infervolt/hardware/roofline.py`, `tests/hardware/__init__.py`, `tests/hardware/test_roofline.py`

- [ ] **Step 1: Write the failing tests**

`tests/hardware/__init__.py`: empty.

`tests/hardware/test_roofline.py`:
```python
import pytest

from infervolt.core.types import ModelInfo
from infervolt.hardware import roofline
from infervolt.hardware.profiles import get_profile


def _qwen8b() -> ModelInfo:
    return ModelInfo(
        id="mock/qwen3-8b", params_b=8.2, num_layers=36, hidden=4096, num_kv_heads=8, head_dim=128
    )


def test_weight_and_kv_bytes() -> None:
    m = _qwen8b()
    assert roofline.weight_bytes(m) == pytest.approx(16.4e9)
    assert roofline.kv_bytes_per_token(m, kv_dtype_bytes=2) == 2 * 36 * 8 * 128 * 2


def test_decode_is_memory_bound_at_batch_one() -> None:
    hw, m = get_profile("a100-80"), _qwen8b()
    step = roofline.decode_step_floor_s(hw, m, batch=1, ctx_tokens=1024)
    mem_only = roofline.weight_bytes(m) / (hw.hbm_bw_gbs * 1e9)
    assert step == pytest.approx(mem_only, rel=0.05)


def test_kv_capacity_shrinks_with_lower_util_and_grows_with_fp8() -> None:
    hw, m = get_profile("rtx4090-24"), _qwen8b()
    fp16 = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.9, kv_dtype_bytes=2)
    fp8 = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.9, kv_dtype_bytes=1)
    low = roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.8, kv_dtype_bytes=2)
    assert fp8 == pytest.approx(2 * fp16)
    assert low < fp16
    assert roofline.kv_capacity_tokens(hw, m, gpu_mem_util=0.5, kv_dtype_bytes=2) == 0.0


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/hardware -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write profiles and roofline**

`src/infervolt/hardware/__init__.py`: empty.

`src/infervolt/hardware/profiles.py`:
```python
"""Named hardware profiles.

Peak numbers are dense BF16 TFLOPs and HBM bandwidth from vendor specs.
"""

from __future__ import annotations

from infervolt.core.types import HardwareProfile

PROFILES: dict[str, HardwareProfile] = {
    "a100-80": HardwareProfile(
        name="a100-80",
        gpu="NVIDIA A100-SXM4-80GB",
        mem_gb=80,
        hbm_bw_gbs=2039,
        peak_tflops=312,
        compute_capability=8.0,
        usd_per_hour=1.5,
    ),
    "h100-80": HardwareProfile(
        name="h100-80",
        gpu="NVIDIA H100 80GB HBM3",
        mem_gb=80,
        hbm_bw_gbs=3350,
        peak_tflops=989,
        compute_capability=9.0,
        usd_per_hour=3.0,
    ),
    "rtx4090-24": HardwareProfile(
        name="rtx4090-24",
        gpu="NVIDIA GeForce RTX 4090",
        mem_gb=24,
        hbm_bw_gbs=1008,
        peak_tflops=165,
        compute_capability=8.9,
        usd_per_hour=0.5,
    ),
    "l4-24": HardwareProfile(
        name="l4-24",
        gpu="NVIDIA L4",
        mem_gb=24,
        hbm_bw_gbs=300,
        peak_tflops=121,
        compute_capability=8.9,
        usd_per_hour=0.6,
    ),
    "m3-8": HardwareProfile(
        name="m3-8",
        gpu="Apple M3 (10-core GPU)",
        mem_gb=8,
        hbm_bw_gbs=100,
        peak_tflops=3.5,
        compute_capability=0.0,
        usd_per_hour=0.0,
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Return a private copy, so callers can adapt a profile without editing the registry."""
    try:
        return PROFILES[name].model_copy(deep=True)
    except KeyError as e:
        raise KeyError(f"unknown hardware profile {name!r}; known: {sorted(PROFILES)}") from e
```

`src/infervolt/hardware/roofline.py`:
```python
"""Roofline estimates for LLM inference. All functions are pure and unit-tested.

Decode is memory-bound: each step streams every weight byte plus the KV cache of every
sequence in the batch through HBM. Prefill mixes a compute term (linear layers, quadratic
attention) with the same weight stream, and takes whichever dominates.

Two conventions hold throughout:

* **Memory is GiB, not GB.** ``HardwareProfile.mem_gb`` is what NVML and the engines
  report -- binary gibibytes -- so byte counts go through :func:`mem_bytes` and
  :func:`reserve_bytes` (``* 2**30``). Model and bandwidth figures stay decimal
  (``params_b * 1e9``, ``hbm_bw_gbs * 1e9``) because that is how they are specified.
* **Every figure is per GPU, i.e. per tensor-parallel shard.** Nothing here knows about
  TP: a caller modelling TP=4 must pass the per-shard model dimensions (divide layers'
  widths, KV heads and parameters itself) before calling.
"""

from __future__ import annotations

from infervolt.core.types import HardwareProfile, ModelInfo

ACTIVATION_RESERVE_GIB = 2.0


def mem_bytes(hw: HardwareProfile) -> float:
    """Total device memory in bytes. ``mem_gb`` is GiB, as NVML reports it."""
    return hw.mem_gb * 2**30


def reserve_bytes() -> float:
    """Bytes held back for activations and fragmentation, i.e. not available for KV."""
    return ACTIVATION_RESERVE_GIB * 2**30


def weight_bytes(m: ModelInfo) -> float:
    return m.params_b * 1e9 * m.weight_bits / 8


def active_params(m: ModelInfo) -> float:
    """Parameters touched per token: the MoE active count when declared, else all of them."""
    b = m.active_params_b if m.active_params_b is not None else m.params_b
    return b * 1e9


def kv_bytes_per_token(m: ModelInfo, kv_dtype_bytes: int = 2) -> float:
    if kv_dtype_bytes <= 0:
        raise ValueError(f"kv_dtype_bytes must be positive, got {kv_dtype_bytes!r}")
    return 2.0 * m.num_layers * m.num_kv_heads * m.head_dim * kv_dtype_bytes


def decode_step_floor_s(
    hw: HardwareProfile, m: ModelInfo, batch: int, ctx_tokens: int, kv_dtype_bytes: int = 2
) -> float:
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch!r}")
    streamed = weight_bytes(m) + batch * kv_bytes_per_token(m, kv_dtype_bytes) * ctx_tokens
    t_mem = streamed / (hw.hbm_bw_gbs * 1e9)
    t_compute = 2.0 * active_params(m) * batch / (hw.peak_tflops * 1e12)
    return max(t_mem, t_compute)


def prefill_floor_s(hw: HardwareProfile, m: ModelInfo, tokens: int) -> float:
    """Lower bound on the time to prefill ``tokens`` in one forward pass.

    Three terms, two of which race:

    * ``linear`` -- ``2 * active_params * tokens`` FLOPs: one multiply-add per active
      parameter per token through the dense/expert projections.
    * ``attn`` -- ``2 * tokens**2 * hidden * num_layers`` FLOPs: the quadratic
      score-and-weighted-sum pair (two matmuls, 2 FLOPs each) that the linear term
      ignores, halved because inference attention is causal -- only the lower triangle
      of the tokens x tokens score matrix is computed, so the full ``4 * L**2`` figure
      overstates the work by 2x. Negligible at short context, dominant at long context.
    * the weight stream -- ``weight_bytes / hbm_bw``: even a one-token prefill must read
      every weight out of HBM once.

    The compute terms share the same SMs, so they add; the result is the larger of that
    sum and the weight stream, since the two overlap.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens!r}")
    linear = 2.0 * active_params(m) * tokens
    attn = 2.0 * tokens**2 * m.hidden * m.num_layers
    t_compute = (linear + attn) / (hw.peak_tflops * 1e12)
    t_mem = weight_bytes(m) / (hw.hbm_bw_gbs * 1e9)
    return max(t_compute, t_mem)


def kv_capacity_tokens(
    hw: HardwareProfile, m: ModelInfo, gpu_mem_util: float, kv_dtype_bytes: int = 2
) -> float:
    """KV-cache tokens that fit once weights and the activation reserve are subtracted."""
    if not 0 < gpu_mem_util <= 1:
        raise ValueError(f"gpu_mem_util must be in (0, 1], got {gpu_mem_util!r}")
    avail = mem_bytes(hw) * gpu_mem_util - weight_bytes(m) - reserve_bytes()
    return max(0.0, avail / kv_bytes_per_token(m, kv_dtype_bytes))
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/hardware -q && uv run ruff check . && uv run mypy src`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/hardware tests/hardware
git commit -m "feat(hardware): named GPU profiles and roofline estimates"
```

---

### Task 4: Model catalog, workload presets, SLO parsing

**Files:**
- Create: `src/infervolt/models/__init__.py`, `src/infervolt/models/catalog.py`, `src/infervolt/workloads/__init__.py`, `src/infervolt/workloads/presets.py`, `tests/test_presets.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_presets.py`:
```python
import pytest

from infervolt.models.catalog import get_model_info
from infervolt.workloads.presets import get_workload, parse_slo


def test_model_catalog_has_mock_models() -> None:
    m = get_model_info("mock/qwen3-8b")
    assert m.params_b == 8.2 and m.num_kv_heads == 8


def test_workload_presets() -> None:
    w = get_workload("chat-4k-512")
    assert w.isl.p50 == 4096 and w.osl.p50 == 512
    assert get_workload("agentic-prefix-16k-512").prefix_share > 0.5


def test_parse_slo() -> None:
    slo = parse_slo("ttft=500ms,itl=30ms")
    assert slo.ttft_ms == 500 and slo.itl_ms == 30 and slo.e2e_ms is None
    assert parse_slo("e2e=2s,p=0.95").e2e_ms == 2000
    assert parse_slo("").ttft_ms is None


def test_parse_slo_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        parse_slo("latency=1ms")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_presets.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write catalog and presets**

`src/infervolt/models/__init__.py`, `src/infervolt/workloads/__init__.py`: empty.

`src/infervolt/models/catalog.py`:
```python
"""Static model catalog. M3 adds resolution of real Hugging Face ids from config.json."""

from __future__ import annotations

from infervolt.core.types import ModelInfo

MODELS: dict[str, ModelInfo] = {
    "mock/qwen3-0.6b": ModelInfo(
        id="mock/qwen3-0.6b",
        arch="qwen3",
        params_b=0.6,
        num_layers=28,
        hidden=1024,
        num_kv_heads=8,
        head_dim=128,
        max_pos=40960,
    ),
    "mock/qwen3-8b": ModelInfo(
        id="mock/qwen3-8b",
        arch="qwen3",
        params_b=8.2,
        num_layers=36,
        hidden=4096,
        num_kv_heads=8,
        head_dim=128,
        max_pos=40960,
    ),
    "mock/llama-70b": ModelInfo(
        id="mock/llama-70b",
        arch="llama",
        params_b=70.0,
        num_layers=80,
        hidden=8192,
        num_kv_heads=8,
        head_dim=128,
        max_pos=131072,
    ),
}


def get_model_info(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id].model_copy(deep=True)
    except KeyError as e:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODELS)}") from e
```

`src/infervolt/workloads/presets.py`:
```python
"""Workload presets and SLO string parsing."""

from __future__ import annotations

from infervolt.core.types import SLO, LoadSpec, TokenDist, Workload

PRESETS: dict[str, Workload] = {
    "chat-4k-512": Workload(
        name="chat-4k-512",
        isl=TokenDist(p50=4096, p99=6000),
        osl=TokenDist(p50=512, p99=1024),
        prefix_share=0.1,
    ),
    "chat-1k-128": Workload(
        name="chat-1k-128",
        isl=TokenDist(p50=1024, p99=2048),
        osl=TokenDist(p50=128, p99=256),
        prefix_share=0.1,
    ),
    "chat-256-512": Workload(
        name="chat-256-512",
        isl=TokenDist(p50=256, p99=512),
        osl=TokenDist(p50=512, p99=768),
        prefix_share=0.0,
    ),
    "rag-16k-64": Workload(
        name="rag-16k-64",
        isl=TokenDist(p50=16384, p99=20000),
        osl=TokenDist(p50=64, p99=128),
        prefix_share=0.0,
        load=LoadSpec(concurrency=[1, 4, 16, 64]),
    ),
    "agentic-prefix-16k-512": Workload(
        name="agentic-prefix-16k-512",
        isl=TokenDist(p50=16384, p99=24000),
        osl=TokenDist(p50=512, p99=1024),
        prefix_share=0.6,
    ),
}


def get_workload(name: str) -> Workload:
    try:
        return PRESETS[name].model_copy(deep=True)
    except KeyError as e:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(PRESETS)}") from e


_UNITS = {"ms": 1.0, "s": 1000.0}
# Longest suffix first, so "500ms" matches "ms" and never the "s" inside it.
_SUFFIXES = sorted(_UNITS, key=len, reverse=True)

_DURATION_KEYS = {"ttft": "ttft_ms", "itl": "itl_ms", "e2e": "e2e_ms"}
_FRACTION_KEYS = {"p": "percentile", "g": "goodput_target"}


def _ms(text: str) -> float:
    """Convert a duration literal ('500ms', '2 s', '250') to milliseconds."""
    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * _UNITS[suffix]
    return float(text)


def parse_slo(text: str) -> SLO:
    """Parse 'ttft=500ms,itl=30ms,e2e=2s,p=0.9,g=0.9' into an SLO. Empty string -> no targets.

    Whitespace around keys and values is ignored. Durations must be non-negative;
    ``p`` and ``g`` are fractions in (0, 1], so a percentile written as ``p=95`` is
    rejected rather than silently accepted as an impossible target.
    """
    fields: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        raw_key, sep, raw_value = part.partition("=")
        key, value = raw_key.strip(), raw_value.strip()
        if key in _DURATION_KEYS:
            if not sep:
                raise ValueError(f"malformed SLO clause {part!r}")
            try:
                ms = _ms(value)
            except ValueError as e:
                raise ValueError(f"malformed SLO clause {part!r}") from e
            if ms < 0:
                raise ValueError(f"malformed SLO clause {part!r}: duration must be >= 0")
            fields[_DURATION_KEYS[key]] = ms
        elif key in _FRACTION_KEYS:
            if not sep:
                raise ValueError(f"malformed SLO clause {part!r}")
            try:
                frac = float(value)
            except ValueError as e:
                raise ValueError(f"malformed SLO clause {part!r}") from e
            if not 0 < frac <= 1:
                raise ValueError(
                    f"malformed SLO clause {part!r}: {key!r} is a fraction in (0, 1], got {frac!r}"
                )
            fields[_FRACTION_KEYS[key]] = frac
        else:
            raise ValueError(f"unknown SLO key {key!r}; use ttft, itl, e2e, p, g")
    return SLO(**fields)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_presets.py -q && uv run ruff check . && uv run mypy src`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/models src/infervolt/workloads tests/test_presets.py
git commit -m "feat: model catalog, workload presets, SLO parser"
```

---

### Task 5: Load-generator protocol and metrics analysis

**Files:**
- Create: `src/infervolt/loadgen/__init__.py`, `src/infervolt/loadgen/base.py`, `src/infervolt/loadgen/analysis.py`, `tests/loadgen/__init__.py`, `tests/loadgen/test_analysis.py`

- [ ] **Step 1: Write the failing tests**

`tests/loadgen/__init__.py`: empty.

`tests/loadgen/test_analysis.py`:
```python
import pytest

from infervolt.core.types import SLO, LoadResult, RequestRecord
from infervolt.hardware.profiles import get_profile
from infervolt.loadgen.analysis import compute_metrics


def _lr() -> LoadResult:
    fast = RequestRecord(ttft_s=0.1, itl_s=[0.01] * 10, output_tokens=10)
    slow = RequestRecord(ttft_s=1.0, itl_s=[0.05] * 10, output_tokens=10)
    bad = RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
    return LoadResult(concurrency=2, duration_s=10.0, requests=[fast, fast, slow, bad])


def test_goodput_counts_only_requests_meeting_every_slo() -> None:
    m = compute_metrics(_lr(), SLO(ttft_ms=500, itl_ms=30), get_profile("a100-80"))
    assert m.goodput_rps == pytest.approx(0.2)  # 2 fast requests / 10 s
    assert m.goodput_frac == pytest.approx(0.5)  # 2 of 4 submitted
    assert m.req_per_s == pytest.approx(0.3)
    assert m.error_rate == pytest.approx(0.25)
    assert m.output_tps == pytest.approx(3.0)


def test_no_slo_means_every_ok_request_is_good() -> None:
    m = compute_metrics(_lr(), SLO(), get_profile("a100-80"))
    assert m.goodput_rps == pytest.approx(m.req_per_s)


def test_cost_per_million_tokens_uses_hourly_price() -> None:
    hw = get_profile("a100-80")  # 1.5 $/h
    m = compute_metrics(_lr(), SLO(), hw)
    assert m.usd_per_m_tokens == pytest.approx(1.5 / 3600 / 3.0 * 1e6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/loadgen -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write loadgen base and analysis**

`src/infervolt/loadgen/__init__.py`: empty.

`src/infervolt/loadgen/base.py`:
```python
"""Load generator protocol.

M1 ships the mock simulator; M2 adds the multi-process HTTP generator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from infervolt.core.types import LoadResult, Workload


@runtime_checkable
class LoadGenerator(Protocol):
    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        """Drive `num_requests` requests at fixed closed-loop `concurrency` and return records."""
        ...
```

`src/infervolt/loadgen/analysis.py`:
```python
"""Turn per-request records into the Metrics the rules and the objective consume.

Percentile units differ between the two layers this module bridges: ``_pct`` and numpy
take a percentile in 0-100, while :attr:`SLO.percentile` is a fraction in 0-1. Every
crossing multiplies by 100 -- keep that conversion at the call site, not in ``_pct``.
"""

from __future__ import annotations

import math

import numpy as np

from infervolt.core.types import SLO, HardwareProfile, LoadResult, Metrics, RequestRecord


def _pct(values: list[float], p: float) -> float:
    """Percentile of ``values``. ``p`` is in 0-100 (numpy's convention), not 0-1."""
    return float(np.percentile(values, p)) if values else 0.0


def request_meets_slo(r: RequestRecord, slo: SLO) -> bool:
    """True when the request did productive work *and* met every configured target.

    Being productive -- succeeding and emitting at least one output token -- is a
    precondition, not a target: goodput measures served work, so a request that failed
    or returned nothing must never count, however fast it was.
    """
    if not r.ok or r.output_tokens <= 0:
        return False
    if slo.ttft_ms is not None and r.ttft_s * 1000 > slo.ttft_ms:
        return False
    if (
        slo.itl_ms is not None
        and r.itl_s
        and _pct(r.itl_s, slo.percentile * 100) * 1000 > slo.itl_ms
    ):
        return False
    return not (slo.e2e_ms is not None and r.e2e_s * 1000 > slo.e2e_ms)


def compute_metrics(lr: LoadResult, slo: SLO, hw: HardwareProfile) -> Metrics:
    """Summarise one load point.

    Latency percentiles are taken over successful requests only; rates divide by
    ``lr.duration_s`` (validated positive). ``usd_per_m_tokens`` prices the node --
    ``hw.usd_per_hour`` is per GPU, so it is multiplied by ``hw.count`` -- against
    *output* tokens alone, and is infinite when no output tokens were produced.
    """
    ok = [r for r in lr.requests if r.ok]
    total = len(lr.requests)
    good = sum(1 for r in ok if request_meets_slo(r, slo))
    dur = lr.duration_s
    ttft = [r.ttft_s * 1000 for r in ok]
    itl = [x * 1000 for r in ok for x in r.itl_s]
    e2e = [r.e2e_s * 1000 for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)
    output_tps = out_tokens / dur
    usd_per_hour = hw.usd_per_hour * hw.count
    usd_per_m = (usd_per_hour / 3600 / output_tps * 1e6) if output_tps > 0 else math.inf
    return Metrics(
        ttft_p50_ms=_pct(ttft, 50),
        ttft_p90_ms=_pct(ttft, 90),
        ttft_p99_ms=_pct(ttft, 99),
        itl_p50_ms=_pct(itl, 50),
        itl_p90_ms=_pct(itl, 90),
        itl_p99_ms=_pct(itl, 99),
        e2e_p50_ms=_pct(e2e, 50),
        e2e_p90_ms=_pct(e2e, 90),
        output_tps=output_tps,
        req_per_s=len(ok) / dur,
        goodput_rps=good / dur,
        goodput_frac=(good / total) if total else 0.0,
        error_rate=((total - len(ok)) / total) if total else 0.0,
        tokens_per_s_per_gpu=output_tps / max(hw.count, 1),
        usd_per_m_tokens=usd_per_m,
    )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/loadgen -q && uv run ruff check . && uv run mypy src`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/loadgen tests/loadgen
git commit -m "feat(loadgen): LoadGenerator protocol and SLO-aware metrics analysis"
```

---

### Task 6: SQLite ledger and run artifact directories

**Files:**
- Create: `src/infervolt/store/__init__.py`, `src/infervolt/store/ledger.py`, `tests/store/__init__.py`, `tests/store/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

`tests/store/__init__.py`: empty.

`tests/store/test_ledger.py`:
```python
from pathlib import Path

from infervolt.core.types import Candidate, EngineConfig, OptimizeSpec, Trial
from infervolt.store.ledger import Ledger


def _trial(run_id: str, idx: int) -> Trial:
    cand = Candidate(id=f"c{idx}", config=EngineConfig(engine="mock", knobs={"a": idx}), origin="tpe")
    return Trial(id=f"t{idx}", run_id=run_id, index=idx, candidate=cand, status="ok")


def test_create_run_and_roundtrip_trials(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs")
    spec = OptimizeSpec(engine="mock", model="mock/qwen3-8b")
    run_id = ledger.create_run(spec)
    assert (tmp_path / "runs" / run_id).is_dir()
    ledger.save_trial(_trial(run_id, 0))
    ledger.save_trial(_trial(run_id, 1))
    t1 = _trial(run_id, 1)
    t1.status = "pruned"
    ledger.save_trial(t1)  # upsert
    trials = ledger.trials(run_id)
    assert [t.index for t in trials] == [0, 1]
    assert trials[1].status == "pruned"
    assert ledger.get_run(run_id).spec.model == "mock/qwen3-8b"


def test_state_and_best_are_persisted(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs")
    run_id = ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b"))
    ledger.set_state(run_id, "search")
    ledger.set_best(run_id, "t3")
    run = ledger.get_run(run_id)
    assert run.state == "search" and run.best_trial_id == "t3"
    assert ledger.trials_jsonl(run_id).name == "trials.jsonl"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/store -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the ledger**

`src/infervolt/store/__init__.py`: empty.

`src/infervolt/store/ledger.py`:
```python
"""SQLite ledger of runs and trials plus a per-run artifact directory.

Schema is intentionally tiny: rows hold pydantic JSON so the models stay the source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel

from infervolt.core.types import OptimizeSpec, RunState, Trial

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, created REAL NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
  best_trial_id TEXT, recipe_path TEXT, diagnosis TEXT
);
CREATE TABLE IF NOT EXISTS trials (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS trials_run ON trials(run_id, idx);
"""


class RunRow(BaseModel):
    id: str
    created: float
    spec: OptimizeSpec
    state: RunState
    best_trial_id: str | None = None
    recipe_path: str | None = None
    diagnosis_json: str | None = None


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


class Ledger:
    def __init__(self, db_path: Path, runs_dir: Path) -> None:
        self.db_path = db_path
        self.runs_dir = runs_dir
        db_path.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # ---- runs
    def create_run(self, spec: OptimizeSpec) -> str:
        run_id = spec.run_id or new_run_id()
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs(id, created, spec, state) VALUES (?,?,?,?)",
                (run_id, time.time(), spec.model_dump_json(), "prepare"),
            )
        return run_id

    def get_run(self, run_id: str) -> RunRow:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created, spec, state, best_trial_id, recipe_path, diagnosis "
                "FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRow(
            id=row[0],
            created=row[1],
            spec=OptimizeSpec.model_validate_json(row[2]),
            state=row[3],
            best_trial_id=row[4],
            recipe_path=row[5],
            diagnosis_json=row[6],
        )

    def _update_run(self, run_id: str, sql: str, value: object) -> None:
        """Apply a single-column update, raising ``KeyError`` when the run does not exist."""
        with self._lock, self._conn:
            cursor = self._conn.execute(sql, (value, run_id))
        if cursor.rowcount == 0:
            raise KeyError(run_id)

    def set_state(self, run_id: str, state: RunState) -> None:
        self._update_run(run_id, "UPDATE runs SET state=? WHERE id=?", state)

    def set_best(self, run_id: str, trial_id: str | None) -> None:
        self._update_run(run_id, "UPDATE runs SET best_trial_id=? WHERE id=?", trial_id)

    def set_recipe(self, run_id: str, path: str) -> None:
        self._update_run(run_id, "UPDATE runs SET recipe_path=? WHERE id=?", path)

    def set_diagnosis(self, run_id: str, diagnosis_json: str) -> None:
        self._update_run(run_id, "UPDATE runs SET diagnosis=? WHERE id=?", diagnosis_json)

    # ---- trials
    def save_trial(self, trial: Trial) -> None:
        """Upsert the trial row and append it to the run's JSONL event log.

        The JSONL file is an append-only event log, not a table: an upsert of an
        already-saved trial appends a second line for the same trial id. Readers must
        therefore take the *last* event per trial id; the SQLite row is the current value.
        """
        self.run_dir(trial.run_id).mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO trials(id, run_id, idx, data) VALUES (?,?,?,?)",
                (trial.id, trial.run_id, trial.index, trial.model_dump_json()),
            )
        with self.trials_jsonl(trial.run_id).open("a") as f:
            f.write(json.dumps({"event": "trial", "trial": trial.model_dump(mode="json")}) + "\n")

    def trials(self, run_id: str) -> list[Trial]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM trials WHERE run_id=? ORDER BY idx, id", (run_id,)
            ).fetchall()
        return [Trial.model_validate_json(r[0]) for r in rows]

    # ---- artifacts
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def trials_jsonl(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trials.jsonl"

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/store -q && uv run ruff check . && uv run mypy src`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/store tests/store
git commit -m "feat(store): SQLite ledger with run and trial persistence"
```

---

### Task 7: Recipe schema, YAML emitter, Markdown report, `recipe validate` CLI

**Files:**
- Create: `src/infervolt/recipes/__init__.py`, `src/infervolt/recipes/schema.py`, `src/infervolt/recipes/emit.py`, `src/infervolt/recipes/templates/report.md.j2`, `examples/recipe.yaml`, `tests/recipes/__init__.py`, `tests/recipes/test_recipe.py`
- Modify: `src/infervolt/cli/main.py`, `pyproject.toml` (package data)

- [ ] **Step 1: Write the failing tests**

`tests/recipes/__init__.py`: empty.

`tests/recipes/test_recipe.py`:
```python
from pathlib import Path

import yaml
from typer.testing import CliRunner

from infervolt.cli.main import app
from infervolt.recipes.emit import render_report, write_recipe
from infervolt.recipes.schema import Recipe, recipe_json_schema

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "recipe.yaml"


def test_example_recipe_validates() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    recipe = Recipe.model_validate(data)
    assert recipe.engine.name == "vllm"
    assert recipe.infervolt.diagnosis.primary == "kv_capacity"


def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    path = write_recipe(recipe, tmp_path)
    assert path.name == "recipe.yaml"
    again = Recipe.model_validate(yaml.safe_load(path.read_text()))
    assert again == recipe


def test_report_renders_key_sections() -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    md = render_report(recipe)
    assert "# infervolt recipe" in md
    assert "kv_capacity" in md and "goodput_rps" in md and "Reproduce" in md


def test_json_schema_export_has_required_top_level_keys() -> None:
    schema = recipe_json_schema()
    assert {"model", "hardware", "engine", "serve", "infervolt"} <= set(schema["required"])


def test_cli_recipe_validate(tmp_path: Path) -> None:
    runner = CliRunner()
    ok = runner.invoke(app, ["recipe", "validate", str(EXAMPLE)])
    assert ok.exit_code == 0, ok.stdout
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {id: x}\n")
    res = runner.invoke(app, ["recipe", "validate", str(bad)])
    assert res.exit_code == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/recipes -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write schema, emitter, template, example**

`src/infervolt/recipes/__init__.py`: empty.

`src/infervolt/recipes/schema.py`:
```python
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


class RecipeMeasured(BaseModel):
    serve_args: dict[str, KnobValue] = Field(default_factory=dict)
    metrics: dict[str, float]


class RecipeQuality(BaseModel):
    guard: str
    tasks: list[str]
    recovery: float


class RecipeResult(BaseModel):
    metrics: dict[str, float]
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
```

`src/infervolt/recipes/emit.py`:
```python
"""Write recipe.yaml and render report.md."""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from infervolt.recipes.schema import Recipe


def yamlish(value: object) -> object:
    """Render booleans the way YAML and engine CLIs spell them; leave everything else alone."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return value


_env = Environment(
    # Autoescape is intentionally off: these templates render Markdown, not HTML, and
    # HTML-escaping would mangle model ids, CLI flags and quoted knob values.
    loader=PackageLoader("infervolt.recipes", "templates"),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters["yamlish"] = yamlish


def write_recipe(recipe: Recipe, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False, width=100))
    return path


def render_report(recipe: Recipe) -> str:
    return _env.get_template("report.md.j2").render(r=recipe)


def write_report(recipe: Recipe, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(render_report(recipe))
    return path
```

`src/infervolt/recipes/templates/report.md.j2`:
```jinja
# infervolt recipe: {{ r.model.id }} on {{ r.hardware.gpu }} ({{ r.engine.name }})

Run `{{ r.infervolt.run_id }}` · created {{ r.infervolt.provenance.created }} · infervolt {{ r.infervolt.provenance.tool_version }}

## Summary

| Metric | Baseline | Tuned |
|---|---|---|
{% for k, v in r.result.metrics.items() %}
| {{ k }} | {{ '%.3f' % r.baseline.metrics.get(k, 0.0) }} | {{ '%.3f' % v }} |
{% endfor %}

{% for k, v in r.result.improvement.items() %}
**{{ k }}:** {{ v }}
{% endfor %}
Verified with {{ r.result.repeats }} interleaved repeats.
{% if r.result.quality %}Quality guard `{{ r.result.quality.guard }}` on {{ r.result.quality.tasks | join(', ') }}: recovery {{ '%.3f' % r.result.quality.recovery }}.{% endif %}

## Diagnosis: {{ r.infervolt.diagnosis.primary }} (confidence {{ '%.2f' % r.infervolt.diagnosis.confidence }})

{{ r.infervolt.rationale }}

| Rule | Score | Evidence |
|---|---|---|
{% for f in r.infervolt.diagnosis.findings %}
| {{ f.rule }} | {{ '%.2f' % f.score }} | {% for e in f.evidence %}`{{ e.key }}`={{ '%.4g' % e.value }}{{ e.unit }}{% if not loop.last %}, {% endif %}{% endfor %} |
{% endfor %}

## Winning configuration

| Knob | Baseline | Tuned |
|---|---|---|
{% for k, v in r.serve.args.items() %}
| {{ k }} | {{ r.baseline.serve_args.get(k, '') | yamlish }} | {{ v | yamlish }} |
{% endfor %}

## Search

{{ r.infervolt.search.trials }} trials ({{ r.infervolt.search.infeasible }} infeasible) with {{ r.infervolt.search.optimizer }} over {{ r.infervolt.search.subspace | join(', ') }}, seed {{ r.infervolt.search.seed }}.
{% if r.infervolt.trials_to_target is not none %}Trials to reach 95% of the final best: {{ r.infervolt.trials_to_target }}.{% endif %}

## Workload and SLO

`{{ r.workload.name }}`: ISL p50 {{ r.workload.isl.p50 }}, OSL p50 {{ r.workload.osl.p50 }}, prefix share {{ r.workload.prefix_share }}.
{% set parts = [] %}
{% if r.slo.ttft_ms is not none %}{% set _ = parts.append('ttft %s ms' % r.slo.ttft_ms) %}{% endif %}
{% if r.slo.itl_ms is not none %}{% set _ = parts.append('itl %s ms' % r.slo.itl_ms) %}{% endif %}
{% if r.slo.e2e_ms is not none %}{% set _ = parts.append('e2e %s ms' % r.slo.e2e_ms) %}{% endif %}
{% if parts %}
SLO: {{ parts | join(', ') }} at p{{ (r.slo.percentile * 100) | int }}, goodput target {{ r.slo.goodput_target }}.
{% else %}
SLO: none (throughput only), goodput target {{ r.slo.goodput_target }}.
{% endif %}

## Reproduce

```bash
{{ r.serve.command }}
```

## Next steps

{% for s in r.infervolt.next_steps %}
- {{ s }}
{% endfor %}

## Caveats

- Measurements come from a synthetic or real load generator as recorded in `artifacts`; GPUs are not bit-reproducible, expect a few percent variance.
- Provenance: llm `{{ r.infervolt.provenance.llm }}`, prompts `{{ r.infervolt.provenance.prompts_sha }}`.
```bash
{{ r.serve.command }}
```

## Next steps

{% for s in r.infervolt.next_steps %}
- {{ s }}
{% endfor %}

## Caveats

- Measurements come from a synthetic or real load generator as recorded in `artifacts`; GPUs are not bit-reproducible, expect a few percent variance.
- Provenance: llm `{{ r.infervolt.provenance.llm }}`, prompts `{{ r.infervolt.provenance.prompts_sha }}`.
```

`examples/recipe.yaml`:
```yaml
schema_version: 1
model: {id: Qwen/Qwen3-8B, revision: abc123, params_b: 8.2, arch: qwen3, moe: false}
hardware: {gpu: NVIDIA A100-SXM4-80GB, count: 1, driver: "580.65", topology: single, provider: thunder-compute}
engine: {name: vllm, version: 0.11.0, image: "vllm/vllm-openai@sha256:deadbeef", commit: abcdef0}
workload:
  name: chat-4k-512
  isl: {p50: 4096, p99: 6000}
  osl: {p50: 512, p99: 1024}
  prefix_share: 0.1
  load: {mode: sweep, concurrency: [1, 4, 16, 64]}
slo: {ttft_ms: 500, itl_ms: 30, percentile: 0.9}
serve:
  args: {max_num_seqs: 128, max_num_batched_tokens: 4096, gpu_memory_utilization: 0.9, kv_cache_dtype: fp8, enable_prefix_caching: true}
  env: {}
  command: "vllm serve Qwen/Qwen3-8B --max-num-seqs 128 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --enable-prefix-caching"
baseline:
  serve_args: {max_num_seqs: 256, max_num_batched_tokens: 2048, gpu_memory_utilization: 0.9, kv_cache_dtype: auto, enable_prefix_caching: true}
  metrics: {goodput_rps: 3.1, ttft_p90_ms: 812, itl_p90_ms: 27.4, output_tps: 1420}
result:
  metrics: {goodput_rps: 5.6, ttft_p90_ms: 430, itl_p90_ms: 24.9, output_tps: 2210}
  repeats: 3
  improvement: {goodput_rps: "+81% (95% CI +62..+97)"}
  quality: {guard: lm-eval, tasks: [gsm8k, arc_challenge], recovery: 0.996}
infervolt:
  run_id: 2026-09-03T14-02-11Z-7f3a
  diagnosis:
    primary: kv_capacity
    confidence: 0.82
    findings:
      - rule: R1
        score: 0.9
        evidence:
          - {source: prometheus, key: preemptions_per_s, value: 2.3, unit: "/s"}
          - {source: prometheus, key: kv_usage_p95, value: 0.97}
  rationale: "Preemptions at c=16 show KV exhaustion at default 0.9 util with fp16 KV; fp8 KV doubles token capacity and removes the preemptions."
  search: {trials: 14, infeasible: 3, subspace: [gpu_memory_utilization, kv_cache_dtype, max_num_seqs, max_model_len], optimizer: optuna-tpe, seed: 7}
  trials_to_target: 6
  warm_start: {from_runs: [], notes_used: []}
  next_steps: ["Try EAGLE-3 draft: decode is now bandwidth-bound (R2 score rose to 0.7 after fix)"]
  artifacts: {report: report.md, trials: trials.jsonl}
  provenance: {tool_version: 0.1.0, llm: claude-opus-5, prompts_sha: 0000000, created: "2026-09-03"}
```

- [ ] **Step 4: Add the `recipe validate` command and package the template**

Append to `src/infervolt/cli/main.py` (after `app = typer.Typer(...)`):
```python
from pathlib import Path  # noqa: E402  (keep imports at top in the real file)

import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from infervolt.recipes.schema import Recipe  # noqa: E402

recipe_app = typer.Typer(help="Recipe utilities.")
app.add_typer(recipe_app, name="recipe")


@recipe_app.command("validate")
def recipe_validate(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]) -> None:
    """Validate a recipe.yaml against the infervolt schema."""
    try:
        Recipe.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError) as e:
        typer.echo(f"INVALID {path}: {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"OK {path}")
```
Put the imports at the top of the file with the others (the `noqa` markers above only exist so the snippet reads in isolation; remove them).

In `pyproject.toml`, under `[tool.hatch.build.targets.wheel]` add `artifacts = ["*.j2"]` (wheel-only; a global `[tool.hatch.build] include` would strip tests/ and examples/ from the sdist).

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/recipes -q && uv run ruff check . && uv run mypy src`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/infervolt/recipes src/infervolt/cli/main.py pyproject.toml examples tests/recipes
git commit -m "feat(recipes): recipe schema, YAML emitter, Markdown report, recipe validate CLI"
```

---

### Task 8: Community files and CI

**Files:**
- Create: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `ROADMAP.md`, `.github/CODEOWNERS`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/workflows/ci.yml`, `.env.example`

- [ ] **Step 1: Write community files**

`CONTRIBUTING.md`:
```markdown
# Contributing to infervolt

Thanks for helping. infervolt is pre-alpha; the fastest way to contribute is to run the mock loop,
file issues with reproductions, and send small PRs.

## Dev setup

```bash
git clone https://github.com/infervolt/infervolt && cd infervolt
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

## Lint gate (CI runs exactly this)

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

## How to add things

- **An engine adapter:** implement `EngineAdapter` from `src/infervolt/engines/base.py`, register it under
  `[project.entry-points."infervolt.engines"]`, add fixtures for its `/metrics` output, and a knob space.
- **A diagnosis rule:** add a function in `src/infervolt/diagnose/rules.py` returning a `Finding`,
  register it in `RULES`, and add a mock scenario that triggers it.
- **A workload preset:** add it to `src/infervolt/workloads/presets.py` with a test.
- **A recipe:** run `infervolt optimize`, then open a PR adding `recipes/<engine>/<model>/<hardware>/`
  with `recipe.yaml`, `report.md`, and `trials.jsonl`. CI validates the YAML.

## Commit style

Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`). Every PR needs tests.
```

`CODE_OF_CONDUCT.md`: the Contributor Covenant v2.1 text. Fetch it verbatim:
```bash
curl -fsSL https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md -o CODE_OF_CONDUCT.md
```
Then replace `[INSERT CONTACT METHOD]` with `security@infervolt.dev` (placeholder mailbox; change when the domain exists).

`SECURITY.md`:
```markdown
# Security policy

infervolt launches inference servers and runs load against them, and can call LLM APIs with your keys.

- Never commit `.env`, `~/.thunder`, or any file containing tokens. A gitleaks pre-commit hook is configured.
- Report vulnerabilities privately via GitHub Security Advisories on this repository. We aim to respond within 7 days.
- Supported versions: the latest minor release.
```

`CHANGELOG.md`:
```markdown
# Changelog

All notable changes are documented here. Format: Keep a Changelog. Versioning: SemVer.

## [Unreleased]

### Added
- Mock engine loop: baseline sweep, rule-based diagnosis, LLM ranking/planning, Optuna search, verify, recipe emission.
```

`ROADMAP.md`:
```markdown
# Roadmap

- **M1 (now):** mock engine loop end to end, CPU-only CI. Interfaces are synchronous; artifacts are JSONL.
- **M2:** llama.cpp adapter (llama-bench stage 1, `llama-server --metrics` stage 2), multi-process HTTP load generator (async), Apple Silicon detection.
- **M3:** vLLM adapter on Thunder Compute (docker by digest, Prometheus name resolution, lm-eval quality guard, budget guard, teardown).
- **M4:** cross-run memory (meta-features, kNN warm start, insight notes), `--resume`, `watch` mode, Parquet + DuckDB `memory stats`.
- **M5:** docs site, committed recipes, PyPI release.
- **Later:** SGLang and TensorRT-LLM adapters, disaggregated prefill/decode and wide-EP recipes.
```

`.github/CODEOWNERS`:
```
* @vageeshaganapaneni
```

`.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug report
about: Something broke
---

**Command run**

**Expected / actual**

**Engine, hardware, model, workload**

**`infervolt --version`, OS, Python**

**Logs / `trials.jsonl` excerpt**
```

`.github/ISSUE_TEMPLATE/feature_request.md`:
```markdown
---
name: Feature request
about: Adapter, rule, workload, or recipe idea
---

**What bottleneck or engine does this cover?**

**Evidence it matters (paper, benchmark, profile)**

**Proposed change**
```

`.github/PULL_REQUEST_TEMPLATE.md`:
```markdown
## What

## Why

## Tests
- [ ] `uv run pytest -q` passes
- [ ] lint gate passes
```

`.env.example`:
```
# Copy to .env. Never commit .env.
ANTHROPIC_API_KEY=
INFERVOLT_ANTHROPIC_MODEL=claude-opus-5
INFERVOLT_OPENAI_BASE_URL=http://localhost:8000/v1
INFERVOLT_OPENAI_MODEL=default
INFERVOLT_OPENAI_API_KEY=EMPTY
```

- [ ] **Step 2: Write the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }
      - run: uv python install ${{ matrix.python }}
      - run: uv sync --extra dev
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -q --cov=infervolt --cov-report=term-missing
      - name: Validate committed recipes
        run: |
          for f in $(find recipes examples -name 'recipe.yaml' 2>/dev/null); do uv run infervolt recipe validate "$f"; done
```

- [ ] **Step 3: Verify locally**

Run: `uv run pre-commit run --all-files && uv run pytest -q`
Expected: hooks pass (ruff may reformat; re-run until clean), all tests pass.

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md CHANGELOG.md ROADMAP.md .github .env.example
git commit -m "chore: community files and CI workflow"
```

M0 is complete when CI is green on this commit pushed to GitHub (create the repo `infervolt/infervolt` or under your account, push `main`).

---

## M1: mock-engine loop end to end

### Task 9: Engine adapter base and registry

**Files:**
- Create: `src/infervolt/engines/__init__.py`, `src/infervolt/engines/base.py`, `src/infervolt/engines/registry.py`, `tests/engines/__init__.py`, `tests/engines/test_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/engines/__init__.py`: empty.

`tests/engines/test_base.py`:
```python
import pytest

from infervolt.engines.base import ExitInfo, classify_log
from infervolt.engines.registry import get_adapter


@pytest.mark.parametrize(
    "code,log,kind",
    [
        (1, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "oom"),
        (1, "No available memory for the cache blocks", "oom"),
        (1, "is larger than the maximum number of tokens that can be stored in KV cache", "oom"),
        (-9, "", "oom"),
        (1, "ggml_metal_graph_compute: failed to allocate", "oom"),
        (1, "RuntimeError: something else", "runtime"),
        (0, "", "none"),
        (124, "", "timeout"),
    ],
)
def test_classify_log(code: int, log: str, kind: str) -> None:
    assert classify_log(ExitInfo(code=code, log_tail=log)) == kind


def test_registry_loads_mock_adapter() -> None:
    adapter = get_adapter("mock")
    assert adapter.name == "mock"


def test_registry_unknown_engine() -> None:
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engines -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write base and registry**

`src/infervolt/engines/__init__.py`: empty.

`src/infervolt/engines/base.py`:
```python
"""Engine adapter contract. Every serving engine (mock, llama.cpp, vLLM, ...) implements this."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from infervolt.core.types import CrashKind, EngineConfig, KnobSpace, KnobValue, RunContext
from infervolt.loadgen.base import LoadGenerator


class EngineVersion(BaseModel):
    name: str
    version: str = ""
    commit: str = ""
    image_digest: str = ""


@dataclass
class ServerHandle:
    """A running server, plus whatever the adapter needs to talk to it.

    ``config`` is the config the server was actually launched with. The runner sets it
    after a successful launch so that every observation taken through this handle can
    record what produced it, without threading the config through each call.
    """

    url: str
    state: Any = None
    config: EngineConfig | None = None


@dataclass
class ExitInfo:
    code: int
    log_tail: str = ""


class LaunchError(Exception):
    def __init__(self, exit: ExitInfo) -> None:
        super().__init__(exit.log_tail[-400:])
        self.exit = exit


OOM_PATTERNS = [
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"No available memory for the cache blocks",
    r"larger than the maximum number of tokens that can be stored in KV cache",
    r"Free memory on device .* less than desired",
    r"failed to allocate",
    r"ggml_metal.*alloc",
    r"kv_cache_init: failed",
]
_OOM_RE = re.compile("|".join(OOM_PATTERNS), re.IGNORECASE)


def classify_log(exit: ExitInfo) -> CrashKind:
    """Engine-agnostic crash classification from exit code and log tail."""
    if exit.code == 0:
        return "none"
    if exit.code == 124:
        return "timeout"
    if exit.code == -9 or _OOM_RE.search(exit.log_tail):
        return "oom"
    return "runtime"


class EngineAdapter(ABC):
    name: str = "abstract"

    @abstractmethod
    def version(self) -> EngineVersion: ...

    @abstractmethod
    def knob_space(self, ctx: RunContext) -> KnobSpace: ...

    @abstractmethod
    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """Static rejections. Non-empty means never launch this config."""

    @abstractmethod
    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        """Start the server. Raise LaunchError with the exit info if it dies during startup."""

    @abstractmethod
    def ready(self, handle: ServerHandle, timeout_s: float) -> bool: ...

    @abstractmethod
    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator: ...

    @abstractmethod
    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        """Canonical engine metrics since the last scrape: kv_usage_p95, num_waiting, num_running,
        preemptions_per_s, queue_time_p90_s, prefill_time_p50_s, prefill_share, prefix_hit_rate,
        max_num_seqs, kv_dtype_bytes."""

    @abstractmethod
    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        """sm_active and dram_active in [0, 1] over the last load window."""

    @abstractmethod
    def stop(self, handle: ServerHandle) -> ExitInfo: ...

    @abstractmethod
    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        """(serve args, reproduction command)."""

    def classify_crash(self, exit: ExitInfo) -> CrashKind:
        return classify_log(exit)
```

`src/infervolt/engines/registry.py`:
```python
"""Adapter discovery through the `infervolt.engines` entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points

from infervolt.engines.base import EngineAdapter


def available_engines() -> list[str]:
    return sorted(ep.name for ep in entry_points(group="infervolt.engines"))


def get_adapter(name: str) -> EngineAdapter:
    for ep in entry_points(group="infervolt.engines"):
        if ep.name == name:
            cls = ep.load()
            adapter = cls()
            assert isinstance(adapter, EngineAdapter)
            return adapter
    raise KeyError(f"unknown engine {name!r}; available: {available_engines()}")
```

The registry test needs the mock adapter from Task 11; until then run only `-k classify_log` (`uv run pytest tests/engines -q -k classify_log`, expected `8 passed`).

- [ ] **Step 4: Lint and commit**

Run: `uv run ruff check . && uv run mypy src`

```bash
git add src/infervolt/engines tests/engines
git commit -m "feat(engines): adapter base class, crash classification, entry-point registry"
```

---

### Task 10: Mock performance model (roofline simulator)

**Files:**
- Create: `src/infervolt/engines/mock/__init__.py`, `src/infervolt/engines/mock/model.py`, `tests/engines/mock/__init__.py`, `tests/engines/mock/test_model.py`

- [ ] **Step 1: Write the failing tests**

`tests/engines/mock/__init__.py`: empty.

`tests/engines/mock/test_model.py`:
```python
import pytest

from infervolt.engines.mock.model import DEFAULT_KNOBS, OomError, PerfModel
from infervolt.hardware.profiles import get_profile
from infervolt.models.catalog import get_model_info
from infervolt.workloads.presets import get_workload


def _pm(hw: str, model: str, workload: str, **knobs: object) -> PerfModel:
    return PerfModel(get_profile(hw), get_model_info(model), get_workload(workload), {**DEFAULT_KNOBS, **knobs})


def test_kv_limited_scenario_preempts_and_queues() -> None:
    pm = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512")
    p = pm.point(16)
    assert p.running < 16 and p.waiting > 0
    assert p.preempt_frac > 0 and p.kv_usage >= 0.9
    assert pm.point(4).preempt_frac == 0


def test_fp8_kv_doubles_admitted_sequences() -> None:
    a = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512").point(64).running
    b = _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", kv_cache_dtype="fp8").point(64).running
    assert b >= 2 * a - 1


def test_decode_bound_scenario_is_dram_heavy() -> None:
    p = _pm("a100-80", "mock/qwen3-8b", "chat-256-512").point(64)
    assert p.dram_active > 0.6 and p.sm_active < 0.5
    assert p.itl_mean_s < 1.3 * p.step_floor_s


def test_prefill_bound_scenario_is_sm_heavy() -> None:
    p = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64").point(16)
    assert p.prefill_share > 0.5 and p.sm_active > 0.7


def test_scheduler_bound_scenario_has_flat_itl() -> None:
    pm = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager=True)
    p1, p8 = pm.point(1), pm.point(8)
    assert abs(p8.itl_mean_s - p1.itl_mean_s) / p1.itl_mean_s < 0.15
    assert p8.sm_active < 0.4 and p8.dram_active < 0.4
    fast = _pm("a100-80", "mock/qwen3-0.6b", "chat-1k-128", enforce_eager=False).point(8)
    assert fast.itl_mean_s < 0.5 * p8.itl_mean_s


def test_speculative_and_quantization_effects() -> None:
    base = _pm("h100-80", "mock/qwen3-8b", "chat-256-512").point(16)
    spec = _pm("h100-80", "mock/qwen3-8b", "chat-256-512", speculative="eagle3").point(16)
    quant = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64", quantization="fp8").point(4)
    plain = _pm("h100-80", "mock/qwen3-8b", "rag-16k-64").point(4)
    assert spec.itl_mean_s < base.itl_mean_s
    assert quant.prefill_s < plain.prefill_s


def test_oom_at_launch() -> None:
    with pytest.raises(OomError, match="CUDA out of memory"):
        _pm("rtx4090-24", "mock/llama-70b", "chat-4k-512").check_launch()
    with pytest.raises(OomError, match="KV cache"):
        _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", max_model_len=32768).check_launch()
    _pm("rtx4090-24", "mock/qwen3-8b", "chat-4k-512", max_model_len=8192).check_launch()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engines/mock -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the performance model**

`src/infervolt/engines/mock/__init__.py`: empty.

`src/infervolt/engines/mock/model.py`:
```python
"""Analytic steady-state simulator of a continuous-batching LLM server.

It is deliberately simple: one closed-loop concurrency level -> admitted sequences, queueing,
prefill/decode interference, KV capacity, preemption, and roofline-derived step times. It is
not meant to be accurate in absolute terms, only to reproduce the *signatures* of each
bottleneck so the diagnosis rules and the search loop can be tested without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from infervolt.core.types import HardwareProfile, KnobValue, ModelInfo, Workload
from infervolt.hardware import roofline

DEFAULT_KNOBS: dict[str, KnobValue] = {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.9,
    # Bare-model default only. MockAdapter.knob_space() overrides this with
    # MockAdapter._default_max_model_len(ctx) -- the shortest offered length that covers
    # the workload -- because a fixed 32768 OOMs at launch on small cards.
    "max_model_len": 32768,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "speculative": "none",
    "quantization": "none",
    "tensor_parallel_size": 1,
}

SPEC_SPEEDUP = {"none": 1.0, "ngram": 1.25, "eagle3": 1.7}
EAGER_OVERHEAD_S = 0.004
GRAPH_OVERHEAD_S = 0.0008
PER_SEQ_OVERHEAD_S = 5e-6


_TRUE_WORDS = frozenset({"true", "1", "yes"})
_FALSE_WORDS = frozenset({"false", "0", "no"})


def _as_bool(v: KnobValue) -> bool:
    """Coerce a knob value to a bool, the way a CLI flag would be read.

    ``bool()`` is wrong here: every non-empty string is truthy, so ``bool("false")``
    is ``True`` and a knob set from YAML, JSON or a command line silently inverts.
    Accepts real bools, the ints 0 and 1, and the usual word spellings in any case;
    anything else raises rather than guessing.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        word = v.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"cannot read {v!r} as a bool; use true/false, 1/0 or yes/no")


class OomError(Exception):
    """Raised by check_launch when the config cannot fit."""


@dataclass
class SimPoint:
    """Steady state at one closed-loop concurrency level.

    Time fields are per request. ``lifetime_s`` is *service* time only -- the span from
    admission to the last token, excluding time spent waiting to be admitted -- so a
    request's end-to-end latency is ``queue_wait_s + lifetime_s``. ``ttft_s``, by
    contrast, is measured from arrival and so already includes ``queue_wait_s``.
    """

    concurrency: int
    running: int
    waiting: int
    queue_wait_s: float
    ttft_s: float
    prefill_s: float
    itl_mean_s: float
    itl_spike_s: float
    n_spikes: int
    lifetime_s: float
    step_floor_s: float
    kv_usage: float
    preempt_frac: float
    prefill_share: float
    sm_active: float
    dram_active: float


class PerfModel:
    def __init__(
        self, hw: HardwareProfile, model: ModelInfo, workload: Workload, knobs: dict[str, KnobValue]
    ) -> None:
        self.hw = hw
        self.workload = workload
        self.knobs = {**DEFAULT_KNOBS, **knobs}
        quant = str(self.knobs["quantization"])
        self.model = model.model_copy(update={"weight_bits": 8}) if quant == "fp8" else model
        self.peak_tflops = hw.peak_tflops * (2.0 if quant == "fp8" else 1.0)
        self.hw_eff = hw.model_copy(update={"peak_tflops": self.peak_tflops})
        self.kv_dtype_bytes = 1 if str(self.knobs["kv_cache_dtype"]) == "fp8" else 2
        self.util = float(self.knobs["gpu_memory_utilization"])
        self.capacity = roofline.kv_capacity_tokens(hw, self.model, self.util, self.kv_dtype_bytes)

    # ---- launch-time checks
    def check_launch(self) -> None:
        need = roofline.weight_bytes(self.model) + roofline.reserve_bytes()
        if need > roofline.mem_bytes(self.hw) * self.util:
            raise OomError(
                f"torch.OutOfMemoryError: CUDA out of memory. "
                f"Tried to allocate {need / 2**30:.1f} GiB"
            )
        max_len = int(self.knobs["max_model_len"])
        if self.capacity < max_len:
            raise OomError(
                f"ValueError: The model's max seq len ({max_len}) is larger than the maximum "
                f"number of tokens that can be stored in KV cache ({int(self.capacity)})."
            )

    # ---- steady state at one concurrency
    def _sched_overhead(self, n: int) -> float:
        base = EAGER_OVERHEAD_S if _as_bool(self.knobs["enforce_eager"]) else GRAPH_OVERHEAD_S
        return base + PER_SEQ_OVERHEAD_S * n

    def _spec_speedup(self) -> float:
        name = str(self.knobs["speculative"])
        try:
            return SPEC_SPEEDUP[name]
        except KeyError as e:
            raise ValueError(
                f"unknown speculative {name!r}; choices: {sorted(SPEC_SPEEDUP)}"
            ) from e

    def point(self, concurrency: int) -> SimPoint:
        w, m = self.workload, self.model
        isl, osl = w.isl.p50, w.osl.p50
        hit = w.prefix_share if _as_bool(self.knobs["enable_prefix_caching"]) else 0.0
        p_tokens = max(1, int(isl * (1 - hit)))
        max_seqs = int(self.knobs["max_num_seqs"])
        by_kv = int(self.capacity // isl) if self.capacity >= isl else 0
        n = max(0, min(concurrency, max_seqs, by_kv))
        if n == 0:
            # Nothing runs. Either the KV cache cannot hold even one request -- a saturated
            # server, which is what the diagnosis rules must see -- or there is simply
            # nothing to run (concurrency 0, max_num_seqs 0), which is an idle one.
            starved = by_kv == 0
            return SimPoint(
                concurrency=concurrency,
                running=0,
                waiting=max(0, concurrency),
                queue_wait_s=0.0,
                ttft_s=0.0,
                prefill_s=0.0,
                itl_mean_s=0.0,
                itl_spike_s=0.0,
                n_spikes=0,
                lifetime_s=0.0,
                step_floor_s=0.0,
                kv_usage=1.0 if starved else 0.0,
                preempt_frac=1.0 if starved else 0.0,
                prefill_share=0.0,
                sm_active=0.0,
                dram_active=0.0,
            )
        waiting = concurrency - n
        ctx = isl + osl // 2
        floor = roofline.decode_step_floor_s(self.hw_eff, m, n, ctx, self.kv_dtype_bytes)
        step = floor + self._sched_overhead(n)
        prefill = roofline.prefill_floor_s(self.hw_eff, m, p_tokens)
        per_tok = prefill / p_tokens
        if _as_bool(self.knobs["enable_chunked_prefill"]):
            chunk = min(p_tokens, int(self.knobs["max_num_batched_tokens"]))
            n_chunks = math.ceil(p_tokens / chunk)
            ttft_core = prefill + n_chunks * step
            spike = chunk * per_tok
        else:
            ttft_core, spike = prefill, prefill
        itl_mean = (step + (n - 1) * prefill / osl) / self._spec_speedup()
        need = n * (isl + osl)
        preempt_frac = max(0.0, (need - self.capacity) / self.capacity)
        lifetime_core = ttft_core + osl * itl_mean
        lifetime = lifetime_core * (1 + preempt_frac)
        queue_wait = (waiting / n) * lifetime if waiting > 0 else 0.0
        mem_part = (
            roofline.weight_bytes(m) + n * roofline.kv_bytes_per_token(m, self.kv_dtype_bytes) * ctx
        ) / (self.hw.hbm_bw_gbs * 1e9)
        comp_part = 2.0 * roofline.active_params(m) * n / (self.peak_tflops * 1e12)
        compute_time = n * prefill + osl * comp_part
        mem_time = osl * mem_part
        return SimPoint(
            concurrency=concurrency,
            running=n,
            waiting=waiting,
            queue_wait_s=queue_wait,
            ttft_s=queue_wait + ttft_core,
            prefill_s=prefill,
            itl_mean_s=itl_mean,
            itl_spike_s=spike,
            n_spikes=min(osl, n - 1),
            lifetime_s=lifetime,
            step_floor_s=floor,
            kv_usage=min(1.0, n * (isl + osl / 2) / self.capacity),
            preempt_frac=preempt_frac,
            prefill_share=min(1.0, n * prefill / lifetime_core),
            sm_active=min(1.0, compute_time / lifetime_core),
            dram_active=min(1.0, mem_time / lifetime_core),
        )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/engines/mock -q && uv run ruff check . && uv run mypy src`
Expected: `7 passed`. If a threshold assertion fails, print the `SimPoint` and adjust the scenario's hardware or workload in the test (not the model) so the signature is unambiguous; the four scenarios must stay distinguishable.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/engines/mock tests/engines/mock
git commit -m "feat(mock): roofline-based steady-state performance model"
```

---

### Task 11: Mock adapter, simulated load generator, scenarios

> **Deviation record (2026-09-02):** `MockAdapter.knob_space()` sets the `max_model_len` default to the shortest choice covering `isl.p99 + osl.p50` instead of a fixed 32768, because on rtx4090-24 + qwen3-8b the KV cache holds ~31.5k tokens and a 32768 default OOMs at launch (the plan's own Task 10 OOM test relies on that). `max_model_len` only gates launch feasibility in the simulator, so nothing downstream loses a findable fix. `DEFAULT_KNOBS` in `model.py` keeps 32768 for the bare model.

**Files:**
- Create: `src/infervolt/engines/mock/adapter.py`, `src/infervolt/engines/mock/scenarios.py`, `tests/engines/mock/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

`tests/engines/mock/test_adapter.py`:
```python
import pytest

from infervolt.core.types import EngineConfig
from infervolt.engines.base import LaunchError
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.engines.registry import get_adapter


def test_scenarios_cover_four_bottlenecks() -> None:
    assert {s.expected for s in SCENARIOS.values()} == {
        "kv_capacity", "decode_bandwidth", "prefill_compute", "scheduler_cpu"
    }


def test_launch_load_scrape_stop() -> None:
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    assert adapter.ready(handle, 1.0)
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency=16, num_requests=32, seed=1)
    assert len(lr.requests) == 32 and lr.duration_s > 0
    assert all(len(r.itl_s) == ctx.workload.osl.p50 for r in lr.requests)
    snap = adapter.scrape(handle)
    assert snap["preemptions_per_s"] > 0 and snap["kv_dtype_bytes"] == 2
    gpu = adapter.gpu_stats(handle)
    assert 0 <= gpu["sm_active"] <= 1
    assert adapter.stop(handle).code == 0


def test_launch_oom_raises_launch_error() -> None:
    adapter = MockAdapter()
    ctx = make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs={**adapter.knob_space(ctx).defaults(), "max_model_len": 32768})
    with pytest.raises(LaunchError) as ei:
        adapter.launch(cfg, ctx)
    assert adapter.classify_crash(ei.value.exit) == "oom"


def test_validate_rejects_fp8_quant_on_ampere() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")  # a100, cc 8.0
    cfg = EngineConfig(engine="mock", knobs={**adapter.knob_space(ctx).defaults(), "quantization": "fp8"})
    assert adapter.validate(cfg, ctx)


def test_deterministic_with_seed() -> None:
    adapter = MockAdapter()
    ctx = make_context("decode", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    h = adapter.launch(cfg, ctx)
    a = adapter.loadgen(h, ctx).run(ctx.workload, 4, 8, seed=3)
    b = adapter.loadgen(h, ctx).run(ctx.workload, 4, 8, seed=3)
    assert a == b


def test_registry_returns_mock() -> None:
    assert get_adapter("mock").name == "mock"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engines/mock/test_adapter.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the adapter and scenarios**

`src/infervolt/engines/mock/adapter.py`:
```python
"""Mock engine: no server, no GPU. Launch builds a PerfModel; the load generator samples from it."""

from __future__ import annotations

import numpy as np

from infervolt.core.types import (
    EngineConfig,
    Knob,
    KnobSpace,
    KnobValue,
    LoadResult,
    RequestRecord,
    RunContext,
    Workload,
)
from infervolt.engines.base import EngineAdapter, EngineVersion, ExitInfo, LaunchError, ServerHandle
from infervolt.engines.mock.model import DEFAULT_KNOBS, OomError, PerfModel, SimPoint, _as_bool
from infervolt.loadgen.base import LoadGenerator

NOISE = 0.03
MAX_MODEL_LEN_CHOICES: list[KnobValue] = [4096, 8192, 16384, 32768]
INT_KNOBS = frozenset({"max_num_seqs", "max_num_batched_tokens"})


def _as_number(value: KnobValue) -> float | None:
    """Read a knob value as a number, or ``None`` if it is not one.

    Bools are rejected outright: ``True`` is numerically 1, but a bool reaching an int
    knob is a config mistake worth reporting rather than silently accepting.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None


class MockState:
    def __init__(self, pm: PerfModel) -> None:
        self.pm = pm
        self.last: SimPoint | None = None


def _state(handle: ServerHandle) -> MockState:
    """Narrow ``ServerHandle.state``, which is deliberately ``Any`` in the base contract."""
    st = handle.state
    assert isinstance(st, MockState), f"handle was not produced by MockAdapter.launch: {st!r}"
    return st


class SimLoadGenerator:
    def __init__(self, state: MockState) -> None:
        self.state = state

    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        pm = self.state.pm
        p = pm.point(concurrency)
        self.state.last = p
        # Seed with the pair rather than a mixed scalar: default_rng hashes the sequence,
        # so neighbouring (seed, concurrency) pairs cannot collide the way seed*1000+c can.
        rng = np.random.default_rng([seed, concurrency])
        osl = workload.osl.p50
        if p.running == 0:
            # Nothing was admitted, so nothing completed. duration_s still has to be
            # positive -- every rate in compute_metrics divides by it -- and one second
            # of a fully failed run is as good a stand-in as any.
            failed = [
                RequestRecord(ttft_s=0.0, itl_s=[], output_tokens=0, ok=False)
                for _ in range(num_requests)
            ]
            return LoadResult(concurrency=concurrency, duration_s=1.0, requests=failed)
        base_itl = max(p.step_floor_s * 0.5, p.itl_mean_s - p.n_spikes * p.itl_spike_s / osl)
        reqs: list[RequestRecord] = []
        for _ in range(num_requests):
            ttft = p.ttft_s * (1 + NOISE * rng.standard_normal())
            itl = base_itl * (1 + NOISE * rng.standard_normal(osl))
            if p.n_spikes:
                idx = rng.choice(osl, size=p.n_spikes, replace=False)
                itl[idx] += p.itl_spike_s
            reqs.append(
                RequestRecord(
                    ttft_s=float(max(ttft, 1e-4)),
                    itl_s=[float(x) for x in itl],
                    output_tokens=osl,
                )
            )
        # Closed loop: ``running`` requests are in service at once and each takes
        # ``lifetime_s``, so completions retire at running/lifetime_s. Queue wait is the
        # time the *waiting* requests spend outside the server -- it lengthens each
        # request's residence time, not the rate the server clears them -- so adding it
        # here would double-count it. Little's law is the check: the resulting rate times
        # (queue_wait_s + lifetime_s) comes back to exactly ``concurrency``.
        duration = num_requests * p.lifetime_s / p.running
        return LoadResult(concurrency=concurrency, duration_s=float(duration), requests=reqs)


class MockAdapter(EngineAdapter):
    name = "mock"

    def version(self) -> EngineVersion:
        return EngineVersion(name="mock", version="1.0", commit="sim")

    @staticmethod
    def _default_max_model_len(ctx: RunContext) -> KnobValue:
        """Shortest offered context that still covers the workload.

        This is workload-aware, not hardware-aware: it looks only at the workload's
        p99 ISL plus p50 OSL, so ``validate`` (which rejects a max_model_len below the
        workload) is satisfied and ``PerfModel.check_launch`` (which rejects one the KV
        cache cannot hold) is given the most headroom the choices allow. It does *not*
        guarantee a launch -- a small card with a long workload can still OOM here, and
        that OOM is a real finding for the search to work around, not a bug.
        """
        need = ctx.workload.isl.p99 + ctx.workload.osl.p50
        for choice in MAX_MODEL_LEN_CHOICES:
            if isinstance(choice, int) and choice >= need:
                return choice
        return MAX_MODEL_LEN_CHOICES[-1]

    def knob_space(self, ctx: RunContext) -> KnobSpace:
        d = DEFAULT_KNOBS
        return KnobSpace(
            knobs=[
                Knob(
                    name="max_num_seqs",
                    kind="int",
                    groups=["kv", "decode", "sched"],
                    default=d["max_num_seqs"],
                    low=8,
                    high=1024,
                    log=True,
                ),
                Knob(
                    name="max_num_batched_tokens",
                    kind="int",
                    groups=["prefill"],
                    default=d["max_num_batched_tokens"],
                    low=512,
                    high=16384,
                    log=True,
                ),
                Knob(
                    name="gpu_memory_utilization",
                    kind="float",
                    groups=["kv"],
                    default=d["gpu_memory_utilization"],
                    low=0.7,
                    high=0.95,
                    step=0.05,
                ),
                Knob(
                    name="max_model_len",
                    kind="cat",
                    groups=["kv"],
                    default=self._default_max_model_len(ctx),
                    choices=MAX_MODEL_LEN_CHOICES,
                ),
                Knob(
                    name="enable_prefix_caching",
                    kind="bool",
                    groups=["kv"],
                    default=d["enable_prefix_caching"],
                ),
                Knob(
                    name="enable_chunked_prefill",
                    kind="bool",
                    groups=["prefill"],
                    default=d["enable_chunked_prefill"],
                ),
                Knob(
                    name="kv_cache_dtype",
                    kind="cat",
                    groups=["kv", "decode"],
                    default=d["kv_cache_dtype"],
                    choices=["auto", "fp8"],
                ),
                Knob(
                    name="enforce_eager", kind="bool", groups=["sched"], default=d["enforce_eager"]
                ),
                Knob(
                    name="speculative",
                    kind="cat",
                    groups=["decode"],
                    default=d["speculative"],
                    choices=["none", "ngram", "eagle3"],
                ),
                Knob(
                    name="quantization",
                    kind="cat",
                    groups=["prefill", "decode"],
                    default=d["quantization"],
                    choices=["none", "fp8"],
                ),
            ]
        )

    @staticmethod
    def _numeric_errors(knob: Knob, value: KnobValue) -> list[str]:
        """Type and range complaints about one int/float knob. Never raises."""
        num = _as_number(value)
        if num is None:
            return [f"{knob.name}={value!r} is not a number"]
        if knob.name in INT_KNOBS and not float(num).is_integer():
            return [f"{knob.name}={value!r} is not an int"]
        if knob.low is not None and knob.high is not None and not knob.low <= num <= knob.high:
            return [f"{knob.name}={value} outside [{knob.low}, {knob.high}]"]
        return []

    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """Every rejection reason for ``cfg``, as strings. This must never raise.

        A caller hands us whatever the search or a user's YAML produced, and a
        malformed knob is exactly what validation exists to report -- so a bad value
        has to come back in the returned list, not out of the stack.
        """
        errs: list[str] = []
        for knob in self.knob_space(ctx).knobs:
            if knob.name not in cfg.knobs:
                continue
            value = cfg.knobs[knob.name]
            if knob.kind == "cat":
                if value not in knob.choices:
                    errs.append(f"{knob.name}={value!r} is not one of {knob.choices!r}")
            elif knob.kind == "bool":
                try:
                    _as_bool(value)
                except ValueError:
                    errs.append(
                        f"{knob.name}={value!r} is not a bool; "
                        f"choices: ['true', 'false', '1', '0', 'yes', 'no']"
                    )
            else:
                errs.extend(self._numeric_errors(knob, value))
        if cfg.knobs.get("quantization") == "fp8" and ctx.hw.compute_capability < 8.9:
            errs.append("fp8 quantization needs compute capability >= 8.9")
        # Only meaningful once max_model_len is known to be one of the offered lengths;
        # the categorical check above has already reported anything else.
        max_len = cfg.knobs.get("max_model_len", MAX_MODEL_LEN_CHOICES[-1])
        if (
            max_len in MAX_MODEL_LEN_CHOICES
            and int(max_len) < ctx.workload.isl.p99 + ctx.workload.osl.p50
        ):
            errs.append("max_model_len shorter than workload p99 ISL + OSL")
        return errs

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        pm = PerfModel(ctx.hw, ctx.model, ctx.workload, cfg.knobs)
        try:
            pm.check_launch()
        except OomError as e:
            raise LaunchError(ExitInfo(code=1, log_tail=str(e))) from e
        return ServerHandle(url="mock://", state=MockState(pm))

    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        return True

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        return SimLoadGenerator(_state(handle))

    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        """Snapshot of the most recent load point, or ``{}`` before any load has run.

        The simulator has no counters accumulating between calls: each ``run`` replaces
        the stored ``SimPoint``, so scraping twice in a row returns the same numbers
        rather than a fresh delta.
        """
        st = _state(handle)
        p = st.last
        if p is None:
            return {}
        w = st.pm.workload
        rate = p.running / p.lifetime_s if p.lifetime_s > 0 else 0.0
        hit = w.prefix_share if _as_bool(st.pm.knobs["enable_prefix_caching"]) else 0.0
        return {
            "kv_usage_p95": p.kv_usage,
            "num_waiting": float(p.waiting),
            "num_running": float(p.running),
            "preemptions_per_s": p.preempt_frac * rate,
            "queue_time_p90_s": p.queue_wait_s,
            "prefill_time_p50_s": p.ttft_s - p.queue_wait_s,
            "prefill_share": p.prefill_share,
            "prefix_hit_rate": hit,
            "max_num_seqs": float(st.pm.knobs["max_num_seqs"]),
            "kv_dtype_bytes": float(st.pm.kv_dtype_bytes),
        }

    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        p = _state(handle).last
        return {"sm_active": p.sm_active, "dram_active": p.dram_active} if p else {}

    def stop(self, handle: ServerHandle) -> ExitInfo:
        return ExitInfo(code=0)

    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(cfg.knobs.items()))
        return dict(cfg.knobs), f"mock-serve {ctx.model.id} {flags}"
```

`src/infervolt/engines/mock/scenarios.py`:
```python
"""Injected-bottleneck scenarios. Used by tests and by the README demo.

Each entry pins hardware, model and workload so that exactly one bottleneck dominates:
the diagnosis rules are expected to name ``expected`` when run against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infervolt.core.types import KnobValue, RunContext
from infervolt.hardware.profiles import get_profile
from infervolt.models.catalog import get_model_info
from infervolt.workloads.presets import get_workload, parse_slo


@dataclass
class Scenario:
    name: str
    hardware: str
    model: str
    workload: str
    slo: str
    expected: str
    baseline: dict[str, KnobValue] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    "kv": Scenario(
        name="kv",
        hardware="rtx4090-24",
        model="mock/qwen3-8b",
        workload="chat-4k-512",
        slo="ttft=600ms,itl=30ms",
        expected="kv_capacity",
    ),
    "decode": Scenario(
        name="decode",
        hardware="a100-80",
        model="mock/qwen3-8b",
        workload="chat-256-512",
        slo="ttft=300ms,itl=30ms",
        expected="decode_bandwidth",
    ),
    "prefill": Scenario(
        name="prefill",
        hardware="h100-80",
        model="mock/qwen3-8b",
        workload="rag-16k-64",
        slo="ttft=1500ms,itl=50ms",
        expected="prefill_compute",
    ),
    "sched": Scenario(
        name="sched",
        hardware="a100-80",
        model="mock/qwen3-0.6b",
        workload="chat-1k-128",
        slo="ttft=200ms,itl=5ms",
        expected="scheduler_cpu",
        baseline={"enforce_eager": True},
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as e:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}") from e


def make_context(name: str, run_dir: str, run_id: str = "test", seed: int = 7) -> RunContext:
    s = get_scenario(name)
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        hw=get_profile(s.hardware),
        model=get_model_info(s.model),
        workload=get_workload(s.workload),
        slo=parse_slo(s.slo),
        seed=seed,
    )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/engines -q && uv run ruff check . && uv run mypy src`
Expected: all pass (including the registry test from Task 9). If `entry_points` does not see `mock`, re-run `uv sync` so the editable install refreshes metadata.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/engines/mock tests/engines/mock
git commit -m "feat(mock): adapter, simulated load generator, four bottleneck scenarios"
```

---

### Task 12: Trial runner (launch → load → scrape → stop → classify)

**Files:**
- Create: `src/infervolt/runner/__init__.py`, `src/infervolt/runner/trial.py`, `tests/runner/__init__.py`, `tests/runner/test_trial.py`

- [ ] **Step 1: Write the failing tests**

`tests/runner/__init__.py`: empty.

`tests/runner/test_trial.py`:
```python
from infervolt.core.types import Candidate, EngineConfig, Trial
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.runner.trial import run_candidate, run_sweep


def _trial(knobs: dict[str, object], idx: int = 0) -> Trial:
    cand = Candidate(id=f"c{idx}", config=EngineConfig(engine="mock", knobs=knobs), origin="baseline")
    return Trial(id=f"t{idx}", run_id="r", index=idx, candidate=cand)


def test_sweep_stops_after_goodput_collapses() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    handle = adapter.launch(cfg, ctx)
    obs, cost_s = run_sweep(adapter, handle, ctx, [1, 4, 16, 64, 128], num_requests=16)
    assert obs[0].load_point == 1 and len(obs) < 5
    assert all(o.valid for o in obs) and cost_s > 0
    assert obs[-1].engine["num_waiting"] > 0


def test_run_candidate_ok_sets_objective_and_best_load_point() -> None:
    adapter, ctx = MockAdapter(), make_context("decode", run_dir="/tmp/x")
    t = run_candidate(adapter, _trial(adapter.knob_space(ctx).defaults()), ctx, [1, 4, 16, 64], num_requests=16)
    assert t.status == "ok" and t.result is not None
    assert t.result.objective == max(o.metrics.goodput_rps for o in t.result.observations)
    assert t.result.best_load_point in {o.load_point for o in t.result.observations}
    assert t.cost_usd > 0 and t.started is not None and t.ended is not None


def test_run_candidate_oom_is_infeasible() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), "max_model_len": 32768}
    t = run_candidate(adapter, _trial(knobs), ctx, [1, 4], num_requests=8)
    assert t.status == "infeasible_oom" and t.crash_kind == "oom" and "KV cache" in t.log_tail


def test_run_candidate_static_rejection_never_launches() -> None:
    adapter, ctx = MockAdapter(), make_context("decode", run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), "quantization": "fp8"}
    t = run_candidate(adapter, _trial(knobs), ctx, [1], num_requests=8)
    assert t.status == "rejected" and "compute capability" in t.log_tail
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runner -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the runner**

`src/infervolt/runner/__init__.py`: empty.

`src/infervolt/runner/trial.py`:
```python
"""Execute one candidate: static validation, launch, sweep, scrape, stop, crash classification.

Every exit from :func:`run_candidate` is a ``Trial`` with a terminal status. Nothing
propagates: a candidate that OOMs, crashes, hangs or is statically rejected is a *result*
the search has to learn from, not an error the caller has to handle.
"""

from __future__ import annotations

import time

from infervolt.core.types import EngineConfig, Observation, Result, RunContext, Trial
from infervolt.engines.base import EngineAdapter, LaunchError, ServerHandle
from infervolt.loadgen.analysis import compute_metrics

# Above any of these the load generator, not the server, is the thing being measured.
CLIENT_CPU_MAX = 0.8
CLIENT_LAG_MAX_MS = 5.0
CLIENT_ERROR_MAX = 0.01
# Sweep stop rules: quit once goodput has fallen this far below the best seen, or once
# the server is shedding more than this share of requests.
SWEEP_COLLAPSE = 0.8
SWEEP_ERROR_MAX = 0.05


def run_load_point(
    adapter: EngineAdapter,
    handle: ServerHandle,
    ctx: RunContext,
    concurrency: int,
    num_requests: int,
) -> tuple[Observation, float]:
    """Drive one concurrency level and return the observation plus its wall-clock seconds."""
    lr = adapter.loadgen(handle, ctx).run(ctx.workload, concurrency, num_requests, ctx.seed)
    obs = Observation(
        load_point=concurrency,
        config=handle.config or EngineConfig(engine=adapter.name),
        metrics=compute_metrics(lr, ctx.slo, ctx.hw),
        engine=adapter.scrape(handle),
        gpu=adapter.gpu_stats(handle),
    )
    h = lr.health
    if not any(r.ok for r in lr.requests):
        # No latency percentile means anything here, and the sweep must not read the
        # resulting zeros as a healthy point that simply scored badly.
        obs.valid, obs.invalid_reason = False, "no successful requests at this load point"
    elif (
        h.worker_cpu > CLIENT_CPU_MAX
        or h.loop_lag_p99_ms > CLIENT_LAG_MAX_MS
        or h.error_rate > CLIENT_ERROR_MAX
    ):
        obs.valid = False
        obs.invalid_reason = (
            f"client artifact: cpu={h.worker_cpu:.2f} "
            f"lag_p99={h.loop_lag_p99_ms:.1f}ms err={h.error_rate:.3f}"
        )
    return obs, lr.duration_s


def run_sweep(
    adapter: EngineAdapter,
    handle: ServerHandle,
    ctx: RunContext,
    concurrencies: list[int],
    num_requests: int,
) -> tuple[list[Observation], float]:
    """Increase concurrency until goodput collapses, errors rise, or the client is the bottleneck.

    Returns every observation taken (including the one that triggered the stop) and the
    total load seconds, which is what the trial is billed for.
    """
    obs: list[Observation] = []
    total_s = 0.0
    best = 0.0
    for c in concurrencies:
        o, dur = run_load_point(adapter, handle, ctx, c, num_requests)
        obs.append(o)
        total_s += dur
        if not o.valid or o.metrics.error_rate > SWEEP_ERROR_MAX:
            break
        best = max(best, o.metrics.goodput_rps)
        if len(obs) > 1 and o.metrics.goodput_rps < SWEEP_COLLAPSE * best:
            break
    return obs, total_s


def run_candidate(
    adapter: EngineAdapter,
    trial: Trial,
    ctx: RunContext,
    concurrencies: list[int],
    num_requests: int,
    ready_timeout_s: float = 900.0,
) -> Trial:
    """Take one candidate from config to a finished trial, in place."""
    cfg = trial.candidate.config
    trial.started = time.time()
    trial.status = "running"
    errs = adapter.validate(cfg, ctx)
    if errs:
        trial.status, trial.log_tail, trial.ended = "rejected", "; ".join(errs), time.time()
        return trial
    try:
        handle = adapter.launch(cfg, ctx)
    except LaunchError as e:
        kind = adapter.classify_crash(e.exit)
        trial.crash_kind = kind
        trial.log_tail = e.exit.log_tail[-2000:]
        trial.status = "infeasible_oom" if kind == "oom" else "crash"
        trial.ended = time.time()
        return trial
    except Exception as e:  # noqa: BLE001 - an adapter bug is still just a failed trial
        # Adapters are contracted to raise LaunchError. One that does not is misbehaving,
        # but taking the whole run down over it would lose every trial already completed.
        trial.crash_kind = "startup"
        trial.log_tail = f"{type(e).__name__}: {e}"
        trial.status = "crash"
        trial.ended = time.time()
        return trial
    handle.config = cfg
    try:
        if not adapter.ready(handle, ready_timeout_s):
            trial.status, trial.crash_kind = "timeout", "timeout"
            return trial
        obs, load_s = run_sweep(adapter, handle, ctx, concurrencies, num_requests)
    finally:
        exit_info = adapter.stop(handle)
        trial.ended = time.time()
    kind = adapter.classify_crash(exit_info)
    if kind != "none":
        # The server died during the sweep; the numbers it produced cannot be trusted.
        trial.crash_kind, trial.log_tail = kind, exit_info.log_tail[-2000:]
        trial.status = "infeasible_oom" if kind == "oom" else "crash"
        return trial
    trial.cost_usd = ctx.hw.usd_per_hour * ctx.hw.count / 3600.0 * load_s
    trial.result = summarize(obs, ctx)
    trial.status = "ok"
    return trial


def summarize(obs: list[Observation], ctx: RunContext) -> Result:
    """Pick the load point with the highest goodput and score the candidate by it.

    Invalid observations are excluded from the choice but kept in ``observations``: the
    diagnosis rules want to see the point where the sweep stopped and why.
    """
    valid = [o for o in obs if o.valid]
    if not valid:
        return Result(
            observations=obs,
            objective=0.0,
            feasible=True,
            slo_met=False,
            best_load_point=obs[0].load_point if obs else 0,
        )
    best = max(valid, key=lambda o: o.metrics.goodput_rps)
    return Result(
        observations=obs,
        objective=best.metrics.goodput_rps,
        feasible=True,
        slo_met=best.metrics.goodput_frac >= ctx.slo.goodput_target,
        best_load_point=best.load_point,
    )
```

Replace the `_cfg_of` placeholder with a clean approach: `run_load_point` needs the config for the observation. Give `ServerHandle` a `config` attribute instead of the `getattr` hack: in `src/infervolt/engines/base.py` change the dataclass to

```python
@dataclass
class ServerHandle:
    url: str
    state: Any = None
    config: EngineConfig | None = None
```

and in `run_load_point` use `config=handle.config or EngineConfig(engine=adapter.name)`. Delete `_cfg_of` and the `# type: ignore[attr-defined]` line (set `handle.config = cfg` plainly after launch).

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/runner tests/engines -q && uv run ruff check . && uv run mypy src`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/runner src/infervolt/engines/base.py tests/runner
git commit -m "feat(runner): candidate execution with sweep, crash classification, cost accounting"
```

---

### Task 13: Diagnosis rules R0–R6

**Files:**
- Create: `src/infervolt/diagnose/__init__.py`, `src/infervolt/diagnose/rules.py`, `tests/diagnose/__init__.py`, `tests/diagnose/test_rules.py`

- [ ] **Step 1: Write the failing tests**

`tests/diagnose/__init__.py`: empty.

`tests/diagnose/test_rules.py`:
```python
import pytest

from infervolt.core.types import Candidate, EngineConfig, Observation, Trial
from infervolt.diagnose.rules import evaluate_rules
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import SCENARIOS, make_context
from infervolt.runner.trial import run_candidate


def _baseline_obs(name: str) -> tuple[list[Observation], EngineConfig]:
    adapter, ctx = MockAdapter(), make_context(name, run_dir="/tmp/x")
    knobs = {**adapter.knob_space(ctx).defaults(), **SCENARIOS[name].baseline}
    cfg = EngineConfig(engine="mock", knobs=knobs)
    t = Trial(id="t0", run_id="r", index=0, candidate=Candidate(id="c0", config=cfg, origin="baseline"))
    t = run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)
    assert t.result is not None
    return t.result.observations, cfg


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_top_finding_matches_injected_bottleneck(name: str) -> None:
    obs, cfg = _baseline_obs(name)
    ctx = make_context(name, run_dir="/tmp/x")
    findings = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert findings, "no findings"
    assert findings[0].bottleneck == SCENARIOS[name].expected, [(f.rule_id, f.score) for f in findings]
    assert findings[0].evidence and findings[0].subspaces


def test_client_artifact_invalidates() -> None:
    obs, cfg = _baseline_obs("decode")
    obs[-1].valid, obs[-1].invalid_reason = False, "client artifact: cpu=0.95"
    ctx = make_context("decode", run_dir="/tmp/x")
    findings = evaluate_rules(obs, ctx, cfg, MockAdapter().knob_space(ctx))
    assert findings[0].bottleneck == "client_artifact"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/diagnose -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the rules**

`src/infervolt/diagnose/__init__.py`: empty.

`src/infervolt/diagnose/rules.py`:
```python
"""Deterministic bottleneck rules. Each rule scores a fraction of weighted sub-conditions in [0, 1]
and attaches the evidence it used. Findings below MIN_SCORE are dropped. Sorting is by score, then
by BOTTLENECK_PRIORITY (capacity problems cap goodput before bandwidth does).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from infervolt.core.types import (
    BOTTLENECK_PRIORITY, EngineConfig, Evidence, Finding, KnobSpace, Observation, RunContext,
)
from infervolt.hardware import roofline

MIN_SCORE = 0.3


@dataclass
class RuleInput:
    obs: list[Observation]
    ctx: RunContext
    cfg: EngineConfig
    space: KnobSpace

    @property
    def valid(self) -> list[Observation]:
        return [o for o in self.obs if o.valid]

    @property
    def top(self) -> Observation:
        return self.valid[-1]

    @property
    def first(self) -> Observation:
        return self.valid[0]

    @property
    def best_slo(self) -> Observation:
        return max(self.valid, key=lambda o: o.metrics.goodput_rps)


def _ev(o: Observation, key: str, source: str = "engine", unit: str = "") -> Evidence:
    table = o.engine if source == "engine" else o.gpu if source == "gpu" else o.metrics.model_dump()
    return Evidence(source=source, key=f"{key}@c{o.load_point}", value=float(table.get(key, 0.0)), unit=unit)


def _score(conds: list[tuple[bool, float]]) -> float:
    return round(sum(w for ok, w in conds if ok), 3)


def _subspaces(space: KnobSpace, *groups: str) -> list[str]:
    return [g for g in groups if g in space.groups()]


# ---- rules


def r0_under_loaded(x: RuleInput) -> Finding | None:
    t = x.top
    e, g, m = t.engine, t.gpu, t.metrics
    conds = [
        (e.get("num_waiting", 0) < 0.5, 0.25),
        (e.get("num_running", 0) < 0.5 * e.get("max_num_seqs", 1), 0.25),
        (g.get("sm_active", 1) < 0.3, 0.25),
        (m.goodput_frac >= x.ctx.slo.goodput_target, 0.25),
    ]
    return Finding(
        rule_id="R0", bottleneck="under_loaded", score=_score(conds),
        evidence=[_ev(t, "num_waiting"), _ev(t, "num_running"), _ev(t, "sm_active", "gpu")],
        subspaces=[], summary="Server is not saturated at the highest load point; extend the sweep.",
    )


def r1_kv_capacity(x: RuleInput) -> Finding | None:
    worst = max(x.valid, key=lambda o: o.engine.get("kv_usage_p95", 0))
    e = worst.engine
    conds = [
        (any(o.engine.get("preemptions_per_s", 0) > 0 for o in x.valid), 0.5),
        (e.get("kv_usage_p95", 0) > 0.9 and e.get("num_waiting", 0) > 0, 0.3),
        (e.get("num_running", 0) < e.get("max_num_seqs", 0) and e.get("num_waiting", 0) > 0, 0.2),
    ]
    return Finding(
        rule_id="R1", bottleneck="kv_capacity", score=_score(conds),
        evidence=[_ev(worst, "preemptions_per_s", unit="/s"), _ev(worst, "kv_usage_p95"), _ev(worst, "num_waiting"), _ev(worst, "num_running")],
        subspaces=_subspaces(x.space, "kv"),
        summary="KV cache is exhausted: requests queue or get preempted before max_num_seqs is reached.",
    )


def r2_decode_bandwidth(x: RuleInput) -> Finding | None:
    o = x.best_slo
    n = max(1, int(o.engine.get("num_running", 1)))
    w = x.ctx.workload
    floor = roofline.decode_step_floor_s(x.ctx.hw, x.ctx.model, n, w.isl.p50 + w.osl.p50 // 2, int(o.engine.get("kv_dtype_bytes", 2)))
    ratio = (o.metrics.itl_p50_ms / 1000) / floor if floor > 0 else 99.0
    top, first = x.top, x.first
    c_ratio = top.load_point / max(first.load_point, 1)
    itl_ratio = top.metrics.itl_p50_ms / max(first.metrics.itl_p50_ms, 1e-6)
    conds = [
        (ratio <= 1.3, 0.5),
        (o.gpu.get("dram_active", 0) > 0.6 and o.gpu.get("sm_active", 1) < 0.5, 0.3),
        (c_ratio > 1 and itl_ratio < 0.5 * c_ratio, 0.2),
    ]
    score = _score(conds)
    if any(ob.engine.get("kv_usage_p95", 0) > 0.9 for ob in x.valid):
        score = round(score * 0.7, 3)
    return Finding(
        rule_id="R2", bottleneck="decode_bandwidth", score=score,
        evidence=[Evidence(source="roofline", key=f"itl_over_floor@c{o.load_point}", value=round(ratio, 3)), _ev(o, "dram_active", "gpu"), _ev(o, "sm_active", "gpu")],
        subspaces=_subspaces(x.space, "decode"),
        summary="Decode runs at the HBM-bandwidth floor: fewer bytes per step (spec decode, FP8 KV, quantization) is the lever.",
    )


def r3_prefill_compute(x: RuleInput) -> Finding | None:
    top, first = x.top, x.first
    c_ratio = top.load_point / max(first.load_point, 1)
    ttft_ratio = top.metrics.ttft_p90_ms / max(first.metrics.ttft_p90_ms, 1e-6)
    conds = [
        (c_ratio > 1 and ttft_ratio >= 0.5 * c_ratio, 0.4),
        (top.engine.get("prefill_share", 0) >= 0.5, 0.4),
        (top.gpu.get("sm_active", 0) >= 0.7, 0.2),
    ]
    return Finding(
        rule_id="R3", bottleneck="prefill_compute", score=_score(conds),
        evidence=[Evidence(source="loadgen", key=f"ttft_p90_growth@c{top.load_point}", value=round(ttft_ratio, 3)), _ev(top, "prefill_share"), _ev(top, "sm_active", "gpu")],
        subspaces=_subspaces(x.space, "prefill"),
        summary="Prefill compute dominates: TTFT grows with concurrency and the GPU is busy on prompt tokens.",
    )


def r4_scheduler_cpu(x: RuleInput) -> Finding | None:
    low = [o for o in x.valid if o.load_point <= 8]
    if len(low) < 2:
        return None
    o1, o8 = low[0], low[-1]
    w = x.ctx.workload
    floor1 = roofline.decode_step_floor_s(x.ctx.hw, x.ctx.model, 1, w.isl.p50 + w.osl.p50 // 2, int(o1.engine.get("kv_dtype_bytes", 2)))
    flat = abs(o8.metrics.itl_p50_ms - o1.metrics.itl_p50_ms) / max(o1.metrics.itl_p50_ms, 1e-6) <= 0.15
    conds = [
        (flat, 0.4),
        (o8.gpu.get("sm_active", 1) < 0.4 and o8.gpu.get("dram_active", 1) < 0.4, 0.3),
        (o1.metrics.itl_p50_ms / 1000 > 2 * floor1, 0.3),
    ]
    return Finding(
        rule_id="R4", bottleneck="scheduler_cpu", score=_score(conds),
        evidence=[Evidence(source="loadgen", key="itl_p50_ms@c1", value=round(o1.metrics.itl_p50_ms, 3), unit="ms"), Evidence(source="loadgen", key=f"itl_p50_ms@c{o8.load_point}", value=round(o8.metrics.itl_p50_ms, 3), unit="ms"), _ev(o8, "sm_active", "gpu"), _ev(o8, "dram_active", "gpu")],
        subspaces=_subspaces(x.space, "sched"),
        summary="Per-step overhead dominates: ITL is flat across low concurrency while the GPU idles.",
    )


def r5_communication(x: RuleInput) -> Finding | None:
    tp = int(x.cfg.knobs.get("tensor_parallel_size", 1))
    conds = [(tp > 1 and x.ctx.hw.interconnect == "pcie", 0.3)]
    return Finding(
        rule_id="R5", bottleneck="communication", score=_score(conds),
        evidence=[Evidence(source="static", key="tensor_parallel_size", value=float(tp))],
        subspaces=_subspaces(x.space, "parallel"),
        summary="Tensor parallel over PCIe; all-reduce likely dominates (low confidence without a profile).",
    )


def r6_client_artifact(x: RuleInput) -> Finding | None:
    bad = [o for o in x.obs if not o.valid]
    if not bad:
        return None
    return Finding(
        rule_id="R6", bottleneck="client_artifact", score=1.0,
        evidence=[Evidence(source="loadgen", key=f"invalid@c{o.load_point}", value=1.0, note=o.invalid_reason) for o in bad],
        subspaces=[], summary="Load generator was the bottleneck; measurements at those load points are invalid.",
    )


RULES: list[Callable[[RuleInput], Finding | None]] = [
    r6_client_artifact, r1_kv_capacity, r3_prefill_compute, r2_decode_bandwidth, r4_scheduler_cpu, r5_communication, r0_under_loaded,
]


def evaluate_rules(obs: list[Observation], ctx: RunContext, cfg: EngineConfig, space: KnobSpace) -> list[Finding]:
    x = RuleInput(obs=obs, ctx=ctx, cfg=cfg, space=space)
    if not x.valid and not any(not o.valid for o in obs):
        return []
    findings = [f for rule in RULES if x.valid or rule is r6_client_artifact for f in [rule(x)] if f and f.score >= MIN_SCORE]
    findings.sort(key=lambda f: (-f.score, BOTTLENECK_PRIORITY[f.bottleneck]))
    return findings
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/diagnose -q && uv run ruff check . && uv run mypy src`
Expected: `5 passed`. If a scenario's top finding is wrong, print `[(f.rule_id, f.score, f.evidence)]` and fix the *scenario* (hardware/workload/SLO in `scenarios.py`) or a threshold constant; do not special-case the rules for the mock.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/diagnose tests/diagnose
git commit -m "feat(diagnose): rules R0-R6 with evidence and sub-space hints"
```

---

### Task 14: LLM client protocol, output schemas, prompts, Fake client

**Files:**
- Create: `src/infervolt/llm/__init__.py`, `src/infervolt/llm/base.py`, `src/infervolt/llm/prompts/rank.j2`, `src/infervolt/llm/prompts/plan.j2`, `src/infervolt/llm/prompts/emit.j2`, `src/infervolt/llm/fake.py`, `tests/llm/__init__.py`, `tests/llm/test_fake.py`

- [ ] **Step 1: Write the failing tests**

`tests/llm/__init__.py`: empty.

`tests/llm/test_fake.py`:
```python
from infervolt.llm.base import (
    SYSTEM_PROMPT, DiagnosisOut, NarrativeOut, SearchPlanOut, extract_context, prompts_sha, render_prompt,
)
from infervolt.llm.fake import FakeLLMClient

FINDINGS = [
    {"rule_id": "R1", "bottleneck": "kv_capacity", "score": 1.0, "summary": "KV exhausted", "subspaces": ["kv"], "evidence": []},
    {"rule_id": "R2", "bottleneck": "decode_bandwidth", "score": 0.7, "summary": "at floor", "subspaces": ["decode"], "evidence": []},
]
KNOBS = [{"name": "kv_cache_dtype", "kind": "cat", "groups": ["kv", "decode"], "choices": ["auto", "fp8"], "default": "auto"},
         {"name": "gpu_memory_utilization", "kind": "float", "groups": ["kv"], "low": 0.7, "high": 0.95, "default": 0.9}]


def test_render_and_extract_context_roundtrip() -> None:
    user = render_prompt("rank", {"findings": FINDINGS, "workload": {"name": "chat-4k-512"}})
    assert "<context>" in user and "R1" in user
    assert extract_context(user)["findings"][0]["rule_id"] == "R1"
    assert len(prompts_sha()) == 12


def test_fake_ranks_by_rule_order() -> None:
    llm = FakeLLMClient()
    user = render_prompt("rank", {"findings": FINDINGS, "workload": {}, "slo": {}, "metrics": []})
    out = llm.structured(system=SYSTEM_PROMPT, user=user, schema=DiagnosisOut)
    assert out.primary_rule_id == "R1" and out.ranked_rule_ids == ["R1", "R2"]
    assert 0.5 <= out.confidence <= 0.95


def test_fake_plans_priors_only_from_known_knobs() -> None:
    llm = FakeLLMClient()
    ctx = {"diagnosis": {"primary": "kv_capacity", "subspaces": ["kv"]}, "knob_space": KNOBS,
           "current": {"kv_cache_dtype": "auto", "gpu_memory_utilization": 0.9}, "budget": {"max_trials": 9}, "priors": [], "notes": []}
    out = llm.structured(system=SYSTEM_PROMPT, user=render_prompt("plan", ctx), schema=SearchPlanOut)
    assert out.subspaces == ["kv"] and out.max_trials == 9
    assert out.priors and all(set(p.knobs) <= {"kv_cache_dtype", "gpu_memory_utilization"} for p in out.priors)
    assert any(p.knobs.get("kv_cache_dtype") == "fp8" for p in out.priors)


def test_fake_narrative_mentions_winning_knobs() -> None:
    llm = FakeLLMClient()
    ctx = {"diagnosis": {"primary": "kv_capacity", "rationale": "KV exhausted"},
           "baseline_metrics": {"goodput_rps": 1.0}, "best_metrics": {"goodput_rps": 1.8},
           "winning_knobs": {"kv_cache_dtype": "fp8"}, "trial_ids": ["t3"]}
    out = llm.structured(system=SYSTEM_PROMPT, user=render_prompt("emit", ctx), schema=NarrativeOut)
    assert "kv_cache_dtype" in out.rationale and out.next_steps
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/llm -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write base, prompts, fake**

`src/infervolt/llm/__init__.py`: empty.

`src/infervolt/llm/base.py`:
```python
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


_env = Environment(loader=PackageLoader("infervolt.llm", "prompts"), autoescape=select_autoescape(default=False), trim_blocks=True, lstrip_blocks=True)


def render_prompt(name: str, context: dict[str, Any]) -> str:
    return _env.get_template(f"{name}.j2").render(context_json=json.dumps(context, indent=1, sort_keys=True, default=str))


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
```

`src/infervolt/llm/prompts/rank.j2`:
```jinja
Rank the bottleneck findings for this serving run.

The deterministic rules already scored each finding from evidence. Your job: choose the PRIMARY
bottleneck (the one that caps goodput under the SLO first), order the rest, explain in <=120 words
citing evidence keys, and list caveats (e.g. measurement artifacts, low load). Use only rule ids
present in the context. Output schema: {"primary_rule_id", "ranked_rule_ids", "rationale",
"confidence" (0-1), "caveats"}.

<context>
{{ context_json }}
</context>
```

`src/infervolt/llm/prompts/plan.j2`:
```jinja
Plan a targeted configuration search for the diagnosed bottleneck.

Pick the sub-spaces (groups) worth searching, propose up to 4 prior candidates as knob dicts using
ONLY knob names from `knob_space` with values inside their bounds/choices, give each a one-line
hypothesis, and set max_trials <= budget.max_trials. Prefer changes that attack the primary
bottleneck; keep the search narrow. Output schema: {"subspaces", "priors": [{"knobs", "hypothesis"}],
"max_trials", "rationale"}.

<context>
{{ context_json }}
</context>
```

`src/infervolt/llm/prompts/emit.j2`:
```jinja
Write the recipe narrative for a verified optimization.

Explain in <=200 words why the winning knobs fix the diagnosed bottleneck, referencing the metric
deltas; propose <=3 next steps; and record <=5 reusable insights, each citing trial ids from
`trial_ids`. Output schema: {"rationale", "next_steps", "insights": [{"text", "cites"}]}.

<context>
{{ context_json }}
</context>
```

`src/infervolt/llm/fake.py`:
```python
"""Deterministic stand-in for an LLM. Reads the <context> JSON and applies fixed heuristics.
Used in CI and as the offline default so the whole loop runs without any API key."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.core.types import KnobValue
from infervolt.llm.base import DiagnosisOut, InsightOut, NarrativeOut, PriorOut, SearchPlanOut, extract_context

T = TypeVar("T", bound=BaseModel)

PRIORS: dict[str, list[tuple[dict[str, KnobValue], str]]] = {
    "kv_capacity": [
        ({"kv_cache_dtype": "fp8"}, "FP8 KV halves bytes per token, doubling KV capacity"),
        ({"gpu_memory_utilization": 0.95}, "Give the KV cache more of the GPU memory"),
        ({"kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.95}, "Both KV levers together"),
    ],
    "decode_bandwidth": [
        ({"speculative": "ngram"}, "N-gram speculation amortizes weight reads over several tokens"),
        ({"speculative": "eagle3"}, "EAGLE-3 draft head gives higher acceptance than n-gram"),
        ({"kv_cache_dtype": "fp8"}, "FP8 KV reduces bytes streamed per decode step"),
    ],
    "prefill_compute": [
        ({"quantization": "fp8"}, "FP8 GEMMs double prefill throughput on Hopper/Ada"),
        ({"max_num_batched_tokens": 8192}, "Larger prefill chunks cut per-chunk scheduling overhead"),
    ],
    "scheduler_cpu": [
        ({"enforce_eager": False}, "CUDA graphs remove per-step launch overhead"),
        ({"enforce_eager": False, "max_num_seqs": 64}, "Graphs plus a smaller batch cap for lower scheduling cost"),
    ],
}


class FakeLLMClient:
    model_id = "fake"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        ctx = extract_context(user)
        if schema is DiagnosisOut:
            return schema.model_validate(self._diagnose(ctx).model_dump())
        if schema is SearchPlanOut:
            return schema.model_validate(self._plan(ctx).model_dump())
        if schema is NarrativeOut:
            return schema.model_validate(self._narrate(ctx).model_dump())
        raise TypeError(f"FakeLLMClient cannot produce {schema.__name__}")

    def _diagnose(self, ctx: dict[str, Any]) -> DiagnosisOut:
        findings = ctx["findings"]
        top = findings[0]
        return DiagnosisOut(
            primary_rule_id=top["rule_id"],
            ranked_rule_ids=[f["rule_id"] for f in findings],
            rationale=f"Rule {top['rule_id']} ({top['bottleneck']}) has the highest evidence score "
            f"{top['score']}: {top['summary']}",
            confidence=min(0.95, 0.5 + float(top["score"]) / 2),
            caveats=["fake-llm: ranking follows rule scores"],
        )

    def _plan(self, ctx: dict[str, Any]) -> SearchPlanOut:
        primary = ctx["diagnosis"]["primary"]
        names = {k["name"] for k in ctx["knob_space"]}
        current = ctx.get("current", {})
        priors = []
        for knobs, hyp in PRIORS.get(primary, []):
            kept = {k: v for k, v in knobs.items() if k in names and current.get(k) != v}
            if kept:
                priors.append(PriorOut(knobs=kept, hypothesis=hyp))
        return SearchPlanOut(
            subspaces=list(ctx["diagnosis"]["subspaces"]), priors=priors[:4],
            max_trials=int(ctx["budget"]["max_trials"]), rationale=f"fake-llm: search the {primary} sub-space",
        )

    def _narrate(self, ctx: dict[str, Any]) -> NarrativeOut:
        d, base, best = ctx["diagnosis"], ctx["baseline_metrics"], ctx["best_metrics"]
        knobs = ", ".join(f"{k}={v}" for k, v in ctx["winning_knobs"].items())
        g0, g1 = float(base.get("goodput_rps", 0)), float(best.get("goodput_rps", 0))
        pct = (g1 - g0) / g0 * 100 if g0 else 0.0
        return NarrativeOut(
            rationale=f"Primary bottleneck {d['primary']}: {d.get('rationale', '')} Changing {knobs} "
            f"raised goodput from {g0:.3f} to {g1:.3f} rps ({pct:+.0f}%).",
            next_steps=["Re-run diagnosis on the tuned config; the next bottleneck may differ.",
                        "Validate on the real engine and hardware before deploying."],
            insights=[InsightOut(text=f"For {d['primary']}, {knobs} helped.", cites=list(ctx.get("trial_ids", [])))],
        )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/llm -q && uv run ruff check . && uv run mypy src`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/llm tests/llm
git commit -m "feat(llm): client protocol, structured schemas, prompt templates, fake client"
```

---

### Task 15: Anthropic, OpenAI-compatible, and replay clients + factory

**Files:**
- Create: `src/infervolt/llm/anthropic_client.py`, `src/infervolt/llm/openai_compat.py`, `src/infervolt/llm/replay.py`, `src/infervolt/llm/factory.py`, `tests/llm/test_clients.py`

- [ ] **Step 1: Write the failing tests**

`tests/llm/test_clients.py`:
```python
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
    out = AnthropicClient(model_id="claude-opus-5", client=stub).structured(system="s", user="u", schema=DiagnosisOut)
    assert out == GOOD
    call = stub.calls[0]
    assert call["model"] == "claude-opus-5" and call["output_format"] is DiagnosisOut and call["system"] == "s"


def test_anthropic_refusal_raises() -> None:
    with pytest.raises(LLMError):
        AnthropicClient(client=_AnthropicStub("refusal")).structured(system="s", user="u", schema=DiagnosisOut)


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
    out = OpenAICompatClient(model_id="m", client=stub).structured(system="s", user="u", schema=DiagnosisOut)
    assert out == GOOD and len(stub.calls) == 2
    assert stub.calls[0]["response_format"]["type"] == "json_schema"


def test_openai_compat_gives_up_after_two_failures() -> None:
    with pytest.raises(LLMError):
        OpenAICompatClient(model_id="m", client=_OpenAIStub(["x", "y"])).structured(system="s", user="u", schema=DiagnosisOut)


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
    assert isinstance(make_llm("fake", Settings(home=tmp_path, llm_cassette=tmp_path / "c.json")), ReplayLLMClient)
    with pytest.raises(KeyError):
        make_llm("nope", s)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/llm/test_clients.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the clients**

`src/infervolt/llm/anthropic_client.py`:
```python
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
        self._client = client

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
```

`src/infervolt/llm/openai_compat.py`:
```python
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
        self, model_id: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key)  # api_key is a plain str here
        self._client = client

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        fmt = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
        last_error = ""
        for _ in range(2):
            resp = self._client.chat.completions.create(model=self.model_id, messages=messages, response_format=fmt)
            text = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_FENCE.sub("", text.strip()))
            except (ValidationError, ValueError) as e:
                last_error = str(e)
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was not valid {schema.__name__} JSON: {last_error}. Reply with only the corrected JSON."},
                ]
        raise LLMError(f"could not obtain valid {schema.__name__}: {last_error}")
```

`src/infervolt/llm/replay.py`:
```python
"""Cassette wrapper: record real LLM replies, replay them in CI without keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from infervolt.llm.base import LLMClient, LLMError

T = TypeVar("T", bound=BaseModel)


class ReplayLLMClient:
    def __init__(self, path: Path, inner: LLMClient | None) -> None:
        self.path = path
        self.inner = inner
        self.model_id = f"replay({inner.model_id if inner else 'offline'})"
        self._data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _key(system: str, user: str, schema: type[BaseModel]) -> str:
        return hashlib.sha256(f"{schema.__name__}\n{system}\n{user}".encode()).hexdigest()

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        key = self._key(system, user, schema)
        if key in self._data:
            return schema.model_validate(self._data[key])
        if self.inner is None:
            raise LLMError(f"no cassette entry for {schema.__name__} and no inner client")
        out = self.inner.structured(system=system, user=user, schema=schema)
        self._data[key] = out.model_dump(mode="json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1, sort_keys=True))
        return out
```

`src/infervolt/llm/factory.py`:
```python
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
        inner = OpenAICompatClient(model_id=settings.openai_model, base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
    else:
        raise KeyError(f"unknown llm {name!r}; use fake, anthropic, or openai")
    if settings.llm_cassette is not None:
        return ReplayLLMClient(settings.llm_cassette, inner=inner)
    return inner
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/llm -q && uv run ruff check . && uv run mypy src`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/llm tests/llm
git commit -m "feat(llm): Anthropic, OpenAI-compatible, replay clients and factory"
```

---

### Task 16: Search (Optuna TPE inside the LLM-chosen sub-space, crash-aware)

**Files:**
- Create: `src/infervolt/search/__init__.py`, `src/infervolt/search/space.py`, `src/infervolt/search/optuna_search.py`, `tests/search/__init__.py`, `tests/search/test_space.py`, `tests/search/test_optuna_search.py`

- [ ] **Step 1: Write the failing tests**

`tests/search/__init__.py`: empty.

`tests/search/test_space.py`:
```python
from infervolt.core.types import Knob, KnobSpace
from infervolt.search.space import Bounds, clamp, is_novel

SPACE = KnobSpace(knobs=[
    Knob(name="gpu_memory_utilization", kind="float", groups=["kv"], default=0.9, low=0.7, high=0.95, step=0.05),
    Knob(name="max_model_len", kind="cat", groups=["kv"], default=32768, choices=[4096, 8192, 16384, 32768]),
    Knob(name="max_num_seqs", kind="int", groups=["kv"], default=256, low=8, high=1024, log=True),
    Knob(name="kv_cache_dtype", kind="cat", groups=["kv"], default="auto", choices=["auto", "fp8"]),
])


def test_tighten_on_oom_lowers_upper_bounds_and_clamps() -> None:
    b = Bounds(SPACE)
    b.tighten_on_oom({"gpu_memory_utilization": 0.95, "max_model_len": 32768, "max_num_seqs": 512, "kv_cache_dtype": "auto"})
    assert b.high["gpu_memory_utilization"] == 0.9 and b.high["max_model_len"] == 16384 and b.high["max_num_seqs"] == 511
    out = clamp({"gpu_memory_utilization": 0.95, "max_model_len": 32768, "max_num_seqs": 900, "kv_cache_dtype": "fp8"}, SPACE, b)
    assert out == {"gpu_memory_utilization": 0.9, "max_model_len": 16384, "max_num_seqs": 511, "kv_cache_dtype": "fp8"}


def test_novelty_uses_normalized_distance() -> None:
    seen = [{"gpu_memory_utilization": 0.9, "max_model_len": 32768, "max_num_seqs": 256, "kv_cache_dtype": "auto"}]
    near = {**seen[0], "gpu_memory_utilization": 0.905}
    far = {**seen[0], "kv_cache_dtype": "fp8"}
    assert not is_novel(near, seen, SPACE) and is_novel(far, seen, SPACE)
```

`tests/search/test_optuna_search.py`:
```python
from pathlib import Path

from infervolt.core.types import Budget, Candidate, EngineConfig, SearchPlan, Trial
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.runner.trial import run_candidate
from infervolt.search.optuna_search import run_search
from infervolt.store.ledger import Ledger


def _baseline(adapter: MockAdapter, ctx) -> Trial:  # type: ignore[no-untyped-def]
    cfg = EngineConfig(engine="mock", knobs=adapter.knob_space(ctx).defaults())
    t = Trial(id="t0", run_id=ctx.run_id, index=0, candidate=Candidate(id="c0", config=cfg, origin="baseline"))
    return run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)


def test_search_improves_kv_scenario_and_handles_oom(tmp_path: Path) -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir=str(tmp_path), run_id="r1")
    ledger = Ledger(tmp_path / "l.sqlite", tmp_path / "runs")
    ledger.create_run(__import__("infervolt.core.types", fromlist=["OptimizeSpec"]).OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="r1"))
    base = _baseline(adapter, ctx)
    space = adapter.knob_space(ctx)
    oom_prior = Candidate(id="p-oom", config=base.candidate.config.with_knobs(max_model_len=32768, gpu_memory_utilization=0.7), origin="llm_prior", hypothesis="will OOM")
    good_prior = Candidate(id="p-fp8", config=base.candidate.config.with_knobs(kv_cache_dtype="fp8"), origin="llm_prior", hypothesis="fp8 kv")
    plan = SearchPlan(subspaces=["kv"], priors=[oom_prior, good_prior], max_trials=8)
    trials = run_search(adapter, ctx, space, plan, base, ledger, Budget(max_trials=8), seed=7)
    statuses = {t.status for t in trials}
    assert "infeasible_oom" in statuses and "ok" in statuses
    best = max((t for t in trials if t.status == "ok" and t.result), key=lambda t: t.result.objective)  # type: ignore[union-attr]
    assert best.result is not None and base.result is not None
    assert best.result.objective > 1.2 * base.result.objective
    assert len(trials) <= 8 and len({t.candidate.config.key() for t in trials}) == len(trials)
    assert len(ledger.trials("r1")) == len(trials)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/search -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write space helpers and the search loop**

`src/infervolt/search/__init__.py`: empty.

`src/infervolt/search/space.py`:
```python
"""Knob-space utilities: OOM-tightened bounds, clamping, novelty."""

from __future__ import annotations

import math

from infervolt.core.types import Knob, KnobSpace, KnobValue

NOVELTY_EPS = 0.05


def _numeric_choices(k: Knob) -> list[float] | None:
    if k.kind == "cat" and k.choices and all(isinstance(c, int | float) and not isinstance(c, bool) for c in k.choices):
        return sorted(float(c) for c in k.choices)
    return None


class Bounds:
    """Upper bounds per knob, lowered whenever a config OOMs (SLO-Guard style)."""

    def __init__(self, space: KnobSpace) -> None:
        self.high: dict[str, float] = {}
        for k in space.knobs:
            if k.kind in ("int", "float") and k.high is not None:
                self.high[k.name] = float(k.high)
            elif (nc := _numeric_choices(k)) is not None:
                self.high[k.name] = nc[-1]

    def tighten_on_oom(self, knobs: dict[str, KnobValue]) -> None:
        for name, value in knobs.items():
            if name not in self.high or isinstance(value, bool | str):
                continue
            v = float(value)
            if name == "gpu_memory_utilization":
                self.high[name] = min(self.high[name], round(v - 0.05, 2))
            elif name == "max_model_len":
                self.high[name] = min(self.high[name], v / 2)
            elif name == "max_num_seqs":
                self.high[name] = min(self.high[name], v - 1)


def clamp(knobs: dict[str, KnobValue], space: KnobSpace, bounds: Bounds) -> dict[str, KnobValue]:
    out: dict[str, KnobValue] = dict(knobs)
    for k in space.knobs:
        if k.name not in out or k.name not in bounds.high:
            continue
        v = out[k.name]
        if isinstance(v, bool | str):
            continue
        hi = bounds.high[k.name]
        if k.kind == "int":
            out[k.name] = int(min(int(v), int(hi)))
        elif k.kind == "float":
            out[k.name] = float(min(float(v), hi))
        else:
            nc = _numeric_choices(k) or []
            allowed = [c for c in nc if c <= hi] or nc[:1]
            out[k.name] = int(min(float(v), allowed[-1])) if all(float(c).is_integer() for c in nc) else min(float(v), allowed[-1])
    return out


def _normalize(k: Knob, v: KnobValue) -> float:
    if k.kind == "bool":
        return 1.0 if v else 0.0
    if k.kind == "cat":
        return k.choices.index(v) / max(len(k.choices) - 1, 1) if v in k.choices else 0.0
    lo, hi = float(k.low or 0), float(k.high or 1)
    x = float(v)
    if k.log and lo > 0 and hi > lo:
        return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (x - lo) / (hi - lo) if hi > lo else 0.0


def is_novel(knobs: dict[str, KnobValue], seen: list[dict[str, KnobValue]], space: KnobSpace, eps: float = NOVELTY_EPS) -> bool:
    for other in seen:
        dist = max((abs(_normalize(k, knobs.get(k.name, k.default)) - _normalize(k, other.get(k.name, k.default))) for k in space.knobs), default=1.0)
        if dist < eps:
            return False
    return True
```

`src/infervolt/search/optuna_search.py`:
```python
"""Optuna TPE search inside the planned sub-space with priors, novelty filter, crash-aware bounds,
and a cheap stage-1 evaluation that prunes below-median candidates before the full sweep."""

from __future__ import annotations

import statistics
import uuid

import optuna

from infervolt.core.types import Budget, Candidate, Knob, KnobSpace, KnobValue, RunContext, SearchPlan, Trial
from infervolt.engines.base import EngineAdapter
from infervolt.runner.trial import run_candidate
from infervolt.search.space import Bounds, clamp, is_novel
from infervolt.store.ledger import Ledger

WORST = -1.0
MAX_SKIPS = 20
STAGE1_REQUESTS = 8
STAGE2_REQUESTS = 16
MIN_STAGE1_BEFORE_PRUNE = 3


def _suggest(trial: optuna.Trial, k: Knob) -> KnobValue:
    if k.kind == "int":
        assert k.low is not None and k.high is not None  # guaranteed by Knob validator
        return trial.suggest_int(k.name, int(k.low), int(k.high), log=k.log)
    if k.kind == "float":
        assert k.low is not None and k.high is not None  # guaranteed by Knob validator
        return trial.suggest_float(k.name, float(k.low), float(k.high), step=k.step)
    if k.kind == "bool":
        return bool(trial.suggest_categorical(k.name, [True, False]))
    return trial.suggest_categorical(k.name, k.choices)  # type: ignore[return-value]


def run_search(
    adapter: EngineAdapter, ctx: RunContext, space: KnobSpace, plan: SearchPlan, baseline: Trial,
    ledger: Ledger, budget: Budget, seed: int, on_trial: object | None = None,
) -> list[Trial]:
    assert baseline.result is not None
    sub = space.subspace(plan.subspaces)
    if not sub.knobs:
        return []
    base_cfg = baseline.candidate.config.with_knobs(**plan.fixed)
    bounds = Bounds(sub)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=3))
    prior_keys: dict[str, str] = {}
    for p in plan.priors:
        params = {k.name: p.config.knobs[k.name] for k in sub.knobs if k.name in p.config.knobs}
        if params:
            study.enqueue_trial(params, skip_if_exists=True)
            prior_keys[base_cfg.with_knobs(**params).key()] = p.hypothesis
    seen: list[dict[str, KnobValue]] = [baseline.candidate.config.knobs] + [t.candidate.config.knobs for t in ledger.trials(ctx.run_id)]
    trials: list[Trial] = []
    stage1_scores: list[float] = []
    max_trials = min(plan.max_trials, budget.max_trials)
    skips = 0
    index = len(ledger.trials(ctx.run_id))
    stage1_c = [baseline.result.best_load_point]
    stage2_c = ctx.workload.load.concurrency
    while len(trials) < max_trials and skips < MAX_SKIPS:
        ot = study.ask()
        params = clamp({k.name: _suggest(ot, k) for k in sub.knobs}, sub, bounds)
        cfg = base_cfg.with_knobs(**params)
        if not is_novel(cfg.knobs, seen, sub):
            study.tell(ot, state=optuna.trial.TrialState.PRUNED)
            skips += 1
            continue
        seen.append(cfg.knobs)
        hyp = prior_keys.get(cfg.key(), "")
        cand = Candidate(id=f"c{uuid.uuid4().hex[:6]}", config=cfg, origin="llm_prior" if hyp else "tpe", hypothesis=hyp, parent_id=baseline.candidate.id)
        index += 1
        trial = Trial(id=f"t{index}", run_id=ctx.run_id, index=index, candidate=cand, stage=1)
        trial = run_candidate(adapter, trial, ctx, stage1_c, STAGE1_REQUESTS)
        if trial.status != "ok" or trial.result is None:
            if trial.crash_kind == "oom":
                bounds.tighten_on_oom(cfg.knobs)
            study.tell(ot, WORST)
            _record(ledger, trial, trials, on_trial)
            continue
        s1 = trial.result.objective
        if len(stage1_scores) >= MIN_STAGE1_BEFORE_PRUNE and s1 < statistics.median(stage1_scores):
            stage1_scores.append(s1)
            trial.status = "pruned"
            study.tell(ot, s1)
            _record(ledger, trial, trials, on_trial)
            continue
        stage1_scores.append(s1)
        trial.stage = 2
        trial = run_candidate(adapter, trial, ctx, stage2_c, STAGE2_REQUESTS)
        study.tell(ot, trial.result.objective if trial.status == "ok" and trial.result else WORST)
        if trial.crash_kind == "oom":
            bounds.tighten_on_oom(cfg.knobs)
        _record(ledger, trial, trials, on_trial)
    return trials


def _record(ledger: Ledger, trial: Trial, trials: list[Trial], on_trial: object | None) -> None:
    ledger.save_trial(trial)
    trials.append(trial)
    if callable(on_trial):
        on_trial(trial)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/search -q && uv run ruff check . && uv run mypy src`
Expected: `3 passed`. mypy may complain about `on_trial: object`; if so type it as `Callable[[Trial], None] | None` and import `Callable` from `collections.abc`.

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/search tests/search
git commit -m "feat(search): Optuna TPE with priors, novelty filter, OOM bound tightening, stage-1 pruning"
```

---

### Task 17: Verify (interleaved repeats with CI) and quality guard

**Files:**
- Create: `src/infervolt/verify/__init__.py`, `src/infervolt/verify/quality.py`, `src/infervolt/verify/verify.py`, `tests/verify/__init__.py`, `tests/verify/test_verify.py`

- [ ] **Step 1: Write the failing tests**

`tests/verify/__init__.py`: empty.

`tests/verify/test_verify.py`:
```python
from infervolt.core.types import Candidate, EngineConfig, Trial
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.runner.trial import run_candidate
from infervolt.verify.quality import MockQualityGuard, needs_quality_guard
from infervolt.verify.verify import paired_ci, verify


def _trial(adapter: MockAdapter, ctx, knobs: dict, idx: int) -> Trial:  # type: ignore[no-untyped-def]
    cfg = EngineConfig(engine="mock", knobs={**adapter.knob_space(ctx).defaults(), **knobs})
    t = Trial(id=f"t{idx}", run_id="r", index=idx, candidate=Candidate(id=f"c{idx}", config=cfg, origin="tpe"))
    return run_candidate(adapter, t, ctx, ctx.workload.load.concurrency, num_requests=16)


def test_paired_ci() -> None:
    lo, hi, mean = paired_ci([1.0, 1.2, 1.1])
    assert lo < mean < hi and lo > 0
    lo2, _, _ = paired_ci([0.1, -0.1, 0.05])
    assert lo2 < 0


def test_verify_accepts_real_improvement_and_rejects_noise() -> None:
    adapter, ctx = MockAdapter(), make_context("kv", run_dir="/tmp/x")
    base = _trial(adapter, ctx, {}, 0)
    better = _trial(adapter, ctx, {"kv_cache_dtype": "fp8"}, 1)
    same = _trial(adapter, ctx, {"enable_chunked_prefill": True}, 2)
    v = verify(adapter, ctx, base, better, MockQualityGuard(), repeats=3)
    assert v.accepted and v.ci_low > 0 and v.repeats == 3 and v.quality is not None
    assert v.quality.recovery >= 0.97
    v2 = verify(adapter, ctx, base, same, MockQualityGuard(), repeats=3)
    assert not v2.accepted


def test_needs_quality_guard_only_for_numerics_changing_knobs() -> None:
    assert needs_quality_guard({"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "fp8"})
    assert needs_quality_guard({"speculative": "none"}, {"speculative": "ngram"})
    assert not needs_quality_guard({"max_num_seqs": 256}, {"max_num_seqs": 128})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/verify -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write quality guard and verify**

`src/infervolt/verify/__init__.py`: empty.

`src/infervolt/verify/quality.py`:
```python
"""Quality guard: accuracy recovery check for knobs that change numerics."""

from __future__ import annotations

from typing import Protocol

from infervolt.core.types import EngineConfig, KnobValue, QualityScore, RunContext

QUALITY_KNOBS = {"kv_cache_dtype", "quantization", "speculative"}
RECOVERY_MIN = {"fp8": 0.99, "int4": 0.97, "default": 0.99}


def needs_quality_guard(before: dict[str, KnobValue], after: dict[str, KnobValue]) -> bool:
    return any(before.get(k) != after.get(k) for k in QUALITY_KNOBS)


class QualityGuard(Protocol):
    name: str

    def evaluate(self, cfg: EngineConfig, ctx: RunContext) -> QualityScore: ...


class MockQualityGuard:
    """Deterministic recovery numbers mirroring the Red Hat 500k-eval study."""

    name = "mock"

    def evaluate(self, cfg: EngineConfig, ctx: RunContext) -> QualityScore:
        rec = 1.0
        if cfg.knobs.get("kv_cache_dtype") == "fp8":
            rec = min(rec, 0.995)
        if cfg.knobs.get("quantization") == "fp8":
            rec = min(rec, 0.992)
        return QualityScore(guard=self.name, tasks=["gsm8k", "arc_challenge"], recovery=rec)


def recovery_threshold(cfg: EngineConfig) -> float:
    q = str(cfg.knobs.get("quantization", "none"))
    return RECOVERY_MIN.get(q, RECOVERY_MIN["default"])
```

`src/infervolt/verify/verify.py`:
```python
"""Interleaved baseline/candidate repeats with a paired-t confidence interval on goodput."""

from __future__ import annotations

import math
import statistics

from pydantic import BaseModel

from infervolt.core.types import QualityScore, RunContext, Trial
from infervolt.engines.base import EngineAdapter, LaunchError
from infervolt.runner.trial import run_load_point
from infervolt.verify.quality import QualityGuard, needs_quality_guard, recovery_threshold

T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 10: 2.262}
VERIFY_REQUESTS = 16


class VerifyResult(BaseModel):
    accepted: bool
    repeats: int
    load_point: int
    baseline_goodput: list[float]
    candidate_goodput: list[float]
    delta_mean: float
    ci_low: float
    ci_high: float
    improvement_pct: float
    quality: QualityScore | None = None
    reason: str = ""


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    n = len(deltas)
    mean = statistics.fmean(deltas)
    if n < 2:
        return mean, mean, mean
    sd = statistics.stdev(deltas)
    t = T_975.get(n, 2.0)
    half = t * sd / math.sqrt(n)
    return mean - half, mean + half, mean


def _goodput_at(adapter: EngineAdapter, ctx: RunContext, trial: Trial, c: int, seed: int) -> float:
    cfg = trial.candidate.config
    try:
        handle = adapter.launch(cfg, ctx)
    except LaunchError:
        return 0.0
    handle.config = cfg
    try:
        adapter.ready(handle, 900.0)
        obs, _ = run_load_point(adapter, handle, ctx.model_copy(update={"seed": seed}), c, VERIFY_REQUESTS)
    finally:
        adapter.stop(handle)
    return obs.metrics.goodput_rps if obs.valid else 0.0


def verify(adapter: EngineAdapter, ctx: RunContext, baseline: Trial, candidate: Trial, guard: QualityGuard, repeats: int = 3) -> VerifyResult:
    assert candidate.result is not None
    c = candidate.result.best_load_point
    b_vals: list[float] = []
    c_vals: list[float] = []
    for i in range(repeats):
        b_vals.append(_goodput_at(adapter, ctx, baseline, c, ctx.seed + 100 + i))
        c_vals.append(_goodput_at(adapter, ctx, candidate, c, ctx.seed + 200 + i))
    deltas = [cv - bv for bv, cv in zip(b_vals, c_vals, strict=True)]
    lo, hi, mean = paired_ci(deltas)
    base_mean = statistics.fmean(b_vals) or 1e-9
    pct = mean / base_mean * 100
    accepted = lo > 0
    reason = "CI-separated improvement" if accepted else "improvement not distinguishable from noise"
    quality: QualityScore | None = None
    if accepted and needs_quality_guard(baseline.candidate.config.knobs, candidate.candidate.config.knobs):
        quality = guard.evaluate(candidate.candidate.config, ctx)
        if quality.recovery < recovery_threshold(candidate.candidate.config):
            accepted, reason = False, f"quality recovery {quality.recovery:.3f} below threshold"
    return VerifyResult(accepted=accepted, repeats=repeats, load_point=c, baseline_goodput=b_vals, candidate_goodput=c_vals,
                        delta_mean=mean, ci_low=lo, ci_high=hi, improvement_pct=pct, quality=quality, reason=reason)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/verify -q && uv run ruff check . && uv run mypy src`
Expected: `3 passed`. If the "same" config is accepted because the mock is too deterministic, raise `NOISE` in the mock adapter to 0.05 (keep the kv scenario passing).

- [ ] **Step 5: Commit**

```bash
git add src/infervolt/verify tests/verify
git commit -m "feat(verify): interleaved repeats with paired-t CI and quality guard"
```

---

### Task 18: Planner state machine, ranker, budget, `optimize`/`report` CLI, integration test, README

**Files:**
- Create: `src/infervolt/diagnose/ranker.py`, `src/infervolt/agent/__init__.py`, `src/infervolt/agent/budget.py`, `src/infervolt/agent/planner.py`, `tests/integration/__init__.py`, `tests/integration/test_mock_loop.py`, `tests/diagnose/test_ranker.py`
- Modify: `src/infervolt/search/optuna_search.py` (add `deadline`), `src/infervolt/cli/main.py`, `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Write the failing tests**

`tests/diagnose/test_ranker.py`:
```python
from infervolt.core.types import Evidence, Finding
from infervolt.diagnose.ranker import rank
from infervolt.engines.mock.scenarios import make_context
from infervolt.llm.base import DiagnosisOut, LLMError
from infervolt.llm.fake import FakeLLMClient

F = [
    Finding(rule_id="R1", bottleneck="kv_capacity", score=1.0, evidence=[Evidence(source="engine", key="kv", value=0.97)], subspaces=["kv"], summary="kv"),
    Finding(rule_id="R2", bottleneck="decode_bandwidth", score=0.6, evidence=[], subspaces=["decode"], summary="bw"),
]


class _BadLLM:
    model_id = "bad"

    def structured(self, *, system: str, user: str, schema):  # type: ignore[no-untyped-def]
        return DiagnosisOut(primary_rule_id="R9", ranked_rule_ids=["R9"], rationale="hallucinated", confidence=0.9)


class _FailingLLM:
    model_id = "fail"

    def structured(self, *, system: str, user: str, schema):  # type: ignore[no-untyped-def]
        raise LLMError("down")


def test_rank_with_fake_llm() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    d = rank(FakeLLMClient(), F, ctx, [], {})
    assert d.primary == "kv_capacity" and d.subspaces == ["kv"] and [f.rule_id for f in d.ranked] == ["R1", "R2"]


def test_rank_falls_back_when_llm_cites_unknown_rule_or_fails() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    for llm in (_BadLLM(), _FailingLLM()):
        d = rank(llm, F, ctx, [], {})
        assert d.primary == "kv_capacity" and any("rule order" in c for c in d.caveats)
```

`tests/integration/__init__.py`: empty.

`tests/integration/test_mock_loop.py`:
```python
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from infervolt.agent.planner import Planner
from infervolt.cli.main import app
from infervolt.config import Settings
from infervolt.core.types import Budget, OptimizeSpec
from infervolt.engines.mock.scenarios import SCENARIOS
from infervolt.llm.fake import FakeLLMClient
from infervolt.recipes.schema import Recipe
from infervolt.store.ledger import Ledger
from infervolt.workloads.presets import parse_slo


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_mock_loop_names_injected_bottleneck_and_emits_recipe(name: str, tmp_path: Path) -> None:
    s = SCENARIOS[name]
    settings = Settings(home=tmp_path)
    spec = OptimizeSpec(engine="mock", model=s.model, hardware=s.hardware, workload=s.workload, slo=parse_slo(s.slo),
                        budget=Budget(max_trials=8), baseline=s.baseline, llm="fake")
    ledger = Ledger(settings.ledger_path, settings.runs_dir)
    outcome = Planner(spec, settings, FakeLLMClient(), ledger, log=lambda m: None).run()
    assert outcome.state == "done", outcome.message
    assert outcome.diagnosis is not None and outcome.diagnosis.primary == s.expected
    assert outcome.accepted, outcome.message  # every scenario has a CI-separated fix in the mock
    assert outcome.improvement_pct is not None and outcome.improvement_pct > 10
    assert outcome.recipe_path and outcome.report_path
    recipe = Recipe.model_validate(yaml.safe_load(Path(outcome.recipe_path).read_text()))
    assert recipe.infervolt.diagnosis.primary == s.expected
    assert recipe.result.metrics["goodput_rps"] > recipe.baseline.metrics["goodput_rps"]
    assert "Diagnosis" in Path(outcome.report_path).read_text()
    assert outcome.trials_to_target is not None and outcome.trials_to_target >= 1
    assert len(ledger.trials(outcome.run_id)) >= 2


@pytest.mark.integration
def test_cli_optimize_and_report(tmp_path: Path) -> None:
    runner = CliRunner()
    args = ["optimize", "--engine", "mock", "--model", "mock/qwen3-8b", "--hardware", "rtx4090-24",
            "--workload", "chat-4k-512", "--slo", "ttft=600ms,itl=30ms", "--llm", "fake", "--max-trials", "6",
            "--home", str(tmp_path)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout
    assert "primary bottleneck: kv_capacity" in res.stdout and "recipe:" in res.stdout
    run_id = next(line.split()[-1] for line in res.stdout.splitlines() if line.startswith("run:"))
    rep = runner.invoke(app, ["report", run_id, "--home", str(tmp_path)])
    assert rep.exit_code == 0 and "# infervolt recipe" in rep.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/diagnose/test_ranker.py tests/integration -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Add `deadline` to `run_search`**

In `src/infervolt/search/optuna_search.py`: add parameter `deadline: float | None = None` to `run_search` (after `seed`), `import time` at the top, and change the loop condition to
```python
    while len(trials) < max_trials and skips < MAX_SKIPS and (deadline is None or time.time() < deadline):
```
Also change `on_trial: object | None` to `on_trial: Callable[[Trial], None] | None` (`from collections.abc import Callable`) and call it directly.

- [ ] **Step 4: Write ranker, budget, planner**

`src/infervolt/diagnose/ranker.py`:
```python
"""LLM ranks and explains rule findings; falls back to rule order if it fails or hallucinates."""

from __future__ import annotations

from typing import Any

from infervolt.core.types import Diagnosis, Finding, KnobValue, Observation, RunContext
from infervolt.llm.base import SYSTEM_PROMPT, DiagnosisOut, LLMClient, LLMError, render_prompt


def rank(llm: LLMClient, findings: list[Finding], ctx: RunContext, obs: list[Observation], knobs: dict[str, KnobValue]) -> Diagnosis:
    if not findings:
        return Diagnosis(primary="under_loaded", ranked=[], rationale="No rule fired; the server was not saturated.", confidence=0.0, subspaces=[])
    context: dict[str, Any] = {
        "findings": [f.model_dump() for f in findings],
        "workload": ctx.workload.model_dump(), "slo": ctx.slo.model_dump(),
        "hardware": ctx.hw.model_dump(), "model": ctx.model.model_dump(), "config": knobs,
        "metrics": [{"concurrency": o.load_point, "valid": o.valid, **o.metrics.model_dump(), "engine": o.engine, "gpu": o.gpu} for o in obs],
    }
    ids = {f.rule_id for f in findings}
    out: DiagnosisOut | None = None
    for _ in range(2):
        try:
            cand = llm.structured(system=SYSTEM_PROMPT, user=render_prompt("rank", context), schema=DiagnosisOut)
        except LLMError:
            continue
        if cand.primary_rule_id in ids and set(cand.ranked_rule_ids) <= ids:
            out = cand
            break
    by_id = {f.rule_id: f for f in findings}
    if out is None:
        primary = findings[0]
        return Diagnosis(primary=primary.bottleneck, ranked=findings, rationale=primary.summary, confidence=primary.score,
                         subspaces=primary.subspaces, caveats=["LLM ranking unavailable or invalid; using rule order"])
    ranked = [by_id[r] for r in out.ranked_rule_ids] + [f for f in findings if f.rule_id not in out.ranked_rule_ids]
    primary = by_id[out.primary_rule_id]
    return Diagnosis(primary=primary.bottleneck, ranked=ranked, rationale=out.rationale, confidence=out.confidence,
                     subspaces=primary.subspaces, caveats=out.caveats)
```

`src/infervolt/agent/__init__.py`: empty.

`src/infervolt/agent/budget.py`:
```python
from __future__ import annotations

import time

from infervolt.core.types import Budget


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.start = time.time()
        self.spent_usd = 0.0
        self.trials = 0

    @property
    def deadline(self) -> float:
        return self.start + self.budget.max_wall_s

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        self.trials += 1

    def exhausted(self) -> str | None:
        if time.time() > self.deadline:
            return "wall-clock budget exhausted"
        if self.budget.max_usd and self.spent_usd > self.budget.max_usd:
            return f"cost budget exhausted (${self.spent_usd:.2f})"
        if self.trials >= self.budget.max_trials:
            return "trial budget exhausted"
        return None
```

`src/infervolt/agent/planner.py`:
```python
"""The optimize loop as an explicit state machine with ledger checkpoints.

PREPARE -> BASELINE -> DIAGNOSE -> PLAN -> SEARCH -> VERIFY -> EMIT -> LEARN -> DONE
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infervolt import __version__
from infervolt.agent.budget import BudgetTracker
from infervolt.config import Settings
from infervolt.core.types import (
    Candidate, Diagnosis, EngineConfig, KnobSpace, KnobValue, OptimizeSpec, RunContext, RunOutcome, SearchPlan, Trial,
)
from infervolt.diagnose.ranker import rank
from infervolt.diagnose.rules import evaluate_rules
from infervolt.engines.base import EngineAdapter
from infervolt.engines.registry import get_adapter
from infervolt.hardware.profiles import get_profile
from infervolt.llm.base import SYSTEM_PROMPT, LLMClient, LLMError, NarrativeOut, SearchPlanOut, prompts_sha, render_prompt
from infervolt.models.catalog import get_model_info
from infervolt.recipes.emit import write_recipe, write_report
from infervolt.recipes.schema import (
    Recipe, RecipeDiagnosis, RecipeDist, RecipeEngine, RecipeFinding, RecipeHardware, RecipeInfervolt, RecipeMeasured,
    RecipeModel, RecipeProvenance, RecipeQuality, RecipeResult, RecipeSearch, RecipeServe, RecipeSLO, RecipeWorkload,
)
from infervolt.runner.trial import run_candidate
from infervolt.search.optuna_search import STAGE2_REQUESTS, run_search
from infervolt.search.space import Bounds, clamp
from infervolt.store.ledger import Ledger
from infervolt.verify.quality import MockQualityGuard, QualityGuard
from infervolt.verify.verify import VerifyResult, verify
from infervolt.workloads.presets import get_workload

REPORT_METRICS = ["goodput_rps", "goodput_frac", "req_per_s", "output_tps", "ttft_p90_ms", "itl_p90_ms", "usd_per_m_tokens"]


class Planner:
    def __init__(self, spec: OptimizeSpec, settings: Settings, llm: LLMClient, ledger: Ledger,
                 adapter: EngineAdapter | None = None, guard: QualityGuard | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.spec, self.settings, self.llm, self.ledger, self.log = spec, settings, llm, ledger, log
        self.adapter = adapter or get_adapter(spec.engine)
        self.guard = guard or MockQualityGuard()

    # ---- entry point
    def run(self) -> RunOutcome:
        run_id = self.ledger.create_run(self.spec)
        try:
            return self._run(run_id)
        except Exception as e:  # noqa: BLE001 - the run must always end in a terminal state
            self.ledger.set_state(run_id, "failed")
            return RunOutcome(run_id=run_id, state="failed", message=f"{type(e).__name__}: {e}")

    def _run(self, run_id: str) -> RunOutcome:
        ctx = self._prepare(run_id)
        space = self.adapter.knob_space(ctx)
        tracker = BudgetTracker(self.spec.budget)
        self.log(f"run: {run_id}")

        self._state(run_id, "baseline")
        baseline = self._baseline(ctx, space)
        if baseline.status != "ok" or baseline.result is None:
            return self._fail(run_id, f"baseline failed: {baseline.status} {baseline.log_tail[:200]}")
        self.log(f"baseline goodput {baseline.result.objective:.3f} rps at c={baseline.result.best_load_point}")

        self._state(run_id, "diagnose")
        findings = evaluate_rules(baseline.result.observations, ctx, baseline.candidate.config, space)
        diagnosis = rank(self.llm, findings, ctx, baseline.result.observations, baseline.candidate.config.knobs)
        self.ledger.set_diagnosis(run_id, diagnosis.model_dump_json())
        self.log(f"primary bottleneck: {diagnosis.primary} (confidence {diagnosis.confidence:.2f}); findings: "
                 + ", ".join(f"{f.rule_id}={f.score:.2f}" for f in diagnosis.ranked))
        if diagnosis.primary in ("client_artifact", "under_loaded") or not diagnosis.subspaces:
            return self._finish_without_change(run_id, ctx, baseline, diagnosis, "no tunable bottleneck identified")

        self._state(run_id, "plan")
        plan = self._plan(ctx, space, diagnosis, baseline.candidate.config)
        self.log(f"search plan: subspaces={plan.subspaces} priors={len(plan.priors)} max_trials={plan.max_trials}")

        self._state(run_id, "search")
        trials = run_search(self.adapter, ctx, space, plan, baseline, self.ledger, self.spec.budget, self.spec.seed,
                            deadline=tracker.deadline, on_trial=lambda t: self._on_trial(t, tracker))
        ok = [t for t in trials if t.status == "ok" and t.result is not None]
        if not ok:
            return self._finish_without_change(run_id, ctx, baseline, diagnosis, "no feasible candidate improved on baseline")
        best = max(ok, key=lambda t: t.result.objective)  # type: ignore[union-attr]
        assert best.result is not None
        if best.result.objective <= baseline.result.objective:
            return self._finish_without_change(run_id, ctx, baseline, diagnosis, "search found nothing better than baseline")

        self._state(run_id, "verify")
        v = verify(self.adapter, ctx, baseline, best, self.guard)
        self.log(f"verify: {'ACCEPTED' if v.accepted else 'rejected'} {v.improvement_pct:+.1f}% (CI {v.ci_low:.3f}..{v.ci_high:.3f}) {v.reason}")
        if not v.accepted:
            return self._finish_without_change(run_id, ctx, baseline, diagnosis, f"verify rejected best trial: {v.reason}")
        self.ledger.set_best(run_id, best.id)

        self._state(run_id, "emit")
        recipe = self._recipe(ctx, space, baseline, best, trials, diagnosis, plan, v)
        out_dir = Path(ctx.run_dir)
        recipe_path, report_path = write_recipe(recipe, out_dir), write_report(recipe, out_dir)
        self.ledger.set_recipe(run_id, str(recipe_path))
        self.log(f"recipe: {recipe_path}\nreport: {report_path}")

        self._state(run_id, "learn")
        ttt = trials_to_target(trials, best.result.objective)
        self._state(run_id, "done")
        return RunOutcome(run_id=run_id, state="done", baseline_trial_id=baseline.id, best_trial_id=best.id, diagnosis=diagnosis,
                          recipe_path=str(recipe_path), report_path=str(report_path), trials_to_target=ttt,
                          improvement_pct=v.improvement_pct, accepted=True, message="ok")

    # ---- states
    def _prepare(self, run_id: str) -> RunContext:
        if self.spec.hardware == "auto":
            raise ValueError("hardware auto-detection arrives in M2; pass --hardware <profile>")
        return RunContext(run_id=run_id, run_dir=str(self.ledger.run_dir(run_id)), hw=get_profile(self.spec.hardware),
                          model=get_model_info(self.spec.model), workload=get_workload(self.spec.workload), slo=self.spec.slo, seed=self.spec.seed)

    def _baseline(self, ctx: RunContext, space: KnobSpace) -> Trial:
        cfg = EngineConfig(engine=self.spec.engine, knobs={**space.defaults(), **self.spec.baseline})
        trial = Trial(id="t0", run_id=ctx.run_id, index=0, candidate=Candidate(id="c0", config=cfg, origin="baseline"), stage=2)
        trial = run_candidate(self.adapter, trial, ctx, ctx.workload.load.concurrency, STAGE2_REQUESTS)
        self.ledger.save_trial(trial)
        return trial

    def _plan(self, ctx: RunContext, space: KnobSpace, diagnosis: Diagnosis, base_cfg: EngineConfig) -> SearchPlan:
        context: dict[str, Any] = {
            "diagnosis": diagnosis.model_dump(), "knob_space": [k.model_dump() for k in space.knobs], "current": base_cfg.knobs,
            "budget": self.spec.budget.model_dump(), "priors": [], "notes": [], "hardware": ctx.hw.model_dump(), "workload": ctx.workload.model_dump(),
        }
        try:
            out = self.llm.structured(system=SYSTEM_PROMPT, user=render_prompt("plan", context), schema=SearchPlanOut)
        except LLMError as e:
            self.log(f"plan: LLM failed ({e}); searching the diagnosis sub-spaces without priors")
            out = SearchPlanOut(subspaces=diagnosis.subspaces, max_trials=self.spec.budget.max_trials)
        groups = set(space.groups())
        subspaces = [g for g in out.subspaces if g in groups] or diagnosis.subspaces
        names = set(space.names())
        bounds = Bounds(space)
        priors: list[Candidate] = []
        for i, p in enumerate(out.priors[:4]):
            unknown = set(p.knobs) - names
            if unknown:
                self.log(f"plan: dropping unknown knobs {sorted(unknown)} from prior {i}")
            knobs = clamp({k: v for k, v in p.knobs.items() if k in names}, space, bounds)
            knobs = {k: v for k, v in knobs.items() if _in_choices(space, k, v)}
            if not knobs:
                continue
            cfg = base_cfg.with_knobs(**knobs)
            if errs := self.adapter.validate(cfg, ctx):
                self.log(f"plan: prior {i} rejected statically: {errs}")
                continue
            priors.append(Candidate(id=f"p{i}", config=cfg, origin="llm_prior", hypothesis=p.hypothesis, parent_id="c0"))
        return SearchPlan(subspaces=subspaces, priors=priors, max_trials=max(1, min(out.max_trials, self.spec.budget.max_trials)), rationale=out.rationale)

    def _on_trial(self, t: Trial, tracker: BudgetTracker) -> None:
        tracker.charge(t.cost_usd)
        obj = f"{t.result.objective:.3f}" if t.result else "-"
        self.log(f"trial {t.id} [{t.candidate.origin}] {t.status} stage={t.stage} objective={obj} {t.candidate.hypothesis}")

    def _recipe(self, ctx: RunContext, space: KnobSpace, baseline: Trial, best: Trial, trials: list[Trial],
                diagnosis: Diagnosis, plan: SearchPlan, v: VerifyResult) -> Recipe:
        assert baseline.result is not None and best.result is not None
        b_obs = next(o for o in baseline.result.observations if o.load_point == baseline.result.best_load_point)
        c_obs = next(o for o in best.result.observations if o.load_point == best.result.best_load_point)
        b_metrics = {k: round(getattr(b_obs.metrics, k), 4) for k in REPORT_METRICS}
        c_metrics = {k: round(getattr(c_obs.metrics, k), 4) for k in REPORT_METRICS}
        winning = {k: val for k, val in best.candidate.config.knobs.items() if baseline.candidate.config.knobs.get(k) != val}
        narrative = self._narrative(diagnosis, b_metrics, c_metrics, winning, [t.id for t in trials])
        args, command = self.adapter.to_recipe_block(best.candidate.config, ctx)
        ver = self.adapter.version()
        w = ctx.workload
        return Recipe(
            model=RecipeModel(id=ctx.model.id, params_b=ctx.model.params_b, arch=ctx.model.arch, moe=ctx.model.moe),
            hardware=RecipeHardware(gpu=ctx.hw.gpu, count=ctx.hw.count, topology=ctx.hw.interconnect, provider=ctx.hw.name),
            engine=RecipeEngine(name=ver.name, version=ver.version, image=ver.image_digest, commit=ver.commit),
            workload=RecipeWorkload(name=w.name, isl=RecipeDist(p50=w.isl.p50, p99=w.isl.p99), osl=RecipeDist(p50=w.osl.p50, p99=w.osl.p99),
                                    prefix_share=w.prefix_share, load={"mode": "sweep", "concurrency": w.load.concurrency}),
            slo=RecipeSLO(**ctx.slo.model_dump()),
            serve=RecipeServe(args=args, command=command),
            baseline=RecipeMeasured(serve_args=dict(baseline.candidate.config.knobs), metrics=b_metrics),
            result=RecipeResult(metrics=c_metrics, repeats=v.repeats,
                                improvement={"goodput_rps": f"{v.improvement_pct:+.0f}% (95% CI {v.ci_low:+.3f}..{v.ci_high:+.3f} rps at c={v.load_point})"},
                                quality=RecipeQuality(**v.quality.model_dump()) if v.quality else None),
            infervolt=RecipeInfervolt(
                run_id=ctx.run_id,
                diagnosis=RecipeDiagnosis(primary=diagnosis.primary, confidence=diagnosis.confidence,
                                          findings=[RecipeFinding(rule=f.rule_id, score=f.score, evidence=f.evidence) for f in diagnosis.ranked]),
                rationale=narrative.rationale,
                search=RecipeSearch(trials=len(trials), infeasible=sum(t.status in ("infeasible_oom", "crash", "rejected") for t in trials),
                                    subspace=[k.name for k in space.subspace(plan.subspaces).knobs], optimizer="optuna-tpe", seed=self.spec.seed),
                trials_to_target=trials_to_target(trials, best.result.objective),
                next_steps=narrative.next_steps,
                artifacts={"report": "report.md", "trials": "trials.jsonl"},
                provenance=RecipeProvenance(tool_version=__version__, llm=self.llm.model_id, prompts_sha=prompts_sha(),
                                            created=datetime.now(UTC).strftime("%Y-%m-%d")),
            ),
        )

    def _narrative(self, diagnosis: Diagnosis, b: dict[str, float], c: dict[str, float], winning: dict[str, KnobValue], trial_ids: list[str]) -> NarrativeOut:
        context = {"diagnosis": diagnosis.model_dump(), "baseline_metrics": b, "best_metrics": c, "winning_knobs": winning, "trial_ids": trial_ids}
        try:
            return self.llm.structured(system=SYSTEM_PROMPT, user=render_prompt("emit", context), schema=NarrativeOut)
        except LLMError:
            return NarrativeOut(rationale=f"{diagnosis.rationale} Winning knobs: {json.dumps(winning)}.", next_steps=["Re-run diagnosis on the tuned config."])

    def _finish_without_change(self, run_id: str, ctx: RunContext, baseline: Trial, diagnosis: Diagnosis, why: str) -> RunOutcome:
        report = Path(ctx.run_dir) / "report.md"
        report.write_text(f"# infervolt run {run_id}: no change recommended\n\n{why}\n\n"
                          f"Primary bottleneck: {diagnosis.primary} (confidence {diagnosis.confidence:.2f})\n\n{diagnosis.rationale}\n")
        self._state(run_id, "done")
        self.log(f"no recipe: {why}\nreport: {report}")
        return RunOutcome(run_id=run_id, state="done", baseline_trial_id=baseline.id, diagnosis=diagnosis, report_path=str(report), accepted=False, message=why)

    def _fail(self, run_id: str, msg: str) -> RunOutcome:
        self._state(run_id, "failed")
        self.log(f"failed: {msg}")
        return RunOutcome(run_id=run_id, state="failed", message=msg)

    def _state(self, run_id: str, state: str) -> None:
        self.ledger.set_state(run_id, state)  # type: ignore[arg-type]


def _in_choices(space: KnobSpace, name: str, value: KnobValue) -> bool:
    k = space.get(name)
    return value in k.choices if k.kind == "cat" else True


def trials_to_target(trials: list[Trial], final_best: float) -> int | None:
    n = 0
    for t in trials:
        n += 1
        if t.status == "ok" and t.result is not None and t.result.objective >= 0.95 * final_best:
            return n
    return None
```

- [ ] **Step 5: Add `optimize` and `report` commands**

Replace `src/infervolt/cli/main.py` with:
```python
"""infervolt command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from infervolt import __version__
from infervolt.config import Settings
from infervolt.core.types import Budget, KnobValue, OptimizeSpec
from infervolt.recipes.schema import Recipe
from infervolt.workloads.presets import parse_slo

app = typer.Typer(help="Measure, diagnose, fix, verify, and remember LLM inference optimizations.")
recipe_app = typer.Typer(help="Recipe utilities.")
app.add_typer(recipe_app, name="recipe")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"infervolt {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show version.")) -> None:
    """infervolt CLI."""


def _settings(home: Path | None) -> Settings:
    return Settings(home=home) if home else Settings()


def _parse_kv(items: list[str]) -> dict[str, KnobValue]:
    out: dict[str, KnobValue] = {}
    for item in items:
        k, _, v = item.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


@app.command()
def optimize(
    engine: str = typer.Option("mock", help="Engine adapter name (see entry points)."),
    model: str = typer.Option(..., help="Model id, e.g. mock/qwen3-8b"),
    hardware: str = typer.Option("auto", help="Hardware profile name (a100-80, h100-80, rtx4090-24, l4-24, m3-8)."),
    workload: str = typer.Option("chat-4k-512", help="Workload preset."),
    slo: str = typer.Option("", help="SLO string, e.g. ttft=500ms,itl=30ms[,e2e=2s,p=0.9]"),
    llm: str = typer.Option("fake", help="fake | anthropic | openai"),
    max_trials: int = typer.Option(12), max_wall_s: float = typer.Option(3600.0), max_usd: float = typer.Option(0.0),
    seed: int = typer.Option(7),
    baseline: list[str] = typer.Option([], "--baseline", help="Baseline knob override k=v (repeatable)."),
    home: Path | None = typer.Option(None, help="State directory (default ~/.infervolt)."),
) -> None:
    """Run the full loop and emit a recipe."""
    from infervolt.agent.planner import Planner
    from infervolt.llm.factory import make_llm
    from infervolt.store.ledger import Ledger

    settings = _settings(home)
    spec = OptimizeSpec(engine=engine, model=model, hardware=hardware, workload=workload, slo=parse_slo(slo),
                        budget=Budget(max_trials=max_trials, max_wall_s=max_wall_s, max_usd=max_usd),
                        baseline=_parse_kv(baseline), seed=seed, llm=llm)
    ledger = Ledger(settings.ledger_path, settings.runs_dir)
    outcome = Planner(spec, settings, make_llm(llm, settings), ledger, log=typer.echo).run()
    if outcome.state != "done":
        raise typer.Exit(code=1)


@app.command()
def report(run_id: str, home: Path | None = typer.Option(None)) -> None:
    """Print the report for a run."""
    from infervolt.store.ledger import Ledger

    settings = _settings(home)
    ledger = Ledger(settings.ledger_path, settings.runs_dir)
    path = ledger.run_dir(run_id) / "report.md"
    if not path.exists():
        typer.echo(f"no report for run {run_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(path.read_text())


@recipe_app.command("validate")
def recipe_validate(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]) -> None:
    """Validate a recipe.yaml against the infervolt schema."""
    try:
        Recipe.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError) as e:
        typer.echo(f"INVALID {path}: {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"OK {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
```

- [ ] **Step 6: Run the whole suite and the timing check**

Run: `time uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: all tests pass; the integration tests take well under 60 s total. If a scenario is not `accepted`, inspect `trials.jsonl` in the tmp run dir: the usual causes are the prior being statically rejected (fix the scenario hardware) or the improvement below the CI bar (raise `max_trials` in the test to 10 or lower `NOISE`).

- [ ] **Step 7: Update README and CHANGELOG**

Replace `README.md`:
```markdown
# infervolt

**Measure → diagnose the bottleneck → targeted fix → re-measure → verify → emit a recipe → remember.**

infervolt is an open-source agent that optimizes inference for open-source LLMs on *your* hardware and
explains *why* the winning configuration wins. It is the real-measurement, explanation-emitting,
engine-agnostic complement to tools that only model (NVIDIA aiconfigurator), only diagnose
(vLLM Doctor), or only search (auto-tuning-vllm).

Status: **pre-alpha**. The loop is complete and tested against a roofline-based mock engine.
llama.cpp and vLLM adapters are next (see ROADMAP.md).

## 60-second demo (no GPU, no API key)

```bash
uv sync --extra dev
uv run infervolt optimize --engine mock --model mock/qwen3-8b --hardware rtx4090-24 \
  --workload chat-4k-512 --slo ttft=600ms,itl=30ms --llm fake --max-trials 8
```

You get a `recipe.yaml` (engine args + before/after metrics + evidence + rationale) and a `report.md`.
Swap `--llm anthropic` (set `ANTHROPIC_API_KEY`) or `--llm openai` (any OpenAI-compatible endpoint,
including a local vLLM) for real reasoning.

## How it works

1. **Baseline sweep** at rising concurrency, recording client-side TTFT/ITL per token, engine counters, and GPU activity.
2. **Rules** (R0–R6) score bottleneck signatures from evidence: KV capacity, prefill compute, decode bandwidth, scheduler/CPU, communication, client artifact.
3. **LLM ranks and explains** the findings and picks the knob sub-space and a few prior candidates. It can only name knobs it was shown; every reply is schema-validated.
4. **Optuna TPE** searches inside that sub-space. OOMs are constraints that tighten bounds. Cheap stage-1 runs prune weak candidates before the full sweep.
5. **Verify**: 3 interleaved baseline/candidate repeats; a win needs a confidence interval on goodput that excludes zero, plus a quality guard when numerics change.
6. **Recipe + report** with provenance. Cross-run memory and continuous re-tuning arrive in M4.

## Compared to

| Tool | Runs real benchmarks | Attributes the bottleneck | Emits a recipe with evidence | Learns across runs |
|---|---|---|---|---|
| NVIDIA aiconfigurator | no (analytic) | no | manifests, no evidence | no |
| vLLM Doctor | reads metrics only | rules | no | local history only |
| auto-tuning-vllm / llm-optimizer | yes | no | config only | per-study |
| **infervolt** | yes | rules + LLM, evidence-backed | yes | M4 |

## Contributing

See CONTRIBUTING.md. Adapters, rules, workloads, and recipes are the four extension points.

## License

Apache-2.0.
```

Append to `CHANGELOG.md` under `[Unreleased]`/`Added`:
```
- CLI: `infervolt optimize`, `infervolt report`, `infervolt recipe validate`.
- Fake, Anthropic, OpenAI-compatible, and replay LLM clients.
```

- [ ] **Step 8: Commit and push**

```bash
git add src/infervolt/agent src/infervolt/diagnose/ranker.py src/infervolt/search/optuna_search.py src/infervolt/cli/main.py tests/integration tests/diagnose/test_ranker.py README.md CHANGELOG.md
git commit -m "feat(agent): planner state machine, optimize/report CLI, end-to-end mock loop"
git push -u origin main
```

M1 acceptance: CI green; `uv run infervolt optimize ...` from the README finishes in under 60 s and prints `primary bottleneck: kv_capacity` and a recipe path; `tests/integration/test_mock_loop.py` passes for all four scenarios.

---

## Self-review notes

- **Spec coverage:** baseline sweep, rules R0–R6, LLM rank/plan/emit with validation, Optuna inside sub-space, novelty filter, OOM tightening, stage-1 prune, interleaved verify with CI, quality guard, recipe YAML + report, ledger checkpoints, CLI, CI, community files: all covered. Deferred to later milestones and recorded in ROADMAP.md: async interfaces and the HTTP load generator (M2), hardware auto-detect (M2), Parquet/DuckDB (M4), `--resume` and memory/warm start (M4), replan-on-plateau LLM call 3 (M4, once memory exists to make it useful).
- **Type consistency:** `ServerHandle.config` is added in Task 12 and used in Tasks 12 and 17; `run_search` gains `deadline` in Task 18 and is called with it only there; `MockQualityGuard` (Task 17) is the planner default; canonical engine metric keys (`kv_usage_p95`, `num_waiting`, `num_running`, `preemptions_per_s`, `queue_time_p90_s`, `prefill_time_p50_s`, `prefill_share`, `prefix_hit_rate`, `max_num_seqs`, `kv_dtype_bytes`) are produced by the mock adapter (Task 11) and consumed by the rules (Task 13) with the same names.
