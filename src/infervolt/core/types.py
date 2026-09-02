"""All shared data models.

Everything else imports from here; this module imports nothing internal.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

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
    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    percentile: float = 0.9


class HardwareProfile(BaseModel):
    name: str
    gpu: str
    count: int = 1
    mem_gb: float
    hbm_bw_gbs: float
    peak_tflops: float
    compute_capability: float
    interconnect: Literal["single", "nvlink", "pcie"] = "single"
    usd_per_hour: float = 0.0


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
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[KnobValue] = Field(default_factory=list)


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
    ttft_s: float
    itl_s: list[float]
    output_tokens: int
    ok: bool = True

    @property
    def e2e_s(self) -> float:
        return self.ttft_s + sum(self.itl_s)


class ClientHealth(BaseModel):
    worker_cpu: float = 0.0
    loop_lag_p99_ms: float = 0.0
    error_rate: float = 0.0


class LoadResult(BaseModel):
    concurrency: int
    duration_s: float
    requests: list[RequestRecord]
    health: ClientHealth = Field(default_factory=ClientHealth)


class Metrics(BaseModel):
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
    usd_per_m_tokens: float


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

BOTTLENECK_PRIORITY: dict[str, int] = {
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
