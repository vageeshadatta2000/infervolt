"""All shared data models.

Everything else imports from here; this module imports nothing internal.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

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

    ``percentile`` is the per-request latency percentile the ``*_ms`` targets are
    measured at (e.g. 0.9 means the p90 of each latency must be under its target).
    ``goodput_target`` is a different axis: the fraction of requests that must meet
    the SLO for the run as a whole to count as "SLO met".
    """

    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    percentile: float = 0.9
    goodput_target: float = 0.9


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
