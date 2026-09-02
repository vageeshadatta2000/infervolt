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
