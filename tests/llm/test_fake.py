from infervolt.llm.base import (
    SYSTEM_PROMPT,
    DiagnosisOut,
    NarrativeOut,
    SearchPlanOut,
    extract_context,
    prompts_sha,
    render_prompt,
)
from infervolt.llm.fake import FakeLLMClient

FINDINGS = [
    {
        "rule_id": "R1",
        "bottleneck": "kv_capacity",
        "score": 1.0,
        "summary": "KV exhausted",
        "subspaces": ["kv"],
        "evidence": [],
    },
    {
        "rule_id": "R2",
        "bottleneck": "decode_bandwidth",
        "score": 0.7,
        "summary": "at floor",
        "subspaces": ["decode"],
        "evidence": [],
    },
]
KNOBS = [
    {
        "name": "kv_cache_dtype",
        "kind": "cat",
        "groups": ["kv", "decode"],
        "choices": ["auto", "fp8"],
        "default": "auto",
    },
    {
        "name": "gpu_memory_utilization",
        "kind": "float",
        "groups": ["kv"],
        "low": 0.7,
        "high": 0.95,
        "default": 0.9,
    },
]


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
    ctx = {
        "diagnosis": {"primary": "kv_capacity", "subspaces": ["kv"]},
        "knob_space": KNOBS,
        "current": {"kv_cache_dtype": "auto", "gpu_memory_utilization": 0.9},
        "budget": {"max_trials": 9},
        "priors": [],
        "notes": [],
    }
    user = render_prompt("plan", ctx)
    out = llm.structured(system=SYSTEM_PROMPT, user=user, schema=SearchPlanOut)
    assert out.subspaces == ["kv"] and out.max_trials == 9
    assert out.priors and all(
        set(p.knobs) <= {"kv_cache_dtype", "gpu_memory_utilization"} for p in out.priors
    )
    assert any(p.knobs.get("kv_cache_dtype") == "fp8" for p in out.priors)


def test_fake_narrative_mentions_winning_knobs() -> None:
    llm = FakeLLMClient()
    ctx = {
        "diagnosis": {"primary": "kv_capacity", "rationale": "KV exhausted"},
        "baseline_metrics": {"goodput_rps": 1.0},
        "best_metrics": {"goodput_rps": 1.8},
        "winning_knobs": {"kv_cache_dtype": "fp8"},
        "trial_ids": ["t3"],
    }
    out = llm.structured(system=SYSTEM_PROMPT, user=render_prompt("emit", ctx), schema=NarrativeOut)
    assert "kv_cache_dtype" in out.rationale and out.next_steps
