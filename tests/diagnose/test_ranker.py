from typing import TypeVar

from pydantic import BaseModel

from infervolt.core.types import Evidence, Finding
from infervolt.diagnose.ranker import rank
from infervolt.engines.mock.scenarios import make_context
from infervolt.llm.base import DiagnosisOut, LLMError
from infervolt.llm.fake import FakeLLMClient

T = TypeVar("T", bound=BaseModel)

F = [
    Finding(
        rule_id="R1",
        bottleneck="kv_capacity",
        score=1.0,
        evidence=[Evidence(source="engine", key="kv", value=0.97)],
        subspaces=["kv"],
        summary="kv",
    ),
    Finding(
        rule_id="R2",
        bottleneck="decode_bandwidth",
        score=0.6,
        evidence=[],
        subspaces=["decode"],
        summary="bw",
    ),
]


class _BadLLM:
    """Cites a rule id that was never in the context."""

    model_id = "bad"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        out = DiagnosisOut(
            primary_rule_id="R9",
            ranked_rule_ids=["R9"],
            rationale="hallucinated",
            confidence=0.9,
        )
        return schema.model_validate(out.model_dump())


class _FailingLLM:
    model_id = "fail"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        raise LLMError("down")


def test_rank_with_fake_llm() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    d = rank(FakeLLMClient(), F, ctx, [], {})
    assert d.primary == "kv_capacity"
    assert d.subspaces == ["kv"]
    assert [f.rule_id for f in d.ranked] == ["R1", "R2"]


def test_rank_falls_back_when_llm_cites_unknown_rule_or_fails() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    for llm in (_BadLLM(), _FailingLLM()):
        d = rank(llm, F, ctx, [], {})
        assert d.primary == "kv_capacity"
        assert any("rule order" in c for c in d.caveats)


def test_rank_without_findings_reports_under_loaded() -> None:
    ctx = make_context("kv", run_dir="/tmp/x")
    d = rank(FakeLLMClient(), [], ctx, [], {})
    assert d.primary == "under_loaded"
    assert d.ranked == []
    assert d.subspaces == []
