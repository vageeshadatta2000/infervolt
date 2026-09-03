"""GreedyEquivalenceGuard: two launches, twenty greedy prompts, one recovery number.

The engine is a stub adapter pointed at an in-process chat server, so the guard's real
HTTP path runs without a GPU.
"""

from __future__ import annotations

import difflib
import statistics
from collections.abc import Callable
from pathlib import Path

import pytest

from infervolt.core.types import (
    EngineConfig,
    KnobSpace,
    KnobValue,
    OptimizeSpec,
    RunContext,
)
from infervolt.engines.base import EngineAdapter, EngineVersion, ExitInfo, ServerHandle
from infervolt.engines.mock.adapter import MockAdapter
from infervolt.engines.mock.scenarios import make_context
from infervolt.loadgen.base import LoadGenerator
from infervolt.verify.quality import (
    DEFAULT_PROMPTS,
    INCOMPLETE_TASK,
    KV_RECOVERY_MIN,
    RECOVERY_MIN,
    GreedyEquivalenceGuard,
    MockQualityGuard,
    default_guard,
    recovery_threshold,
)
from tests.fake_http import FakeServer


class StubAdapter(EngineAdapter):
    """A launchable server that is really the fake chat endpoint.

    Each launch swaps in the next completion function, so the baseline and the candidate
    can be made to answer differently without two servers.
    """

    name = "stub"

    def __init__(
        self,
        server: FakeServer,
        answers: list[Callable[[str], str]],
        ready_ok: bool = True,
        launch_error: Exception | None = None,
    ) -> None:
        self.server = server
        self.answers = answers
        self.ready_ok = ready_ok
        self.launch_error = launch_error
        self.launched: list[EngineConfig] = []
        self.stopped = 0

    def version(self) -> EngineVersion:
        return EngineVersion(name=self.name)

    def knob_space(self, ctx: RunContext) -> KnobSpace:
        return MockAdapter().knob_space(ctx)

    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        return []

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        if self.launch_error is not None:
            raise self.launch_error
        self.server.completion = self.answers[min(len(self.launched), len(self.answers) - 1)]
        self.launched.append(cfg)
        return ServerHandle(url=self.server.url)

    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        return self.ready_ok

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        raise NotImplementedError

    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        return {}

    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        return {}

    def stop(self, handle: ServerHandle) -> ExitInfo:
        self.stopped += 1
        return ExitInfo(code=0)

    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        return dict(cfg.knobs), "stub-serve"


def cfg(**knobs: KnobValue) -> EngineConfig:
    return EngineConfig(engine="stub", knobs=knobs)


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    return make_context("kv", run_dir=str(tmp_path))


def identical(prompt: str) -> str:
    return f"The answer to {prompt} is forty two."


def test_identical_answers_recover_everything(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg(kv_cache_dtype="auto"))
        score = guard.evaluate(cfg(kv_cache_dtype="fp8"), ctx)
    assert score.guard == "greedy-equivalence"
    assert score.tasks == ["greedy-20"]
    assert score.recovery == pytest.approx(1.0)


def test_divergent_answers_score_the_token_overlap(ctx: RunContext) -> None:
    def drifted(prompt: str) -> str:
        return f"The answer to {prompt} is forty three point five."

    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, drifted])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg())
        score = guard.evaluate(cfg(quantization="fp8"), ctx)
    expected = statistics.fmean(
        difflib.SequenceMatcher(None, identical(p).split(), drifted(p).split()).ratio()
        for p in DEFAULT_PROMPTS
    )
    assert score.recovery == pytest.approx(expected)
    assert 0.0 < score.recovery < 1.0


def test_a_completely_different_answer_recovers_nothing(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, lambda p: "zzz qqq"])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg())
        score = guard.evaluate(cfg(quantization="fp8"), ctx)
    assert score.recovery == pytest.approx(0.0)


def test_the_requests_are_greedy_non_streaming_and_capped(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg(), max_tokens=32)
        guard.evaluate(cfg(quantization="fp8"), ctx)
    assert len(server.chat_requests) == 2 * len(DEFAULT_PROMPTS)
    first = server.chat_requests[0]
    assert first["temperature"] == 0
    assert first["max_tokens"] == 32
    assert first["stream"] is False
    assert first["model"] == ctx.model.id
    assert first["messages"] == [{"role": "user", "content": DEFAULT_PROMPTS[0]}]


def test_the_baseline_and_the_candidate_are_launched_one_after_the_other(
    ctx: RunContext,
) -> None:
    """Sequentially: two servers on one card would each be measuring the other."""
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        baseline = cfg(kv_cache_dtype="auto")
        candidate = cfg(kv_cache_dtype="fp8")
        GreedyEquivalenceGuard(adapter, baseline_cfg=baseline).evaluate(candidate, ctx)
    assert adapter.launched == [baseline, candidate]
    assert adapter.stopped == 2


def test_a_shorter_prompt_set_names_itself(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg(), prompts=["a?", "b?"])
        score = guard.evaluate(cfg(quantization="fp8"), ctx)
    assert score.tasks == ["greedy-2"]
    assert len(server.chat_requests) == 4


def test_a_launch_that_fails_scores_zero_rather_than_raising(ctx: RunContext) -> None:
    """The run has already spent its GPU budget; an unverifiable win is a rejection."""
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical], launch_error=RuntimeError("no GPU"))
        score = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg()).evaluate(
            cfg(quantization="fp8"), ctx
        )
    assert score.recovery == 0.0
    assert score.tasks == ["greedy-20", INCOMPLETE_TASK]


def test_a_server_that_never_becomes_ready_scores_zero(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical], ready_ok=False)
        score = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg()).evaluate(
            cfg(quantization="fp8"), ctx
        )
    assert score.recovery == 0.0
    assert INCOMPLETE_TASK in score.tasks
    assert adapter.stopped == 1  # the one it did launch was still torn down


def test_an_unreachable_endpoint_scores_zero(ctx: RunContext) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(adapter, baseline_cfg=cfg(), request_timeout_s=1.0)
        server.stop()  # the server dies between launch and the first prompt
        score = guard.evaluate(cfg(quantization="fp8"), ctx)
    assert score.recovery == 0.0
    assert INCOMPLETE_TASK in score.tasks


def test_the_stored_context_is_used_when_evaluate_is_called_without_one(
    ctx: RunContext,
) -> None:
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(adapter, ctx, baseline_cfg=cfg(), prompts=["a?"])
        assert guard.evaluate(cfg(quantization="fp8")).recovery == pytest.approx(1.0)


def test_without_a_context_anywhere_the_guard_says_so() -> None:
    with FakeServer() as server:
        guard = GreedyEquivalenceGuard(StubAdapter(server, [identical]))
        with pytest.raises(ValueError, match="RunContext"):
            guard.evaluate(cfg())


def test_the_baseline_defaults_to_the_runs_own_baseline_config(ctx: RunContext) -> None:
    """No explicit baseline: rebuild it the way the planner does, defaults plus overrides."""
    with FakeServer() as server:
        adapter = StubAdapter(server, [identical, identical])
        guard = GreedyEquivalenceGuard(
            adapter, baseline_knobs={"enforce_eager": True}, prompts=["a?"]
        )
        guard.evaluate(cfg(quantization="fp8"), ctx)
    launched = adapter.launched[0]
    defaults = MockAdapter().knob_space(ctx).defaults()
    assert launched.knobs["enforce_eager"] is True
    assert launched.knobs["max_num_seqs"] == defaults["max_num_seqs"]


# ---------------------------------------------------------------- thresholds and selection


def test_recovery_threshold_takes_the_loosest_applicable_floor() -> None:
    assert recovery_threshold(cfg()) == RECOVERY_MIN["default"]
    assert recovery_threshold(cfg(kv_cache_dtype="fp8")) == KV_RECOVERY_MIN["fp8"]
    assert recovery_threshold(cfg(quantization="int4")) == RECOVERY_MIN["int4"]
    assert recovery_threshold(cfg(quantization="q4")) == RECOVERY_MIN["q4"]
    assert recovery_threshold(cfg(speculative="ngram")) == 0.99
    # Weights and KV cache both changed: the more forgiving floor wins.
    assert recovery_threshold(cfg(quantization="int4", kv_cache_dtype="fp8")) == 0.95
    assert recovery_threshold(cfg(quantization="fp8", kv_cache_dtype="fp8")) == 0.97


def test_the_mock_engine_keeps_the_table_driven_guard() -> None:
    spec = OptimizeSpec(engine="mock", model="mock/qwen3-8b")
    assert isinstance(default_guard(MockAdapter(), spec), MockQualityGuard)


def test_every_real_engine_gets_the_greedy_guard() -> None:
    with FakeServer() as server:
        spec = OptimizeSpec(engine="stub", model="mock/qwen3-8b", baseline={"enforce_eager": True})
        guard = default_guard(StubAdapter(server, [identical]), spec)
    assert isinstance(guard, GreedyEquivalenceGuard)
    assert guard.baseline_knobs == {"enforce_eager": True}


def test_the_planner_picks_the_guard_from_the_adapter(tmp_path: Path) -> None:
    from infervolt.agent.planner import Planner
    from infervolt.config import Settings
    from infervolt.llm.fake import FakeLLMClient
    from infervolt.store.ledger import Ledger

    settings = Settings(home=tmp_path)
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        with FakeServer() as server:
            real = Planner(
                OptimizeSpec(engine="stub", model="mock/qwen3-8b"),
                settings,
                FakeLLMClient(),
                ledger,
                adapter=StubAdapter(server, [identical]),
            )
        mock = Planner(
            OptimizeSpec(engine="mock", model="mock/qwen3-8b"),
            settings,
            FakeLLMClient(),
            ledger,
            adapter=MockAdapter(),
        )
    assert isinstance(mock.guard, MockQualityGuard)
    assert isinstance(real.guard, GreedyEquivalenceGuard)
