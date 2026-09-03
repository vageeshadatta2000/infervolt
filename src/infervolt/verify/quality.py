"""Quality guard: accuracy recovery check for knobs that change numerics.

A speed win bought by changing what the model computes is not a win until someone
checks the answers still hold. Only a few knobs can do that, so only they trigger an
eval -- see :data:`QUALITY_KNOBS`.

Two guards ship. :class:`MockQualityGuard` returns table-driven numbers for the
simulator, which has no text to compare. :class:`GreedyEquivalenceGuard` is the one real
engines get: it asks the baseline and the candidate the same twenty questions at
temperature 0 and measures how much of the baseline's answer survives. It is cheap --
two launches and forty short completions, no eval harness, no dataset download -- and it
catches the failure mode that matters here, a knob that quietly changes the output.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import statistics
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from infervolt.core.types import EngineConfig, KnobValue, OptimizeSpec, QualityScore, RunContext
from infervolt.engines.base import EngineAdapter

QUALITY_KNOBS = {"kv_cache_dtype", "quantization", "speculative"}
"""Knobs that change the numerics of generation, and so can move accuracy.

Everything else -- batch sizes, memory fractions, scheduling -- changes *when* work is
done, not what it computes, and needs no eval.
"""

RECOVERY_MIN = {"fp8": 0.99, "int4": 0.95, "q4": 0.95, "awq": 0.95, "gptq": 0.95, "default": 0.99}
"""Minimum share of the baseline a config must recover, by *weight* quantization.

Four-bit weights are allowed to lose the most, because the speedup they buy is the
largest and because greedy text agreement is a harsher measure than an eval score: a
single diverging token at position three costs the rest of the sentence.
"""

KV_RECOVERY_MIN = {"fp8": 0.97}
"""Same, keyed on the *KV cache* dtype. An fp8 cache perturbs attention over the whole
context, so identical prefixes drift apart later in a way fp16 weights do not."""

SPECULATIVE_RECOVERY_MIN = 0.99
"""Speculative decoding is supposed to be output-preserving by construction: the target
model verifies every drafted token. Anything below the default floor is a bug, not a
tradeoff, so it gets no extra tolerance."""


def needs_quality_guard(before: dict[str, KnobValue], after: dict[str, KnobValue]) -> bool:
    """True when the move touched a numerics-changing knob.

    Compared with ``.get`` on both sides so a knob that appears or disappears counts as
    a change, not as equal-by-absence.
    """
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


DEFAULT_PROMPTS: tuple[str, ...] = (
    "What is the capital of France?",
    "List the first eight prime numbers.",
    "Write a Python function that reverses a string.",
    "Explain what a hash table is in two sentences.",
    "If a train travels 60 km in 45 minutes, what is its average speed in km/h?",
    "Name three states of matter and give one example of each.",
    "Translate 'good morning' into Spanish, French, and German.",
    "What is the difference between a list and a tuple in Python?",
    "Summarise the water cycle in three sentences.",
    "Who wrote 'Pride and Prejudice', and in what year was it published?",
    "Compute 17 * 23 and show the intermediate steps.",
    "Write a SQL query that returns the ten most recent rows of a table called orders.",
    "What does the acronym HTTP stand for, and what is it used for?",
    "Give two advantages and two disadvantages of solar power.",
    "Explain recursion to someone who has never programmed.",
    "What is the boiling point of water at sea level in Celsius and Fahrenheit?",
    "Write a regular expression that matches an email address.",
    "Describe the difference between weather and climate.",
    "Sort the numbers 42, 7, 19, 3, 88, 15 in ascending order.",
    "In one sentence, what problem does version control solve?",
)
"""Twenty short prompts spanning recall, arithmetic, code and summarisation.

Short on purpose: the point is not to be a benchmark but to make divergence *visible*.
Greedy decoding is deterministic given the same numerics, so two servers that compute
the same thing return the same tokens; a knob that changes the numerics shows up as
answers that start together and come apart.
"""

INCOMPLETE_TASK = "incomplete"
"""Marker appended to :attr:`QualityScore.tasks` when the comparison could not be run."""


class GreedyEquivalenceGuard:
    """How much of the baseline's greedy output does this config still produce?

    ``evaluate`` launches the baseline, asks it the prompts at ``temperature: 0``, stops
    it, then does the same for the candidate, and scores the pairs with
    :class:`difflib.SequenceMatcher` over whitespace tokens. Sequentially, not
    side by side: the two servers would otherwise be sharing a GPU, and a quality check
    that changes the memory available to the config under test is measuring the wrong
    thing.

    A comparison that could not be completed -- a server that would not start, an
    endpoint that errored -- scores 0.0 rather than raising. The caller
    (:func:`infervolt.verify.verify.verify`) is deciding whether to ship a recipe, and
    "we could not check" has to block that decision the same way a real regression does,
    without taking down a run that has already spent its GPU budget.
    """

    name = "greedy-equivalence"

    def __init__(
        self,
        adapter: EngineAdapter,
        ctx: RunContext | None = None,
        baseline_cfg: EngineConfig | None = None,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
        max_tokens: int = 64,
        baseline_knobs: dict[str, KnobValue] | None = None,
        ready_timeout_s: float = 900.0,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.adapter = adapter
        self.ctx = ctx
        self.baseline_cfg = baseline_cfg
        self.prompts = list(prompts)
        self.max_tokens = max_tokens
        self.baseline_knobs = dict(baseline_knobs or {})
        self.ready_timeout_s = ready_timeout_s
        self.request_timeout_s = request_timeout_s

    @property
    def task(self) -> str:
        return f"greedy-{len(self.prompts)}"

    def baseline_for(self, ctx: RunContext) -> EngineConfig:
        """The config to compare against: the one given, else the run's own baseline.

        Reconstructed the same way the planner builds it -- the adapter's knob-space
        defaults with the user's ``--baseline`` overrides on top -- so a guard built
        before the run context existed still compares against what the run measured.
        """
        if self.baseline_cfg is not None:
            return self.baseline_cfg
        defaults = self.adapter.knob_space(ctx).defaults()
        return EngineConfig(engine=self.adapter.name, knobs={**defaults, **self.baseline_knobs})

    def evaluate(self, cfg: EngineConfig, ctx: RunContext | None = None) -> QualityScore:
        """Compare ``cfg``'s answers with the baseline's. ``ctx`` defaults to the stored one.

        The argument is optional only so a guard constructed with its run context can be
        called without repeating it; :func:`~infervolt.verify.verify.verify` always
        passes the context it is running under, and that one wins.
        """
        run_ctx = ctx if ctx is not None else self.ctx
        if run_ctx is None:
            raise ValueError("GreedyEquivalenceGuard needs a RunContext to launch against")
        try:
            baseline = self._completions(self.baseline_for(run_ctx), run_ctx)
            candidate = self._completions(cfg, run_ctx)
        except QualityCheckError:
            return QualityScore(guard=self.name, tasks=[self.task, INCOMPLETE_TASK], recovery=0.0)
        ratios = [
            difflib.SequenceMatcher(None, b.split(), c.split()).ratio()
            for b, c in zip(baseline, candidate, strict=True)
        ]
        recovery = statistics.fmean(ratios) if ratios else 1.0
        return QualityScore(guard=self.name, tasks=[self.task], recovery=recovery)

    def _completions(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """Launch ``cfg``, ask every prompt, and tear it down again."""
        try:
            handle = self.adapter.launch(cfg, ctx)
        except Exception as e:  # noqa: BLE001 - a config that will not start cannot be scored
            raise QualityCheckError(f"launch failed: {type(e).__name__}: {e}") from e
        handle.config = cfg
        model = self.model_name(ctx)
        try:
            if not self.adapter.ready(handle, self.ready_timeout_s):
                raise QualityCheckError("server never became ready")
            return [self._complete(handle.url, model, p) for p in self.prompts]
        finally:
            # Suppressed, not reported: raising here would replace whatever the body was
            # already saying. A server that would not die still makes itself known --
            # it is holding the GPU the next launch needs.
            with contextlib.suppress(Exception):
                self.adapter.stop(handle)

    @staticmethod
    def model_name(ctx: RunContext) -> str:
        """What to put in the request's ``model`` field: the id the run is about."""
        return ctx.model.id

    def _complete(self, base_url: str, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Greedy: the whole method rests on the server being deterministic, so any
            # sampling here would show up as a quality regression that is really noise.
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        body = _post_json(f"{base_url}/v1/chat/completions", payload, self.request_timeout_s)
        try:
            choices = body["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise QualityCheckError(f"unexpected completion response: {body!r}") from e
        return "" if content is None else str(content)


class QualityCheckError(Exception):
    """The comparison could not be made. Never escapes :meth:`GreedyEquivalenceGuard.evaluate`."""


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - the URL is the handle we just launched
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise QualityCheckError(f"{url}: {type(e).__name__}: {e}") from e
    if not isinstance(body, dict):
        raise QualityCheckError(f"{url}: expected a JSON object, got {type(body).__name__}")
    return body


def default_guard(adapter: EngineAdapter, spec: OptimizeSpec) -> QualityGuard:
    """The guard an engine gets when the caller did not pick one.

    The simulator has no text to compare, so it keeps the table-driven mock. Every real
    engine serves an OpenAI-compatible endpoint and gets the greedy comparison, told
    which knobs the run's baseline was pinned to so it compares against the same config
    the planner measured.
    """
    if adapter.name == "mock":
        return MockQualityGuard()
    return GreedyEquivalenceGuard(adapter, baseline_knobs=dict(spec.baseline))


def recovery_threshold(cfg: EngineConfig) -> float:
    """The recovery floor for ``cfg``: the loosest of the floors its knobs imply.

    A config that quantizes weights *and* the KV cache is allowed the tolerance of the
    more forgiving of the two, not the intersection -- each numerics change costs some
    agreement, and stacking them cannot make the config held to a stricter bar than
    either change alone.
    """
    floors = [RECOVERY_MIN.get(str(cfg.knobs.get("quantization", "none")), RECOVERY_MIN["default"])]
    kv = str(cfg.knobs.get("kv_cache_dtype", "auto"))
    if kv in KV_RECOVERY_MIN:
        floors.append(KV_RECOVERY_MIN[kv])
    if str(cfg.knobs.get("speculative", "none")) != "none":
        floors.append(SPECULATIVE_RECOVERY_MIN)
    return min(floors)
