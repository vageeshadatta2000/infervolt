"""Deterministic prompt synthesis for the HTTP load generator.

Real user traffic is neither available nor reproducible, so requests carry text drawn
from a fixed 200-word bank. Two properties are what make the synthesis worth having:

* **Determinism** -- the same ``(tokens, seed, prefix, words_per_token)`` always yields
  the same string, so rerunning a trial sends byte-identical work to the server and any
  difference in the measurement belongs to the config, not to the prompts.
* **Length control in tokens** -- prompts are built in *words* and converted with a
  ``words_per_token`` ratio, calibrated once per run against the server's own tokenizer
  when one is reachable (:func:`calibrate_words_per_token`). Word counts are the only
  length knob a tokenizer-free client has; the ratio is what makes them mean something.

A caller that wants prefix caching exercised passes a ``prefix`` shared across requests
(see :func:`shared_prefix`). Bodies are independent random draws, so *no* prefix is
shared unless one was asked for -- the prefix hit rate an engine reports is then a
property of the workload rather than an accident of the word bank.
"""

from __future__ import annotations

import random
from collections.abc import Callable

_BANK_TEXT = """
about above across after again against almost alone along already
although always among ancient another answer anything appear around arrive
autumn average balance basket because become before behind believe below
beneath beside better between beyond bridge bright bring broken builder
candle canyon careful carry castle center certain chapter circle clever
climate closer colour common compass complete concert consider copper corner
cotton council counter country couple courage create crystal current custom
danger daughter decide degree delight depend desert design detail develop
differ direct distant divide double dragon during eastern effort either
element engine enough escape evening event ever every example expect
explain factor family famous farmer feather figure filter finger finish
flavour follow forest forget fortune forward foster fragile freedom friend
garden gather gentle glacier golden govern granite gravel ground guitar
habit handle harbour harvest header health hidden higher history hollow
honest hunter husband ignore image imagine impact improve include indeed
inside island jacket journey judgement junior kitchen ladder lantern latter
leader legend length lesson letter level library lighter listen little
lonely longer machine magnet manner marble market master matter meadow
measure member memory mention merchant method middle mineral minute mirror
modern moment morning mountain narrow nation native nature nearly needle
"""

WORD_BANK: tuple[str, ...] = tuple(_BANK_TEXT.split())
"""The vocabulary every synthesised prompt is drawn from."""

DEFAULT_WORDS_PER_TOKEN = 0.75
"""Fallback ratio when no tokenizer endpoint is reachable.

English BPE vocabularies land near 1.3 tokens per word, so a word is about 0.75 tokens.
It is only a starting point: :func:`calibrate_words_per_token` replaces it with the
server's own answer whenever one can be obtained.
"""

MIN_WORDS_PER_TOKEN = 0.05
MAX_WORDS_PER_TOKEN = 4.0
"""Clamp on a calibrated ratio.

A tokenizer endpoint that answers with the wrong units (characters, say, or a count for
a different string) would otherwise turn a 4k-token prompt into either one word or
several million of them -- neither of which measures the workload that was asked for.
"""

CALIBRATION_WORDS = 300
"""Sample size for calibration: long enough to average over word-length variation,
short enough that the round trip costs nothing."""

_PREFIX_SALT = 0x9E3779B9
"""Keeps a shared prefix's words distinct from a body built with the same seed."""


def _draw(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(WORD_BANK) for _ in range(count)]


def build_prompt(
    tokens: int,
    seed: int,
    prefix: str = "",
    words_per_token: float = DEFAULT_WORDS_PER_TOKEN,
) -> str:
    """A prompt of roughly ``tokens`` tokens, starting with ``prefix``.

    ``prefix`` counts towards the length rather than being added on top of it, so a
    workload's ISL still describes the whole prompt once prefix sharing is switched on.
    A prefix longer than the whole budget is never truncated -- truncating it would
    break the sharing it exists to create -- so such a prompt is longer than asked.
    """
    total_words = max(1, round(tokens * words_per_token))
    prefix_words = len(prefix.split())
    body = " ".join(_draw(max(1, total_words - prefix_words), seed))
    return f"{prefix} {body}" if prefix else body


def shared_prefix(tokens: int, seed: int, words_per_token: float = DEFAULT_WORDS_PER_TOKEN) -> str:
    """The common prefix every request in a prefix-sharing workload starts with."""
    return build_prompt(tokens, seed ^ _PREFIX_SALT, words_per_token=words_per_token)


def calibrate_words_per_token(count_tokens: Callable[[str], int]) -> float:
    """Measure the server's words-per-token ratio on a fixed sample.

    ``count_tokens`` is anything that turns text into a token count -- typically a POST
    to the engine's tokenizer endpoint. A count of zero or less means the endpoint did
    not answer usefully, and the default ratio is used instead; exceptions are left to
    the caller, which knows whether a failed round trip is fatal.
    """
    sample = " ".join(_draw(CALIBRATION_WORDS, seed=0))
    n = count_tokens(sample)
    if n <= 0:
        return DEFAULT_WORDS_PER_TOKEN
    return min(max(CALIBRATION_WORDS / n, MIN_WORDS_PER_TOKEN), MAX_WORDS_PER_TOKEN)
