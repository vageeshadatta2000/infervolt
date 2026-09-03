from __future__ import annotations

import pytest

from infervolt.loadgen.prompts import (
    CALIBRATION_WORDS,
    DEFAULT_WORDS_PER_TOKEN,
    MAX_WORDS_PER_TOKEN,
    MIN_WORDS_PER_TOKEN,
    WORD_BANK,
    build_prompt,
    calibrate_words_per_token,
    shared_prefix,
)


def test_bank_is_a_deterministic_two_hundred_word_vocabulary() -> None:
    assert len(WORD_BANK) == 200
    assert len(set(WORD_BANK)) == 200
    assert all(w.isalpha() and w.islower() for w in WORD_BANK)


def test_same_seed_same_prompt_different_seed_different_prompt() -> None:
    a = build_prompt(128, seed=1)
    b = build_prompt(128, seed=1)
    c = build_prompt(128, seed=2)
    assert a == b
    assert a != c


def test_length_tracks_tokens_times_words_per_token() -> None:
    for tokens in (16, 64, 512):
        for wpt in (0.5, 0.75, 1.0):
            words = build_prompt(tokens, seed=3, words_per_token=wpt).split()
            assert len(words) == round(tokens * wpt)


def test_tiny_requests_still_get_at_least_one_word() -> None:
    assert build_prompt(0, seed=0).split() != []
    assert build_prompt(1, seed=0, words_per_token=0.1).split() != []


def test_prefix_is_kept_verbatim_at_the_front_and_counts_towards_length() -> None:
    prefix = shared_prefix(64, seed=5)
    prompt = build_prompt(256, seed=7, prefix=prefix)
    assert prompt.startswith(prefix + " ")
    # The prefix is part of the budget, not extra on top of it.
    assert len(prompt.split()) == round(256 * DEFAULT_WORDS_PER_TOKEN)


def test_prefix_longer_than_the_budget_is_not_truncated() -> None:
    prefix = shared_prefix(512, seed=5)
    prompt = build_prompt(8, seed=7, prefix=prefix)
    assert prompt.startswith(prefix)


def test_shared_prefix_is_stable_and_seed_dependent() -> None:
    assert shared_prefix(32, seed=1) == shared_prefix(32, seed=1)
    assert shared_prefix(32, seed=1) != shared_prefix(32, seed=2)


def test_calibration_uses_a_three_hundred_word_sample() -> None:
    seen: list[str] = []

    def count(text: str) -> int:
        seen.append(text)
        return len(text.split()) * 2

    assert calibrate_words_per_token(count) == pytest.approx(0.5)
    assert len(seen) == 1
    assert len(seen[0].split()) == CALIBRATION_WORDS


def test_calibration_falls_back_when_the_tokenizer_reports_nothing() -> None:
    assert calibrate_words_per_token(lambda _: 0) == DEFAULT_WORDS_PER_TOKEN
    assert calibrate_words_per_token(lambda _: -3) == DEFAULT_WORDS_PER_TOKEN


def test_calibration_is_clamped_to_a_sane_band() -> None:
    assert calibrate_words_per_token(lambda _: 10_000_000) == MIN_WORDS_PER_TOKEN
    assert calibrate_words_per_token(lambda _: 1) == MAX_WORDS_PER_TOKEN
