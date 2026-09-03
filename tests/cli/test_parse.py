"""``--baseline`` parsing and the option validation that happens before a run opens."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from infervolt.cli.main import LLM_NAMES, _parse_kv, app

runner = CliRunner()

USAGE_ERROR = 2
"""Click's exit code for a bad invocation, as opposed to 1 for a run that failed."""


def test_parse_kv_narrows_to_the_tightest_type() -> None:
    parsed = _parse_kv(
        [
            "enforce_eager=true",
            "enable_prefix_caching=False",
            "max_num_seqs=64",
            "gpu_memory_utilization=0.9",
            "kv_cache_dtype=fp8",
        ]
    )
    assert parsed == {
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.9,
        "kv_cache_dtype": "fp8",
    }
    # Types, not just values: a bool that arrived as 1 would pass ``== True``.
    assert isinstance(parsed["enforce_eager"], bool)
    assert isinstance(parsed["max_num_seqs"], int)
    assert isinstance(parsed["gpu_memory_utilization"], float)
    assert isinstance(parsed["kv_cache_dtype"], str)


def test_parse_kv_keeps_an_explicitly_empty_value() -> None:
    """``k=`` is a deliberate empty string; only a missing ``=`` is the mistake."""
    assert _parse_kv(["speculative="]) == {"speculative": ""}


@pytest.mark.parametrize("item", ["nonsense", "=64", ""])
def test_parse_kv_rejects_items_that_are_not_key_equals_value(item: str) -> None:
    with pytest.raises(typer.BadParameter):
        _parse_kv([item])


def test_cli_rejects_a_baseline_without_an_equals_sign(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "mock/qwen3-8b",
            "--hardware",
            "rtx4090-24",
            "--baseline",
            "nonsense",
            "--home",
            str(tmp_path),
        ],
    )
    assert res.exit_code == USAGE_ERROR, res.output
    assert "nonsense" in res.output
    assert not (tmp_path / "runs").exists()  # rejected before any state was written


def test_cli_rejects_an_unknown_llm(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "mock/qwen3-8b",
            "--hardware",
            "rtx4090-24",
            "--llm",
            "bogus",
            "--home",
            str(tmp_path),
        ],
    )
    assert res.exit_code == USAGE_ERROR, res.output
    assert all(name in res.output for name in LLM_NAMES)


def test_cli_rejects_hardware_auto_until_detection_lands(tmp_path: Path) -> None:
    """``auto`` is the default so M2 can switch it on; today it is the one illegal value."""
    res = runner.invoke(app, ["optimize", "--model", "mock/qwen3-8b", "--home", str(tmp_path)])
    assert res.exit_code == USAGE_ERROR, res.output
    assert "M2" in res.output and "rtx4090-24" in res.output


def test_cli_report_says_the_run_is_unknown(tmp_path: Path) -> None:
    res = runner.invoke(app, ["report", "no-such-run", "--home", str(tmp_path)])
    assert res.exit_code == 1
    assert "unknown run no-such-run" in res.output
    assert "Traceback" not in res.output
