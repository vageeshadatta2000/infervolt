from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from infervolt.agent.planner import Planner
from infervolt.cli.main import app
from infervolt.config import Settings
from infervolt.core.types import Budget, OptimizeSpec
from infervolt.engines.mock.scenarios import SCENARIOS
from infervolt.llm.fake import FakeLLMClient
from infervolt.recipes.schema import Recipe
from infervolt.store.ledger import Ledger
from infervolt.workloads.presets import parse_slo

MIN_IMPROVEMENT_PCT = 10.0
"""Floor the verified win must clear in every scenario.

Verification drives both arms at the *candidate's* best load point, so the verified
delta is not the objective gap the search saw. The narrowest of the four is `decode`
at about +70%; `kv` and `sched` clear it by a wide margin because their baselines serve
nothing at all at that load point.
"""


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_mock_loop_names_injected_bottleneck_and_emits_recipe(name: str, tmp_path: Path) -> None:
    s = SCENARIOS[name]
    settings = Settings(home=tmp_path)
    spec = OptimizeSpec(
        engine="mock",
        model=s.model,
        hardware=s.hardware,
        workload=s.workload,
        slo=parse_slo(s.slo),
        budget=Budget(max_trials=8),
        baseline=s.baseline,
        llm="fake",
    )
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        outcome = Planner(spec, settings, FakeLLMClient(), ledger, log=lambda m: None).run()
        assert outcome.state == "done", outcome.message
        assert outcome.diagnosis is not None and outcome.diagnosis.primary == s.expected
        assert outcome.accepted, outcome.message  # every scenario has a CI-separated fix
        assert outcome.improvement_pct is not None
        assert outcome.improvement_pct > MIN_IMPROVEMENT_PCT
        assert outcome.recipe_path and outcome.report_path
        recipe = Recipe.model_validate(yaml.safe_load(Path(outcome.recipe_path).read_text()))
        assert recipe.infervolt.diagnosis.primary == s.expected
        assert recipe.result.metrics["goodput_rps"] > recipe.baseline.metrics["goodput_rps"]
        # The narrative is written from the verified arms, both measured at the
        # candidate's load point. `kv`'s baseline serves nothing there, so there is no
        # rate to be a percentage of and the prose must say so in absolute terms;
        # `decode`'s baseline does serve, so a percentage is the honest summary.
        rationale = recipe.infervolt.rationale
        if name == "kv":
            assert "%" not in rationale, rationale
            assert "served nothing" in rationale, rationale
        elif name == "decode":
            assert "%" in rationale, rationale
        assert "Diagnosis" in Path(outcome.report_path).read_text()
        assert outcome.trials_to_target is not None and outcome.trials_to_target >= 1
        assert len(ledger.trials(outcome.run_id)) >= 2


@pytest.mark.integration
def test_cli_optimize_and_report(tmp_path: Path) -> None:
    runner = CliRunner()
    args = [
        "optimize",
        "--engine",
        "mock",
        "--model",
        "mock/qwen3-8b",
        "--hardware",
        "rtx4090-24",
        "--workload",
        "chat-4k-512",
        "--slo",
        "ttft=600ms,itl=30ms",
        "--llm",
        "fake",
        "--max-trials",
        "6",
        "--home",
        str(tmp_path),
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout
    assert "primary bottleneck: kv_capacity" in res.stdout
    assert "recipe:" in res.stdout
    run_id = next(line.split()[-1] for line in res.stdout.splitlines() if line.startswith("run:"))
    rep = runner.invoke(app, ["report", run_id, "--home", str(tmp_path)])
    assert rep.exit_code == 0
    assert "# infervolt recipe" in rep.stdout


@pytest.mark.integration
def test_cli_report_exits_nonzero_for_unknown_run(tmp_path: Path) -> None:
    res = CliRunner().invoke(app, ["report", "nope", "--home", str(tmp_path)])
    assert res.exit_code == 1
