from pathlib import Path

import yaml
from typer.testing import CliRunner

from infervolt.cli.main import app
from infervolt.core.types import Evidence
from infervolt.recipes.emit import render_report, write_recipe
from infervolt.recipes.schema import Recipe, recipe_json_schema

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "recipe.yaml"


def test_example_recipe_validates() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    recipe = Recipe.model_validate(data)
    assert recipe.engine.name == "vllm"
    assert recipe.infervolt.diagnosis.primary == "kv_capacity"


def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    path = write_recipe(recipe, tmp_path)
    assert path.name == "recipe.yaml"
    again = Recipe.model_validate(yaml.safe_load(path.read_text()))
    assert again == recipe


def test_report_renders_key_sections() -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    md = render_report(recipe)
    assert "# infervolt recipe" in md
    assert "kv_capacity" in md and "goodput_rps" in md and "Reproduce" in md


def test_report_renders_evidence_notes_when_a_rule_left_one() -> None:
    """A note is the rule's own caveat about a number; dropping it loses the caveat.

    ``prefix_hit_rate=0`` reads as a cold cache until the note says prefix caching was
    off, so the note has to travel with the value into the report.
    """
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    recipe.infervolt.diagnosis.findings[0].evidence.append(
        Evidence(source="prometheus", key="prefix_hit_rate", value=0.0, note="did not fire")
    )
    md = render_report(recipe)
    assert "`prefix_hit_rate`=0 (did not fire)" in md
    # The example's own evidence carries no notes, and an empty one must add nothing --
    # not a trailing pair of empty parentheses.
    assert "`kv_usage_p95`=0.97" in md
    assert "()" not in md


def test_report_never_prints_none_and_shows_the_goodput_target() -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    assert recipe.slo.e2e_ms is None  # the example leaves one SLO target unset
    md = render_report(recipe)
    assert "None" not in md
    assert "goodput target" in md


def test_report_renders_booleans_yaml_style() -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    assert recipe.serve.args["enable_prefix_caching"] is True
    md = render_report(recipe)
    assert "| enable_prefix_caching | true | true |" in md
    assert "True" not in md


def test_report_says_throughput_only_when_no_slo_target_is_set() -> None:
    recipe = Recipe.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    recipe.slo.ttft_ms = None
    recipe.slo.itl_ms = None
    md = render_report(recipe)
    assert "SLO: none (throughput only)" in md


def test_json_schema_export_has_required_top_level_keys() -> None:
    schema = recipe_json_schema()
    assert {"model", "hardware", "engine", "serve", "infervolt"} <= set(schema["required"])


def test_cli_recipe_validate(tmp_path: Path) -> None:
    runner = CliRunner()
    ok = runner.invoke(app, ["recipe", "validate", str(EXAMPLE)])
    assert ok.exit_code == 0, ok.stdout
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {id: x}\n")
    res = runner.invoke(app, ["recipe", "validate", str(bad)])
    assert res.exit_code == 1


def test_cli_recipe_validate_missing_file_is_a_clean_usage_error(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["recipe", "validate", str(tmp_path / "nope.yaml")])
    assert res.exit_code == 2, res.output  # click's usage-error code, not a crash
    assert not isinstance(res.exception, OSError)
    assert "Traceback" not in res.output
