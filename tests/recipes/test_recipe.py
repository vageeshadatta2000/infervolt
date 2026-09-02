from pathlib import Path

import yaml
from typer.testing import CliRunner

from infervolt.cli.main import app
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
