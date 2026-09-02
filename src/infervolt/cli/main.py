"""infervolt command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from infervolt import __version__
from infervolt.recipes.schema import Recipe

app = typer.Typer(help="Measure, diagnose, fix, verify, and remember LLM inference optimizations.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"infervolt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """infervolt CLI."""


recipe_app = typer.Typer(help="Recipe utilities.")
app.add_typer(recipe_app, name="recipe")


@recipe_app.command("validate")
def recipe_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a recipe.yaml against the infervolt schema."""
    try:
        Recipe.model_validate(yaml.safe_load(path.read_text()))
    except (ValidationError, yaml.YAMLError, OSError) as e:
        typer.echo(f"INVALID {path}: {e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"OK {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
