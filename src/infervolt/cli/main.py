"""infervolt command-line interface."""

from __future__ import annotations

import typer

from infervolt import __version__

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


if __name__ == "__main__":  # pragma: no cover
    app()
