"""infervolt command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from infervolt import __version__
from infervolt.cli.infra import infra_app
from infervolt.config import Settings
from infervolt.core.types import Budget, KnobValue, OptimizeSpec
from infervolt.hardware.profiles import PROFILES
from infervolt.recipes.schema import Recipe
from infervolt.workloads.presets import parse_slo

app = typer.Typer(help="Measure, diagnose, fix, verify, and remember LLM inference optimizations.")

HOME_HELP = "State directory holding the ledger and run artifacts (default ~/.infervolt)."


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
app.add_typer(infra_app, name="infra")


def _settings(home: Path | None) -> Settings:
    return Settings(home=home) if home else Settings()


LLM_NAMES = ("fake", "anthropic", "openai")


def _parse_kv(items: list[str]) -> dict[str, KnobValue]:
    """Parse ``k=v`` overrides, narrowing each value to the tightest type it parses as.

    Order matters: ``true``/``false`` before numbers (Python would read ``True`` as 1),
    ints before floats (``64`` is a sequence count, not 64.0), and anything left is a
    string -- which is what categorical knobs such as ``kv_cache_dtype=fp8`` want.

    An item with no ``=`` is a usage error, not an empty-string override: ``--baseline
    max_num_seqs 64`` (a space instead of an equals sign) would otherwise silently set
    the knob to ``""`` and measure something nobody asked for.
    """
    out: dict[str, KnobValue] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k:
            raise typer.BadParameter(f"expected knob=value, got {item!r}", param_hint="--baseline")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


@app.command()
def optimize(
    engine: str = typer.Option("mock", help="Engine adapter name (see entry points)."),
    model: str = typer.Option(..., help="Model id, e.g. mock/qwen3-8b"),
    hardware: str = typer.Option(
        "auto", help="Hardware profile (a100-80, h100-80, rtx4090-24, l4-24, m3-8)."
    ),
    workload: str = typer.Option("chat-4k-512", help="Workload preset."),
    slo: str = typer.Option("", help="SLO string, e.g. ttft=500ms,itl=30ms[,e2e=2s,p=0.9]"),
    llm: str = typer.Option("fake", help="fake | anthropic | openai"),
    max_trials: int = typer.Option(12, help="Trial budget for the search."),
    max_wall_s: float = typer.Option(3600.0, help="Wall-clock budget in seconds."),
    max_usd: float = typer.Option(
        0.0,
        help="Cost budget in USD; 0 means unlimited (cost accounting for real engines "
        "arrives in M3; mock cost uses the profile's usd_per_hour).",
    ),
    seed: int = typer.Option(7, help="Sampler and load-generator seed."),
    baseline: Annotated[
        list[str] | None,
        typer.Option("--baseline", help="Baseline knob override k=v (repeatable)."),
    ] = None,
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Run the full loop and emit a recipe."""
    import warnings

    import optuna

    from infervolt.agent.planner import Planner
    from infervolt.llm.factory import make_llm
    from infervolt.store.ledger import Ledger

    # ``TPESampler(multivariate=True)`` is a deliberate choice, not an accident to be
    # warned about once per call site on every run. pytest silences it through
    # ``filterwarnings``; the CLI has no such config, and a user's first impression of the
    # tool should not be a stack of Optuna warnings above their diagnosis. Imported here
    # rather than at module scope so ``--version`` and ``recipe validate`` stay fast.
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    if llm not in LLM_NAMES:
        raise typer.BadParameter(
            f"unknown llm {llm!r}; use {', '.join(LLM_NAMES)}", param_hint="--llm"
        )
    if hardware == "auto":
        # The default is "auto" so that M2 can turn it on without changing anyone's
        # command line; until then it is the one value the loop cannot serve, and saying
        # so here is cheaper than a run that dies after opening a ledger row.
        raise typer.BadParameter(
            f"auto-detect arrives in M2; pass a profile name ({', '.join(sorted(PROFILES))})",
            param_hint="--hardware",
        )
    settings = _settings(home)
    spec = OptimizeSpec(
        engine=engine,
        model=model,
        hardware=hardware,
        workload=workload,
        slo=parse_slo(slo),
        budget=Budget(max_trials=max_trials, max_wall_s=max_wall_s, max_usd=max_usd),
        baseline=_parse_kv(baseline or []),
        seed=seed,
        llm=llm,
    )
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        outcome = Planner(spec, settings, make_llm(llm, settings), ledger, log=typer.echo).run()
    if outcome.state != "done":
        raise typer.Exit(code=1)


@app.command()
def report(
    run_id: str,
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Print the report for a run."""
    from infervolt.store.ledger import Ledger

    settings = _settings(home)
    # The ledger is the authority on which runs exist, so an unknown id and a run that
    # exists but produced no report get different answers -- "never heard of it" and "it
    # got as far as <state>" are different problems with different next steps.
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        try:
            row = ledger.get_run(run_id)
        except KeyError:
            typer.echo(f"unknown run {run_id}", err=True)
            raise typer.Exit(code=1) from None
        state = row.state
    path = settings.runs_dir / run_id / "report.md"
    if not path.exists():
        typer.echo(f"no report for run {run_id} (state: {state})", err=True)
        raise typer.Exit(code=1)
    typer.echo(path.read_text())


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
