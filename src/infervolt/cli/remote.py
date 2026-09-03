"""``infervolt remote`` -- run the loop on a machine that has the GPU.

The command is a passthrough by design: everything after ``--`` is handed to the remote
``infervolt optimize`` untouched, so there is exactly one place where optimize flags are
defined and this file never has to grow a copy of them. That is what
``allow_extra_args``/``ignore_unknown_options`` buy -- a ``--max-trials`` added to
``optimize`` tomorrow works remotely today.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated

import typer

from infervolt.config import Settings
from infervolt.infra.base import Provider
from infervolt.infra.registry import get_provider
from infervolt.infra.ssh import SshSession
from infervolt.infra.types import InstanceSpec
from infervolt.remote.runner import RemoteRun
from infervolt.store.ledger import Ledger

remote_app = typer.Typer(help="Run infervolt on rented compute and bring the results home.")

HOME_HELP = "State directory holding the ledger and run artifacts (default ~/.infervolt)."
PROVIDER_HELP = "Provider name (see the infervolt.providers entry points)."
ENGINE_INSTALLS = ("venv", "docker")


def _settings(home: Path | None) -> Settings:
    return Settings(home=home) if home else Settings()


def _provider(name: str, ledger: Ledger | None = None) -> Provider:
    try:
        return get_provider(name, ledger)
    except KeyError as e:
        raise typer.BadParameter(str(e), param_hint="--provider") from None


def _passthrough(args: list[str]) -> list[str]:
    """Everything after ``--``, with the separator itself dropped.

    Click usually eats the separator, but not in every parsing mode, and an argv that
    starts with a bare ``--`` would reach the remote optimize as an empty option.
    """
    trimmed = args[1:] if args and args[0] == "--" else args
    if not trimmed:
        raise typer.BadParameter(
            "nothing to run remotely; put the optimize arguments after '--', e.g. "
            "remote optimize --provider thunder -- --engine vllm --model Qwen/Qwen3-8B"
        )
    return trimmed


@remote_app.command(
    "optimize",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def remote_optimize(
    ctx: typer.Context,
    provider: str = typer.Option("thunder", "--provider", help=PROVIDER_HELP),
    gpu: str = typer.Option("a100xl", "--gpu", help="GPU family, e.g. a100xl or h100."),
    count: int = typer.Option(1, "--count", help="GPUs on the instance."),
    disk: int = typer.Option(100, "--disk", help="Disk in GB; model weights live here."),
    max_usd: float = typer.Option(10.0, "--max-usd", help="Hard cost cap; 0 means no cap."),
    keep: Annotated[
        bool, typer.Option("--keep", help="Leave the instance running for the next run.")
    ] = False,
    instance: Annotated[
        str | None, typer.Option("--instance", help="Reuse this instance instead of renting one.")
    ] = None,
    ref: Annotated[
        str | None,
        typer.Option("--ref", help="Commit to install on the box (default: this checkout's HEAD)."),
    ] = None,
    engine_install: str = typer.Option(
        "venv", "--engine-install", help="How the engine gets onto the box: venv or docker."
    ),
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Provision a box, run optimize on it, pull the artifacts back, and terminate it.

    Everything after '--' is passed to the remote optimize unchanged. The instance is
    terminated on every exit path unless --keep is given, and the cost cap terminates it
    even then.
    """
    args = _passthrough(list(ctx.args))
    if engine_install not in ENGINE_INSTALLS:
        raise typer.BadParameter(
            f"expected {' or '.join(ENGINE_INSTALLS)}", param_hint="--engine-install"
        )
    settings = _settings(home)
    # On a container provider the image *is* the engine, so the docker path names one here
    # rather than letting the box guess; the venv path leaves it unset.
    image = f"vllm/vllm-openai:v{settings.vllm_version}" if engine_install == "docker" else None
    spec = InstanceSpec(gpu=gpu, count=count, disk_gb=disk, image=image)
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        outcome = RemoteRun(
            _provider(provider, ledger),
            spec,
            args,
            max_usd=max_usd,
            keep=keep,
            ref=ref,
            engine_install="docker" if engine_install == "docker" else "venv",
            instance_id=instance,
            ledger=ledger,
            settings=settings,
            log=typer.echo,
        ).run()

    typer.echo(f"{'ok' if outcome.ok else 'FAILED'}: {outcome.message}")
    typer.echo(
        f"instance {outcome.instance_id or '-'}  ${outcome.cost_usd:.3f}  {outcome.elapsed_s:.0f}s"
    )
    if outcome.local_run_dir is not None:
        typer.echo(f"run {outcome.run_id} -> {outcome.local_run_dir}")
    if not outcome.ok:
        raise typer.Exit(code=1)


@remote_app.command("shell")
def remote_shell(
    instance: str = typer.Option(..., "--instance", help="Instance id as shown by 'infra list'."),
    provider: str = typer.Option("thunder", "--provider", help=PROVIDER_HELP),
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Print the ssh command for an instance, so a human can go and look at the box.

    Printed rather than executed: the caller decides whether to run it, pipe it, or paste
    it into another terminal, and nothing here has to own a pty.
    """
    settings = _settings(home)
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        rows = [i for i in ledger.instances(provider=provider) if i.id == instance]
        if not rows:
            typer.echo(f"no live {provider} instance {instance!r}", err=True)
            raise typer.Exit(code=1)
        inst = _provider(provider, ledger).refresh(rows[0])
    if inst.ssh is None:
        typer.echo(f"{provider}:{instance} has no ssh address yet ({inst.status})", err=True)
        raise typer.Exit(code=1)
    typer.echo(shlex.join(SshSession(inst.ssh).ssh_argv()))
