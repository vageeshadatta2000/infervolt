"""``infervolt infra`` -- see what compute costs, what we are renting, and stop renting it.

Everything here is read-only except ``terminate`` and ``gc``, which destroy instances and
therefore ask first. There is no ``provision`` command on purpose: an instance nobody is
running a job on is a bill nobody is watching, so boxes are created by ``remote optimize``
(which owns their teardown) and this group exists to clean up after it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from infervolt.config import Settings
from infervolt.infra.base import Provider
from infervolt.infra.registry import available_providers, get_provider
from infervolt.infra.types import InfraError, Instance
from infervolt.store.ledger import Ledger

infra_app = typer.Typer(help="Rent, list and release GPU instances.")

HOME_HELP = "State directory holding the ledger and run artifacts (default ~/.infervolt)."
PROVIDER_HELP = "Provider name (see the infervolt.providers entry points)."


def _settings(home: Path | None) -> Settings:
    return Settings(home=home) if home else Settings()


def _provider(name: str, ledger: Ledger | None = None) -> Provider:
    try:
        return get_provider(name, ledger)
    except KeyError as e:
        raise typer.BadParameter(str(e), param_hint="--provider") from None


def _confirm(yes: bool, question: str) -> None:
    if not yes and not typer.confirm(question):
        typer.echo("aborted")
        raise typer.Exit(code=1)


@infra_app.command("providers")
def providers() -> None:
    """List the installed providers."""
    for name in available_providers():
        typer.echo(name)


@infra_app.command("offers")
def offers(
    provider: str = typer.Option("thunder", "--provider", help=PROVIDER_HELP),
    gpu: Annotated[str | None, typer.Option(help="Filter by GPU family, e.g. a100.")] = None,
) -> None:
    """Show what the provider is selling, at today's prices."""
    try:
        found = _provider(provider).list_offers(gpu)
    except InfraError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None
    if not found:
        typer.echo(f"no offers from {provider}" + (f" matching {gpu!r}" if gpu else ""))
        return
    typer.echo(f"{'OFFER':<16} {'GPU':<24} {'MEM':>7} {'N':>3} {'USD/H':>8}")
    for offer in found:
        typer.echo(
            f"{offer.raw_id:<16} {offer.gpu:<24} {offer.gpu_mem_gb:>6.0f}G "
            f"{offer.count:>3} {offer.usd_per_hour:>8.2f}"
        )


@infra_app.command("list")
def list_instances(
    provider: Annotated[str | None, typer.Option("--provider", help=PROVIDER_HELP)] = None,
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """List instances the ledger still considers live, with fresh status from the provider.

    The ledger is the source of truth for *what we are paying for*, and the provider for
    *what state it is in*. A row the provider no longer knows about shows as ``terminated``
    rather than disappearing, because a stale row is exactly what ``gc`` is for.
    """
    settings = _settings(home)
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        rows = ledger.instances(active_only=True, provider=provider)
        if not rows:
            typer.echo("no active instances")
            return
        typer.echo(f"{'PROVIDER':<10} {'ID':<14} {'GPU':<14} {'N':>3} {'USD/H':>7}  STATUS")
        for inst in rows:
            typer.echo(_row(_refreshed(inst, ledger)))


def _refreshed(inst: Instance, ledger: Ledger) -> Instance:
    """Best-effort status update. A provider that cannot be reached is not a reason to
    hide the row -- the row is how the user gets their money back."""
    try:
        return _provider(inst.provider, ledger).refresh(inst)
    except (InfraError, typer.BadParameter):
        return inst.model_copy(update={"status": f"{inst.status} (unreachable)"})


def _row(inst: Instance) -> str:
    host = inst.ssh.host if inst.ssh else "-"
    return (
        f"{inst.provider:<10} {inst.id:<14} {inst.gpu:<14} {inst.count:>3} "
        f"{inst.usd_per_hour:>7.2f}  {inst.status} {host}"
    )


@infra_app.command("terminate")
def terminate(
    instance_id: Annotated[str, typer.Argument(help="Instance id as shown by 'infra list'.")],
    provider: str = typer.Option(..., "--provider", help=PROVIDER_HELP),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Destroy one instance and close its ledger row."""
    settings = _settings(home)
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        known = {i.id: i for i in ledger.instances(active_only=True, provider=provider)}
        inst = known.get(instance_id) or Instance(
            provider=provider, id=instance_id, gpu="unknown", status="unknown"
        )
        _confirm(yes, f"terminate {provider} instance {instance_id}?")
        try:
            _provider(provider, ledger).terminate(inst)
        except InfraError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from None
        # Even an instance the ledger never knew about gets its row closed, so a second
        # terminate of the same id is quiet rather than a second API call.
        ledger.mark_terminated(instance_id, provider=provider)
    typer.echo(f"terminated {provider}:{instance_id}")


@infra_app.command("gc")
def gc(
    provider: Annotated[str | None, typer.Option("--provider", help=PROVIDER_HELP)] = None,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    home: Annotated[Path | None, typer.Option(help=HOME_HELP)] = None,
) -> None:
    """Terminate every instance the ledger still lists as live.

    The backstop for a controller that died before its ``finally`` ran. It reports each
    instance separately and keeps going after a failure: one provider being down must not
    leave the others billing.
    """
    settings = _settings(home)
    failures = 0
    with Ledger(settings.ledger_path, settings.runs_dir) as ledger:
        rows = ledger.instances(active_only=True, provider=provider)
        if not rows:
            typer.echo("nothing to collect")
            return
        _confirm(yes, f"terminate {len(rows)} instance(s)?")
        for inst in rows:
            try:
                _provider(inst.provider, ledger).terminate(inst)
            except (InfraError, typer.BadParameter) as e:
                failures += 1
                typer.echo(f"FAILED {inst.provider}:{inst.id}: {e}", err=True)
                continue
            typer.echo(f"terminated {inst.provider}:{inst.id} ({inst.gpu} x{inst.count})")
    if failures:
        raise typer.Exit(code=1)
