"""What the runner does to a box, in what order, and what it does when that goes wrong.

The interesting assertions are the ones about money: every path out of ``run()`` -- success,
a bootstrap that never finished, a run that outlived its budget -- has to end with the
instance terminated, because the alternative is a GPU billing overnight.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infervolt.config import Settings
from infervolt.infra.types import InstanceSpec
from infervolt.remote.bootstrap import BOOTSTRAP_OK
from infervolt.remote.runner import RemoteRun
from infervolt.store.ledger import Ledger
from tests.remote.conftest import RUN_ID, WORK, FakeProvider, FakeSession

SHA = "0123456789abcdef0123456789abcdef01234567"
ARGS = ["--engine", "vllm", "--model", "Qwen/Qwen3-8B", "--hardware", "a100-80"]


def make_run(
    provider: FakeProvider,
    tmp_path: Path,
    *,
    log: list[str] | None = None,
    keep: bool = False,
) -> RemoteRun:
    sink = log if log is not None else []
    return RemoteRun(
        provider,
        InstanceSpec(gpu="a100xl", disk_gb=100),
        ARGS,
        max_usd=10.0,
        keep=keep,
        ref=SHA,
        settings=Settings(home=tmp_path),
        log=sink.append,
    )


def kinds(session: FakeSession) -> list[str]:
    """Label each recorded command by what it obviously is."""
    out = []
    for cmd in session.commands:
        if "pkill" in cmd:
            out.append("kill")
        elif BOOTSTRAP_OK in cmd:
            out.append("bootstrap")
        elif "optimize" in cmd:
            out.append("optimize")
        else:  # pragma: no cover - the runner issues nothing else
            out.append("other")
    return out


def test_a_happy_run_provisions_bootstraps_optimizes_pulls_and_terminates(
    provider: FakeProvider, session: FakeSession, tmp_path: Path
) -> None:
    outcome = make_run(provider, tmp_path).run()

    assert [s.gpu for s in provider.provisioned] == ["a100xl"]
    assert provider.waited == ["i-fake"]
    assert kinds(session) == ["bootstrap", "optimize"]
    assert session.gets == [(f"{WORK}/state/runs/{RUN_ID}", tmp_path / "runs" / RUN_ID)]
    assert provider.terminated == ["i-fake"]
    assert session.closed

    assert outcome.ok and outcome.run_id == RUN_ID
    assert outcome.instance_id == "i-fake"
    assert outcome.local_run_dir == tmp_path / "runs" / RUN_ID
    assert (tmp_path / "runs" / RUN_ID / "recipe.yaml").exists()
    assert outcome.elapsed_s >= 0.0 and outcome.cost_usd >= 0.0


def test_the_bootstrap_pins_the_ref_and_the_optimize_args_are_passed_through(
    provider: FakeProvider, session: FakeSession, tmp_path: Path
) -> None:
    make_run(provider, tmp_path).run()
    bootstrap, optimize = session.commands
    assert f"git+https://github.com/vageeshadatta2000/infervolt@{SHA}" in bootstrap
    assert f"{WORK}/venv/bin/infervolt optimize --home '{WORK}/state'" in optimize
    assert "--model Qwen/Qwen3-8B" in optimize


def test_keep_leaves_the_instance_running(
    provider: FakeProvider, session: FakeSession, tmp_path: Path
) -> None:
    outcome = make_run(provider, tmp_path, keep=True).run()
    assert provider.terminated == []
    assert outcome.ok and "kept" in outcome.message


def test_a_bootstrap_without_its_marker_never_starts_a_paid_run(
    tmp_path: Path,
) -> None:
    session = FakeSession(bootstrap_ok=False)
    provider = FakeProvider(session=session)
    outcome = make_run(provider, tmp_path).run()

    assert kinds(session) == ["bootstrap"]  # no optimize
    assert provider.terminated == ["i-fake"]
    assert not outcome.ok and "bootstrap" in outcome.message
    assert outcome.run_id is None and outcome.local_run_dir is None


def test_a_bootstrap_that_exits_nonzero_is_the_same_kind_of_abort(tmp_path: Path) -> None:
    session = FakeSession(bootstrap_code=1)
    provider = FakeProvider(session=session)
    outcome = make_run(provider, tmp_path).run()
    assert kinds(session) == ["bootstrap"]
    assert provider.terminated == ["i-fake"] and not outcome.ok


def test_the_cost_cap_kills_the_run_and_terminates_the_box(tmp_path: Path) -> None:
    session = FakeSession(block_optimize=True)
    # $3600/h is $1 a second, so a one-cent cap fires while the fake run is still going.
    provider = FakeProvider(session=session, usd_per_hour=3600.0)
    outcome = RemoteRun(
        provider,
        InstanceSpec(gpu="a100xl"),
        ARGS,
        max_usd=0.01,
        ref=SHA,
        settings=Settings(home=tmp_path),
        guard_interval_s=0.01,
    ).run()

    assert kinds(session) == ["bootstrap", "optimize", "kill"]
    assert provider.terminated == ["i-fake"]
    assert not outcome.ok and outcome.message == "cost cap reached"
    assert outcome.cost_usd >= 0.01


def test_the_cost_cap_terminates_even_when_keep_was_asked_for(tmp_path: Path) -> None:
    session = FakeSession(block_optimize=True)
    provider = FakeProvider(session=session, usd_per_hour=3600.0)
    outcome = RemoteRun(
        provider,
        InstanceSpec(gpu="a100xl"),
        ARGS,
        max_usd=0.01,
        ref=SHA,
        keep=True,
        settings=Settings(home=tmp_path),
        guard_interval_s=0.01,
    ).run()
    # --keep is a convenience; the cap is a promise.
    assert provider.terminated == ["i-fake"] and not outcome.ok


def test_an_existing_instance_is_reused_instead_of_provisioning(
    session: FakeSession, tmp_path: Path
) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider = FakeProvider(ledger, session=session)
        kept = provider.provision(InstanceSpec(gpu="a100xl"))
        provider.provisioned.clear()

        outcome = RemoteRun(
            provider,
            InstanceSpec(gpu="a100xl"),
            ARGS,
            instance_id=kept.id,
            ref=SHA,
            keep=True,
            ledger=ledger,
            settings=Settings(home=tmp_path),
        ).run()

    assert provider.provisioned == []
    assert provider.waited == ["i-fake"]
    assert outcome.ok and outcome.instance_id == "i-fake"


def test_an_unknown_instance_id_is_refused_before_anything_is_rented(
    provider: FakeProvider, tmp_path: Path
) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        outcome = RemoteRun(
            provider,
            InstanceSpec(gpu="a100xl"),
            ARGS,
            instance_id="i-nope",
            ref=SHA,
            ledger=ledger,
            settings=Settings(home=tmp_path),
        ).run()
    assert not outcome.ok and "i-nope" in outcome.message
    assert provider.provisioned == [] and provider.terminated == []


def test_a_provisioning_failure_is_reported_not_raised(tmp_path: Path) -> None:
    provider = FakeProvider(fail_provision=True)
    outcome = make_run(provider, tmp_path).run()
    assert not outcome.ok and "capacity" in outcome.message
    assert outcome.instance_id == ""


def test_a_failing_optimize_still_pulls_the_artifacts_back(tmp_path: Path) -> None:
    # A run that ended in "no win" or crashed mid-search still wrote a run dir, and that
    # dir is the only evidence of what the GPU minutes bought.
    session = FakeSession(optimize_code=1)
    provider = FakeProvider(session=session)
    outcome = make_run(provider, tmp_path).run()
    assert session.gets and outcome.run_id == RUN_ID
    assert not outcome.ok and "exited 1" in outcome.message
    assert provider.terminated == ["i-fake"]


def test_a_run_that_printed_no_run_id_pulls_nothing(tmp_path: Path) -> None:
    session = FakeSession(run_id=None, optimize_code=2)
    provider = FakeProvider(session=session)
    outcome = make_run(provider, tmp_path).run()
    assert session.gets == [] and outcome.run_id is None
    assert provider.terminated == ["i-fake"]


def test_every_transition_is_logged_with_elapsed_time_and_running_cost(
    provider: FakeProvider, tmp_path: Path
) -> None:
    lines: list[str] = []
    make_run(provider, tmp_path, log=lines).run()
    prefixed = [line for line in lines if line.startswith("[")]
    assert [line.split("] ", 1)[1].split(" ", 1)[0] for line in prefixed] == [
        "provision",
        "ready",
        "bootstrap",
        "optimize",
        "pull",
        "terminate",
        "done",
    ]
    assert all("s $" in line for line in prefixed)
    # The remote stream is passed through unprefixed, so the user sees the run's own output.
    assert f"run: {RUN_ID}" in lines


def test_without_a_ref_and_without_a_checkout_the_run_refuses_to_start(
    provider: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infervolt.remote import runner as runner_mod

    monkeypatch.setattr(runner_mod, "controller_sha", lambda cwd=None: None)
    outcome = RemoteRun(
        provider, InstanceSpec(gpu="a100xl"), ARGS, settings=Settings(home=tmp_path)
    ).run()
    assert not outcome.ok and "--ref" in outcome.message
    assert provider.provisioned == []


def test_the_ref_defaults_to_the_controllers_own_commit(
    provider: FakeProvider, session: FakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infervolt.remote import runner as runner_mod

    monkeypatch.setattr(runner_mod, "controller_sha", lambda cwd=None: "f" * 40)
    RemoteRun(provider, InstanceSpec(gpu="a100xl"), ARGS, settings=Settings(home=tmp_path)).run()
    assert f"@{'f' * 40}" in session.commands[0]


def test_only_environment_variables_that_are_set_reach_the_box(
    provider: FakeProvider, session: FakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_controller")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    make_run(provider, tmp_path).run()
    assert "export HF_TOKEN='hf_from_controller'" in session.commands[0]
    assert "ANTHROPIC_API_KEY" not in session.commands[0]


def test_the_docker_engine_install_uses_the_specs_image(
    session: FakeSession, tmp_path: Path
) -> None:
    provider = FakeProvider(session=session)
    RemoteRun(
        provider,
        InstanceSpec(gpu="a100xl", image="vllm/vllm-openai:v0.11.0"),
        ARGS,
        ref=SHA,
        engine_install="docker",
        settings=Settings(home=tmp_path),
    ).run()
    assert "docker pull 'vllm/vllm-openai:v0.11.0'" in session.commands[0]
