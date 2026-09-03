import shutil
import threading
from pathlib import Path

import pytest

from infervolt.core.types import Candidate, EngineConfig, OptimizeSpec, Trial
from infervolt.store.ledger import Ledger


def _trial(run_id: str, idx: int) -> Trial:
    cand = Candidate(
        id=f"c{idx}", config=EngineConfig(engine="mock", knobs={"a": idx}), origin="tpe"
    )
    return Trial(id=f"t{idx}", run_id=run_id, index=idx, candidate=cand, status="ok")


def test_create_run_and_roundtrip_trials(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        spec = OptimizeSpec(engine="mock", model="mock/qwen3-8b")
        run_id = ledger.create_run(spec)
        assert (tmp_path / "runs" / run_id).is_dir()
        ledger.save_trial(_trial(run_id, 0))
        ledger.save_trial(_trial(run_id, 1))
        t1 = _trial(run_id, 1)
        t1.status = "pruned"
        ledger.save_trial(t1)  # upsert
        trials = ledger.trials(run_id)
        assert [t.index for t in trials] == [0, 1]
        assert trials[1].status == "pruned"
        assert ledger.get_run(run_id).spec.model == "mock/qwen3-8b"


def test_state_and_best_are_persisted(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        run_id = ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b"))
        ledger.set_state(run_id, "search")
        ledger.set_best(run_id, "t3")
        run = ledger.get_run(run_id)
        assert run.state == "search" and run.best_trial_id == "t3"
        assert ledger.trials_jsonl(run_id).name == "trials.jsonl"


def test_save_trial_recreates_a_missing_run_dir(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        run_id = ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b"))
        shutil.rmtree(ledger.run_dir(run_id))
        ledger.save_trial(_trial(run_id, 0))
        assert [t.index for t in ledger.trials(run_id)] == [0]
        assert ledger.trials_jsonl(run_id).read_text().count("\n") == 1


def test_create_run_twice_does_not_reset_state(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        spec = OptimizeSpec(engine="mock", model="mock/qwen3-8b", run_id="fixed-run")
        assert ledger.create_run(spec) == "fixed-run"
        ledger.set_state("fixed-run", "search")
        assert ledger.create_run(spec) == "fixed-run"
        assert ledger.get_run("fixed-run").state == "search"


def test_save_trial_works_from_another_thread(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger:
        run_id = ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b"))
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                ledger.save_trial(_trial(run_id, 0))
            except BaseException as exc:  # noqa: BLE001 - reported to the main thread
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert errors == []
        assert [t.index for t in ledger.trials(run_id)] == [0]


def test_set_state_on_an_unknown_run_raises(tmp_path: Path) -> None:
    with (
        Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs") as ledger,
        pytest.raises(KeyError),
    ):
        ledger.set_state("no-such-run", "search")
