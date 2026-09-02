from pathlib import Path

from infervolt.core.types import Candidate, EngineConfig, OptimizeSpec, Trial
from infervolt.store.ledger import Ledger


def _trial(run_id: str, idx: int) -> Trial:
    cand = Candidate(
        id=f"c{idx}", config=EngineConfig(engine="mock", knobs={"a": idx}), origin="tpe"
    )
    return Trial(id=f"t{idx}", run_id=run_id, index=idx, candidate=cand, status="ok")


def test_create_run_and_roundtrip_trials(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs")
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
    ledger = Ledger(tmp_path / "ledger.sqlite", tmp_path / "runs")
    run_id = ledger.create_run(OptimizeSpec(engine="mock", model="mock/qwen3-8b"))
    ledger.set_state(run_id, "search")
    ledger.set_best(run_id, "t3")
    run = ledger.get_run(run_id)
    assert run.state == "search" and run.best_trial_id == "t3"
    assert ledger.trials_jsonl(run_id).name == "trials.jsonl"
