"""SQLite ledger of runs and trials plus a per-run artifact directory.

Schema is intentionally tiny: rows hold pydantic JSON so the models stay the source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from infervolt.core.types import OptimizeSpec, RunState, Trial

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, created REAL NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
  best_trial_id TEXT, recipe_path TEXT, diagnosis TEXT
);
CREATE TABLE IF NOT EXISTS trials (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS trials_run ON trials(run_id, idx);
"""


class RunRow(BaseModel):
    id: str
    created: float
    spec: OptimizeSpec
    state: RunState
    best_trial_id: str | None = None
    recipe_path: str | None = None
    diagnosis_json: str | None = None


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


class Ledger:
    def __init__(self, db_path: Path, runs_dir: Path) -> None:
        self.db_path = db_path
        self.runs_dir = runs_dir
        db_path.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)

    # ---- runs
    def create_run(self, spec: OptimizeSpec) -> str:
        run_id = spec.run_id or new_run_id()
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(id, created, spec, state) VALUES (?,?,?,?)",
                (run_id, time.time(), spec.model_dump_json(), "prepare"),
            )
        return run_id

    def get_run(self, run_id: str) -> RunRow:
        row = self._conn.execute(
            "SELECT id, created, spec, state, best_trial_id, recipe_path, diagnosis "
            "FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRow(
            id=row[0],
            created=row[1],
            spec=OptimizeSpec.model_validate_json(row[2]),
            state=row[3],
            best_trial_id=row[4],
            recipe_path=row[5],
            diagnosis_json=row[6],
        )

    def set_state(self, run_id: str, state: RunState) -> None:
        with self._conn:
            self._conn.execute("UPDATE runs SET state=? WHERE id=?", (state, run_id))

    def set_best(self, run_id: str, trial_id: str | None) -> None:
        with self._conn:
            self._conn.execute("UPDATE runs SET best_trial_id=? WHERE id=?", (trial_id, run_id))

    def set_recipe(self, run_id: str, path: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE runs SET recipe_path=? WHERE id=?", (path, run_id))

    def set_diagnosis(self, run_id: str, diagnosis_json: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE runs SET diagnosis=? WHERE id=?", (diagnosis_json, run_id))

    # ---- trials
    def save_trial(self, trial: Trial) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO trials(id, run_id, idx, data) VALUES (?,?,?,?)",
                (trial.id, trial.run_id, trial.index, trial.model_dump_json()),
            )
        with self.trials_jsonl(trial.run_id).open("a") as f:
            f.write(json.dumps({"event": "trial", "trial": trial.model_dump(mode="json")}) + "\n")

    def trials(self, run_id: str) -> list[Trial]:
        rows = self._conn.execute(
            "SELECT data FROM trials WHERE run_id=? ORDER BY idx", (run_id,)
        ).fetchall()
        return [Trial.model_validate_json(r[0]) for r in rows]

    # ---- artifacts
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def trials_jsonl(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trials.jsonl"

    def close(self) -> None:
        self._conn.close()
