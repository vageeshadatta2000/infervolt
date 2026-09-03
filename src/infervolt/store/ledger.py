"""SQLite ledger of runs and trials plus a per-run artifact directory.

Schema is intentionally tiny: rows hold pydantic JSON so the models stay the source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel

from infervolt.core.types import OptimizeSpec, RunState, Trial

# ``infra.types`` imports nothing internal, so persisting rented boxes here does not put
# the ledger inside the provider import cycle.
from infervolt.infra.types import Instance

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
CREATE TABLE IF NOT EXISTS instances (
  id TEXT NOT NULL, provider TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL,
  terminated REAL, PRIMARY KEY (provider, id)
);
CREATE INDEX IF NOT EXISTS instances_live ON instances(terminated);
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
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # ---- runs
    def create_run(self, spec: OptimizeSpec) -> str:
        run_id = spec.run_id or new_run_id()
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs(id, created, spec, state) VALUES (?,?,?,?)",
                (run_id, time.time(), spec.model_dump_json(), "prepare"),
            )
        return run_id

    def get_run(self, run_id: str) -> RunRow:
        with self._lock:
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

    def _update_run(self, run_id: str, sql: str, value: object) -> None:
        """Apply a single-column update, raising ``KeyError`` when the run does not exist."""
        with self._lock, self._conn:
            cursor = self._conn.execute(sql, (value, run_id))
        if cursor.rowcount == 0:
            raise KeyError(run_id)

    def set_state(self, run_id: str, state: RunState) -> None:
        self._update_run(run_id, "UPDATE runs SET state=? WHERE id=?", state)

    def set_best(self, run_id: str, trial_id: str | None) -> None:
        self._update_run(run_id, "UPDATE runs SET best_trial_id=? WHERE id=?", trial_id)

    def set_recipe(self, run_id: str, path: str) -> None:
        self._update_run(run_id, "UPDATE runs SET recipe_path=? WHERE id=?", path)

    def set_diagnosis(self, run_id: str, diagnosis_json: str) -> None:
        self._update_run(run_id, "UPDATE runs SET diagnosis=? WHERE id=?", diagnosis_json)

    # ---- trials
    def save_trial(self, trial: Trial) -> None:
        """Upsert the trial row and append it to the run's JSONL event log.

        The JSONL file is an append-only event log, not a table: an upsert of an
        already-saved trial appends a second line for the same trial id. Readers must
        therefore take the *last* event per trial id; the SQLite row is the current value.
        """
        self.run_dir(trial.run_id).mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO trials(id, run_id, idx, data) VALUES (?,?,?,?)",
                (trial.id, trial.run_id, trial.index, trial.model_dump_json()),
            )
        with self.trials_jsonl(trial.run_id).open("a") as f:
            f.write(json.dumps({"event": "trial", "trial": trial.model_dump(mode="json")}) + "\n")

    def trials(self, run_id: str) -> list[Trial]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM trials WHERE run_id=? ORDER BY idx, id", (run_id,)
            ).fetchall()
        return [Trial.model_validate_json(r[0]) for r in rows]

    # ---- instances
    def save_instance(self, instance: Instance) -> None:
        """Write the row that lets a *different* process terminate this box.

        That is the whole point of persisting it: a controller which crashes mid-run is
        the case where a rented GPU bills all night, and the only thing that can stop it
        is a row on disk naming the provider and the id. Re-saving an instance updates
        the payload but keeps ``created`` -- and keeps ``terminated``, so a refresh after
        teardown cannot resurrect a dead row into ``infra gc``.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO instances(id, provider, data, created, terminated) "
                "VALUES (?,?,?,?,NULL) "
                "ON CONFLICT(provider, id) DO UPDATE SET data=excluded.data",
                (
                    instance.id,
                    instance.provider,
                    instance.model_dump_json(),
                    instance.created_at,
                ),
            )

    def instances(self, active_only: bool = True, provider: str | None = None) -> list[Instance]:
        sql = "SELECT data FROM instances"
        clauses: list[str] = []
        params: list[object] = []
        if active_only:
            clauses.append("terminated IS NULL")
        if provider is not None:
            clauses.append("provider=?")
            params.append(provider)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Instance.model_validate_json(r[0]) for r in rows]

    def mark_terminated(self, instance_id: str, provider: str | None = None) -> None:
        """Close the row. Silent about unknown ids: termination is called on paths that
        must not raise, and "it is not in the ledger" is the state we wanted anyway."""
        sql = "UPDATE instances SET terminated=? WHERE id=? AND terminated IS NULL"
        params: list[object] = [time.time(), instance_id]
        if provider is not None:
            sql += " AND provider=?"
            params.append(provider)
        with self._lock, self._conn:
            self._conn.execute(sql, params)

    # ---- artifacts
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def trials_jsonl(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trials.jsonl"

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
