"""Experiment tracking with SQLite for evaluation runs."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Stored experiment record."""

    experiment_id: int
    name: str
    created_at: str
    seed: int
    run_hash: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    notes: str = ""
    artifacts: Dict[str, Any] = None


class ExperimentTracker:
    """Track experiment metadata and metrics in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    run_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    notes TEXT,
                    artifacts_json TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def compute_run_hash(config: Dict[str, Any], seed: int) -> str:
        """Compute deterministic hash for config and seed."""
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = sha256(f"{payload}:{seed}".encode("utf-8")).hexdigest()
        return digest

    def log_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        seed: int,
        notes: str = "",
        artifacts: Optional[Dict[str, Any]] = None
    ) -> int:
        """Insert a new experiment record and return its ID."""
        created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        run_hash = self.compute_run_hash(config, seed)
        config_json = json.dumps(config, sort_keys=True)
        metrics_json = json.dumps(metrics, sort_keys=True)
        artifacts_json = json.dumps(artifacts or {}, sort_keys=True)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                (name, created_at, seed, run_hash, config_json, metrics_json, notes, artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, created_at, seed, run_hash, config_json, metrics_json, notes, artifacts_json)
            )
            conn.commit()
            experiment_id = cursor.lastrowid
        finally:
            conn.close()

        logger.info("Logged experiment %s with id %s", name, experiment_id)
        return int(experiment_id)

    def get_experiment(self, experiment_id: int) -> Optional[ExperimentRecord]:
        """Fetch a single experiment record."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT id, name, created_at, seed, run_hash, config_json, metrics_json, notes, artifacts_json
                FROM experiments
                WHERE id = ?
                """,
                (experiment_id,)
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return None

        return ExperimentRecord(
            experiment_id=row[0],
            name=row[1],
            created_at=row[2],
            seed=row[3],
            run_hash=row[4],
            config=json.loads(row[5]),
            metrics=json.loads(row[6]),
            notes=row[7] or "",
            artifacts=json.loads(row[8]) if row[8] else {}
        )

    def list_experiments(self, limit: int = 50) -> Iterable[ExperimentRecord]:
        """List recent experiments."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT id, name, created_at, seed, run_hash, config_json, metrics_json, notes, artifacts_json
                FROM experiments
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            yield ExperimentRecord(
                experiment_id=row[0],
                name=row[1],
                created_at=row[2],
                seed=row[3],
                run_hash=row[4],
                config=json.loads(row[5]),
                metrics=json.loads(row[6]),
                notes=row[7] or "",
                artifacts=json.loads(row[8]) if row[8] else {}
            )
