"""Job store: SQLite for state, plain files for clips and previews.

Queue order: normal-priority jobs before low-priority (backfill) ones, and within a priority the NEWEST first, so a
fresh detection is never stuck behind a backlog.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PENDING, RUNNING, READY, FAILED = "pending", "running", "ready", "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  detection_id INTEGER PRIMARY KEY,
  state        TEXT NOT NULL,
  priority     INTEGER NOT NULL DEFAULT 0,
  created_at   REAL NOT NULL,
  started_at   REAL,
  finished_at  REAL,
  species      TEXT,
  scientific   TEXT,
  camera       TEXT,
  clip_path    TEXT,
  info         TEXT,
  error        TEXT,
  attempts     INTEGER NOT NULL DEFAULT 0,
  nbytes       INTEGER NOT NULL DEFAULT 0,
  cleaned      INTEGER NOT NULL DEFAULT 0,
  took_s       REAL,
  device       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (state, priority, created_at);
CREATE INDEX IF NOT EXISTS jobs_finished ON jobs (finished_at);
"""


def iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    detection_id: int
    state: str
    priority: int
    created_at: float
    started_at: float | None
    finished_at: float | None
    species: str | None
    scientific: str | None
    camera: str | None
    clip_path: str | None
    info: dict[str, Any] | None
    error: str | None
    attempts: int
    nbytes: int
    cleaned: bool
    took_s: float | None
    device: str | None


def _row_to_job(r: sqlite3.Row) -> Job:
    return Job(r["detection_id"], r["state"], r["priority"], r["created_at"], r["started_at"], r["finished_at"], r["species"],
               r["scientific"], r["camera"], r["clip_path"], json.loads(r["info"]) if r["info"] else None, r["error"],
               r["attempts"], r["nbytes"], bool(r["cleaned"]), r["took_s"], r["device"])


class Store:
    def __init__(self, data_dir: Path, inbox: Path, previews: Path, db_path: Path):
        self.inbox, self.previews = inbox, previews
        for d in (data_dir, inbox, previews):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    # ------------------------------------------------------------------ paths
    def clip_path(self, det: int) -> Path:
        return self.inbox / f"{det}.clip"

    def preview_path(self, det: int) -> Path:
        return self.previews / f"{det}.m4a"

    def info_path(self, det: int) -> Path:
        return self.previews / f"{det}.json"

    # ------------------------------------------------------------------ writes
    def submit(self, det: int, clip: bytes, *, species: str, scientific: str | None, camera: str | None, low_priority: bool = False,
               force: bool = False, now: float | None = None) -> tuple[Job, bool]:
        """Queue a job. Idempotent per detection id: a repeat returns the existing job unchanged (created=False) unless
        `force`, which replaces it. Returns (job, created)."""
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE detection_id=?", (det,)).fetchone()
            if row is not None:
                if not force or row["state"] == RUNNING:
                    return _row_to_job(row), False
                self._remove_files(det)
            path = self.clip_path(det)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(clip)
            tmp.replace(path)
            self._db.execute(
                "INSERT OR REPLACE INTO jobs (detection_id,state,priority,created_at,species,scientific,camera,clip_path,attempts)"
                " VALUES (?,?,?,?,?,?,?,?,0)",
                (det, PENDING, 1 if low_priority else 0, now, species, scientific, camera, str(path)))
            return self.get(det), True

    def claim_next(self, now: float | None = None) -> Job | None:
        """Take the next job to run (marks it running)."""
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute(
                "SELECT detection_id FROM jobs WHERE state=? ORDER BY priority ASC, created_at DESC LIMIT 1", (PENDING,)).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET state=?, started_at=?, attempts=attempts+1 WHERE detection_id=?", (RUNNING, now, row[0]))
            return self.get(row[0])

    def finish(self, det: int, info: dict[str, Any], *, nbytes: int, cleaned: bool, took_s: float, device: str,
               now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET state=?, finished_at=?, info=?, error=NULL, nbytes=?, cleaned=?, took_s=?, device=? WHERE detection_id=?",
                (READY, now, json.dumps(info), nbytes, 1 if cleaned else 0, took_s, device, det))
            self._drop_clip(det)
        self.info_path(det).write_text(json.dumps(info))

    def fail(self, det: int, error: str, *, took_s: float | None = None, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute("UPDATE jobs SET state=?, finished_at=?, error=?, took_s=? WHERE detection_id=?",
                             (FAILED, now, error[:500], took_s, det))
            self._drop_clip(det)

    def requeue(self, det: int) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET state=?, started_at=NULL WHERE detection_id=? AND state=?", (PENDING, det, RUNNING))

    def delete(self, det: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM jobs WHERE detection_id=?", (det,))
            self._remove_files(det)
            return cur.rowcount > 0

    def recover(self) -> int:
        """After a restart: jobs that were running go back to the queue if their clip is still there, else fail."""
        n = 0
        with self._lock:
            for r in self._db.execute("SELECT detection_id, clip_path FROM jobs WHERE state=?", (RUNNING,)).fetchall():
                if r["clip_path"] and Path(r["clip_path"]).exists():
                    self._db.execute("UPDATE jobs SET state=?, started_at=NULL WHERE detection_id=?", (PENDING, r["detection_id"]))
                else:
                    self._db.execute("UPDATE jobs SET state=?, error=? WHERE detection_id=?", (FAILED, "interrupted by a restart", r["detection_id"]))
                n += 1
        return n

    # ------------------------------------------------------------------ reads
    def get(self, det: int) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE detection_id=?", (det,)).fetchone()
        return _row_to_job(row) if row else None

    def queue_position(self, job: Job) -> int | None:
        """1 = next in line. None unless the job is waiting."""
        if job.state != PENDING:
            return None
        with self._lock:
            n = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state=? AND (priority<? OR (priority=? AND created_at>?))",
                (PENDING, job.priority, job.priority, job.created_at)).fetchone()[0]
        return int(n) + 1

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT state, COUNT(*) c FROM jobs GROUP BY state").fetchall()
            cleaned = self._db.execute("SELECT COUNT(*) FROM jobs WHERE state=? AND cleaned=1", (READY,)).fetchone()[0]
        by = {r["state"]: r["c"] for r in rows}
        return {"ready": by.get(READY, 0), "failed": by.get(FAILED, 0), "cleaned": int(cleaned),
                "pending": by.get(PENDING, 0), "running": by.get(RUNNING, 0)}

    def today(self, since: float) -> dict[str, int]:
        with self._lock:
            r = self._db.execute(
                "SELECT SUM(state=?) ready, SUM(state=? AND cleaned=1) cleaned, SUM(state=?) failed FROM jobs WHERE finished_at>=?",
                (READY, READY, FAILED, since)).fetchone()
        ready, cleaned, failed = (r[0] or 0), (r[1] or 0), (r[2] or 0)
        return {"processed": int(ready + failed), "cleaned": int(cleaned), "failed": int(failed)}

    def timing(self, n: int = 50) -> dict[str, float | None]:
        with self._lock:
            rows = self._db.execute(
                "SELECT took_s FROM jobs WHERE state=? AND took_s IS NOT NULL ORDER BY finished_at DESC LIMIT ?", (READY, n)).fetchall()
        v = sorted(r[0] for r in rows)
        return {"medianS": round(v[len(v) // 2], 2) if v else None, "lastS": round(rows[0][0], 2) if rows else None}

    def last_errors(self, n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT detection_id, finished_at, error FROM jobs WHERE state=? ORDER BY finished_at DESC LIMIT ?", (FAILED, n)).fetchall()
        return [{"at": iso(r["finished_at"]), "detectionId": r["detection_id"], "error": r["error"]} for r in rows]

    def storage(self) -> dict[str, int]:
        with self._lock:
            r = self._db.execute("SELECT COUNT(*), COALESCE(SUM(nbytes),0) FROM jobs WHERE state=?", (READY,)).fetchone()
        return {"previews": int(r[0]), "bytes": int(r[1])}

    # ------------------------------------------------------------------ retention
    def purge(self, *, max_age_days: float, cap_bytes: int, now: float | None = None) -> list[int]:
        """Delete finished jobs older than `max_age_days`, then the oldest ones until previews fit in `cap_bytes`."""
        now = time.time() if now is None else now
        removed: list[int] = []
        with self._lock:
            cutoff = now - max_age_days * 86400
            old = [r[0] for r in self._db.execute(
                "SELECT detection_id FROM jobs WHERE state IN (?,?) AND COALESCE(finished_at, created_at) < ?", (READY, FAILED, cutoff)).fetchall()]
            for det in old:
                self.delete(det)
                removed.append(det)
            total = self._db.execute("SELECT COALESCE(SUM(nbytes),0) FROM jobs WHERE state=?", (READY,)).fetchone()[0]
            if total > cap_bytes:
                for r in self._db.execute("SELECT detection_id, nbytes FROM jobs WHERE state=? ORDER BY finished_at ASC", (READY,)).fetchall():
                    if total <= cap_bytes:
                        break
                    self.delete(r["detection_id"])
                    removed.append(r["detection_id"])
                    total -= r["nbytes"]
        return removed

    # ------------------------------------------------------------------ files
    def _drop_clip(self, det: int) -> None:
        self.clip_path(det).unlink(missing_ok=True)

    def _remove_files(self, det: int) -> None:
        for p in (self.clip_path(det), self.preview_path(det), self.info_path(det)):
            p.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._db.close()
