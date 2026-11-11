"""The record store of the right-sizing loop: one SQLite file in the plugin's own data folder.

Every turn is one row, and every model attempt inside a turn is one row, failures included.
Nothing leaves this file unless the user runs the export command.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS rs_turns (
    turn_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    platform TEXT,
    owner TEXT,
    exact TEXT NOT NULL,
    template TEXT NOT NULL,
    contract TEXT NOT NULL,
    fp TEXT NOT NULL,
    text TEXT,
    reply TEXT,
    base_model TEXT,
    rung TEXT,
    model TEXT,
    mode TEXT,
    reason TEXT,
    split TEXT NOT NULL,
    api_calls INTEGER DEFAULT 0,
    api_errors INTEGER DEFAULT 0,
    tok_in INTEGER DEFAULT 0,
    tok_out INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    tool_errors INTEGER DEFAULT 0,
    tools_used TEXT DEFAULT '[]',
    tools_total INTEGER DEFAULT 0,
    tools_kept INTEGER DEFAULT 0,
    tool_tokens_saved INTEGER DEFAULT 0,
    duration_s REAL DEFAULT 0,
    finish TEXT,
    escalated INTEGER DEFAULT 0,
    recalled TEXT,
    score REAL,
    weight REAL,
    source TEXT,
    closed INTEGER DEFAULT 0,
    imported INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS rs_turns_session ON rs_turns(session_id, seq);
CREATE INDEX IF NOT EXISTS rs_turns_exact ON rs_turns(exact);
CREATE TABLE IF NOT EXISTS rs_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_key TEXT NOT NULL,
    ts REAL NOT NULL,
    api_request_id TEXT,
    model TEXT,
    ok INTEGER NOT NULL,
    error TEXT,
    tok_in INTEGER DEFAULT 0,
    tok_out INTEGER DEFAULT 0,
    duration_s REAL DEFAULT 0,
    finish TEXT,
    tool_calls INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS rs_attempts_turn ON rs_attempts(turn_key);
CREATE TABLE IF NOT EXISTS rs_policy (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rs_policy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""

TURN_COLUMNS = (
    "turn_key", "session_id", "seq", "ts", "platform", "owner", "exact", "template", "contract", "fp", "text", "reply",
    "base_model", "rung", "model", "mode", "reason", "split", "api_calls", "api_errors", "tok_in", "tok_out",
    "tool_calls", "tool_errors", "tools_used", "tools_total", "tools_kept", "tool_tokens_saved", "duration_s",
    "finish", "escalated", "recalled", "score", "weight", "source", "closed", "imported",
)


class RecordStore:
    """Thread-safe store. One connection, one lock; every write is committed at once."""

    def __init__(self, path: str | Path) -> None:
        self.in_memory = str(path) == ":memory:"  # no file is made: Continuum is off and has no record file
        self.path = Path(path)
        if not self.in_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # A short wait: a turn must not hang on a record file that another process holds.
        self._db = sqlite3.connect(":memory:" if self.in_memory else str(self.path), check_same_thread=False,
                                   timeout=1.0)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            try:
                try:
                    self._db.execute("PRAGMA journal_mode=WAL")
                    self._db.execute("PRAGMA secure_delete=ON")  # removed text is overwritten, not only unlinked
                except sqlite3.OperationalError:
                    pass
                self._db.executescript(SCHEMA)
                have = {r[1] for r in self._db.execute("PRAGMA table_info(rs_turns)")}
                if "owner" not in have:  # a record file from before the column existed
                    self._db.execute("ALTER TABLE rs_turns ADD COLUMN owner TEXT")
                self._db.commit()
            except sqlite3.DatabaseError:
                self._db.close()
                raise

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One write, committed at once. A write that fails (for example on a file another process holds)
        is rolled back, so this connection keeps no open transaction and never reads an old state after it."""
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                try:
                    self._db.rollback()
                except sqlite3.Error:
                    pass
                raise

    # ---- turns ----
    def next_seq(self, session_id: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM rs_turns WHERE session_id = ?",
                                   (session_id,)).fetchone()
        return int(row[0])

    def open_turn(self, row: Mapping[str, Any]) -> None:
        cols = [c for c in TURN_COLUMNS if c in row]
        with self._write():
            self._db.execute(
                f"INSERT OR REPLACE INTO rs_turns ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [row[c] for c in cols])

    def update_turn(self, turn_key: str, **fields: Any) -> None:
        if not fields:
            return
        bad = [k for k in fields if k not in TURN_COLUMNS]
        if bad:
            raise ValueError(f"unknown turn fields: {bad}")
        with self._write():
            self._db.execute(f"UPDATE rs_turns SET {', '.join(k + ' = ?' for k in fields)} WHERE turn_key = ?",
                             [*fields.values(), turn_key])

    def bump_turn(self, turn_key: str, **deltas: float) -> None:
        bad = [k for k in deltas if k not in TURN_COLUMNS]
        if bad:
            raise ValueError(f"unknown turn fields: {bad}")
        with self._write():
            self._db.execute(
                f"UPDATE rs_turns SET {', '.join(f'{k} = COALESCE({k}, 0) + ?' for k in deltas)} WHERE turn_key = ?",
                [*deltas.values(), turn_key])

    def turn(self, turn_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM rs_turns WHERE turn_key = ?", (turn_key,)).fetchone()
        return dict(row) if row else None

    def last_turn(self, session_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if session_id:
                row = self._db.execute("SELECT * FROM rs_turns WHERE session_id = ? ORDER BY seq DESC LIMIT 1",
                                       (session_id,)).fetchone()
            else:
                row = self._db.execute("SELECT * FROM rs_turns WHERE imported = 0 ORDER BY ts DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def add_turns(self, rows: list[Mapping[str, Any]]) -> int:
        """Write many finished rows in one transaction (an import)."""
        if not rows:
            return 0
        with self._write():
            for row in rows:
                cols = [c for c in TURN_COLUMNS if c in row]
                self._db.execute(
                    f"INSERT OR REPLACE INTO rs_turns ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    [row[c] for c in cols])
        return len(rows)

    def scored_turns(self, limit: int, split: str | None = None, own_only: bool = False,
                     imported_only: bool = False) -> list[dict[str, Any]]:
        """Newest scored turns, without their text."""
        sql = ("SELECT turn_key, session_id, ts, platform, owner, exact, template, contract, fp, base_model, rung, model,"
               " mode, split, score, weight, source, tools_used, tok_in, tok_out, api_calls, escalated, imported,"
               " reply IS NOT NULL AS has_reply"
               " FROM rs_turns WHERE score IS NOT NULL AND weight > 0 AND rung IS NOT NULL"
               " AND COALESCE(mode, '') != 'mixed'")
        args: list[Any] = []
        if own_only:
            sql += " AND imported = 0"
        if imported_only:
            sql += " AND imported = 1"
        if split:
            sql += " AND split = ?"
            args.append(split)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["fp"] = json.loads(d["fp"])
                d["tools_used"] = json.loads(d["tools_used"] or "[]")
            except ValueError:
                continue  # an unreadable row is left out, never a reason to stop
            out.append(d)
        return out

    def open_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM rs_turns WHERE session_id = ? AND closed != 1 ORDER BY seq",
                                    (session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---- attempts ----
    def add_attempt(self, turn_key: str, **fields: Any) -> None:
        cols = ["turn_key", "ts", *fields.keys()]
        with self._write():
            self._db.execute(
                f"INSERT INTO rs_attempts ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [turn_key, time.time(), *fields.values()])

    def attempts(self, turn_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM rs_attempts WHERE turn_key = ? ORDER BY id", (turn_key,)).fetchall()
        return [dict(r) for r in rows]

    # ---- policy state ----
    def get_policy(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value FROM rs_policy WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_policy(self, key: str, value: Any) -> None:
        with self._write():
            self._db.execute("INSERT OR REPLACE INTO rs_policy (key, value, ts) VALUES (?, ?, ?)",
                             (key, json.dumps(value), time.time()))

    def set_policies(self, values: Mapping[str, Any]) -> None:
        """Write several policy values in one transaction: all of them, or none."""
        with self._write():
            for key, value in values.items():
                self._db.execute("INSERT OR REPLACE INTO rs_policy (key, value, ts) VALUES (?, ?, ?)",
                                 (key, json.dumps(value), time.time()))

    def change_policy(self, key: str, change: Callable[[Any], Any], default: Any = None) -> Any:
        """Read a policy value, change it and write it back in one transaction. Two Hermes processes on one
        record file then never write over each other's change (a pin, a block, a step of the floor).
        Returns the value that was written."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT value FROM rs_policy WHERE key = ?", (key,)).fetchone()
                value = change(json.loads(row[0]) if row else copy.deepcopy(default))
                self._db.execute("INSERT OR REPLACE INTO rs_policy (key, value, ts) VALUES (?, ?, ?)",
                                 (key, json.dumps(value), time.time()))
                self._db.commit()
            except BaseException:
                try:
                    self._db.rollback()
                except sqlite3.Error:
                    pass
                raise
            return value

    def log_policy(self, event: str, detail: Mapping[str, Any]) -> None:
        with self._write():
            self._db.execute("INSERT INTO rs_policy_log (ts, event, detail) VALUES (?, ?, ?)",
                             (time.time(), event, json.dumps(detail)))

    def policy_log(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT ts, event, detail FROM rs_policy_log ORDER BY id DESC LIMIT ?",
                                    (int(limit),)).fetchall()
        return [{"ts": r[0], "event": r[1], "detail": json.loads(r[2])} for r in rows]

    # ---- totals ----
    def totals(self) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(score IS NOT NULL), 0), COALESCE(SUM(tok_in), 0),"
                " COALESCE(SUM(tok_out), 0), COALESCE(SUM(api_calls), 0), COALESCE(SUM(api_errors), 0),"
                " COALESCE(SUM(escalated), 0), COALESCE(SUM(tool_tokens_saved), 0), COALESCE(SUM(recalled IS NOT NULL), 0)"
                " FROM rs_turns WHERE imported = 0").fetchone()
            by = self._db.execute(
                "SELECT rung, mode, COUNT(*), COALESCE(SUM(tok_in), 0), COALESCE(SUM(tok_out), 0),"
                " COALESCE(SUM(score * weight), 0), COALESCE(SUM(weight), 0)"
                " FROM rs_turns WHERE imported = 0 GROUP BY rung, mode").fetchall()
            attempts = self._db.execute("SELECT COUNT(*), COALESCE(SUM(ok = 0), 0) FROM rs_attempts").fetchone()
            imported = self._db.execute("SELECT COUNT(*) FROM rs_turns WHERE imported = 1").fetchone()
            checks = self._db.execute("SELECT COUNT(*), COALESCE(SUM(score = 0), 0) FROM rs_turns"
                                      " WHERE source = 'reference_check'").fetchone()
        return {
            "turns": row[0], "scored": row[1], "tok_in": row[2], "tok_out": row[3], "api_calls": row[4],
            "api_errors": row[5], "escalated": row[6], "tool_tokens_saved": row[7], "recalled": row[8],
            "attempts": attempts[0], "failed_attempts": attempts[1], "imported": imported[0],
            "checks": checks[0], "checks_failed": checks[1],
            "by_rung": [{"rung": r[0], "mode": r[1], "turns": r[2], "tok_in": r[3], "tok_out": r[4],
                         "passes": r[5], "evidence": r[6]} for r in by],
        }

    # ---- export and import ----
    def export_rows(self) -> Iterable[dict[str, Any]]:
        """Scored turns as fingerprints and outcomes only. No Task text and no reply text.

        The two text hashes of a fingerprint (exact and template) are left out: they are the easiest
        part to test a guess against. The rest (sketch, word buckets, shape) is still a fingerprint of
        the text, so the file is not for people who must not learn what the Tasks were about."""
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, contract, fp, model, score, weight, source, tools_used, tok_in, tok_out, api_calls"
                " FROM rs_turns WHERE score IS NOT NULL AND weight > 0 AND imported = 0"
                " AND COALESCE(mode, '') != 'mixed' ORDER BY ts").fetchall()
        for r in rows:
            d = dict(r)
            try:
                fp = json.loads(d["fp"])
            except ValueError:
                continue
            d["fp"] = {k: fp.get(k) for k in ("contract", "sketch", "words", "profile", "n_words")}
            d["tools_used"] = json.loads(d["tools_used"] or "[]")
            yield d

    def size_bytes(self) -> int:
        """Size of the record file with its side file, in bytes (0 with no file)."""
        if self.in_memory:
            return 0
        total = 0
        for name in (str(self.path), str(self.path) + "-wal"):
            try:
                total += os.path.getsize(name)
            except OSError:
                pass
        return total

    def forget_text(self, before: float | None = None, compact: bool = True, max_bytes: int = 0) -> int:
        """Remove the stored Task text and reply text of every turn, or of the turns older than the time
        `before`. Fingerprints and outcomes stay.

        The rows are cleared at once (with `secure_delete` the old bytes of the changed pages are overwritten).
        `compact` also rewrites the file, so that no old copy of a page stays in it or in its side file. That
        rewrite reads the whole file, so it is not done on the path of a user's turn: `compact=False` leaves a
        mark, and the next start of Hermes or the next `forget-text` does it. With `max_bytes` set, a file
        larger than that is not rewritten here either (the start of Hermes must stay short). See `compact`.

        The mark is written in the same transaction as the removal, so it cannot be lost."""
        with self._write():
            cur = self._db.execute("UPDATE rs_turns SET text = NULL, reply = NULL"
                                   " WHERE (text IS NOT NULL OR reply IS NOT NULL)"
                                   + (" AND ts < ?" if before is not None else ""),
                                   (float(before),) if before is not None else ())
            n = int(cur.rowcount or 0)
            if n:
                self._db.execute("INSERT OR REPLACE INTO rs_policy (key, value, ts) VALUES (?, ?, ?)",
                                 ("compact_pending", json.dumps(True), time.time()))
        if compact and (n or self.compact_pending()) and not (max_bytes and self.size_bytes() > max_bytes):
            self.compact()
        return n

    def compact_pending(self) -> bool:
        """True when text was removed and the file was not rewritten since."""
        try:
            return bool(self.get_policy("compact_pending", False))
        except sqlite3.Error:
            return False

    def _mark_compact(self, pending: bool) -> None:
        try:
            if self.compact_pending() != pending:
                self.set_policy("compact_pending", pending)
        except sqlite3.Error:
            pass  # the mark is a reminder only; the rows are already cleared

    def compact(self) -> bool:
        """Rewrite the file and empty its side file, so that removed text stays in no old page. Returns False
        when another process holds the file; the mark then stays and the next start tries again."""
        if self.in_memory:
            return True
        with self._lock:
            done = False
            try:
                self._db.execute("VACUUM")
                row = self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                done = bool(row) and int(row[0]) == 0  # first column: 1 when a reader or writer blocked it
            except sqlite3.Error:
                try:
                    self._db.rollback()
                except sqlite3.Error:
                    pass
        self._mark_compact(not done)
        return done

    def text_turns(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM rs_turns WHERE text IS NOT NULL OR reply IS NOT NULL").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def read_snapshot(path: str | Path, recent: int = 12) -> dict[str, Any]:
    """Read-only view of the record file for the dashboard. Opens the file read-only and never creates it.

    The view holds counts, outcomes and choices. It holds no Task text and no reply text.
    """
    path = Path(path)
    empty = {"present": False, "turns": 0, "scored": 0, "by_model": [], "by_mode": [], "by_source": [], "recent": [],
             "policy": {}, "policy_log": [], "checks": 0, "checks_failed": 0, "escalated": 0, "tool_tokens_saved": 0,
             "imported": 0}
    if not path.is_file():
        return empty
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return empty
    try:
        one = lambda sql: db.execute(sql).fetchone()  # noqa: E731
        head = one("SELECT COUNT(*), COALESCE(SUM(score IS NOT NULL), 0), COALESCE(SUM(escalated), 0),"
                   " COALESCE(SUM(tool_tokens_saved), 0) FROM rs_turns WHERE imported = 0")
        checks = one("SELECT COUNT(*), COALESCE(SUM(score = 0), 0) FROM rs_turns WHERE source = 'reference_check'")
        by_model = db.execute(
            "SELECT COALESCE(model, ''), COUNT(*), COALESCE(SUM(tok_in), 0), COALESCE(SUM(tok_out), 0),"
            " COALESCE(SUM(score * weight), 0), COALESCE(SUM(weight), 0), COALESCE(SUM(duration_s), 0)"
            " FROM rs_turns WHERE imported = 0 AND COALESCE(mode, '') != 'mixed'"  # two models answered: no model's
            " GROUP BY model ORDER BY COUNT(*) DESC").fetchall()
        by_mode = db.execute("SELECT COALESCE(mode, ''), COUNT(*) FROM rs_turns WHERE imported = 0"
                             " GROUP BY mode ORDER BY COUNT(*) DESC").fetchall()
        by_source = db.execute("SELECT COALESCE(source, 'open'), COUNT(*) FROM rs_turns WHERE imported = 0"
                               " GROUP BY source ORDER BY COUNT(*) DESC").fetchall()
        last = db.execute(
            "SELECT ts, model, mode, reason, score, weight, source, tok_in, tok_out, escalated, tools_total, tools_kept"
            " FROM rs_turns WHERE imported = 0 ORDER BY ts DESC LIMIT ?", (max(0, int(recent)),)).fetchall()
        policy = {r[0]: r[1] for r in db.execute("SELECT key, value FROM rs_policy")}
        log = db.execute("SELECT ts, event, detail FROM rs_policy_log ORDER BY id DESC LIMIT 8").fetchall()
        imported = one("SELECT COUNT(*) FROM rs_turns WHERE imported = 1")[0]
    except sqlite3.Error:
        return empty
    finally:
        db.close()
    return {
        "present": True, "turns": head[0], "scored": head[1], "escalated": head[2], "tool_tokens_saved": head[3],
        "checks": checks[0], "checks_failed": checks[1], "imported": imported,
        "by_model": [{"model": r[0] or "(not recorded)", "turns": r[1], "tok_in": r[2], "tok_out": r[3],
                      "pass_rate": (r[4] / r[5]) if r[5] else None, "evidence": r[5], "seconds": r[6]} for r in by_model],
        "by_mode": [{"mode": r[0] or "(none)", "turns": r[1]} for r in by_mode],
        "by_source": [{"source": r[0], "turns": r[1]} for r in by_source],
        "recent": [{"ts": r[0], "model": r[1], "mode": r[2], "reason": (r[3] or "")[:160], "score": r[4],
                    "weight": r[5], "source": r[6], "tok_in": r[7], "tok_out": r[8], "escalated": r[9],
                    "tools_total": r[10], "tools_kept": r[11]} for r in last],
        "policy": policy,
        "policy_log": [{"ts": r[0], "event": r[1], "detail": json.loads(r[2])} for r in log],
    }
