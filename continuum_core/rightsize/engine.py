"""The right-sizing loop for one Hermes profile.

    start_turn      fingerprint the Task, read the records of similar Tasks, choose a rung
    shape_request   apply the choice to the provider request (model, settings, tool list)
    record_attempt  record every model call of the turn, failures included
    record_tool     record every tool call of the turn
    end_turn        store the result; settle the outcome when the turn itself shows it
    feedback        an explicit outcome from the user or from a check
    end_session     close what is open, audit the policy on held-out records, tune the similarity weights

With the setting `mode: off` the loop records nothing new and changes no request; with no record file yet it
makes none. The age limit of stored text (`text_days`) still holds in off mode. Only `auto` and `observe` turn the
loop on: every other value of `mode` (a typing mistake, or the `false` that YAML makes of an unquoted `off`) is
read as off. A request is changed (its model, its tool list, an added past answer) only in `mode: auto` and
only while the host says the switch is granted.

The loop never raises into Hermes: the plugin wraps each call. It makes no network call of its own.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from . import chooser as ch
from . import fingerprint as fpm
from .store import RecordStore

logger = logging.getLogger("continuum.rightsize")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "rightsize.json"
MEMORY = ":memory:"
SQLITE_HEADER = b"SQLite format 3\x00"  # a store that writes no file: used when Continuum is off and no record file exists


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        fn = tool.get("function")
        if isinstance(fn, Mapping) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
    return ""


def _approx_tokens(value: Any) -> int:
    try:
        return len(json.dumps(value, default=str)) // 4
    except Exception:
        return 0


def _num(value: Any, default: float) -> float:
    """A setting typed by a person: a wrong value falls back to the default and never stops a turn."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


_YES = ("true", "yes", "on", "1")
_NO = ("false", "no", "off", "0")


def _flag(value: Any, default: bool, unknown: bool = False) -> bool:
    """A yes/no setting typed by a person. `"false"`, `"no"`, `"off"` and `"0"` are no, also as text. A value
    that is neither a yes nor a no is read as `unknown`, which is the careful side for that setting."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        name = value.strip().lower()
        if name in _YES:
            return True
        if name in _NO:
            return False
    return unknown


def _error_kind(error: Any, reason: Any = "", status: Any = 0) -> str:
    """What is kept of a failed model call: short names only, never the provider's message text."""
    kind = error if isinstance(error, str) else type(error).__name__ if error is not None else ""
    kind = kind if re.fullmatch(r"[A-Za-z_][\w.]{0,59}", kind or "") else "error"  # a name, not a sentence
    parts = [kind, _reason_word(reason), str(int(status)) if status else ""]
    return " ".join(p for p in parts if p)


def _reason_word(reason: Any) -> str:
    """The host's reason for a failed call, when it is one short word such as `rate_limit`. Anything else (a
    sentence, a part of the provider's message) is kept as "other", so that no provider text is stored."""
    word = str(reason or "").strip()
    return word if re.fullmatch(r"[a-z0-9_.-]{1,40}", word) else "other" if word else ""


_CLEAN_ADDRESS = re.compile(r"(?:[a-z][a-z0-9+.-]*://[^/#?@\sA-Z]+|unreadable)(?:#p=[0-9a-f]{8})?")


def _mark(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]


def _clean_address(url: Any) -> str:
    """What is kept of a provider address: scheme, host and port, and a short one-way mark of the path. The
    user name, password, query and fragment are dropped. A key that someone wrote anywhere into an address
    is not stored; two addresses with different paths still compare as different. Cleaning a cleaned value
    changes nothing."""
    raw = str(url or "").strip()
    if not raw or _CLEAN_ADDRESS.fullmatch(raw):
        return raw
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        if not parts.scheme or not host:
            return "unreadable#p=" + _mark(raw)  # not an address this code can read: only the mark is kept
        if ":" in host:
            host = "[" + host + "]"
        port = ":" + str(parts.port) if parts.port else ""
        path = parts.path.rstrip("/")
    except ValueError:
        return "unreadable#p=" + _mark(raw)
    return parts.scheme.lower() + "://" + host + port + ("#p=" + _mark(path) if path else "")


MARK_SOURCES = ch.MARK_SOURCES  # a user's mark and a trusted check: the only outcomes that can move a Task


def owner_of(platform: str, sender_id: str) -> str:
    """A short one-way mark of who sent a message on which platform. The sender id itself is not stored."""
    return hashlib.sha256(f"{platform or ''}\x1f{sender_id or ''}".encode("utf-8")).hexdigest()[:16]


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _merge(dict(base[k]), v)
        else:
            base[k] = copy.deepcopy(v)
    return base


class RightSizer:
    def __init__(self, db_path: str | Path, settings: Mapping[str, Any] | None = None,
                 config: Mapping[str, Any] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.cfg = copy.deepcopy(dict(config)) if config else load_config()
        self.settings = {**self.cfg["settings"], **{k: v for k, v in (settings or {}).items() if v is not None}}
        self.clock = clock
        self.store_note = ""
        self.store_note_names_file = False  # a note that names a file on the machine is for the local user only
        self.store = self._open_store(Path(db_path))
        self._lock = threading.RLock()
        self._live: dict[str, dict[str, Any]] = {}
        self._pools: dict[str, ch.Pool] | None = None
        self._text_checked = float("-inf")  # when the age limit of stored text was last applied
        self._new_outcomes = 1  # outcomes written since the last audit; 1 so that the first audit always runs
        self._held: set[str] = set()  # sessions kept on the user's own model after a refused request
        self._tool_miss: set[str] = set()  # sessions whose last turn needed a tool that was not sent
        self._unwritten: dict[str, dict[str, Any]] = {}  # end-of-turn fields that could not be written yet
        self.defaults = {"floor": float(self.cfg["chooser"]["floor"]),
                         "min_evidence": float(self.cfg["chooser"]["min_evidence"])}
        # The host decides whether this plugin may change the model of a request. Hermes wires its consent
        # check in here (capability llm.model_override); without consent the loop records and does not switch.
        self.switch_allowed: Callable[[], bool] = lambda: True
        self.switch_denied_note = "the model switch is not granted"
        try:
            # The age limit of stored text holds in every mode, off included. The start of Hermes is also the
            # place where the file is rewritten after text was removed (never on the path of a turn). Every
            # process that loads the plugin passes here, also a short `hermes continuum` command beside a
            # running gateway, so the rewrite at a start is for small files only: it then takes a moment and
            # no turn of another process waits long for it. A larger file is rewritten by `forget-text`.
            # The limit is a value of config/rightsize.json, not a Hermes setting. 0 or less: never at a start.
            mib = _num(self.cfg["settings"].get("rewrite_at_start_max_mib"), 32.0)
            limit = max(1, int(max(0.0, mib) * 1024 * 1024))
            self.expire_text(force=True, compact=False)
            if self.store.compact_pending() and self.store.size_bytes() <= limit:
                self.store.compact()  # one try; a file that is held keeps its mark for the next start
        except (sqlite3.Error, OSError):
            pass  # another process holds the file: the text is removed at the end of a session or within the
            # hour, and the rewrite waits for the next start, `forget-text` or `compact`

    # ---------- settings ----------
    def _open_store(self, path: Path) -> RecordStore:
        """Open the record file. Off with no file yet: nothing is created. A file that is in use, or a folder
        that cannot be written, is left as it is and Continuum is off for this run. Only a file that is not a
        record file at all is set aside (kept under another name, never deleted) and a new one is started."""
        if self.mode == "off" and not path.exists():
            return RecordStore(MEMORY)
        try:
            # A file with another header is never handed to SQLite: SQLite removes the side files of a file
            # it cannot read, and they are kept with it here.
            with path.open("rb") as fh:
                head = fh.read(16)
            if head and head != SQLITE_HEADER:
                raise sqlite3.DatabaseError("file is not a database")
        except FileNotFoundError:
            pass
        except sqlite3.DatabaseError as exc:
            if self.mode == "off":  # off changes nothing: the file is not renamed and no new one is made
                return self._off_for_this_run(f"the record file could not be read ({type(exc).__name__}); it was "
                                              "left as it is")
            return self._set_aside(path, exc)
        except OSError as exc:
            return self._off_for_this_run(f"the record file could not be opened ({type(exc).__name__}); it was "
                                          "left as it is")
        try:
            return RecordStore(path)
        except (sqlite3.OperationalError, OSError) as exc:
            return self._off_for_this_run(f"the record file could not be opened ({type(exc).__name__}); it was "
                                          "left as it is")
        except sqlite3.DatabaseError as exc:
            if self.mode == "off":
                return self._off_for_this_run(f"the record file could not be read ({type(exc).__name__}); it was "
                                              "left as it is")
            return self._set_aside(path, exc)

    def _set_aside(self, path: Path, exc: Exception) -> RecordStore:
        """Keep a file that is not a record file under another name, with its side files, and start a new one."""
        aside = path.with_name(f"{path.stem}.unreadable-{int(self.clock())}{path.suffix}")
        try:
            path.rename(aside)
            for tail in ("-wal", "-shm"):  # the side files belong to the old file
                side = path.with_name(path.name + tail)
                if side.exists():
                    side.rename(aside.with_name(aside.name + tail))
            store = RecordStore(path)
        except (sqlite3.Error, OSError):
            return self._off_for_this_run(f"the record file could not be read ({type(exc).__name__}) and "
                                          "could not be set aside; it was left as it is")
        self.store_note_names_file = True
        self.store_note = (f"the record file could not be read ({type(exc).__name__}); it was kept as "
                           f"{aside.name} and a new one was started")
        store.log_policy("record-file-set-aside", {"kept_as": aside.name})
        return store

    def _off_for_this_run(self, why: str) -> RecordStore:
        self.settings["mode"] = "off"
        self.store_note = why + ", and Continuum is off until Hermes starts again"
        return RecordStore(MEMORY)

    @property
    def mode(self) -> str:
        """auto, observe or off. Every value that is not `auto` or `observe` is off."""
        raw = self.settings.get("mode", "auto")
        name = raw.strip().lower() if isinstance(raw, str) else ""
        return name if name in ("auto", "observe") else "off"

    def mode_note(self) -> str:
        raw = self.settings.get("mode", "auto")
        if self.mode != "off" or raw is False or (isinstance(raw, str) and raw.strip().lower() == "off"):
            return ""
        return f"the setting `mode` has the value {str(raw)[:40]!r}, which is not auto, observe or off; it is read as off"

    def flag(self, name: str) -> bool:
        """A yes/no setting. A value that is neither yes nor no turns the setting off: no text is stored, no
        record is added to a turn, and the policy is not changed by Continuum itself."""
        return _flag(self.settings.get(name), bool(self.cfg["settings"].get(name, False)))

    def text_days(self) -> float:
        """Age limit of stored text in days. 0 keeps the text; a value that is no number, or below 0, is the default."""
        default = float(self.cfg["settings"].get("text_days", 90))
        days = _num(self.settings.get("text_days"), default)
        return days if days >= 0 else default

    def strictness(self) -> dict[str, float]:
        return self.store.get_policy("strictness", dict(self.defaults))

    def preset(self) -> str:
        name = self.store.get_policy("preset", self.cfg["fingerprint"]["default_preset"])
        return name if name in self.cfg["fingerprint"]["presets"] else self.cfg["fingerprint"]["default_preset"]

    def rungs(self, base_model: str) -> list[ch.Rung]:
        """Cheapest first, the reference last. A rung is named by the model it runs, so the records of
        one model are never read as the records of another."""
        raw = self.settings.get("rungs") or ""
        entries: list[Mapping[str, Any]] = []
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                entries = [e for e in parsed if isinstance(e, Mapping)] if isinstance(parsed, list) else []
            except ValueError:
                entries = []
        elif isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, Mapping)]
        small = self._small_model()
        if not entries and small:
            entries = [{"model": small, "cost": _num(self.settings.get("small_model_cost_ratio"), 0.2),
                        "max_context": self.settings.get("small_model_max_context")}]
        blocked = self.blocked()
        out: list[ch.Rung] = []
        for e in entries:
            model = str(e.get("model") or "").strip()
            cost = _num(e.get("cost"), 0.5)
            if not model or model in ("*", base_model) or not 0.0 < cost < 1.0 or model in blocked:
                continue  # a rung must be another model, cheaper than the reference, and not refused by the provider
            params = e.get("params")
            out.append(ch.Rung(name=model, model=model, cost=cost, params=dict(params) if isinstance(params, Mapping) else {},
                               max_context=max(0, int(_num(e.get("max_context"), 0)))))
        out.sort(key=lambda r: r.cost)
        out.append(ch.Rung(name=base_model or "reference", model="", cost=1.0))
        return out

    def _small_model(self) -> str:
        raw = self.settings.get("small_model")
        return raw.strip() if isinstance(raw, str) else ""  # a list or a number is no model name

    def _rung_signature(self) -> str:
        """A short hash of the rung settings. Blocks and the provider of the cheaper model hold for one set of
        rung settings; the settings themselves (they can carry request parameters) are not stored."""
        raw = json.dumps([self._small_model(), self.settings.get("rungs") or ""], default=str, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def pinned(self) -> list[str]:
        """Sessions the user pinned to their own model, oldest pin first. Read from the record file each time,
        so a pin set in another Hermes process on the same file holds here too."""
        raw = self.store.get_policy("pinned", [])
        return [str(x) for x in raw] if isinstance(raw, list) else []

    def blocked(self) -> dict[str, Any]:
        """Cheaper models that the provider refused. A block lasts until the rung settings change or the
        user runs /continuum unblock."""
        blocks = self.store.get_policy("blocked_rungs", {})
        sig = self._rung_signature()
        return {m: b for m, b in blocks.items() if isinstance(b, Mapping) and b.get("settings") == sig}

    def unblock(self) -> list[str]:
        with self._lock:
            names = sorted(self.blocked())
            self.store.set_policies({"blocked_rungs": {}, "rung_strikes": {}, "rung_home": {}})
            if names:
                self.store.log_policy("rung-unblocked", {"models": names})
            self._held.clear()
            return names

    def _home(self) -> Mapping[str, Any] | None:
        home = self.store.get_policy("rung_home", {})
        return home if isinstance(home, Mapping) and home.get("settings") == self._rung_signature() else None

    def _at_home(self, live: dict[str, Any], provider: str, base_url: str) -> bool:
        """A cheaper model is named for one provider. The provider and address where a cheaper model first
        answered are kept (see record_attempt: a request that failed there keeps nothing). A turn that runs
        somewhere else (the user changed the provider, or Hermes fell back to another one) is left alone,
        because the model name may mean nothing there."""
        here = {"provider": str(provider or ""), "base_url": _clean_address(base_url)}
        if not here["provider"] and not here["base_url"]:
            return True  # the host named no provider: nothing to compare
        home = self._home()
        if home is None:
            live["home_candidate"] = here  # kept when this request is answered
            return True
        return home.get("provider") == here["provider"] and _clean_address(home.get("base_url")) == here["base_url"]

    def _recall_allowed(self, platform: str, sender_id: str) -> bool:
        """Recorded text goes back only to a sender the host names. A platform with one local user (the list is
        in the config file) needs no sender id; on every other platform an empty sender id gets no recall."""
        return bool(str(sender_id or "").strip()) or str(platform or "") in self.cfg["recall"].get("single_user_platforms", [])

    def _may_switch(self) -> bool:
        try:
            return bool(self.switch_allowed())
        except Exception:  # unreadable consent never grants
            return False

    def _may_change(self) -> bool:
        """True when Continuum may change a request at all: its model, its tool list, or added context."""
        return self.mode == "auto" and self._may_switch()

    def may_change(self) -> bool:
        return self._may_change()

    # ---------- pools ----------
    def _pool(self, kind: str) -> ch.Pool:
        with self._lock:
            if self._pools is None:
                limit = int(self.cfg["chooser"]["max_pool"])
                # Imported records have their own share of the pool, so no import file can push the user's
                # own records out of it.
                rows = self.store.scored_turns(limit, own_only=True) + self.store.scored_turns(
                    int(self.cfg.get("import", {}).get("max_pool", 1000)), imported_only=True)
                trust = float(self.cfg.get("import", {}).get("trust", 0.5))
                strong = float(self.cfg["chooser"].get("strong_weight", 0.5))
                for r in rows:
                    r["rung"] = r["model"] or r["rung"]
                    # Strong is about the kind of signal, so it is decided before the import discount (see
                    # chooser.is_strong: a PASS of the user's own model and a thank-you are always weak).
                    r["strong"] = ch.is_strong(r, strong)
                    if r["imported"]:
                        r["weight"] *= trust
                fp_cfg = self.cfg["fingerprint"]
                weights = fp_cfg["presets"][self.preset()]
                self._pools = {"fit": ch.Pool([r for r in rows if r["split"] == "fit"], fp_cfg, weights),
                               "any": ch.Pool(rows, fp_cfg, weights)}
            return self._pools[kind]

    def _dirty(self) -> None:
        with self._lock:
            self._pools = None

    def _forget_mark_since(self, t0: float) -> bool | None:
        """True when forget-text ran (in this or another process) at or after `t0`, False when it did not, None
        when the mark cannot be read."""
        try:
            mark = self.store.get_policy("text_forgotten_at", 0)
            return isinstance(mark, (int, float)) and bool(mark) and float(mark) >= float(t0)
        except (sqlite3.Error, ValueError, TypeError, OverflowError):
            return None

    def _forgotten_since(self, t0: float) -> bool:
        """For an answer: True when forget-text ran at or after `t0`, and also when the mark cannot be read (the
        answer is then not stored; the rest of the turn's end is written as usual)."""
        return self._forget_mark_since(t0) is not False

    def _marks_recently(self, scope: tuple[str, str]) -> bool:
        """True when this user (platform and sender) marked one of their last `explore_needs_marks` scored turns
        (a mark or a trusted check, MARK_SOURCES).

        A trial turn on a cheaper model teaches Continuum something only when somebody marks it: a cheaper model
        moves only on checked outcomes, and a PASS from the user's own model is not one (see `choose`). Without
        recent marks a trial turn would cost accuracy and could never move a Task, so none is made."""
        window = int(_num(self.cfg["chooser"].get("explore_needs_marks"), 0))
        if window <= 0:
            return True
        seen = 0
        for r in self._pool("any").records:  # newest first; the user's own records come before imported ones
            if r.get("imported"):
                break
            if (str(r.get("platform") or ""), str(r.get("owner") or "")) != scope:
                continue
            if r.get("source") in MARK_SOURCES:
                return True
            seen += 1
            if seen >= window:
                break
        return False

    # ---------- the turn ----------
    def start_turn(self, session_id: str, user_message: str, model: str = "", platform: str = "",
                   sender_id: str = "") -> ch.Decision:
        session_id = session_id or "default"
        try:
            self.expire_text()  # at most once an hour: a gateway that runs for months still drops old text
        except (sqlite3.Error, OSError):
            pass
        if self.mode == "off":  # off is off: no fingerprint, no record, no change
            with self._lock:
                self._live.pop(session_id, None)
                self._tool_miss.discard(session_id)
            return ch.Decision(model or "reference", "", "off", "Continuum is off")
        with self._lock:  # the state of the turn before never shapes this turn, whatever happens below
            previous = self._live.pop(session_id, None)
        text = str(user_message or "")
        tokens = len(text) // 3  # a careful guess of the size of the message; the request is measured later
        scope = (str(platform or ""), owner_of(platform, sender_id))
        fp = fpm.fingerprint(text, self.cfg["fingerprint"])
        with self._lock:
            restore_tools = bool((previous and previous.get("tool_miss")) or (session_id in self._tool_miss))
            self._tool_miss.discard(session_id)
            self._settle_previous(session_id, fp, text)
            if len(self._live) > 200:  # sessions that never ended
                for stale in sorted(self._live, key=lambda k: self._live[k]["t0"])[:100]:
                    self._live.pop(stale, None)
            seq = self.store.next_seq(session_id)
            turn_key = f"{session_id}:{seq}"
            rungs = self.rungs(model)
            state = self.strictness()
            if session_id in self.pinned():
                dec = ch.Decision(rungs[-1].name, "", "pinned", "this session is pinned to your own model")
            elif session_id in self._held:
                dec = ch.Decision(rungs[-1].name, "", "pinned", "the provider refused a request on a cheaper model "
                                  "earlier in this session; the session stays on your own model")
            else:
                explore = min(1.0, max(0.0, _num(self.settings.get("explore_share"), 0.1)))
                trials_wait = explore > 0 and not self._marks_recently(scope)
                dec = ch.choose(fp, text, rungs, self._pool("fit"), self.cfg["chooser"], floor=state["floor"],
                                min_evidence=state["min_evidence"], explore_share=explore,
                                u=ch.uniform(turn_key), request_tokens=tokens, trials=not trials_wait)
                if dec.trials_waiting and self.mode == "auto" and self._may_switch():
                    dec.reason += "; trial turns wait until you mark one of your recent turns (/continuum good|bad)"
                # Observe mode, or no consent for the model: nothing is switched and no check question is
                # sent (a turn kept on the user's own model for a check is part of the switching loop).
                if dec.mode in ("exploit", "explore", "reference-check") and (
                        self.mode == "observe" or not self._may_switch()):
                    note = "observe mode" if self.mode == "observe" else self.switch_denied_note
                    dec = ch.Decision(rungs[-1].name, "", "observe", note + "; would have chosen: " + dec.reason,
                                      floor=dec.floor, best_similarity=dec.best_similarity, neighbours=dec.neighbours,
                                      estimates=dec.estimates)
            # Adding a past answer changes the request, so it needs the same grant as a model switch. The past
            # answer must come from the same platform and the same sender.
            # A turn that is kept on the user's own model (pin, hold, never-downsize) is not changed at all.
            recall_ok = self._recall_on() and self._recall_allowed(platform, sender_id)
            if recall_ok and dec.mode != "pinned" and self._may_change():
                dec.recall_key, dec.recall_similarity = ch.find_recall(fp, self._pool("any"), self.cfg["recall"], scope)
            keep_text = self.flag("store_text")
            now = self.clock()  # one start time for the row and the turn: forget-text compares with it
            self.store.open_turn({
                "turn_key": turn_key, "session_id": session_id, "seq": seq, "ts": now, "platform": scope[0],
                "owner": scope[1],
                "exact": fp["exact"], "template": fp["template"], "contract": fp["contract"], "fp": json.dumps(fp),
                "text": text if keep_text else None, "base_model": model, "rung": dec.rung,
                "model": dec.model or model, "mode": dec.mode, "reason": dec.reason,
                "split": ch.split_of(fp["template"], float(self.cfg["holdout"]["share"])),
                "recalled": dec.recall_key,
            })
            self._live[session_id] = {"turn_key": turn_key, "decision": dec, "fp": fp, "text": text, "t0": now,
                                      "rungs": rungs, "escalated": False, "tools_done": False, "tools_used": [],
                                      "small_calls": 0, "last_finish": "", "tool_miss": False,
                                      "restore_tools": restore_tools, "scope": scope, "recall_ok": recall_ok}
        return dec

    def _recall_on(self) -> bool:
        """Recall needs stored text: with `store_text: false` it is off, also for text stored earlier."""
        return self.flag("recall") and self.flag("store_text")

    def scope_of(self, session_id: str) -> tuple[str, str] | None:
        """Platform and sender mark of the turn that is running in a session, if that turn may get records."""
        live = self._live.get(session_id or "default")
        return live["scope"] if live and live.get("recall_ok") else None

    def recall_context(self, session_id: str) -> str | None:
        """Text for Hermes to add to the turn: the stored reply of a very similar Task that was marked good."""
        live = self._live.get(session_id or "default")
        if not live or not live["decision"].recall_key or not live.get("recall_ok") or not self._may_change():
            return None
        row = self.store.turn(live["decision"].recall_key)
        if not row or not row.get("reply") or not ch.in_scope(row, live["scope"]):
            return None
        cap = int(self.cfg["recall"]["max_chars"])
        task = (row.get("text") or "")[: cap // 2]
        return ("[Continuum record] A very similar Task was answered before and the answer was marked good.\n"
                f"Past Task: {task}\nPast answer: {row['reply'][:cap]}\n"
                "Use it only where it fits the current Task; the current Task decides.")

    def shape_request(self, request: Mapping[str, Any], session_id: str, model: str = "",
                      api_call_count: int = 0, provider: str = "", base_url: str = "") -> tuple[dict[str, Any], str] | None:
        """Return the changed provider request and the reason, or None to leave the request as it is."""
        with self._lock:
            live = self._live.get(session_id or "default")
            if not live or live.get("ended") or not isinstance(request, Mapping):
                return None  # after the turn has ended, a request (a reference check) goes out unchanged
            try:
                return self._shape(live, request, model, provider, base_url)
            except Exception:
                # The request goes out as it is, and the turn is recorded as left alone, so that the calls
                # that follow are never counted for the cheaper model.
                logger.warning("continuum: shaping a request failed; the turn is left alone", exc_info=True)
                live["tools_done"], live["tools_keep"] = True, None
                dec = live["decision"]
                if dec.mode in ("exploit", "explore") and not live["escalated"]:
                    try:
                        self._leave_alone(live, model, "Continuum had a fault while it shaped a request")
                    except Exception:
                        dec.mode = "mixed" if live.get("small_calls") else "observe"
                        live["row_fields"] = {**live.get("row_fields", {}), "mode": dec.mode}
                return None

    def _shape(self, live: dict[str, Any], request: Mapping[str, Any], model: str, provider: str,
               base_url: str) -> tuple[dict[str, Any], str] | None:
        if True:  # (kept as a block: the caller holds the lock of the loop)
            dec: ch.Decision = live["decision"]
            new: dict[str, Any] | None = None
            reasons: list[str] = []
            # tools: decided once per turn, at its first request
            tools = request.get("tools")
            if not live["tools_done"]:
                live["tools_done"] = True
                live["tools_keep"] = None
                size_tools = self.settings.get("size_tools", "observe")
                if isinstance(size_tools, bool):  # `size_tools: on` without quotes is a yes/no in YAML
                    size_tools = "on" if size_tools else "off"
                size_tools = str(size_tools).strip().lower()
                names = [_tool_name(t) for t in tools] if isinstance(tools, list) else []
                if names and all(names) and size_tools in ("observe", "on") and not live.get("restore_tools"):
                    keep = ch.choose_tools(live["fp"], live["text"], self._pool("any"), names, self.cfg["tools"])
                    if keep is not None:
                        # without the grant, and on a turn kept on the user's own model: record only
                        if size_tools == "on" and dec.mode != "pinned" and self._may_change():
                            live["tools_keep"] = set(keep)
                        self.store.update_turn(live["turn_key"], tools_total=len(names), tools_kept=len(keep))
                    else:
                        self.store.update_turn(live["turn_key"], tools_total=len(names), tools_kept=len(names))
            cheaper = dec.mode in ("exploit", "explore") and not live["escalated"]
            if (cheaper or live.get("tools_keep") is not None) and not self._may_change():
                # The consent was taken back while the turn ran: from here on the request goes out as it is.
                live["tools_keep"] = None
                if cheaper:
                    self._leave_alone(live, model, "the consent was taken back during this turn")
            if live.get("tools_keep") is not None and isinstance(tools, list):
                new = dict(request)
                new["tools"] = [t for t in tools if _tool_name(t) in live["tools_keep"]]
                reasons.append(f"kept {len(new['tools'])} of {len(tools)} tools")
                self.store.bump_turn(live["turn_key"], tool_tokens_saved=_approx_tokens(
                    [t for t in tools if _tool_name(t) not in live["tools_keep"]]))
            # rung
            rung = next((r for r in live["rungs"] if r.name == dec.rung), None)
            if rung is not None and not rung.is_reference and dec.mode in ("exploit", "explore") \
                    and not live["escalated"] and not self._at_home(live, provider, base_url):
                # Another provider than the one the cheaper model was set for: this turn is left alone, and
                # nothing is learned about the cheaper model from it.
                self._leave_alone(live, model, "this turn runs on another provider than the cheaper model was set for")
                rung = None
            if rung is not None and not rung.is_reference and dec.mode in ("exploit", "explore") and not live["escalated"]:
                live["small_calls"] += 1
                size = _approx_tokens(request.get("messages")) + _approx_tokens(request.get("tools"))
                too_long = rung.max_context and size > float(self.cfg["chooser"]["context_fill"]) * rung.max_context
                if live["small_calls"] > int(self.cfg["chooser"]["max_calls_on_small"]) or too_long:
                    self._escalate(live, "too long for the rung" if too_long else "too many calls on the rung")
                else:
                    new = dict(new if new is not None else request)
                    new["model"] = rung.model
                    _merge(new, rung.params)
                    reasons.append(dec.reason)
            live["sent_model"] = str((new if new is not None else request).get("model") or "")
            if new is None:
                return None
            return new, "; ".join(reasons)

    def _leave_alone(self, live: dict[str, Any], model: str, why: str) -> None:
        """Turn a choice for a cheaper model into an observed turn on the user's own model, and record why."""
        dec: ch.Decision = live["decision"]
        ref = live["rungs"][-1]
        # `row_fields` is written again when the turn ends: if the write below fails (another process holds
        # the file), the record must still not credit the cheaper model with an answer of the user's own model.
        if live.get("small_calls"):
            # Part of this turn already ran on the cheaper model. The row keeps that model and is marked
            # mixed: a turn that two models answered is never counted for or against either of them.
            dec.reason = f"{why}; the first {live['small_calls']} model calls ran on {dec.model or dec.rung}"
            dec.mode = "mixed"
            live["row_fields"] = {"mode": "mixed", "reason": dec.reason}
        else:
            dec.reason = f"{why}; would have chosen: {dec.reason}"
            dec.rung, dec.model, dec.mode = ref.name, "", "observe"
            live["row_fields"] = {"rung": ref.name, "model": model or ref.name, "mode": "observe",
                                  "reason": dec.reason}
        self.store.update_turn(live["turn_key"], **live["row_fields"])

    def _escalate(self, live: dict[str, Any], why: str) -> None:
        live["escalated"] = True  # set first: the next request returns to the user's own model even if the
        live["escalate_reason"] = why  # write below fails; the row gets the mark again when the turn ends
        self.store.update_turn(live["turn_key"], escalated=1)

    def record_attempt(self, session_id: str, *, ok: bool, model: str = "", usage: Mapping[str, Any] | None = None,
                       duration: float = 0.0, finish: str = "", tool_calls: int = 0, error: str = "",
                       api_request_id: str = "", retryable: bool | None = None, status: int = 0,
                       reason: str = "") -> None:
        with self._lock:
            live = self._live.get(session_id or "default")
            if not live:
                return
            usage = usage if isinstance(usage, Mapping) else {}
            if not ok:  # the host names its own model on a failed call; the record names the model that was called
                model = live.get("sent_model") or model
            tok_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            tok_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            # Of a failed call only the kind of the error is kept (a class name, the host's reason word, the
            # status code), never the provider's message: that text can repeat a part of the request.
            error = "" if ok else _error_kind(error, reason, status)
            try:
                self.store.add_attempt(live["turn_key"], api_request_id=str(api_request_id or ""),
                                       model=str(model or ""), ok=int(bool(ok)), error=error, tok_in=tok_in,
                                       tok_out=tok_out, duration_s=round(float(duration or 0.0), 4),
                                       finish=str(finish or ""), tool_calls=int(tool_calls or 0))
                self.store.bump_turn(live["turn_key"], api_calls=1, api_errors=0 if ok else 1, tok_in=tok_in,
                                     tok_out=tok_out)
            except sqlite3.Error:
                # The file is held by another process. The count of this call is lost; what follows (the return
                # to the user's own model after a failed call) must still happen.
                logger.warning("continuum: a model call could not be recorded", exc_info=True)
            live["last_finish"] = str(finish or "")
            if ok and live["decision"].mode in ("exploit", "explore") and not live["escalated"] \
                    and live.get("small_calls"):
                rung = live["decision"].rung
                if self.store.get_policy("rung_strikes", {}).get(rung):
                    # this cheaper model answered: its earlier refusals are over (other models keep theirs)
                    self.store.change_policy("rung_strikes", lambda cur: {
                        k: v for k, v in (cur if isinstance(cur, Mapping) else {}).items() if k != rung}, {})
                here = live.pop("home_candidate", None)
                if here and self._home() is None:  # the cheaper model answered here: this is its provider
                    sig = self._rung_signature()
                    # read and written in one transaction: the first process that sees an answer sets it
                    self.store.change_policy("rung_home", lambda cur: cur if isinstance(cur, Mapping) and cur.get(
                        "settings") == sig and (cur.get("provider") or cur.get("base_url")) else {**here, "settings": sig}, {})
            if not ok and live["decision"].mode in ("exploit", "explore") and not live["escalated"] \
                    and not live.get("ended"):
                reason = _reason_word(reason)  # a reason word, never a sentence
                if retryable is False:
                    # The host will not send this request again, so the turn ends with the provider's error.
                    # Nothing about answer quality was learned. The session stays on the user's own model.
                    # The cheaper model is blocked at once only when the provider refused the model itself
                    # (no such model, no access to it). A fault of the account or of the request never blocks
                    # it. Any other refusal blocks it after `block_after_refusals` refusals in a row.
                    rung = live["decision"].rung
                    self._held.add(session_id or "default")
                    live["ended"] = True
                    # The row is closed first, with no outcome: if a later write fails, the turn is never
                    # scored against the cheaper model at the end of the session.
                    self.store.update_turn(live["turn_key"], closed=1, escalated=1, source="host_error")
                    cc = self.cfg["chooser"]
                    kept = reason not in cc.get("block_reasons", [])
                    if kept and reason not in cc.get("no_block_reasons", []):
                        limit = int(cc.get("block_after_refusals", 3))
                        count = [0]

                        def strike(cur: Any) -> dict[str, Any]:  # read and written in one transaction
                            strikes = dict(cur) if isinstance(cur, Mapping) else {}
                            count[0] = int(strikes.get(rung, 0)) + 1
                            strikes[rung] = count[0]
                            if count[0] >= limit:
                                strikes.pop(rung)
                            return strikes

                        self.store.change_policy("rung_strikes", strike, {})
                        kept = count[0] < limit
                    if not kept:
                        self.store.update_turn(live["turn_key"], source="rung_blocked")
                        sig = self._rung_signature()
                        block = {"ts": self.clock(), "error": error, "reason": reason,
                                 "status": int(status or 0), "settings": sig}
                        self.store.change_policy("blocked_rungs", lambda cur: {
                            **{m: b for m, b in (cur if isinstance(cur, Mapping) else {}).items()
                               if isinstance(b, Mapping) and b.get("settings") == sig}, rung: block}, {})
                        self.store.log_policy("rung-blocked", {"model": rung, "reason": reason, "status": int(status or 0),
                                                               "turn": live["turn_key"]})
                    self._dirty()
                    return
                live["api_error_on_small"] = "api_error_transient" if reason in self.cfg["outcomes"].get(
                    "api_error_transient", {}).get("reasons", []) else "api_error"
                self._escalate(live, "the model call failed on the rung")

    def record_tool(self, session_id: str, tool_name: str, result: Any = None) -> None:
        with self._lock:
            live = self._live.get(session_id or "default")
            if not live:
                return
            text = result if isinstance(result, str) else json.dumps(result, default=str) if result is not None else ""
            failed = bool(re.match(r'\s*(\{\s*"error"|error\b|traceback)', text[:200], re.I))
            if tool_name and tool_name not in live["tools_used"]:
                live["tools_used"].append(tool_name)
            self.store.bump_turn(live["turn_key"], tool_calls=1, tool_errors=1 if failed else 0)

    def end_turn(self, session_id: str, assistant_response: str = "") -> dict[str, Any] | None:
        """Store the result of the turn. Returns the turn key and whether a reference check is wanted."""
        with self._lock:
            live = self._live.get(session_id or "default")
            if not live or live.get("ended"):
                return None
            live["ended"] = True
            reply = str(assistant_response or "")
            fields: dict[str, Any] = {"duration_s": round(self.clock() - live["t0"], 3), "finish": live["last_finish"],
                                      "tools_used": json.dumps(live["tools_used"])}
            if self.flag("store_text"):
                ran = self._forget_mark_since(live["t0"])
                if ran is not False:  # forget-text ran during the turn, or the mark cannot be read: no answer
                    fields["reply"] = None
                    if ran:  # it did run: the message text inserted at the start of the turn goes too
                        fields["text"] = None
                else:
                    fields["reply"] = reply
            fields.update(live.get("row_fields") or {})  # a turn that was left alone: written again, see _leave_alone
            if live["escalated"]:
                fields["escalated"] = 1
            # What the turn itself says about its outcome is worked out before the write, from what this
            # process saw, so that it is not lost when the write fails.
            oc = self.cfg["outcomes"]
            signal = None
            if not reply.strip():
                signal = "empty"
            elif live["escalated"]:
                signal = live.get("api_error_on_small") or "escalated"
            elif live["last_finish"] in ("length", "max_tokens"):
                signal = "cut_off"
            if re.search(self.cfg["tools"]["miss_pattern"], reply, re.I) and live.get("tools_keep") is not None:
                live["tool_miss"] = True
                if len(self._tool_miss) >= 1000:  # a flag outlives its session by design; the set is kept small
                    self._tool_miss.clear()
                self._tool_miss.add(session_id or "default")
            try:
                self.store.update_turn(live["turn_key"], closed=2, **fields)  # 2 = ended, outcome still open
                if "reply" in fields and self._forgotten_since(live["t0"]):
                    self.store.update_turn(live["turn_key"], reply=None)  # forget-text ran while this write waited
                if not signal:
                    row = self.store.turn(live["turn_key"]) or {}
                    if row.get("tool_calls") and row.get("tool_errors", 0) / row["tool_calls"] >= oc["tool_errors"]["share"]:
                        signal = "tool_errors"
                if signal:
                    self._score(live["turn_key"], oc[signal]["score"], oc[signal]["weight"], signal)
            except sqlite3.Error:
                # Another process holds the file. The row would stay open and look like a turn with no reply,
                # under the model that was chosen first. The fields and the signal are kept and written at
                # the next message or at the end of the session; until then the row is not scored (see
                # _write_kept). A turn that failed on the cheaper model keeps its failure: it can never
                # become a pass for that model through a later "thanks".
                if fields.get("reply") and self._forgotten_since(live["t0"]):
                    fields["reply"] = None  # the answer may already be in the row: the next write clears it
                if len(self._unwritten) >= 256:
                    self._unwritten.pop(next(iter(self._unwritten)))
                self._unwritten[live["turn_key"]] = {"fields": fields, "signal": signal, "final": False,
                                                     "session": session_id or "default"}
                raise
            # A sample of the turns that a cheaper model answered is checked by the user's own model. A trial
            # turn that was kept on the user's own model for the comparison (reference-check) is always checked.
            mode = live["decision"].mode
            on_cheaper = mode in ("exploit", "explore") and not live["escalated"]
            share = _num(self.settings.get("check_share"), 0.0)
            want = bool(not signal and share > 0 and (
                mode == "reference-check" or (on_cheaper and ch.uniform(live["turn_key"] + ":check") < share)))
            return {"turn_key": live["turn_key"], "check": want, "prompt": oc["reference_check"]["prompt"],
                    "timeout_s": float(oc["reference_check"]["timeout_s"])}

    def check_result(self, turn_key: str, answer: Any, duration: float = 0.0) -> str:
        """The user's own model said PASS or FAIL about a turn. It replaces only the weak signals; an explicit
        outcome, a user's correction and a recorded fault stay. The answer must begin with one of the two
        words and must not hold the other one: "PASS? No, FAIL" is not read at all."""
        found = set(re.findall(r"\b(PASS|FAIL)\b", str(answer or "").upper()))
        first = re.match(r"[^A-Za-z]*(PASS|FAIL)\b", str(answer or "").upper())
        word = first.group(1) if first and len(found) == 1 else ""
        with self._lock:
            row = self.store.turn(turn_key)
            if not row:
                return "unread"
            # The check is a model call, so it is on the record like every other attempt.
            self.store.add_attempt(turn_key, api_request_id="", model=str(row.get("base_model") or ""),
                                   ok=int(word in ("PASS", "FAIL")), error="" if word in ("PASS", "FAIL") else "unread",
                                   tok_in=0, tok_out=0, duration_s=round(float(duration or 0.0), 4),
                                   finish="reference_check", tool_calls=0)
            if word not in ("PASS", "FAIL"):
                self.store.log_policy("check_unread", {"turn": turn_key})
                return "unread"
            oc = self.cfg["outcomes"]["reference_check"]
            if row["score"] is not None and row["source"] not in oc["replaces"]:
                return "kept " + str(row["source"])
            # A FAIL is strong evidence; a PASS is weak (`pass_weight`): in our runs the user's own model passed
            # 224 of 270 wrong answers and failed 5 of 208 right ones (tests/benchmark/*/RESULTS.md).
            weight = oc["weight"] if word == "FAIL" else _num(oc.get("pass_weight"), oc["weight"])
            self._score(turn_key, 1.0 if word == "PASS" else 0.0, weight, "reference_check")
            return word.lower()

    def _score(self, turn_key: str, score: float, weight: float, source: str) -> None:
        self.store.update_turn(turn_key, score=float(score), weight=float(weight), source=source, closed=1)
        self._new_outcomes += 1
        self._dirty()

    def _settle_previous(self, session_id: str, fp: Mapping[str, Any], text: str) -> None:
        """The next message of a session is the weak outcome signal for the turn before it."""
        oc = self.cfg["outcomes"]
        self._write_kept()
        rows = self.store.open_turns(session_id)
        last = self.store.last_turn(session_id)
        if last and last["closed"] == 1 and last["source"] in ("session_end", "reference_check"):
            rows.append(last)  # the next message says more than the end of a session; a user's "no" beats a check
        for row in rows:
            if row["turn_key"] in self._unwritten:
                continue  # the turn did end; its end could not be written yet, so it is not scored as empty
            if row["closed"] == 0:  # no end of the turn is on record
                self._close_unended(row)
                continue
            prev_fp = json.loads(row["fp"])
            parts = fpm.level_similarities(fp, prev_fp, fpm.profile_scales([fp["profile"], prev_fp["profile"]]))
            sim = fpm.fuse(parts, self.cfg["fingerprint"], self.cfg["fingerprint"]["presets"][self.preset()])
            # The same Task again, or a rewording of it. The same template with other numbers is a new Task.
            # A very short message ("ok", "next", "yes") sent twice is not a complaint about the answer.
            long_enough = min(int(fp.get("n_words") or 0), int(prev_fp.get("n_words") or 0)) >= int(
                oc["asked_again"].get("min_words", 0))
            again = long_enough and (parts["exact"] or (not parts["template"] and sim >= oc["asked_again"]["similarity"]))
            if re.search(oc["correction"]["pattern"], text, re.I):
                signal = "correction"
            elif again:
                signal = "asked_again"
            elif re.search(oc["thanks"]["pattern"], text, re.I):
                signal = "thanks"
            else:
                signal = "moved_on"
            if row["source"] == "reference_check" and signal not in ("correction", "asked_again"):
                continue
            self._score(row["turn_key"], oc[signal]["score"], oc[signal]["weight"], signal)

    def _close_unended(self, row: Mapping[str, Any]) -> None:
        """A row with no end on record. A row that shows a fall back to the user's own model keeps that
        failure for the cheaper model. With a model call that worked on record, the end was lost (Hermes
        stopped, or the end could not be written and this process no longer holds it): the row is closed
        with no outcome, because a record that is not complete says nothing for or against a model. With
        no call that worked, no reply reached the user: that is the "empty" outcome."""
        if int(row.get("escalated") or 0):
            oc = self.cfg["outcomes"]["escalated"]
            self._score(row["turn_key"], oc["score"], oc["weight"], "escalated")
            return
        answered = int(row.get("api_calls") or 0) > int(row.get("api_errors") or 0)
        if int(row.get("tok_out") or 0) > 0 or answered:
            self.store.update_turn(row["turn_key"], closed=1, source="unwritten")
            return
        oc = self.cfg["outcomes"]["empty"]
        self._score(row["turn_key"], oc["score"], oc["weight"], "empty")

    def _write_kept(self) -> None:
        """Write the end-of-turn fields that an earlier write could not store (the file was held).

        A row that was closed in the meantime (the user marked it, or the turn was cut short) keeps its
        outcome: only the fields are added. The loop stops at the first write that fails, so a message waits
        for one write at most, not for one per kept row."""
        for key in list(self._unwritten):
            kept = self._unwritten[key]
            fields = dict(kept.get("fields") or {})
            signal = kept.get("signal")
            try:
                row = self.store.turn(key)
                began = float(row.get("ts") or 0) if row is not None else 0.0
                if row is not None and "reply" in fields and self._forgotten_since(began):
                    fields["reply"] = None  # forget-text ran after this turn began: write no answer, clear one
                if row is None:
                    pass  # the row is gone (the file was replaced): nothing to write
                elif row["closed"] == 1:
                    self.store.update_turn(key, **fields)
                    self._dirty()  # the fields may change which model the row counts for
                else:
                    if not signal and row.get("tool_calls") and (
                            (row.get("tool_errors") or 0) / row["tool_calls"]
                            >= self.cfg["outcomes"]["tool_errors"]["share"]):
                        signal = "tool_errors"  # the same rule as on the normal path, from the row itself
                if row is None or row["closed"] == 1:
                    pass
                elif signal:
                    oc = self.cfg["outcomes"][signal]
                    self.store.update_turn(key, score=float(oc["score"]), weight=float(oc["weight"]), source=signal,
                                           closed=1, **fields)
                    self._new_outcomes += 1
                    self._dirty()
                elif kept.get("final"):  # its session has ended: closed with no outcome
                    self.store.update_turn(key, closed=1, source="unwritten", **fields)
                else:
                    self.store.update_turn(key, closed=2, **fields)
                if row is not None and fields.get("reply") and self._forgotten_since(began):
                    self.store.update_turn(key, reply=None)  # forget-text ran while this write waited
            except sqlite3.Error:
                break  # still held: tried again at the next message or at the end of a session
            self._unwritten.pop(key, None)

    def feedback(self, good: bool, session_id: str | None = None, source: str = "explicit",
                 turn_key: str | None = None) -> dict[str, Any] | None:
        """An explicit outcome. It replaces any weaker signal on the turn. With a session named, only a turn
        of that session is marked; the newest turn of any session is used only when no session is named."""
        with self._lock:
            if self.mode == "off":
                return None
            self._write_kept()
            row = self.store.turn(turn_key) if turn_key else self.store.last_turn(session_id or None)
            if not row:
                return None
            if row.get("source") in ("host_error", "rung_blocked"):
                return row
            kept_fields = (self._unwritten.get(row["turn_key"]) or {}).get("fields") or {}
            if (int(row.get("escalated") or 0) or int(kept_fields.get("escalated") or 0)) and good:
                # The answer came from the user's own model after the cheaper one failed. A "good" for that
                # answer is not a pass for the cheaper model: the row keeps the failure.
                if row.get("score") is None or float(row["score"]) > 0.0:
                    oc = self.cfg["outcomes"]["escalated"]
                    self._score(row["turn_key"], oc["score"], oc["weight"], "escalated")
                return self.store.turn(row["turn_key"])
            weight = float(self.cfg["outcomes"].get(source, self.cfg["outcomes"]["explicit"])["weight"])
            self._score(row["turn_key"], 1.0 if good else 0.0, weight, source)
            return self.store.turn(row["turn_key"])

    def end_session(self, session_id: str) -> dict[str, Any]:
        session_id = session_id or "default"
        with self._lock:
            self._live.pop(session_id, None)
            self._held.discard(session_id)
            off = self.mode == "off"
            if not off:
                # A turn of this session whose end could not be written yet is closed now, with its own
                # failure signal when it has one and with no outcome otherwise: a record that is not
                # complete says nothing for or against a model. If the file is still held, the turn stays
                # kept and the next write that works closes it.
                for kept in self._unwritten.values():
                    if kept.get("session") == session_id:
                        kept["final"] = True
                self._write_kept()
                for row in self.store.open_turns(session_id):
                    if row["turn_key"] in self._unwritten:
                        continue
                    if row["closed"] == 0:
                        self._close_unended(row)
                        continue
                    oc = self.cfg["outcomes"]["session_end"]
                    self._score(row["turn_key"], oc["score"], oc["weight"], "session_end")
        try:
            self.expire_text(force=True)
        except (sqlite3.Error, OSError):
            pass  # another process holds the file: the audit and the tuning below still run
        if off:
            return {}
        out = {"audit": self.audit_now()}  # a pin stays until /continuum unpin
        if self.flag("self_tune"):
            out["tune"] = self.tune_now(due_only=True)
        return out

    def expire_text(self, force: bool = False, compact: bool = False, max_bytes: int = 0) -> int:
        """Remove stored Task and answer text that is older than `text_days`. Fingerprints and outcomes stay.

        It runs when Hermes starts, at the end of each session, and at most once an hour at the start of a
        turn, in every mode: `mode: off` stops new records, it does not keep old text for ever. On the path of
        a turn and at a session end it clears the rows only; the rewrite of the whole file (`compact`) is done
        when Hermes starts and by `forget-text`, so that no user's turn waits for it."""
        now = float(self.clock())
        with self._lock:
            if not force and now - self._text_checked < 3600.0:
                return 0
            self._text_checked = now
        days = self.text_days()
        if days <= 0 or self.store.in_memory:
            return 0
        old = self.store.forget_text(before=now - days * 86400.0, compact=compact,
                                     max_bytes=max_bytes)  # not under the lock of the loop
        if old:
            self.store.log_policy("old-text-removed", {"turns": old, "older_than_days": days})
            self._dirty()
        return old

    def drop_open(self, session_id: str, why: str) -> None:
        """A turn that the user cut short says nothing about the model: close it with no outcome."""
        with self._lock:
            for row in self.store.open_turns(session_id):
                if row["closed"] == 0 and row["turn_key"] not in self._unwritten:  # a kept turn did end
                    self.store.update_turn(row["turn_key"], closed=1, source=why)
            live = self._live.get(session_id)
            if live and not live.get("ended"):
                self._live.pop(session_id, None)

    def pin(self, session_id: str, on: bool = True) -> None:
        """Keep a session on the user's own model. The pin is stored, so it holds after a restart."""
        sid = session_id or "default"

        def change(cur: Any) -> list[str]:  # merged with the pins of another process; the oldest pins go first
            names = [str(x) for x in cur if str(x) != sid] if isinstance(cur, list) else []
            return (names + [sid] if on else names)[-500:]

        self.store.change_policy("pinned", change, [])

    def forget_text(self) -> int:
        with self._lock:
            for live in self._live.values():  # a running turn must not point at a record that has no text now
                live["decision"].recall_key = None
            for kept in self._unwritten.values():  # an earlier answer waiting to be written is forgotten first
                fields = kept.get("fields")
                if isinstance(fields, dict):
                    fields.pop("reply", None)
        # Another Hermes process (a gateway beside this shell command) may hold answers it could not write yet.
        # It reads this mark before it writes them and drops the answer of every turn that started before it.
        self.store.set_policy("text_forgotten_at", self.clock())
        n = self.store.forget_text()  # not under the lock of the loop: a turn does not wait for the clean-up
        self.store.log_policy("text-forgotten", {"turns": n, "file_rewritten": not self.store.compact_pending()})
        self._dirty()
        return n

    # ---------- learning safeguards ----------
    def audit_now(self) -> dict[str, Any]:
        """Test the policy on its own outcomes and move the floor one step at most.

        The audit acts once on a body of evidence: after a step it waits until `audit_new_evidence` units of
        new cheaper-model outcomes exist. Without that rule the same records would move the floor again at the
        end of every turn. With `self_tune: false` it reports what it would do and changes nothing."""
        if self.mode == "off":
            return {"action": "none", "views": {}, "state": self.strictness(), "note": "Continuum is off"}
        with self._lock:
            self._new_outcomes = 0
        hold = self.cfg["holdout"]
        rows = self.store.scored_turns(int(hold["audit_window"]), own_only=True)
        if not rows:
            return {"action": "none", "views": {}, "state": self.strictness()}
        for r in rows:  # the reference is the model the user had set for that turn
            r["rung"] = "reference" if (not r["model"] or r["model"] == r["base_model"]) else r["model"]
        before = self.strictness()
        report = ch.audit(rows, "reference", before, self.defaults, hold, float(self.cfg["chooser"]["z"]))
        cheap = [r for r in rows if r["mode"] == "exploit" and r["rung"] != "reference"]
        mark = float(self.store.get_policy("audit_mark", {"ts": -1.0}).get("ts", -1.0))
        report["new_evidence"] = round(sum(r["weight"] for r in cheap if r["ts"] > mark), 2)
        if report["action"] != "none":
            wait = report["new_evidence"] < float(hold.get("audit_new_evidence", 2.0))
            if wait or not self.flag("self_tune"):
                report["advised"], report["action"], report["state"] = report["action"], "none", before
                report["note"] = ("waits for new cheaper-model outcomes since the last step" if wait
                                  else "self_tune is off: nothing was changed")
            else:
                # The step is taken only from the state it was computed on. If another Hermes process on the
                # same record file moved the floor in between, its state stays and this step is dropped.
                written = self.store.change_policy(
                    "strictness", lambda cur: report["state"] if cur == before else cur, dict(self.defaults))
                if written == report["state"]:
                    self.store.set_policy("audit_mark", {"ts": max(r["ts"] for r in cheap)})
                    self.store.log_policy("audit-" + report["action"], {"before": before, **report})
                else:
                    report["advised"], report["action"], report["state"] = report["action"], "none", written
                    report["note"] = "another Hermes process changed the policy first: nothing was changed"
        return report

    def audit_if_new(self) -> dict[str, Any] | None:
        """The audit at the end of a turn: it runs only when an outcome was written since the last audit."""
        return self.audit_now() if self._new_outcomes else None

    def tune_now(self, due_only: bool = False) -> dict[str, Any]:
        """Choose the similarity weights that predict the user's own recorded outcomes best.

        `/continuum tune` runs it at once. At the end of a session it runs only when `tune.every` new scored
        records exist since the last run (`due_only`), and it scores at most `tune.max_queries` Tasks, so the
        end of a session never waits long for it, however large the record file is."""
        tune = self.cfg["tune"]
        rows = self.store.scored_turns(int(tune["max_records"]), own_only=True)  # the user's own records only
        if due_only and rows:
            mark = float(self.store.get_policy("tune_mark", {"ts": -1.0}).get("ts", -1.0))
            fresh, every = sum(1 for r in rows if r["ts"] > mark), int(tune.get("every", 25))
            if fresh < every:
                return {"chosen": self.preset(), "skipped": f"waits for {every} new scored records, has {fresh}"}
        for r in rows:
            r["rung"] = r["model"] or r["rung"]
        fit = [r for r in rows if r["split"] == "fit"]
        hold = [r for r in rows if r["split"] == "holdout"]
        if len(fit) < int(tune["min_records"]):
            return {"chosen": self.preset(), "skipped": f"needs {tune['min_records']} scored fit records, has {len(fit)}"}
        current = self.preset()
        max_q = int(tune.get("max_queries", 120)) if due_only else 0
        report = ch.tune_presets(fit, hold, self.cfg["fingerprint"], self.cfg["chooser"], current,
                                 float(tune["holdout_margin"]), max_q)
        report["holdout_records"] = len(hold)
        if report["chosen"] != current:
            self.store.set_policy("preset", report["chosen"])
            self._dirty()
            self.store.log_policy("tune-preset", report)
        try:
            self.store.set_policy("tune_mark", {"ts": max(r["ts"] for r in rows)})
        except (sqlite3.Error, OSError):
            pass  # the mark is a saving, not a rule: without it the next session end tunes again
        return report

    # ---------- reading ----------
    def recall(self, query: str, limit: int = 3, scope: tuple[str, str] | None = None) -> list[dict[str, Any]]:
        """Recorded Tasks like the query, with their answers. Only records of the given platform and sender
        are returned; with no scope (no running turn to take it from) nothing is returned."""
        if self.mode == "off" or scope is None or not self._recall_on():
            return []
        fp = fpm.fingerprint(query, self.cfg["fingerprint"])
        out: list[dict[str, Any]] = []
        want = max(1, min(10, int(limit)))
        for sim, r in self._pool("any").rank(fp, min_similarity=0.2):
            if len(out) >= want:
                break
            if not ch.in_scope(r, scope):
                continue
            row = self.store.turn(r["turn_key"]) or {}
            out.append({"similarity": round(sim, 3), "task": (row.get("text") or "")[:600],
                        "answer": (row.get("reply") or "")[:1500], "outcome": r["score"], "outcome_source": r["source"],
                        "model": r["model"], "tools_used": r["tools_used"]})
        return out

    def status(self) -> dict[str, Any]:
        totals = self.store.totals()
        small_tokens = ref_equiv_saved = 0.0
        small_turns = small_good = 0
        costs = {r.name: r.cost for r in self.rungs("")}
        by_model: dict[str, dict[str, float]] = {}
        for row in self.store.scored_turns(int(self.cfg["chooser"]["max_pool"]), own_only=True):
            m = by_model.setdefault(row["model"] or "?", {"turns": 0, "passes": 0.0, "evidence": 0.0, "tokens": 0})
            m["turns"] += 1
            m["passes"] += row["score"] * row["weight"]
            m["evidence"] += row["weight"]
            m["tokens"] += (row["tok_in"] or 0) + (row["tok_out"] or 0)
            if row["model"] in costs and row["mode"] in ("exploit", "explore") and not row["escalated"]:
                tokens = (row["tok_in"] or 0) + (row["tok_out"] or 0)
                small_tokens += tokens
                small_turns += 1
                # A saving is counted only where someone marked the cheaper model's answer good: a turn that
                # failed, or that nobody checked, may have to be done again on the user's own model.
                if ch.is_mark(row) and float(row["score"]) >= 0.5:
                    small_good += 1
                    ref_equiv_saved += tokens * (1.0 - costs[row["model"]])
        return {"mode": self.mode, "size_tools": self.settings.get("size_tools"), "preset": self.preset(),
                "strictness": self.strictness(), "defaults": self.defaults, "totals": totals, "by_model": by_model,
                "small_rung_tokens": int(small_tokens), "estimated_reference_tokens_saved": int(ref_equiv_saved),
                "small_turns": small_turns, "small_turns_marked_good": small_good,
                "rungs": [{"model": r.model or "(your model)", "cost": r.cost} for r in self.rungs("")],
                "blocked": self.blocked(), "store_note": self.store_note,
                "policy_log": self.store.policy_log(5)}

    def status_text(self, detail: bool = True) -> str:
        """The status as text. `detail=False` (a chat platform) leaves out the provider's name and address and
        the reason of a block: they describe the account of the person who runs Hermes."""
        s = self.status()
        t = s["totals"]
        lines = [f"Continuum right-sizing · mode {s['mode']} · tools {s['size_tools']} · similarity preset {s['preset']}",
                 "Rungs: " + " -> ".join(f"{r['model']} (cost {r['cost']:g})" for r in s["rungs"]),
                 f"Records: {t['turns']} turns ({t['scored']} with an outcome), {t['attempts']} model attempts "
                 f"({t['failed_attempts']} failed), {t['imported']} imported",
                 f"Policy: floor {s['strictness']['floor']:.2f}, evidence needed {s['strictness']['min_evidence']:.1f}"]
        for model, m in sorted(s["by_model"].items()):
            rate = m["passes"] / m["evidence"] if m["evidence"] else 0.0
            lines.append(f"  {model}: {m['turns']} turns, pass rate {rate:.0%} on {m['evidence']:.1f} units of evidence, "
                         f"{m['tokens']:,} tokens")
        if s["small_turns"]:
            line = (f"Cheaper models answered {s['small_turns']} turn{'' if s['small_turns'] == 1 else 's'} "
                    f"({s['small_turns_marked_good']} marked good), "
                    f"{s['small_rung_tokens']:,} tokens")
            if s["estimated_reference_tokens_saved"]:
                line += (f"; estimated saving on the turns marked good: {s['estimated_reference_tokens_saved']:,} "
                         f"reference-token units (assumes equal token counts)")
            lines.append(line)
        if t["tool_tokens_saved"]:
            lines.append(f"Tool list sizing removed about {t['tool_tokens_saved']:,} prompt tokens")
        if t["escalated"]:
            lines.append(f"Escalated to the reference inside a turn: {t['escalated']} times")
        if t["checks"]:
            lines.append(f"Checks by your own model: {t['checks']} "
                         f"({t['checks_failed']} said FAIL)")
        for model, b in sorted(s["blocked"].items()):
            why = f" (the provider refused it: {b.get('reason') or b.get('error') or 'error'})" if detail else ""
            lines.append(f"Blocked: {model}{why}. "
                         "Turns stay on your own model. After you fix the model name or access: /continuum unblock")
        home = self.store.get_policy("rung_home", {})
        if detail and len(s["rungs"]) > 1 and isinstance(home, Mapping) and home.get("settings") == self._rung_signature():
            lines.append(f"Cheaper models are used on provider {home.get('provider') or '(not named)'} only; "
                         "after a change of provider: /continuum unblock")
        if detail and self.store.compact_pending():
            lines.append("Text was removed and the record file is not rewritten yet, so old copies of that text "
                         "can still be in the file. To rewrite it now: hermes continuum compact")
        if s["store_note"]:  # it names a file on the machine: shown to the local user only
            lines.append("Note: " + s["store_note"] + "." if detail or not self.store_note_names_file else
                         "Note: Continuum started a new record file in this run; the local user can see why.")
        if self.mode_note():
            lines.append("Note: " + self.mode_note() + ".")
        if s["mode"] == "off":
            lines.append("Mode off: Continuum records nothing and changes nothing.")
        if len(s["rungs"]) > 1 and s["mode"] == "auto" and not self._may_switch():
            lines.append("Continuum records and does not switch models: " + self.switch_denied_note + ".")
        if not len(s["rungs"]) > 1 and not s["blocked"]:
            lines.append("No cheaper rung is set: Continuum records and fingerprints, and does not switch models. "
                         "Set plugins.entries.continuum.settings.small_model to a cheaper model on your endpoint.")
        return "\n".join(lines)

    def why_text(self, session_id: str | None = None) -> str:
        row = self.store.last_turn(session_id or None)
        if not row:
            return "Continuum has no recorded turn in this session yet." if session_id else "Continuum has no recorded turn yet."
        fp = json.loads(row["fp"])
        live = self._live.get(row["session_id"])
        dec = live["decision"] if live and live["turn_key"] == row["turn_key"] else None
        lines = [f"Continuum · turn {row['turn_key']}",
                 f"Task     {row['contract']} · {fp['n_words']} words · template {row['template'][:8]} · "
                 f"{'held-out' if row['split'] == 'holdout' else 'fit'} split"]
        if dec:
            lines.append(f"Nearest  {dec.neighbours} recorded Tasks above the similarity floor, best {dec.best_similarity:.2f}")
            for name, est in dec.estimates.items():
                lines.append(f"         {name}: pass rate {est['p']:.2f} (lower bound {est['lb']:.2f}) on {est['n']:.1f} units")
        try:  # every level against the nearest other record, the LSH band match included
            pool = self._pool("any")
            near = next(((s, r) for s, r in pool.rank(fp) if r["turn_key"] != row["turn_key"]), None)
            if near:
                lv = fpm.level_similarities(fp, near[1]["fp"], pool.scales)
                lines.append(f"Levels   nearest record {near[0]:.2f}: exact {lv['exact']:.0f} · template {lv['template']:.0f} · "
                             f"contract {lv['contract']:.0f} · LSH band {lv['lsh']:.0f} · sketch {lv['sketch']:.2f} · "
                             f"words {lv['words']:.2f} · profile {lv['profile']:.2f}")
        except (KeyError, TypeError, ValueError):
            pass
        lines.append(f"Choice   {row['model'] or row['rung']} [{row['mode']}] {row['reason'] or ''}".rstrip())
        if row["tools_total"]:
            applied = bool(row.get("tool_tokens_saved")) or (
                str(self.settings.get("size_tools")).strip().lower() in ("on", "true")
                and row["mode"] not in ("pinned", "observe") and self._may_change())
            lines.append(f"Tools    {row['tools_kept']} of {row['tools_total']} would be kept"
                         if not applied else
                         f"Tools    kept {row['tools_kept']} of {row['tools_total']}")
        if row["recalled"]:
            lines.append(f"Recall   added the record of a similar good answer ({row['recalled']})")
        lines.append(f"Result   {row['api_calls']} model calls ({row['api_errors']} failed) · {row['tok_in']:,} in / "
                     f"{row['tok_out']:,} out tokens · {row['tool_calls']} tool calls · {row['duration_s']:.1f} s"
                     + (" · escalated to the reference" if row["escalated"] else ""))
        if row["score"] is not None:
            outcome = f"{'pass' if row['score'] >= 0.5 else 'fail'} from '{row['source']}' with weight {row['weight']:.1f}"
        elif int(row.get("closed") or 0) == 1:
            outcome = f"none ({row.get('source') or 'closed'}; not counted)"
        else:
            outcome = "open (the next message or /continuum good|bad settles it)"
        lines.append(f"Outcome  {outcome}")
        return "\n".join(lines)

    # ---------- export and import ----------
    def export(self, folder: str | Path) -> Path:
        """Write scored records as fingerprints and outcomes only: no Task text and no reply text."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"continuum-records-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(self.clock()))}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"schema": "continuum.records/2", "holds": "fingerprints and outcomes, no text",
                                "note": "a fingerprint is not text, but a person who guesses a Task can test the guess "
                                        "against it"}) + "\n")
            for row in self.store.export_rows():
                f.write(json.dumps(row) + "\n")
        return path

    def import_file(self, path: str | Path) -> int:
        """Read an export. Imported records are labelled, carry reduced weight, never hold text and never
        decide a choice alone. Every field is checked: a line that does not fit is left out."""
        path = Path(path)
        fp_cfg = self.cfg["fingerprint"]
        size, buckets = int(fp_cfg.get("minhash_size", 32)), int(fp_cfg.get("word_buckets", 512))
        limit = int(self.cfg.get("import", {}).get("max_records", 20000))
        session = f"import:{path.stem}"[:120]
        now = float(self.clock())
        ready: list[dict[str, Any]] = []
        if self.mode == "off":
            return 0
        if True:  # the file is read and checked outside the lock of the loop: a long import never holds a turn
            with path.open(encoding="utf-8", errors="replace") as lines:
                for i, line in enumerate(lines):
                    if len(ready) >= limit:
                        break
                    try:
                        row = json.loads(line)
                        raw = row["fp"]
                        score, weight = float(row["score"]), float(row["weight"])
                        model = str(row["model"]).strip()[:200]
                        sketch = [int(x) for x in raw["sketch"]]
                        profile = [float(x) for x in raw["profile"]]
                        words = {str(k): float(v) for k, v in dict(raw["words"]).items()}
                        if not (0.0 <= score <= 1.0 and 0.0 < weight <= 1.0) or not model \
                                or len(sketch) != size or len(profile) != len(fpm.PROFILE_NAMES) \
                                or len(words) > buckets or any(v != v or abs(v) > 1e6 for v in profile) \
                                or any(not 0.0 <= v <= 1.0 for v in words.values()):
                            continue
                        mark = hashlib.sha256(json.dumps([sketch, profile]).encode()).hexdigest()[:28]
                        # The two text hashes are never taken from a file: stand-ins keep the levels working.
                        fp = {"exact": "imp-" + hashlib.sha256(f"{session}:{i}".encode()).hexdigest()[:28],
                              "template": "imp-" + mark, "contract": str(raw["contract"])[:60], "sketch": sketch,
                              "words": words, "profile": profile, "n_words": max(0, int(raw.get("n_words") or 0))}
                        ts = _num(row.get("ts"), now)  # a time in the future would look newer than every own record
                        ready.append({
                            "turn_key": f"{session}:{i}", "session_id": session, "seq": i,
                            "ts": min(now, max(0.0, ts)), "platform": "import", "owner": "import",
                            "exact": fp["exact"], "template": fp["template"], "contract": fp["contract"],
                            "fp": json.dumps(fp), "base_model": "", "rung": model, "model": model, "mode": "import",
                            "reason": "imported", "split": ch.split_of(fp["template"], float(self.cfg["holdout"]["share"])),
                            "tools_used": json.dumps([str(t)[:80] for t in list(row.get("tools_used") or [])[:50]]),
                            "tok_in": max(0, int(row.get("tok_in") or 0)), "tok_out": max(0, int(row.get("tok_out") or 0)),
                            "score": score, "weight": weight, "source": ("import:" + str(row.get("source") or ""))[:60],
                            "closed": 1, "imported": 1,
                        })
                    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                        continue
        n = self.store.add_turns(ready)
        self._dirty()
        return n
