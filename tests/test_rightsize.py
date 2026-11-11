"""Tests of the right-sizing loop: fingerprint levels, record store, chooser, safeguards, engine and Hermes wiring."""

from __future__ import annotations

import threading
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuum_core.rightsize import (  # noqa: E402
    Pool, RecordStore, RightSizer, Rung, audit, choose, fingerprint, fuse, level_similarities, load_config,
    split_of, tune_presets, wilson,
)
fpm = sys.modules["continuum_core.rightsize.fingerprint"]  # the package exports the function of the same name
from continuum_core.rightsize import hermes as wiring  # noqa: E402
from continuum_core.rightsize.engine import owner_of  # noqa: E402

CFG = load_config()
FP = CFG["fingerprint"]


@pytest.fixture(autouse=True)
def _trials_without_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests reach the cheaper model through trial turns on a new record file. The shipped rule that trial
    turns wait for a recent mark of the user (`chooser.explore_needs_marks`) is off here; the tests of that rule
    pass the shipped config itself (`config=load_config()`, read before this patch) and so keep it on."""
    import continuum_core.rightsize.engine as engine_mod

    def without_rule(path: Any = None) -> dict[str, Any]:
        cfg = load_config(path)
        cfg["chooser"]["explore_needs_marks"] = 0
        return cfg

    monkeypatch.setattr(engine_mod, "load_config", without_rule)
BALANCED = FP["presets"][FP["default_preset"]]


def sim(a: str, b: str) -> float:
    fa, fb = fingerprint(a, FP), fingerprint(b, FP)
    return fuse(level_similarities(fa, fb, fpm.profile_scales([fa["profile"], fb["profile"]])), FP, BALANCED)


class Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def make(tmp_path: Path, **settings: Any) -> RightSizer:
    """A sizer as the local terminal user sees it: a turn is on platform `cli` unless the test names another."""
    rs = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", **settings}, clock=Clock())
    start = rs.start_turn

    def start_turn(session_id: str, user_message: str, model: str = "", platform: str = "cli",
                   sender_id: str = "") -> Any:
        return start(session_id, user_message, model=model, platform=platform, sender_id=sender_id)

    rs.start_turn = start_turn  # type: ignore[method-assign]
    return rs


def run_turn(rs: RightSizer, session: str, text: str, reply: str = "ok", good: bool | None = None,
             model: str = "big-m", fail_small: bool = False) -> Any:
    dec = rs.start_turn(session, text, model=model)
    shaped = rs.shape_request({"model": model, "messages": [{"role": "user", "content": text}]}, session, model)
    sent = shaped[0]["model"] if shaped else model
    if fail_small and sent != model:
        rs.record_attempt(session, ok=False, model=sent, error="timeout")
        shaped = rs.shape_request({"model": model, "messages": []}, session, model)
        sent = shaped[0]["model"] if shaped else model
    rs.record_attempt(session, ok=True, model=sent, usage={"prompt_tokens": 100, "completion_tokens": 20},
                      finish="stop")
    rs.end_turn(session, reply)
    if good is not None:
        rs.feedback(good, session, source="check")
    return dec


# ---------------- fingerprint levels ----------------
def test_levels_from_exact_to_continuous() -> None:
    a = fingerprint("What is 12 * 7?", FP)
    same = fingerprint("  what is 12 * 7? ", FP)
    numbers = fingerprint("What is 31 * 9?", FP)
    other = fingerprint("Write a Python function that reverses a linked list.", FP)
    scales = fpm.profile_scales([a["profile"], numbers["profile"], other["profile"]])
    assert level_similarities(a, same, scales)["exact"] == 1.0
    lv = level_similarities(a, numbers, scales)
    assert lv["exact"] == 0.0 and lv["template"] == 1.0 and lv["contract"] == 1.0
    assert a["contract"] == "in:expression|out:number" or a["contract"].endswith("out:number")
    far = level_similarities(a, other, scales)
    assert far["template"] == 0.0 and far["contract"] == 0.0
    assert lv["profile"] > far["profile"]
    assert fuse(lv, FP, BALANCED) >= FP["template_floor"] > fuse(far, FP, BALANCED)


def test_similarity_is_continuous_not_only_exact() -> None:
    base = "Summarize the quarterly sales report for the board in five bullet points."
    near = "Summarize the quarterly marketing report for the board in six bullet points."
    far = "Fix the failing unit test in tests/test_auth.py and run pytest."
    assert 1.0 > sim(base, near) > sim(base, far)
    assert sim(base, base) == 1.0


def test_fingerprint_is_stable_plain_json_and_holds_no_text() -> None:
    text = "Email the invoice 4471 to the supplier in Rotterdam."
    one, two = fingerprint(text, FP), fingerprint(text, FP)
    assert one == two and json.loads(json.dumps(one)) == one
    assert len(one["sketch"]) == FP["minhash_size"] and len(one["profile"]) == len(fpm.PROFILE_NAMES)
    dumped = json.dumps(one).lower()
    assert "rotterdam" not in dumped and "invoice" not in dumped


def test_empty_task_matches_nothing() -> None:
    empty = fingerprint("", FP)
    assert empty["sketch"] == [] and fpm.sketch_similarity(empty["sketch"], empty["sketch"]) == 0.0


def test_lsh_band_is_shared_by_near_duplicates() -> None:
    a = fingerprint("Translate this sentence into French: the cat sits on the mat near the door", FP)
    b = fingerprint("Translate this sentence into French: the cat sits on the mat near the window", FP)
    c = fingerprint("Deploy the staging server and tail the logs", FP)
    assert fpm.sketch_similarity(a["sketch"], b["sketch"]) > fpm.sketch_similarity(a["sketch"], c["sketch"])
    assert not fpm.shares_band(a["sketch"], c["sketch"], FP["band_count"])


# ---------------- store ----------------
def test_store_records_every_attempt_and_exports_no_text(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s1", "My secret project Bluebird needs a budget of 4000.", reply="Bluebird answer", good=True)
    run_turn(rs, "s1", "Another private question about Bluebird.", reply="x", fail_small=True, good=False)
    totals = rs.store.totals()
    assert totals["turns"] == 2 and totals["attempts"] >= 2
    path = rs.export(tmp_path / "exports")
    body = path.read_text(encoding="utf-8")
    assert "Bluebird" not in body and "bluebird" not in body
    rows = [json.loads(line) for line in body.splitlines()]
    assert rows[0]["schema"] == "continuum.records/2" and len(rows) == 3
    assert "exact" not in rows[1]["fp"] and "template" not in rows[1]["fp"]  # the two text hashes stay at home
    assert set(rows[1]) >= {"fp", "score", "weight", "model", "source"} and "text" not in rows[1] and "reply" not in rows[1]


def test_store_text_off_keeps_fingerprints_only(tmp_path: Path) -> None:
    rs = make(tmp_path, store_text=False)
    run_turn(rs, "s1", "Private words here.", reply="Private reply.", good=True)
    row = rs.store.last_turn("s1")
    assert row["text"] is None and row["reply"] is None and row["score"] == 1.0


def test_import_is_labelled_downweighted_and_textless(tmp_path: Path) -> None:
    src = make(tmp_path / "a")
    for i in range(5):
        run_turn(src, f"s{i}", f"What is {i + 2} * 7?", good=True)
    path = src.export(tmp_path / "exports")
    dst = make(tmp_path / "b")
    assert dst.import_file(path) == 5
    assert dst.store.totals()["imported"] == 5 and dst.store.totals()["turns"] == 0
    rows = dst.store.scored_turns(50)
    assert all(r["imported"] and r["source"].startswith("import:") for r in rows)
    pool = dst._pool("any")
    assert all(r["weight"] == pytest.approx(CFG["import"]["trust"]) for r in pool.records)


# ---------------- chooser ----------------
def test_wilson_bounds() -> None:
    lo, hi = wilson(0.9, 20, 1.0)
    assert 0.0 < lo < 0.9 < hi <= 1.0
    assert wilson(0.9, 2, 1.0)[0] < lo
    assert wilson(0.5, 0, 1.0) == (0.0, 1.0)


def test_split_is_stable_and_near_the_share() -> None:
    labels = [split_of(f"template-{i}", 0.2) for i in range(4000)]
    assert labels == [split_of(f"template-{i}", 0.2) for i in range(4000)]
    assert 0.17 < labels.count("holdout") / len(labels) < 0.23


def _records(n: int, rung: str, score: float, stem: str = "What is {} * 7?") -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        fp = fingerprint(stem.format(i + 2), FP)
        out.append({"turn_key": f"{rung}:{stem[:5]}:{i}", "fp": fp, "exact": fp["exact"], "template": fp["template"],
                    "contract": fp["contract"], "rung": rung, "model": rung, "score": score, "weight": 1.0,
                    "split": "fit", "source": "check", "tools_used": [], "imported": 0, "has_reply": 0})
    return out


RUNGS = [Rung("small-m", "small-m", 0.2), Rung("big-m", "", 1.0)]
STRICT = {"floor": CFG["chooser"]["floor"], "min_evidence": CFG["chooser"]["min_evidence"], "explore_share": 0.1}


def test_no_evidence_means_the_reference() -> None:
    dec = choose(fingerprint("What is 5 * 7?", FP), "What is 5 * 7?", RUNGS, Pool([], FP, BALANCED), CFG["chooser"],
                 u=0.9, **STRICT)
    assert dec.mode == "reference" and dec.model == ""


def test_similar_passes_let_the_cheaper_rung_take_the_task() -> None:
    pool = Pool(_records(10, "small-m", 1.0) + _records(6, "big-m", 1.0), FP, BALANCED)
    dec = choose(fingerprint("What is 99 * 7?", FP), "What is 99 * 7?", RUNGS, pool, CFG["chooser"], u=0.9, **STRICT)
    assert dec.mode == "exploit" and dec.model == "small-m" and dec.estimates["small-m"]["lb"] >= STRICT["floor"]


def test_similar_failures_rule_the_cheaper_rung_out() -> None:
    pool = Pool(_records(10, "small-m", 0.0) + _records(6, "big-m", 1.0), FP, BALANCED)
    for u in (0.0, 0.04, 0.5, 0.99):  # never, not even as exploration
        dec = choose(fingerprint("What is 99 * 7?", FP), "What is 99 * 7?", RUNGS, pool, CFG["chooser"], u=u, **STRICT)
        assert dec.model == "" and dec.mode == "reference"


def _weak(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same records as outcomes that nobody checked ("the session ended"): a weak pass."""
    return [{**r, "score": 1.0, "weight": 0.2, "source": "session_end", "strong": False} for r in records]


def test_unchecked_turns_of_your_own_model_do_not_rule_a_cheaper_model_out() -> None:
    small = _records(12, "small-m", 1.0)
    small[5]["score"] = 0.0  # 11 of 12 on checks
    ask = lambda pool: choose(fingerprint("What is 99 * 7?", FP), "What is 99 * 7?", RUNGS, pool, CFG["chooser"],
                              u=0.9, **STRICT)
    unchecked = ask(Pool(small + _weak(_records(30, "big-m", 1.0)), FP, BALANCED))  # "100%" that nobody checked
    assert unchecked.estimates["big-m"]["checked"] == 0.0 and unchecked.mode == "exploit"
    checked = ask(Pool(small + _records(30, "big-m", 1.0), FP, BALANCED))  # 100% on checks: like for like
    assert checked.estimates["big-m"]["checked"] == 1.0 and checked.mode == "reference"


def test_checked_outcomes_take_over_from_unchecked_ones() -> None:
    stem = "What is {} * 7?"
    weak_passes = _weak(_records(40, "small-m", 1.0, stem=stem))
    fp = fingerprint("What is 99 * 7?", FP)
    only_weak = Pool(weak_passes, FP, BALANCED)
    est = only_weak.estimate(only_weak.rank(fp), "small-m", CFG["chooser"])
    assert est["checked"] == 0.0 and est["p"] > 0.9  # nothing checked yet: the weak passes are all there is
    mixed = Pool(weak_passes + _records(6, "small-m", 0.0, stem="What is {} * 7 ?"), FP, BALANCED)
    est = mixed.estimate(mixed.rank(fp), "small-m", CFG["chooser"])
    assert est["checked"] == 1.0 and est["p"] < 0.2  # six checked failures are not hidden by forty unchecked turns
    dec = choose(fp, "What is 99 * 7?", RUNGS, mixed, CFG["chooser"], u=0.9, **STRICT)
    assert dec.mode == "reference"


def test_audit_reads_your_own_model_only_where_it_was_checked() -> None:
    defaults = {"floor": 0.7, "min_evidence": 3.0}
    cheap = [{"rung": "small-m", "mode": "exploit", "score": s, "weight": 0.8, "split": "holdout"}
             for s in [1.0] * 15 + [0.0] * 5]  # 75% on checks
    unchecked = [{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 0.2, "split": "holdout"}] * 60
    rep = audit(cheap + unchecked, "reference", defaults, defaults, CFG["holdout"], 1.0)
    assert rep["action"] == "none" and rep["views"]["all"]["reference_evidence"] == 0.0
    marked = [{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 1.0, "split": "holdout",
               "source": "explicit"}] * 60
    assert audit(cheap + marked, "reference", defaults, defaults, CFG["holdout"], 1.0)["action"] == "tighten"


def test_evidence_does_not_carry_to_unlike_tasks() -> None:
    pool = Pool(_records(10, "small-m", 1.0), FP, BALANCED)
    text = "Draft a product announcement for the new mobile app and suggest three titles."
    dec = choose(fingerprint(text, FP), text, RUNGS, pool, CFG["chooser"], u=0.9, **STRICT)
    assert dec.mode == "reference" and dec.estimates["small-m"]["n"] < STRICT["min_evidence"]


def test_exploration_is_bounded_and_reference_is_rechecked() -> None:
    empty = Pool([], FP, BALANCED)
    fp = fingerprint("What is 5 * 7?", FP)
    modes = [choose(fp, "What is 5 * 7?", RUNGS, empty, CFG["chooser"], u=i / 1000, **STRICT).mode for i in range(1000)]
    assert modes.count("explore") == 100 and modes.count("reference") == 900
    good = Pool(_records(10, "small-m", 1.0), FP, BALANCED)
    modes = [choose(fp, "What is 5 * 7?", RUNGS, good, CFG["chooser"], u=i / 1000, **STRICT).mode for i in range(1000)]
    assert modes.count("reference-check") == 50 and modes.count("exploit") == 950


def test_never_downsize_pattern_pins_the_reference() -> None:
    pool = Pool(_records(10, "small-m", 1.0, "Delete the production database backup number {}"), FP, BALANCED)
    text = "Delete the production database backup number 4"
    dec = choose(fingerprint(text, FP), text, RUNGS, pool, CFG["chooser"], u=0.9, **STRICT)
    assert dec.mode == "pinned" and dec.model == ""


# ---------------- safeguards ----------------
def test_audit_tightens_when_cheaper_rungs_fall_behind_and_relaxes_after() -> None:
    hold = CFG["holdout"]
    defaults = {"floor": 0.7, "min_evidence": 3.0}
    bad = [{"rung": "small-m", "mode": "exploit", "score": 0.0, "weight": 1.0, "split": "holdout"} for _ in range(8)] + \
          [{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 1.0, "split": "holdout"} for _ in range(8)]
    rep = audit(bad, "reference", defaults, defaults, hold, 1.0)
    assert rep["action"] == "tighten" and rep["state"]["floor"] > 0.7 and rep["state"]["min_evidence"] > 3.0
    good = [{"rung": "small-m", "mode": "exploit", "score": 1.0, "weight": 1.0, "split": "holdout"} for _ in range(30)] + \
           [{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 1.0, "split": "holdout"} for _ in range(30)]
    rep2 = audit(good, "reference", rep["state"], defaults, hold, 1.0)
    assert rep2["action"] == "relax" and rep2["state"]["floor"] < rep["state"]["floor"]
    rep3 = audit(good, "reference", defaults, defaults, hold, 1.0)
    assert rep3["action"] == "none" and rep3["state"] == defaults  # never looser than the defaults
    state = dict(defaults)
    for _ in range(40):
        state = audit(bad, "reference", state, defaults, hold, 1.0)["state"]
    assert state["floor"] <= hold["max_floor"] and state["min_evidence"] <= hold["max_evidence"]


def test_audit_needs_evidence_before_it_acts() -> None:
    defaults = {"floor": 0.7, "min_evidence": 3.0}
    few = [{"rung": "small-m", "mode": "exploit", "score": 0.0, "weight": 1.0, "split": "holdout"} for _ in range(2)] + \
          [{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 1.0, "split": "holdout"} for _ in range(2)]
    assert audit(few, "reference", defaults, defaults, CFG["holdout"], 1.0)["action"] == "none"


def test_tune_adopts_a_preset_only_with_held_out_support() -> None:
    rng = random.Random(7)
    fit, hold = [], []
    stems = [("What is {} * 7?", 1.0), ("Refactor module number {} into services and explain the design.", 0.0)]
    for i in range(80):
        stem, score = stems[i % 2]
        rec = _records(1, "small-m", score, stem.replace("{}", str(rng.randrange(10_000))))[0]
        rec["turn_key"] = f"t{i}"
        (hold if i % 5 == 0 else fit).append(rec)
    rep = tune_presets(fit, hold, FP, CFG["chooser"], FP["default_preset"], CFG["tune"]["holdout_margin"])
    assert rep["chosen"] in FP["presets"] and set(rep["fit_error"]) == set(FP["presets"])
    if rep["chosen"] != rep["current"]:
        assert rep["held_out"][rep["chosen"]] <= rep["held_out"][rep["current"]] + CFG["tune"]["holdout_margin"]


# ---------------- engine ----------------
def test_engine_learns_then_downsizes_and_keeps_held_out_records_out_of_the_choice(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=0.3)
    modes = []
    for i in range(60):
        dec = rs.start_turn(f"s{i}", f"What is {i + 2} * 7?", model="big-m")
        modes.append(dec.mode)
        shaped = rs.shape_request({"model": "big-m", "messages": []}, f"s{i}", "big-m")
        sent = shaped[0]["model"] if shaped else "big-m"
        assert (sent == "small-m") == (dec.mode in ("exploit", "explore"))
        rs.record_attempt(f"s{i}", ok=True, model=sent, usage={"prompt_tokens": 50, "completion_tokens": 5})
        rs.end_turn(f"s{i}", str((i + 2) * 7))
        rs.feedback(True, f"s{i}", source="check")
    assert modes[0] in ("reference", "explore") and "exploit" in modes[20:]
    assert modes.index("exploit") > modes.index("explore")  # evidence came from bounded exploration first
    assert all(r["split"] == "fit" for r in rs._pool("fit").records)
    assert "small-m" in rs.status_text() and "pass rate" in rs.status_text()
    assert "Choice" in rs.why_text()


def test_all_one_template_in_holdout_never_downsizes(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=0.0)
    rs.cfg["holdout"]["share"] = 1.0  # every record is held out: the chooser must see nothing
    for i in range(30):
        dec = run_turn(rs, f"s{i}", f"What is {i + 2} * 7?", good=True)
        assert dec.mode == "reference"
    assert rs._pool("fit").records == []


def test_implicit_outcomes_from_the_next_message(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "Write a haiku about the sea.")
    run_turn(rs, "s", "Thanks, that works. Now list three rivers in Europe.")
    run_turn(rs, "s", "No, that's wrong. The Nile is not in Europe.")
    run_turn(rs, "s", "Convert 30 degrees Celsius to Fahrenheit.")
    run_turn(rs, "s", "Convert 45 degrees Celsius to Fahrenheit.")   # same template, other number: a new Task
    run_turn(rs, "s", "Explain what a mutex is in two sentences.")
    run_turn(rs, "s", "explain what a mutex is in two sentences.")   # the same Task again
    rs.end_session("s")
    sources = [rs.store.turn(f"s:{i}")["source"] for i in range(1, 8)]
    assert sources == ["thanks", "correction", "moved_on", "moved_on", "moved_on", "asked_again", "session_end"]
    weights = [rs.store.turn(f"s:{i}")["weight"] for i in range(1, 8)]
    assert max(weights) < 1.0  # an implicit signal never weighs as much as an explicit one


def test_explicit_feedback_replaces_a_weak_signal(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "Write a haiku about the sea.")
    run_turn(rs, "s", "List three rivers in Europe.")
    assert rs.store.turn("s:1")["source"] == "moved_on"
    rs.feedback(False, turn_key="s:1")
    row = rs.store.turn("s:1")
    assert row["score"] == 0.0 and row["weight"] == 1.0 and row["source"] == "explicit"


def test_failed_call_on_a_cheaper_rung_escalates_inside_the_turn(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)  # force a try of the cheaper rung
    dec = run_turn(rs, "s", "What is 3 * 7?", fail_small=True)
    assert dec.mode == "explore"
    row = rs.store.turn("s:1")
    attempts = rs.store.attempts("s:1")
    assert [a["ok"] for a in attempts] == [0, 1] and [a["model"] for a in attempts] == ["small-m", "big-m"]
    assert row["escalated"] == 1 and row["score"] == 0.0 and row["source"] == "api_error"


def test_too_many_calls_on_a_cheaper_rung_escalate(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    sent = []
    for _ in range(CFG["chooser"]["max_calls_on_small"] + 3):
        shaped = rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
        sent.append(shaped[0]["model"] if shaped else "big-m")
    assert sent[:CFG["chooser"]["max_calls_on_small"]] == ["small-m"] * CFG["chooser"]["max_calls_on_small"]
    assert set(sent[CFG["chooser"]["max_calls_on_small"]:]) == {"big-m"}


def test_observe_and_off_and_pin_change_nothing(tmp_path: Path) -> None:
    for settings in ({"mode": "observe", "explore_share": 1.0}, {"mode": "off", "explore_share": 1.0}):
        rs = make(tmp_path / settings["mode"], **settings)
        rs.start_turn("s", "What is 3 * 7?", model="big-m")
        assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None
    rs = make(tmp_path / "pin", explore_share=1.0)
    rs.pin("s", True)
    assert rs.start_turn("s", "What is 3 * 7?", model="big-m").mode == "pinned"
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None


def test_no_cheaper_rung_means_records_only(tmp_path: Path) -> None:
    rs = RightSizer(tmp_path / "r.db", clock=Clock())
    dec = rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert dec.mode == "observe" and rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None
    assert "No cheaper rung is set" in rs.status_text()


def test_recall_adds_a_good_past_answer_only(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "a", "What is the capital of Australia?", reply="Canberra.", good=True)
    run_turn(rs, "b", "What is the tallest mountain in Africa?", reply="Mount Kenya.", good=False)
    rs.start_turn("c", "What is the capital of Australia?", model="big-m")
    assert "Canberra." in (rs.recall_context("c") or "")
    rs.start_turn("d", "What is the tallest mountain in Africa?", model="big-m")
    assert rs.recall_context("d") is None
    found = rs.recall("capital of Australia", 3, scope=rs.scope_of("d"))
    assert found and found[0]["answer"] == "Canberra." and found[0]["outcome"] == 1.0
    assert rs.recall("capital of Australia", 3) == []  # no running turn to take the user from: nothing is shown


def test_recall_stays_with_the_same_platform_and_sender(tmp_path: Path) -> None:
    rs = make(tmp_path)
    task = "What is the door code of my office?"
    rs.start_turn("a", task, model="big-m", platform="telegram", sender_id="alice")
    rs.record_attempt("a", ok=True, model="big-m")
    rs.end_turn("a", "It is 4411.")
    rs.feedback(True, "a")
    for session, platform, sender, seen in (("b", "telegram", "bob", False), ("c", "discord", "alice", False),
                                            ("d", "telegram", "alice", True)):
        rs.start_turn(session, task, model="big-m", platform=platform, sender_id=sender)
        assert bool(rs.recall_context(session)) is seen, session
        assert bool(rs.recall("door code of my office", 3, scope=rs.scope_of(session))) is seen, session
    row = rs.store.turn("a:1")
    assert row["platform"] == "telegram" and row["owner"] == owner_of("telegram", "alice")
    assert "alice" not in json.dumps(row, default=str)  # the sender is stored as a short hash only


def test_tool_list_sizing_keeps_what_similar_good_tasks_used(tmp_path: Path) -> None:
    rs = make(tmp_path, size_tools="on", explore_share=0.0)
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}}
             for n in ("terminal", "read_file", "web_search", "browser", "image_gen", "calendar")]
    for i in range(8):
        rs.start_turn(f"s{i}", f"Search the web for the price of item {i + 100} and report it.", model="big-m")
        rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, f"s{i}", "big-m")
        rs.record_tool(f"s{i}", "web_search", "{}")
        rs.record_attempt(f"s{i}", ok=True, model="big-m")
        rs.end_turn(f"s{i}", "done")
        rs.feedback(True, f"s{i}", source="check")
    rs.start_turn("new", "Search the web for the price of item 977 and report it.", model="big-m")
    shaped = rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, "new", "big-m")
    kept = {t["function"]["name"] for t in shaped[0]["tools"]}
    assert "web_search" in kept and "terminal" in kept and "image_gen" not in kept and "calendar" not in kept
    # the model says it lacks a tool: every tool returns on the next turn
    rs.record_attempt("new", ok=True, model="big-m")
    rs.end_turn("new", "I don't have a tool to open the calendar.")
    rs.start_turn("new", "Then open the calendar.", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, "new", "big-m") is None


def test_records_survive_a_restart(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=0.3)
    for i in range(40):
        run_turn(rs, f"s{i}", f"What is {i + 2} * 7?", good=True)
    rs.store.close()
    again = make(tmp_path, explore_share=0.0)
    dec = again.start_turn("later", "What is 123 * 7?", model="big-m")
    assert dec.mode in ("exploit", "reference-check") and again.store.totals()["turns"] == 41


# ---------------- Hermes wiring ----------------
class FakeCtx:
    def __init__(self, granted: tuple[str, ...] = ("llm.model_override",), **config: Any) -> None:
        self.config = config
        self.granted = set(granted)
        self.side_answers: list[Any] = []
        self.side_prompts: list[str] = []
        self.hooks: dict[str, list[Any]] = {}
        self.middleware: dict[str, Any] = {}
        self.tools: dict[str, dict[str, Any]] = {}
        self.commands: dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def has_capability(self, capability: str) -> bool:
        return capability in self.granted

    def side_question(self, token: str, prompt: str, *, timeout: float = 15.0) -> Any:
        self.side_prompts.append(prompt)
        return self.side_answers.pop(0) if self.side_answers else None

    def register_hook(self, name: str, callback: Any, **_: Any) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_middleware(self, kind: str, callback: Any, **_: Any) -> None:
        self.middleware[kind] = callback

    def register_tool(self, name: str, **kwargs: Any) -> None:
        self.tools[name] = kwargs

    def register_command(self, name: str, handler: Any, **_: Any) -> None:
        self.commands[name] = handler

    def register_skill(self, *args: Any, **kwargs: Any) -> None:
        pass

    def register_context_engine(self, *_: Any) -> None:
        raise AssertionError("the lab is off")

    register_memory_provider = register_delegation_provider = register_context_engine

    def fire(self, name: str, **kwargs: Any) -> list[Any]:
        return [cb(**kwargs) for cb in self.hooks.get(name, [])]


BOTH = (wiring.CAPABILITY, wiring.CHECK_CAPABILITY)


@pytest.fixture()
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import continuum_core.storage as storage_mod

    monkeypatch.setattr(storage_mod, "resolve_continuum_db_path", lambda hermes_home=None: tmp_path / "continuum.db")
    spec = __import__("importlib.util").util.spec_from_file_location("continuum_plugin_v2", REPO_ROOT / "__init__.py")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_registers_what_the_manifest_declares(plugin: Any) -> None:
    import yaml  # type: ignore[import-untyped]

    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    ctx = FakeCtx()
    plugin.register(ctx)
    assert set(ctx.hooks) == set(manifest["provides_hooks"]) == set(wiring.HOOKS)
    # `hermes plugins validate` compares the registered middleware with this field and fails without it.
    assert set(ctx.middleware) == set(manifest["provides_middleware"]) == {"llm_request"}
    assert set(ctx.tools) == set(manifest["provides_tools"])
    visible = [n for n, kw in ctx.tools.items() if kw.get("check_fn") is None or kw["check_fn"]()]
    assert visible == ["continuum_recall"]
    for key in plugin.RIGHTSIZE_SETTINGS:
        assert key in manifest["config_schema"], key
        assert manifest["config_schema"][key]["default"] == CFG["settings"][key], key
    assert manifest["capabilities"] == [wiring.CAPABILITY, wiring.CHECK_CAPABILITY]


def test_no_model_switch_without_the_users_consent(plugin: Any) -> None:
    ctx = FakeCtx(granted=(), small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    request = {"model": "big-m", "messages": [], "tools": []}
    ctx.fire("pre_llm_call", session_id="s", user_message="What is 6 * 7?", model="big-m")
    out = ctx.middleware["llm_request"](request=request, session_id="s", model="big-m", api_call_count=1)
    assert out is None or out["request"]["model"] == "big-m"
    row = rt.sizer.store.turn("s:1")
    assert row["mode"] == "observe" and row["model"] == "big-m" and "hermes plugins enable" in row["reason"]
    assert "not granted" in ctx.commands["continuum"]("status")
    ctx.fire("post_llm_call", session_id="s", assistant_response="42")
    ctx.granted.add(wiring.CAPABILITY)  # consent given: the next turn may switch
    ctx.fire("pre_llm_call", session_id="s", user_message="What is 8 * 9?", model="big-m")
    out = ctx.middleware["llm_request"](request=request, session_id="s", model="big-m", api_call_count=1)
    assert out["request"]["model"] == "small-m"


def test_full_turn_through_the_hermes_callbacks(plugin: Any, tmp_path: Path) -> None:
    ctx = FakeCtx(small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    ctx.fire("pre_llm_call", session_id="s", user_message="What is 6 * 7?", model="big-m", platform="cli",
             is_first_turn=True, conversation_history=[], task_id="t", turn_id="u")
    out = ctx.middleware["llm_request"](request={"model": "big-m", "messages": [], "temperature": 0.2},
                                        original_request={}, session_id="s", model="big-m", api_call_count=1)
    assert out["request"]["model"] == "small-m" and out["request"]["temperature"] == 0.2 and out["source"] == "continuum"
    ctx.fire("post_api_request", session_id="s", model="big-m", response_model="small-m", api_duration=0.4,
             usage={"prompt_tokens": 30, "completion_tokens": 3}, finish_reason="stop", assistant_tool_call_count=0)
    ctx.fire("post_tool_call", session_id="s", tool_name="terminal", result="42", status="ok")
    ctx.fire("post_llm_call", session_id="s", user_message="What is 6 * 7?", assistant_response="42", model="big-m")
    ctx.fire("on_session_end", session_id="s", completed=True, interrupted=False)
    def cmd(args: str, **kw: Any) -> str:
        return ctx.commands["continuum"](args, platform="cli", **kw)

    assert "small-m" in cmd("status") and "Choice" in cmd("why", session_id="s")
    assert "marked good" in cmd("good", session_id="s")
    ctx.fire("on_session_finalize", session_id="s", platform="cli")
    row = rt.sizer.store.turn("s:1")
    assert row["model"] == "small-m" and row["tok_in"] == 30 and row["tool_calls"] == 1 and row["score"] == 1.0
    assert "Audit" in cmd("audit") and "Tune" in cmd("tune") and "Continuum commands" in cmd("help")
    assert "no Task text" in cmd("export") and list((tmp_path / "exports").glob("*.jsonl"))
    assert "inbox" in cmd("import ../../etc/passwd")  # only file names inside the inbox are read
    assert "pin" in cmd("nonsense") or "Continuum commands" in cmd("nonsense")
    handler = ctx.tools["continuum_recall"]["handler"]
    assert json.loads(handler({"query": "what is 6 * 7"}))["records"] == []  # no running turn: nothing is shown
    ctx.fire("pre_llm_call", session_id="s2", user_message="And what is 6 * 8?", model="big-m", platform="cli")
    recall = json.loads(handler({"query": "what is 6 * 7"}, session_id="s2", task_id="t"))
    assert recall["records"][0]["answer"] == "42"
    ctx.fire("pre_llm_call", session_id="s3", user_message="And what is 6 * 9?", model="big-m", platform="telegram",
             sender_id="someone-else")
    assert json.loads(handler({"query": "what is 6 * 7"}, session_id="s3"))["records"] == []


def test_callbacks_never_raise_into_hermes(plugin: Any) -> None:
    ctx = FakeCtx(small_model="small-m")
    plugin.register(ctx)
    junk: list[dict[str, Any]] = [{}, {"session_id": None, "user_message": None, "request": None, "usage": "x"},
                                  {"session_id": 7, "user_message": [{"type": "text", "text": "hi"}], "error": "boom",
                                   "result": object(), "assistant_response": {"a": 1}, "model": None, "status": 3}]
    for kwargs in junk:
        for name in wiring.HOOKS:
            ctx.fire(name, **kwargs)
        ctx.middleware["llm_request"](**kwargs)
    assert isinstance(ctx.commands["continuum"](None), str)


def test_failed_call_is_recorded_against_the_model_that_was_called(plugin: Any) -> None:
    ctx = FakeCtx(small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    request = {"model": "big-m", "messages": []}
    ctx.fire("pre_llm_call", session_id="s", user_message="Name three colours.", model="big-m")
    assert ctx.middleware["llm_request"](request=request, session_id="s", model="big-m")["request"]["model"] == "small-m"
    ctx.fire("api_request_error", session_id="s", model="big-m", error={"type": "InternalServerError"})
    assert ctx.middleware["llm_request"](request=request, session_id="s", model="big-m") is None  # back to the user's model
    ctx.fire("post_api_request", session_id="s", model="big-m", response_model="big-m", usage={"prompt_tokens": 9})
    ctx.fire("post_llm_call", session_id="s", assistant_response="red, green, blue")
    attempts = [(a["model"], a["ok"]) for a in rt.sizer.store.attempts("s:1")]
    row = rt.sizer.store.turn("s:1")
    assert attempts == [("small-m", 0), ("big-m", 1)]
    assert (row["escalated"], row["source"], row["score"], row["rung"]) == (1, "api_error", 0.0, "small-m")


def _cheap_turn(ctx: Any, session: str, text: str, reply: str, token: Any = "tok") -> Any:
    ctx.fire("pre_llm_call", session_id=session, user_message=text, model="big-m")
    out = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id=session, model="big-m")
    assert out["request"]["model"] == "small-m"
    ctx.fire("post_api_request", session_id=session, model="big-m", response_model="small-m", usage={})
    asked = len(ctx.side_prompts)
    assert ctx.fire("post_llm_call", session_id=session, assistant_response=reply, side_question_token=token) == [None]
    for thread in threading.enumerate():  # the check runs on its own thread; the hook itself returns nothing
        if thread.name == "continuum-check":
            thread.join(5)
    return True if len(ctx.side_prompts) > asked else None


def test_reference_check_by_the_users_own_model(plugin: Any) -> None:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0, check_share=1.0)
    rt = plugin.register(ctx)
    store = rt.sizer.store
    ctx.side_answers = ["FAIL. The sum is wrong.", "pass", "I am not sure.", "PASS"]
    assert _cheap_turn(ctx, "a", "What is 6 * 7?", "41") is not None
    row = store.turn("a:1")
    assert (row["source"], row["score"], row["weight"]) == ("reference_check", 0.0, 0.8)
    assert "PASS or FAIL" in ctx.side_prompts[0]
    # after the turn has ended the middleware leaves a request alone, so the check runs on the user's own model
    assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="a", model="big-m") is None
    _cheap_turn(ctx, "b", "Name the capital of France.", "Paris")
    assert (store.turn("b:1")["source"], store.turn("b:1")["score"]) == ("reference_check", 1.0)
    _cheap_turn(ctx, "c", "Translate 'cat' to Spanish.", "gato")
    assert store.turn("c:1")["score"] is None  # an answer that is neither PASS nor FAIL records no outcome
    assert rt.sizer.check_result("c:1", "PASS? No, FAIL.") == "unread"  # both words: not read at all
    assert [a["finish"] for a in store.attempts("c:1")][-2:] == ["reference_check"] * 2  # each check is on the record
    # a user's correction beats a PASS from the check; a plain next message does not replace it
    _cheap_turn(ctx, "d", "List two prime numbers.", "4 and 6")
    assert store.turn("d:1")["source"] == "reference_check"
    ctx.fire("pre_llm_call", session_id="b", user_message="Now name the capital of Peru.", model="big-m")
    assert store.turn("b:1")["source"] == "reference_check"
    ctx.fire("pre_llm_call", session_id="d", user_message="No, that is wrong.", model="big-m")
    assert (store.turn("d:1")["source"], store.turn("d:1")["score"]) == ("correction", 0.0)
    # an explicit outcome is never replaced by a check
    assert rt.sizer.feedback(True, turn_key="a:1")["source"] == "explicit"
    assert rt.sizer.check_result("a:1", "FAIL") == "kept explicit"
    assert "Checks by your own model" in ctx.commands["continuum"]("status")


def test_no_reference_check_without_a_token_or_the_grant_or_on_a_pinned_session(plugin: Any) -> None:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0, check_share=1.0)
    rt = plugin.register(ctx)
    assert _cheap_turn(ctx, "a", "What is 6 * 7?", "42", token=None) is None  # no consent: Hermes gives no token
    ctx.granted.discard(wiring.CHECK_CAPABILITY)
    assert _cheap_turn(ctx, "b", "What is 6 * 8?", "48") is None  # a token, but no llm.side_question grant
    ctx.granted.add(wiring.CHECK_CAPABILITY)
    ctx.commands["continuum"]("pin", session_id="p")
    ctx.fire("pre_llm_call", session_id="p", user_message="What is 6 * 7?", model="big-m")
    assert ctx.fire("post_llm_call", session_id="p", assistant_response="42", side_question_token="tok") == [None]
    assert not ctx.side_prompts and rt.sizer.store.totals()["checks"] == 0


def test_resumed_session_upgrades_the_weak_outcome(plugin: Any) -> None:
    ctx = FakeCtx(small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    for text, reply in (("What is 6 * 7?", "42"), ("No, that is wrong. Explain it again.", "It is 42.")):
        ctx.fire("pre_llm_call", session_id="s", user_message=text, model="big-m")
        ctx.fire("post_llm_call", session_id="s", assistant_response=reply)
        ctx.fire("on_session_finalize", session_id="s")  # every one-shot run closes its session
        last = rt.sizer.store.last_turn("s")
        assert (last["source"], last["weight"]) == ("session_end", 0.2)
    first = rt.sizer.store.turn("s:1")
    assert (first["source"], first["score"], first["weight"]) == ("correction", 0.0, 0.5)


def test_interrupted_turn_leaves_no_outcome(plugin: Any) -> None:
    ctx = FakeCtx(small_model="small-m")
    rt = plugin.register(ctx)
    ctx.fire("pre_llm_call", session_id="s", user_message="Write a long essay.", model="big-m")
    ctx.fire("on_session_end", session_id="s", completed=False, interrupted=True)
    ctx.fire("pre_llm_call", session_id="s", user_message="Something else now.", model="big-m")
    row = rt.sizer.store.turn("s:1")
    assert row["score"] is None and row["source"] == "interrupted"


def test_dashboard_snapshot_is_read_only_and_holds_no_text(tmp_path: Path) -> None:
    from continuum_core.rightsize import read_snapshot

    missing = tmp_path / "none" / "rightsize.db"
    assert read_snapshot(missing)["present"] is False
    assert not missing.parent.exists()  # reading never creates the file or its folder

    rs = make(tmp_path, small_model="small")
    run_turn(rs, "s1", "My secret Task text about invoices", reply="A private reply", good=True)
    run_turn(rs, "s2", "Another secret Task", reply="Another private reply", good=False)
    snap = read_snapshot(rs.store.path)
    assert snap["present"] and snap["turns"] == 2 and snap["scored"] == 2
    assert sum(m["turns"] for m in snap["by_model"]) == 2 and len(snap["recent"]) == 2
    flat = json.dumps(snap)
    assert "secret" not in flat and "private" not in flat and "invoices" not in flat


# ---------------- consent, off mode, refused requests, audit steps, import checks ----------------
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}}
         for n in ("terminal", "read_file", "web_search", "browser", "image_gen", "calendar")]
SEARCH = "Search the web for the price of item {} and report it."


def _seed_search_turns(folder: Path) -> None:
    rs = make(folder, size_tools="on", explore_share=0.0)
    for i in range(8):
        rs.start_turn(f"s{i}", SEARCH.format(i + 100), model="big-m")
        rs.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, f"s{i}", "big-m")
        rs.record_tool(f"s{i}", "web_search", "{}")
        rs.record_attempt(f"s{i}", ok=True, model="big-m")
        rs.end_turn(f"s{i}", f"Item {i + 100} costs 12 dollars.")
        rs.feedback(True, f"s{i}", source="check")
    rs.store.close()


def test_nothing_in_a_request_changes_without_the_grant_or_outside_auto(tmp_path: Path) -> None:
    request = {"model": "big-m", "messages": [], "tools": TOOLS}
    cases = (("auto-granted", {}, True, True), ("auto-no-grant", {}, False, False),
             ("observe", {"mode": "observe"}, True, False), ("off", {"mode": "off"}, True, False))
    for name, settings, granted, changes in cases:
        _seed_search_turns(tmp_path / name)
        rs = make(tmp_path / name, size_tools="on", explore_share=1.0, **settings)
        rs.switch_allowed = lambda granted=granted: granted
        before = rs.store.totals()["turns"]
        rs.start_turn("new", SEARCH.format(100), model="big-m")  # the same Task as a recorded good one
        shaped = rs.shape_request(request, "new", "big-m")
        assert bool(rs.recall_context("new")) is changes, name
        assert (shaped is not None) is changes, name
        if changes:
            assert len(shaped[0]["tools"]) < len(TOOLS)
        if name == "off":
            assert rs.store.totals()["turns"] == before and rs.scope_of("new") is None
            assert rs.recall(SEARCH.format(100), 3, scope=("", owner_of("", ""))) == []
        else:
            row = rs.store.turn("new:1")  # what it would keep is still on the record
            assert row["tools_total"] == len(TOOLS) and row["tools_kept"] < len(TOOLS), name
        rs.store.close()


def test_mode_off_stores_nothing_and_changes_nothing(plugin: Any) -> None:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0, check_share=1.0, mode="off")
    rt = plugin.register(ctx)
    cmd = ctx.commands["continuum"]
    assert ctx.fire("pre_llm_call", session_id="s", user_message="My private note about Bluebird.", model="big-m") == [None]
    assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": [], "tools": TOOLS}, session_id="s",
                                         model="big-m") is None
    ctx.fire("post_api_request", session_id="s", model="big-m", response_model="big-m", usage={"prompt_tokens": 9})
    ctx.fire("api_request_error", session_id="s", model="big-m", error={"type": "X"}, retryable=False)
    ctx.fire("post_tool_call", session_id="s", tool_name="terminal", result="x")
    assert ctx.fire("post_llm_call", session_id="s", assistant_response="A private reply.", side_question_token="tok") == [None]
    ctx.fire("on_session_end", session_id="s")
    ctx.fire("on_session_finalize", session_id="s")
    totals = rt.sizer.store.totals()
    assert totals["turns"] == 0 and totals["attempts"] == 0 and not ctx.side_prompts
    assert rt.sizer.store.policy_log(10) == [] and rt.sizer.blocked() == {}
    assert "records nothing" in cmd("status") and "is off" in cmd("good", session_id="s")
    assert json.loads(ctx.tools["continuum_recall"]["handler"]({"query": "Bluebird"}, session_id="s"))["records"] == []
    assert rt.sizer.audit_now()["action"] == "none"


def test_no_plugin_data_folder_means_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugin: Any) -> None:
    import continuum_core.storage as storage_mod

    shared = tmp_path / "continuum-standalone"
    shared.mkdir()
    monkeypatch.setattr(storage_mod, "resolve_continuum_db_path", lambda hermes_home=None: shared / "continuum.db")
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    ctx.fire("pre_llm_call", session_id="s", user_message="A message that must not land in a shared folder.", model="big-m")
    assert rt.sizer.mode == "off" and rt.sizer.store.totals()["turns"] == 0
    assert "stores nothing" in ctx.commands["continuum"]("status")


def _refuse(rs: RightSizer, session: str, text: str, **error: Any) -> None:
    rs.start_turn(session, text, model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, session, "big-m")[0]["model"] == "small-m"
    rs.record_attempt(session, ok=False, model="small-m", **error)


def test_refused_request_blocks_the_cheaper_model_and_holds_the_session(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    _refuse(rs, "s", "What is 3 * 7?", error="NotFoundError", retryable=False, status=404, reason="model_not_found")
    row = rs.store.turn("s:1")
    assert (row["score"], row["source"], row["closed"], row["api_errors"]) == (None, "rung_blocked", 1, 1)
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None  # nothing more in this turn
    assert rs.end_turn("s", "") is None and rs.store.turn("s:1")["score"] is None  # no outcome was learned
    assert list(rs.blocked()) == ["small-m"] and "Blocked: small-m" in rs.status_text()
    assert rs.store.policy_log(5)[0]["event"] == "rung-blocked"
    dec = rs.start_turn("s", "What is 4 * 7?", model="big-m")  # the same session stays on the user's own model
    assert dec.mode == "pinned" and "refused" in dec.reason
    assert rs.store.turn("s:1")["score"] is None  # the next message is no outcome for a turn that had no answer
    other = rs.start_turn("t", "What is 5 * 7?", model="big-m")  # other sessions: the blocked model is no rung
    assert other.mode == "observe" and rs.shape_request({"model": "big-m", "messages": []}, "t", "big-m") is None
    rs.store.close()
    again = make(tmp_path, explore_share=1.0)  # the block holds after a restart
    assert list(again.blocked()) == ["small-m"]
    assert again.unblock() == ["small-m"] and again.blocked() == {}
    assert again.start_turn("u", "What is 6 * 7?", model="big-m").mode == "explore"
    again.store.close()
    changed = make(tmp_path / "x", explore_share=1.0)
    _refuse(changed, "s", "What is 3 * 7?", error="AuthError", retryable=False, status=403, reason="model_entitlement")
    changed.store.close()
    renamed = make(tmp_path / "x", explore_share=1.0, small_model="other-m")  # new rung settings end the block
    assert renamed.blocked() == {} and renamed.start_turn("v", "What is 7 * 7?", model="big-m").model == "other-m"


def test_a_fault_in_the_request_does_not_block_the_model(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    _refuse(rs, "s", "What is 3 * 7?", error="BadRequest", retryable=False, status=400, reason="context_overflow")
    row = rs.store.turn("s:1")
    assert (row["score"], row["source"]) == (None, "host_error") and rs.blocked() == {}
    assert rs.start_turn("s", "What is 4 * 7?", model="big-m").mode == "pinned"  # this session still stays put
    assert rs.start_turn("t", "What is 5 * 7?", model="big-m").mode == "explore"


def test_a_passing_fault_weighs_less_than_a_real_failure(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    for session, reason, source, weight in (("a", "rate_limit", "api_error_transient", 0.2),
                                            ("b", "", "api_error", 0.6)):
        _refuse(rs, session, f"What is 3 * 7 in session {session}?", error="E", retryable=True, reason=reason)
        assert rs.shape_request({"model": "big-m", "messages": []}, session, "big-m") is None  # back on your own model
        rs.record_attempt(session, ok=True, model="big-m")
        rs.end_turn(session, "21")
        row = rs.store.turn(f"{session}:1")
        assert (row["source"], row["score"], row["weight"], row["escalated"]) == (source, 0.0, weight, 1)


def _put(rs: RightSizer, key: str, model: str, mode: str, score: float, ts: float, imported: int = 0,
         split: str = "holdout") -> None:
    fp = fingerprint(f"Stored Task number {key}", FP)
    rs.store.open_turn({"turn_key": key, "session_id": key, "seq": 1, "ts": ts, "platform": "", "owner": "",
                        "exact": fp["exact"], "template": fp["template"], "contract": fp["contract"],
                        "fp": json.dumps(fp), "base_model": "big-m", "rung": model, "model": model, "mode": mode,
                        "reason": "", "split": split, "score": score, "weight": 1.0, "source": "check", "closed": 1,
                        "imported": imported})


def _behind(rs: RightSizer, start: int, n: int, imported: int = 0) -> None:
    for i in range(start, start + n):
        _put(rs, f"c{i}", "small-m", "exploit", 0.0, 100.0 + i, imported)
        _put(rs, f"r{i}", "big-m", "reference", 1.0, 100.0 + i, imported)


def test_audit_steps_once_on_a_body_of_evidence(tmp_path: Path) -> None:
    rs = make(tmp_path)
    _behind(rs, 0, 8)
    assert rs.audit_now()["action"] == "tighten"
    state, log = rs.strictness(), len(rs.store.policy_log(100))
    assert state["floor"] == pytest.approx(0.75) and state["min_evidence"] == pytest.approx(4.0)
    for _ in range(25):  # Hermes fires the hook at the end of every turn
        rep = rs.audit_now()
        assert rep["action"] == "none" and rep["advised"] == "tighten" and rep["new_evidence"] == 0
    assert rs.strictness() == state and len(rs.store.policy_log(100)) == log
    _behind(rs, 8, 2)  # two new units of cheaper-model outcomes: one more step, and only one
    assert rs.audit_now()["action"] == "tighten" and rs.audit_now()["action"] == "none"
    assert rs.strictness()["floor"] == pytest.approx(0.80)


def test_audit_respects_self_tune_off_and_never_reads_imported_records(tmp_path: Path) -> None:
    fixed = make(tmp_path / "fixed", self_tune=False)
    _behind(fixed, 0, 8)
    rep = fixed.audit_now()
    assert (rep["action"], rep["advised"]) == ("none", "tighten") and "self_tune is off" in rep["note"]
    assert fixed.strictness() == fixed.defaults and fixed.store.policy_log(10) == []
    assert "advise" in wiring.command(fixed, tmp_path, "audit", platform="cli")
    imported = make(tmp_path / "imported")
    _behind(imported, 0, 8, imported=1)
    assert imported.audit_now()["action"] == "none" and imported.strictness() == imported.defaults


def test_import_checks_every_field_and_a_bad_line_never_stops_a_turn(tmp_path: Path) -> None:
    src = make(tmp_path / "a")
    for i in range(10):
        run_turn(src, f"s{i}", f"What is {i + 2} * 7?", good=True)
    good = [json.loads(line) for line in src.export(tmp_path / "exports").read_text().splitlines()]
    one = good[1]
    bad = [
        {**one, "fp": {k: v for k, v in one["fp"].items() if k != "profile"}},      # no shape profile
        {**one, "fp": {**one["fp"], "sketch": one["fp"]["sketch"][:5]}},            # a short sketch
        {**one, "fp": {**one["fp"], "words": {"1": 7.5}}},                          # a weight out of range
        {**one, "fp": {**one["fp"], "profile": ["x"] * len(one["fp"]["profile"])}},  # not numbers
        {**one, "score": 3.0}, {**one, "weight": 50.0}, {**one, "model": ""}, {**one, "fp": "text"}, [1, 2], "x",
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [good[0], *bad, *good[1:]]) + "\nnot json\n", encoding="utf-8")
    dst = make(tmp_path / "b", explore_share=0.0)
    assert dst.import_file(path) == 10
    rows = dst.store.scored_turns(50)
    assert all(r["mode"] == "import" and r["platform"] == "import" and r["exact"].startswith("imp-") for r in rows)
    dec = run_turn(dst, "mine", "What is 44 * 7?")  # a normal turn is chosen and recorded as before
    assert dst.store.turn("mine:1")["closed"] == 2 and dst.store.totals()["turns"] == 1
    # Ten imported passes are capped below the evidence a choice needs: they never move a Task alone.
    assert dec.mode == "reference" and dec.estimates["small-m"]["n"] <= CFG["chooser"]["import_max_evidence"]
    assert dst.recall("What is 4 * 7?", 3, scope=dst.scope_of("mine")) == []  # imported records are never recalled


def test_never_downsize_reads_inflected_words() -> None:
    empty = Pool([], FP, BALANCED)
    for text in ("Deploy the fix to the staging server.", "We are deploying tonight.", "List the open invoices.",
                 "Summarise the payments of March.", "Where are the passwords kept?", "Run the two migrations.",
                 "Read this medical report.", "Check the credentials file."):
        assert choose(fingerprint(text, FP), text, RUNGS, empty, CFG["chooser"], u=0.01, **STRICT).mode == "pinned", text
    for text in ("What is 5 * 7?", "Write a haiku about the sea.", "Employ a simpler word here."):
        assert choose(fingerprint(text, FP), text, RUNGS, empty, CFG["chooser"], u=0.01, **STRICT).mode == "explore", text


def test_reference_check_turns_stay_on_your_own_model_and_are_checked(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0, check_share=1.0)
    for i in range(12):
        run_turn(rs, f"s{i}", f"What is {i + 2} * 7?", good=True)
    seen = set()
    for i in range(40):
        dec = rs.start_turn(f"n{i}", f"What is {i + 50} * 7?", model="big-m")
        seen.add(dec.mode)
        shaped = rs.shape_request({"model": "big-m", "messages": []}, f"n{i}", "big-m")
        assert (shaped is None) == (dec.mode == "reference-check")
        rs.record_attempt(f"n{i}", ok=True, model=shaped[0]["model"] if shaped else "big-m")
        ended = rs.end_turn(f"n{i}", "ok")
        assert ended["check"] is True  # both kinds of turn get a check when the share is 1
        if dec.mode == "reference-check":
            assert rs.check_result(ended["turn_key"], "PASS") == "pass"
            row = rs.store.turn(ended["turn_key"])
            assert row["model"] == "big-m" and row["source"] == "reference_check"
    assert seen == {"exploit", "reference-check"}
    assert rs.status()["small_rung_tokens"] == 0 or "reference-check" not in {
        r["mode"] for r in rs.store.scored_turns(500) if r["model"] == "small-m"}


def test_weak_signals_ignore_polite_no_and_short_repeats(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "Write a haiku about the sea.")
    run_turn(rs, "s", "No problem, now list three rivers in Europe.")
    run_turn(rs, "s", "No worries. What is the capital of Peru?")
    run_turn(rs, "s", "ok")
    run_turn(rs, "s", "ok")
    run_turn(rs, "s", "No, the capital is Lima.")
    run_turn(rs, "s", "next")
    sources = [rs.store.turn(f"s:{i}")["source"] for i in range(1, 7)]
    assert sources == ["moved_on", "moved_on", "moved_on", "moved_on", "correction", "moved_on"]


def test_marks_and_reasons_never_fall_back_to_another_session(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "alice", "What is 3 * 7?")
    assert rs.feedback(False, "bob") is None and rs.store.turn("alice:1")["source"] is None
    assert "alice" not in rs.why_text("bob") and "3 * 7" not in rs.why_text("bob")
    assert "no recorded turn" in wiring.command(rs, tmp_path, "bad", session_id="bob")
    # With no session id the newest turn is meant, and only where one local user works: never on a chat platform.
    assert "no session id" in wiring.command(rs, tmp_path, "bad", platform="telegram")
    assert "no session id" in wiring.command(rs, tmp_path, "why", platform="telegram")
    assert "no session id" in wiring.command(rs, tmp_path, "pin", platform="discord")
    assert rs.store.turn("alice:1")["source"] is None and rs.store.get_policy("pinned", []) == []
    assert "marked bad" in wiring.command(rs, tmp_path, "bad", platform="cli")
    assert rs.store.turn("alice:1")["source"] == "explicit"


def test_forget_text_keeps_fingerprints_and_outcomes(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "My project Bluebird needs a budget of 4000.", reply="Bluebird answer", good=True)
    assert "1 turns" in wiring.command(rs, tmp_path, "forget-text", platform="cli")
    row = rs.store.turn("s:1")
    assert row["text"] is None and row["reply"] is None and row["score"] == 1.0 and json.loads(row["fp"])["sketch"]
    assert rs.store.policy_log(1)[0]["event"] == "text-forgotten"
    rs.start_turn("t", "My project Bluebird needs a budget of 4000.", model="big-m")
    assert rs.recall_context("t") is None  # no stored answer is left to add


def test_pin_holds_after_a_restart(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    rs.pin("s", True)
    rs.store.close()
    again = make(tmp_path, explore_share=1.0)
    assert again.start_turn("s", "What is 3 * 7?", model="big-m").mode == "pinned"
    again.pin("s", False)
    assert again.start_turn("s", "What is 4 * 7?", model="big-m").mode == "explore"


def test_unreadable_record_file_is_kept_aside_and_a_new_one_starts(tmp_path: Path) -> None:
    (tmp_path / "rightsize.db").write_bytes(b"this is not a record file\n" * 200)
    rs = make(tmp_path)
    kept = list(tmp_path.glob("rightsize.unreadable-*.db"))
    assert len(kept) == 1 and kept[0].read_bytes().startswith(b"this is not a record file")  # never deleted
    assert "kept as" in rs.store_note and "Note:" in rs.status_text()
    run_turn(rs, "s", "What is 3 * 7?")
    assert rs.store.totals()["turns"] == 1


def test_locked_record_file_never_breaks_a_turn(plugin: Any, tmp_path: Path) -> None:
    import sqlite3

    ctx = FakeCtx(small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    other = sqlite3.connect(tmp_path / "rightsize.db", timeout=0.1)
    other.execute("BEGIN EXCLUSIVE")
    try:
        assert ctx.fire("pre_llm_call", session_id="s", user_message="What is 6 * 7?", model="big-m") == [None]
        assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="s",
                                             model="big-m") is None  # the request goes out unchanged
        ctx.fire("post_api_request", session_id="s", model="big-m", response_model="big-m", usage={})
        assert ctx.fire("post_llm_call", session_id="s", assistant_response="42") == [None]
        assert isinstance(ctx.commands["continuum"]("status"), str)
    finally:
        other.rollback()
        other.close()
    ctx.fire("pre_llm_call", session_id="s", user_message="What is 6 * 8?", model="big-m")  # works again afterwards
    assert rt.sizer.store.totals()["turns"] == 1


def test_words_in_any_script_are_counted() -> None:
    fp = fingerprint("Привет мир, как дела сегодня?", FP)
    assert fp["n_words"] == 5 and sim("Привет мир, как дела сегодня?", "Привет мир, как дела завтра?") > 0.5
    assert sim("東京の天気を教えてください", "Write a haiku about the sea.") < 0.3
    names = fpm.PROFILE_NAMES
    assert len(fp["profile"]) == len(names) and any(v > 0 for v in fp["profile"])


# ---------------- second review: mode, blocks, providers, ids, imports, files ----------------
@pytest.mark.parametrize("value", [False, "false", "False", "auot", "on", 0, "", "OFF ", [], "none"])
def test_every_mode_that_is_not_auto_or_observe_is_off(plugin: Any, tmp_path: Path, value: Any) -> None:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0, check_share=1.0, mode=value)
    rt = plugin.register(ctx)
    assert rt.sizer.mode == "off"
    assert ctx.fire("pre_llm_call", session_id="s", user_message="A private note about Bluebird.", model="big-m") == [None]
    assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": [], "tools": TOOLS}, session_id="s",
                                         model="big-m") is None
    ctx.fire("post_api_request", session_id="s", model="big-m", response_model="big-m", usage={"prompt_tokens": 9})
    assert ctx.fire("post_llm_call", session_id="s", assistant_response="A private reply.", side_question_token="t") == [None]
    ctx.fire("on_session_finalize", session_id="s")
    assert rt.sizer.store.totals()["turns"] == 0 and not ctx.side_prompts
    assert not (tmp_path / "rightsize.db").exists()  # off with no record file: none is made
    status = ctx.commands["continuum"]("status")
    assert "records nothing" in status
    plain_off = value is False or value == "OFF "  # these two mean off; every other value gets a note
    assert ("is read as off" in status) == (not plain_off)


def test_modes_auto_and_observe_are_read_in_any_case() -> None:
    for value, want in (("AUTO", "auto"), (" Observe ", "observe"), ("auto", "auto")):
        rs = RightSizer(":memory:", settings={"mode": value})
        assert rs.mode == want and rs.mode_note() == ""


def test_commands_write_nothing_when_off(plugin: Any, tmp_path: Path) -> None:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", mode="off")
    rt = plugin.register(ctx)
    cmd = ctx.commands["continuum"]
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "x.jsonl").write_text('{"schema": "continuum.records/2"}\n', encoding="utf-8")
    for sub in ("good", "bad", "pin", "unpin", "unblock", "audit", "tune", "import x.jsonl"):
        assert "is off" in cmd(sub, session_id="s", platform="cli"), sub
    assert "0 turns" in cmd("forget-text", platform="cli")
    assert rt.sizer.store.totals()["turns"] == 0 and rt.sizer.store.get_policy("pinned", []) == []
    assert not (tmp_path / "rightsize.db").exists()


def _wired(plugin: Any, **config: Any) -> tuple[FakeCtx, Any]:
    ctx = FakeCtx(granted=BOTH, small_model="small-m", explore_share=1.0, check_share=0.0, **config)
    return ctx, plugin.register(ctx)


def _hermes_refusal(ctx: FakeCtx, session: str, text: str, reason: str, status: int = 400, **extra: Any) -> Any:
    """One turn through the real hook callbacks: the cheaper model is asked and the provider refuses for good."""
    ctx.fire("pre_llm_call", session_id=session, user_message=text, model="big-m", platform="cli")
    shaped = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id=session,
                                           model="big-m", provider="prov-a", base_url="https://a.example/v1", **extra)
    if shaped:
        ctx.fire("api_request_error", session_id=session, model="big-m", request=shaped["request"],
                 error={"type": "ProviderError"}, retryable=False, status_code=status, reason=reason)
    return shaped


@pytest.mark.parametrize("reason", ["auth", "auth_permanent", "billing", "data_policy_denied", "upstream_blocked",
                                    "format_error", "content_policy_blocked", "context_overflow"])
def test_account_and_request_faults_never_block_the_cheaper_model(plugin: Any, reason: str) -> None:
    ctx, rt = _wired(plugin)
    for i in range(5):  # also not after many of them
        assert _hermes_refusal(ctx, f"s{i}", f"What is {i + 3} * 7?", reason, status=402)["request"]["model"] == "small-m"
    assert rt.sizer.blocked() == {} and "Blocked" not in ctx.commands["continuum"]("status")
    assert rt.sizer.store.turn("s0:1")["source"] == "host_error" and rt.sizer.store.turn("s0:1")["score"] is None


@pytest.mark.parametrize("reason", ["model_not_found", "model_entitlement"])
def test_a_refused_model_is_blocked_at_once_through_the_hooks(plugin: Any, reason: str) -> None:
    ctx, rt = _wired(plugin)
    _hermes_refusal(ctx, "s", "What is 3 * 7?", reason, status=404)
    assert list(rt.sizer.blocked()) == ["small-m"] and rt.sizer.blocked()["small-m"]["reason"] == reason
    assert _hermes_refusal(ctx, "t", "What is 4 * 7?", reason) is None  # no turn goes there any more


def test_other_refusals_block_only_after_three_in_a_row(plugin: Any) -> None:
    ctx, rt = _wired(plugin)
    _hermes_refusal(ctx, "a", "What is 3 * 7?", "unknown")
    _hermes_refusal(ctx, "b", "What is 4 * 7?", "")
    assert rt.sizer.blocked() == {}
    # the cheaper model answers one turn: the refusals before it are over
    ctx.fire("pre_llm_call", session_id="c", user_message="What is 5 * 7?", model="big-m", platform="cli")
    shaped = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="c", model="big-m",
                                           provider="prov-a", base_url="https://a.example/v1")
    ctx.fire("post_api_request", session_id="c", model="big-m", response_model=shaped["request"]["model"], usage={})
    ctx.fire("post_llm_call", session_id="c", assistant_response="35")
    for i, session in enumerate("def"):
        assert rt.sizer.blocked() == {}
        _hermes_refusal(ctx, session, f"What is {i + 6} * 7?", "unknown")
    assert list(rt.sizer.blocked()) == ["small-m"]


def test_a_turn_on_another_provider_is_left_alone(plugin: Any) -> None:
    ctx, rt = _wired(plugin)

    def ask(session: str, text: str, provider: str, base_url: str) -> Any:
        ctx.fire("pre_llm_call", session_id=session, user_message=text, model="big-m", platform="cli")
        out = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id=session,
                                            model="big-m", provider=provider, base_url=base_url)
        ctx.fire("post_api_request", session_id=session, model="big-m",
                 response_model=out["request"]["model"] if out else "big-m", usage={})
        ctx.fire("post_llm_call", session_id=session, assistant_response="ok")
        return out

    assert ask("a", "What is 3 * 7?", "prov-a", "https://a.example/v1")["request"]["model"] == "small-m"
    assert ask("b", "What is 4 * 7?", "prov-b", "https://b.example/v1") is None  # a fallback or a changed provider
    row = rt.sizer.store.turn("b:1")
    assert row["mode"] == "observe" and "another provider" in row["reason"] and row["model"] == "big-m"
    assert ask("c", "What is 5 * 7?", "prov-a", "https://a.example/v1/")["request"]["model"] == "small-m"
    assert ask("d", "What is 6 * 7?", "prov-a", "https://other.example/v1") is None  # the same name, another address
    assert "provider prov-a only" in ctx.commands["continuum"]("status", platform="cli")
    assert "prov-a" not in ctx.commands["continuum"]("status", platform="discord")  # not for a chat user
    ctx.commands["continuum"]("unblock", platform="cli")  # the user moved for good: the next provider becomes the home
    assert ask("e", "What is 8 * 7?", "prov-b", "https://b.example/v1")["request"]["model"] == "small-m"


def test_no_recall_for_an_unnamed_sender_on_a_chat_platform(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=0.0)
    task = "What is the capital of Australia?"
    for platform in ("telegram", "cli"):
        rs.start_turn(f"first-{platform}", task, model="big-m", platform=platform, sender_id="")
        rs.record_attempt(f"first-{platform}", ok=True, model="big-m")
        rs.end_turn(f"first-{platform}", "Canberra.")
        rs.feedback(True, f"first-{platform}")
    rs.start_turn("t2", task, model="big-m", platform="telegram", sender_id="")
    assert rs.recall_context("t2") is None and rs.scope_of("t2") is None  # two unnamed senders may be two people
    rs.start_turn("c2", task, model="big-m", platform="cli", sender_id="")
    assert "Canberra." in rs.recall_context("c2") and rs.scope_of("c2") is not None  # one local user
    rs.start_turn("t3", task, model="big-m", platform="telegram", sender_id="alice")
    assert rs.recall_context("t3") is None  # a named sender never gets what an unnamed one wrote


def _export_lines(rs: RightSizer, tmp_path: Path) -> list[dict[str, Any]]:
    path = rs.export(tmp_path / "exports")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_imported_time_stamps_are_bounded_and_own_records_stay_in_the_pool(tmp_path: Path) -> None:
    src = make(tmp_path / "src")
    for i in range(30):
        run_turn(src, f"s{i}", f"What is {i + 2} * 7?", good=True)
    head, *rows = _export_lines(src, tmp_path)
    for i, row in enumerate(rows):
        row["ts"] = [9e18, float("inf"), "NaN", -5, None, "soon"][i % 6]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(x) for x in [head, *rows]) + "\n", encoding="utf-8")
    rs = make(tmp_path / "dst")
    rs.cfg["chooser"]["max_pool"] = 5
    rs.cfg["import"]["max_pool"] = 3
    for i in range(5):
        run_turn(rs, f"own{i}", f"Summarise the notes of meeting {i}.", good=True)
    assert rs.import_file(bad) == 30
    now = rs.clock()
    stamps = [r["ts"] for r in rs.store.scored_turns(100, imported_only=True)]
    assert len(stamps) == 30 and all(0.0 <= t <= now for t in stamps)
    pool = rs._pool("any").records
    assert sum(1 for r in pool if not r.get("imported")) == 5 and sum(1 for r in pool if r.get("imported")) == 3


def test_imported_records_never_cut_the_tool_list(tmp_path: Path) -> None:
    src = make(tmp_path / "src", size_tools="on", explore_share=0.0)
    for i in range(8):
        src.start_turn(f"s{i}", SEARCH.format(i + 100), model="big-m")
        src.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, f"s{i}", "big-m")
        src.record_tool(f"s{i}", "web_search", "{}")
        src.record_attempt(f"s{i}", ok=True, model="big-m")
        src.end_turn(f"s{i}", "done")
        src.feedback(True, f"s{i}", source="check")
    src.start_turn("probe", SEARCH.format(977), model="big-m")
    assert src.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, "probe", "big-m") is not None
    path = src.export(tmp_path / "exports")
    rs = make(tmp_path / "dst", size_tools="on", explore_share=0.0)
    assert rs.import_file(path) == 8
    rs.start_turn("new", SEARCH.format(977), model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, "new", "big-m") is None
    assert rs.store.turn("new:1")["tools_kept"] == rs.store.turn("new:1")["tools_total"] == len(TOOLS)


def test_a_long_import_does_not_hold_a_turn(tmp_path: Path) -> None:
    import threading

    src = make(tmp_path / "src")
    for i in range(20):
        run_turn(src, f"s{i}", f"What is {i + 2} * 7?", good=True)
    head, *rows = _export_lines(src, tmp_path)
    big = tmp_path / "big.jsonl"
    big.write_text("\n".join(json.dumps(x) for x in [head] + rows * 150) + "\n", encoding="utf-8")
    rs = make(tmp_path / "dst")
    held = threading.Event()
    seen: list[bool] = []
    real = rs.store.add_turns

    def slow_add(ready: list[dict[str, Any]]) -> int:  # the file is read by now; a turn must be able to start here
        worker = threading.Thread(target=lambda: (rs.start_turn("live", "What is 9 * 9?", model="big-m"), held.set()))
        worker.start()
        seen.append(held.wait(5.0))
        return real(ready)

    rs.store.add_turns = slow_add  # type: ignore[method-assign]
    assert rs.import_file(big) == 3000 and seen == [True]


def test_a_check_request_is_never_changed_or_counted_as_a_turn_attempt(plugin: Any) -> None:
    ctx, rt = _wired(plugin)
    ctx.fire("pre_llm_call", session_id="s", user_message="What is 3 * 7?", model="big-m", platform="cli")
    request = {"model": "big-m", "messages": [], "tools": TOOLS}
    # the check of the turn before arrives while this turn runs, under the same session id
    for ids in ({"turn_id": "turn-1:side_question"}, {"api_request_id": "turn-1:side_question:api:1"}):
        assert ctx.middleware["llm_request"](request=request, session_id="s", model="big-m", **ids) is None
    ctx.fire("post_api_request", session_id="s", model="big-m", response_model="big-m", usage={"prompt_tokens": 5},
             api_request_id="turn-1:side_question:api:1")
    ctx.fire("api_request_error", session_id="s", model="big-m", error={"type": "X"}, retryable=False,
             reason="model_not_found", turn_id="turn-1:side_question")
    assert rt.sizer.store.totals()["attempts"] == 0 and rt.sizer.blocked() == {}
    assert ctx.middleware["llm_request"](request=request, session_id="s", model="big-m")["request"]["model"] == "small-m"


def test_that_is_fine_is_no_correction(tmp_path: Path) -> None:
    rs = make(tmp_path)
    for i, nxt in enumerate(["No, that is fine.", "no that's fine, thanks", "No, all good", "No. It's ok"]):
        run_turn(rs, f"s{i}", f"Do you need more detail on point {i}?")
        rs.start_turn(f"s{i}", nxt, model="big-m")
        assert rs.store.turn(f"s{i}:1")["source"] == "moved_on", nxt
    run_turn(rs, "x", "What is 3 * 7?")
    rs.start_turn("x", "No, that is wrong.", model="big-m")
    assert rs.store.turn("x:1")["source"] == "correction"


def test_a_pin_stays_after_the_session_is_closed(plugin: Any, tmp_path: Path) -> None:
    ctx, rt = _wired(plugin)
    assert "stays on your own model" in ctx.commands["continuum"]("pin", session_id="s", platform="cli")
    ctx.fire("on_session_end", session_id="s")
    ctx.fire("on_session_finalize", session_id="s")
    assert rt.sizer.start_turn("s", "What is 3 * 7?", model="big-m").mode == "pinned"
    rt.sizer.store.close()
    ctx2, rt2 = _wired(plugin)  # Hermes starts again and the user resumes the session
    assert rt2.sizer.start_turn("s", "What is 4 * 7?", model="big-m").mode == "pinned"
    ctx2.commands["continuum"]("unpin", session_id="s", platform="cli")
    assert rt2.sizer.start_turn("s", "What is 5 * 7?", model="big-m").mode == "explore"


def test_a_record_file_another_process_writes_to_is_left_alone(plugin: Any, tmp_path: Path) -> None:
    import sqlite3

    first = make(tmp_path)
    run_turn(first, "s", "What is 3 * 7?", good=True)
    first.store.close()
    other = sqlite3.connect(tmp_path / "rightsize.db", timeout=0.1)
    other.execute("BEGIN IMMEDIATE")  # another Hermes process is in the middle of a write
    other.execute("UPDATE rs_turns SET reason = 'theirs'")
    try:
        ctx, rt = _wired(plugin)  # the plugin still loads, and it stays on
        assert rt.sizer.mode == "auto" and not rt.sizer.store_note
        # a turn in this moment cannot be recorded: it goes out as Hermes built it, and nothing is raised
        assert ctx.fire("pre_llm_call", session_id="t", user_message="What is 6 * 7?", model="big-m") == [None]
        assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="t",
                                             model="big-m") is None
    finally:
        other.commit()
        other.close()
    assert not list(tmp_path.glob("rightsize.unreadable-*"))  # never renamed while another process uses it
    assert rt.sizer.store.turn("s:1")["reason"] == "theirs"  # the other process's write is there
    ctx.fire("pre_llm_call", session_id="u", user_message="What is 7 * 7?", model="big-m")  # and the next turn records
    assert rt.sizer.store.turn("u:1") is not None


def test_a_record_file_locked_at_start_is_left_alone(plugin: Any, tmp_path: Path) -> None:
    import sqlite3

    first = make(tmp_path)
    run_turn(first, "s", "What is 3 * 7?", good=True)
    first.store.close()
    other = sqlite3.connect(tmp_path / "rightsize.db", timeout=0.1)
    other.execute("PRAGMA journal_mode=DELETE")  # a file whose lock also shuts readers out
    other.commit()
    before = (tmp_path / "rightsize.db").read_bytes()
    other.execute("BEGIN EXCLUSIVE")
    try:
        ctx, rt = _wired(plugin)  # the plugin still loads
        assert rt.sizer.mode == "off" and "left as it is" in rt.sizer.store_note
        assert ctx.fire("pre_llm_call", session_id="t", user_message="What is 6 * 7?", model="big-m") == [None]
        assert "Note:" in ctx.commands["continuum"]("status")
    finally:
        other.rollback()
        other.close()
    assert not list(tmp_path.glob("rightsize.unreadable-*"))  # never renamed while another process uses it
    assert (tmp_path / "rightsize.db").read_bytes() == before
    again = make(tmp_path)
    assert again.store.totals()["turns"] == 1  # the records are all there at the next start


def test_an_unreadable_file_that_cannot_be_set_aside_never_stops_the_plugin(plugin: Any, tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "rightsize.db").write_bytes(b"this is not a record file\n" * 200)

    def no_rename(self: Path, target: Any) -> Any:
        raise PermissionError("read-only folder")

    monkeypatch.setattr(Path, "rename", no_rename)
    ctx, rt = _wired(plugin)
    assert rt.sizer.mode == "off" and "could not be set aside" in rt.sizer.store_note
    assert (tmp_path / "rightsize.db").read_bytes().startswith(b"this is not a record file")
    assert ctx.fire("pre_llm_call", session_id="t", user_message="What is 6 * 7?", model="big-m") == [None]


def test_the_side_files_of_an_unreadable_record_file_are_kept_with_it(tmp_path: Path) -> None:
    (tmp_path / "rightsize.db").write_bytes(b"this is not a record file\n" * 200)
    (tmp_path / "rightsize.db-wal").write_bytes(b"old wal")
    rs = make(tmp_path)
    kept = sorted(p.name for p in tmp_path.glob("rightsize.unreadable-*"))
    assert len(kept) == 2 and kept[1] == kept[0] + "-wal"
    run_turn(rs, "s", "What is 3 * 7?")
    assert rs.store.totals()["turns"] == 1


def test_forgotten_text_is_gone_from_the_file_bytes(tmp_path: Path) -> None:
    rs = make(tmp_path)
    for i in range(30):
        run_turn(rs, f"s{i}", f"My project Bluebird-{i} needs a budget of {i}000 for Zanzibar.", reply=f"Quokka answer {i}",
                 good=True)

    def all_bytes() -> bytes:
        return b"".join(p.read_bytes() for p in tmp_path.glob("rightsize.db*"))

    assert b"Bluebird" in all_bytes() and b"Quokka" in all_bytes()
    assert "30 turns" in wiring.command(rs, tmp_path, "forget-text", platform="cli")
    assert b"Bluebird" not in all_bytes() and b"Zanzibar" not in all_bytes() and b"Quokka" not in all_bytes()
    assert rs.store.totals()["turns"] == 30


def test_stored_text_has_an_age_limit(tmp_path: Path) -> None:
    rs = make(tmp_path, text_days=30)
    run_turn(rs, "old", "My old note about Bluebird.", reply="old reply", good=True)
    rs.clock.t += 31 * 86400
    run_turn(rs, "new", "My new note about Quokka.", reply="new reply", good=True)
    rs.end_session("new")
    old, new = rs.store.turn("old:1"), rs.store.turn("new:1")
    assert old["text"] is None and old["reply"] is None and old["score"] == 1.0
    assert new["text"] and new["reply"] == "new reply"
    assert rs.store.policy_log(5)[-1]["event"] == "old-text-removed" or any(
        e["event"] == "old-text-removed" for e in rs.store.policy_log(5))
    keep = make(tmp_path / "keep", text_days=0)
    run_turn(keep, "old", "My old note about Bluebird.", good=True)
    keep.clock.t += 400 * 86400
    keep.end_session("old")
    assert keep.store.turn("old:1")["text"]


def test_a_turn_kept_on_the_own_model_is_not_changed_at_all(tmp_path: Path) -> None:
    rs = make(tmp_path, size_tools="on", explore_share=0.0)
    for i in range(8):
        rs.start_turn(f"s{i}", SEARCH.format(i + 100), model="big-m")
        rs.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, f"s{i}", "big-m")
        rs.record_tool(f"s{i}", "web_search", "{}")
        rs.record_attempt(f"s{i}", ok=True, model="big-m")
        rs.end_turn(f"s{i}", "The price is 12.")
        rs.feedback(True, f"s{i}")
    rs.pin("p", True)
    rs.start_turn("p", SEARCH.format(100), model="big-m")  # the same Task as a recorded good one
    assert rs.recall_context("p") is None
    assert rs.shape_request({"model": "big-m", "messages": [], "tools": TOOLS}, "p", "big-m") is None
    rs.start_turn("free", SEARCH.format(100), model="big-m")  # the same Task in a session that is not pinned
    assert rs.recall_context("free") and rs.shape_request({"model": "big-m", "messages": [], "tools": TOOLS},
                                                          "free", "big-m") is not None


def test_store_text_false_turns_recall_off_for_text_stored_before(tmp_path: Path) -> None:
    rs = make(tmp_path)
    task = "What is the capital of Australia?"
    run_turn(rs, "a", task, reply="Canberra.", good=True)
    rs.start_turn("b", task, model="big-m")
    assert "Canberra." in rs.recall_context("b") and rs.recall(task, scope=rs.scope_of("b"))
    rs.store.close()
    again = make(tmp_path, store_text=False)
    again.start_turn("c", task, model="big-m")
    assert again.recall_context("c") is None and again.scope_of("c") is None
    assert again.recall(task, scope=("", owner_of("", ""))) == []
    assert again.store.turn("c:1")["text"] is None


def test_a_turn_that_fails_to_start_leaves_no_old_state_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum_core.rightsize import engine as engine_mod

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")[0]["model"] == "small-m"

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("fingerprint failed")

    monkeypatch.setattr(engine_mod.fpm, "fingerprint", broken)
    with pytest.raises(RuntimeError):
        rs.start_turn("s", "Deploy the payment service to production.", model="big-m")
    # the new turn has no choice of its own: it goes out as Hermes built it, not with the choice of the turn before
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None


def test_a_request_too_long_for_the_small_model_returns_to_the_own_model(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0, small_model_max_context=1000)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    short = {"model": "big-m", "messages": [{"role": "user", "content": "What is 3 * 7?"}]}
    assert rs.shape_request(short, "s", "big-m")[0]["model"] == "small-m"
    long = {"model": "big-m", "messages": [{"role": "user", "content": "word " * 4000}]}
    assert rs.shape_request(long, "s", "big-m") is None and rs.store.turn("s:1")["escalated"] == 1
    # a message that is already too long is never tried on the small model
    dec = rs.start_turn("t", "Summarise this: " + "word " * 4000, model="big-m")
    assert dec.mode == "reference" and dec.model == ""


def test_why_shows_every_level_and_the_lsh_band_match(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "a", "What is 3 * 7?", good=True)
    run_turn(rs, "b", "What is 4 * 7?", good=True)
    why = rs.why_text("b")
    assert "LSH band" in why and "template 1" in why and "exact 0" in why and "profile" in why


def test_wrong_settings_never_stop_a_turn(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share="a lot", check_share=[], small_model_cost_ratio="cheap", text_days="x",
              small_model_max_context="big")
    assert run_turn(rs, "s", "What is 3 * 7?") is not None
    rs.end_session("s")


def test_the_lab_setting_registers_tools_only(plugin: Any) -> None:
    ctx = FakeCtx(enable_lab_tools=True)  # FakeCtx fails the test when a provider is registered
    plugin.register(ctx)
    assert set(ctx.hooks) == set(wiring.HOOKS) and all(len(v) == 1 for v in ctx.hooks.values())
    assert len([n for n, kw in ctx.tools.items() if kw.get("check_fn") is None or kw["check_fn"]()]) == 19


def test_the_recall_tool_shows_records_only_where_continuum_may_change_a_turn(plugin: Any, tmp_path: Path) -> None:
    seed = make(tmp_path)
    run_turn(seed, "a", "What is the capital of Australia?", reply="Canberra.", good=True)
    seed.store.close()
    for name, granted, settings, shown in (("auto", BOTH, {}, True), ("no grant", (), {}, False),
                                           ("observe", BOTH, {"mode": "observe"}, False)):
        ctx = FakeCtx(granted=granted, small_model="small-m", explore_share=0.0, **settings)
        rt = plugin.register(ctx)
        ctx.fire("pre_llm_call", session_id="b", user_message="Tell me about Australia.", model="big-m",
                 platform="cli")
        out = json.loads(ctx.tools["continuum_recall"]["handler"]({"query": "capital of Australia"}, session_id="b"))
        assert bool(out["records"]) is shown, name
        assert ("only observes" in out.get("note", "")) is (not shown), name
        rt.sizer.store.close()


def test_a_grant_taken_back_in_the_middle_of_a_turn_stops_every_change(tmp_path: Path) -> None:
    _seed_search_turns(tmp_path)
    rs = make(tmp_path, size_tools="on", explore_share=1.0)
    granted = [True]
    rs.switch_allowed = lambda: granted[0]
    request = {"model": "big-m", "messages": [], "tools": TOOLS}
    rs.start_turn("new", SEARCH.format(100), model="big-m")
    assert rs.shape_request(request, "new", "big-m") is not None
    granted[0] = False  # the user takes the consent back while the turn runs
    assert rs.shape_request(request, "new", "big-m", api_call_count=2) is None
    assert rs.recall_context("new") is None
    row = rs.store.turn("new:1")
    # the first model call went out on the cheaper model, so the turn is marked as answered by two models
    assert row["mode"] == "mixed" and "consent was taken back" in row["reason"] and "ran on small-m" in row["reason"]
    granted[0] = True  # given again in the same turn: the turn stays as it is
    assert rs.shape_request(request, "new", "big-m", api_call_count=3) is None


# ---- the third review: settings typed as text, the age limit of text, whole-file commands, two processes ----

def test_a_yes_or_no_setting_typed_as_text_is_read_as_the_user_meant_it(tmp_path: Path) -> None:
    for i, (value, stored) in enumerate((("false", False), ("No", False), ("off", False), ("0", False), (0, False),
                                         ("true", True), ("YES", True), (1, True), (None, True),
                                         ("maybe", False), ([], False), (2, False))):
        rs = make(tmp_path / str(i), store_text=value)
        run_turn(rs, "s", "What is 3 * 7?", reply="21")
        row = rs.store.turn("s:1")
        assert (row["text"] is not None) is stored and (row["reply"] is not None) is stored, repr(value)
        rs.store.close()
    quiet = make(tmp_path / "recall", recall="false")
    run_turn(quiet, "a", "What is the capital of Australia?", reply="Canberra.", good=True)
    quiet.start_turn("b", "What is the capital of Australia?", model="big-m")
    assert quiet.recall_context("b") is None
    assert make(tmp_path / "tune", self_tune="false").flag("self_tune") is False
    assert make(tmp_path / "lab", enable_lab_tools="perhaps").flag("enable_lab_tools") is False
    assert make(tmp_path / "model", small_model=["small-m"]).start_turn("s", "What is 3 * 7?", model="big-m").mode == "observe"


def test_no_check_question_in_observe_mode_or_without_the_model_grant(plugin: Any) -> None:
    for name, granted, settings in (("observe", BOTH, {"mode": "observe"}),
                                    ("no model grant", (wiring.CHECK_CAPABILITY,), {})):
        ctx = FakeCtx(granted=granted, small_model="small-m", explore_share=1.0, check_share=1.0, **settings)
        rt = plugin.register(ctx)
        ctx.fire("pre_llm_call", session_id="a", user_message="What is 6 * 7?", model="big-m", platform="cli")
        assert ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="a",
                                             model="big-m") is None, name
        ctx.fire("post_api_request", session_id="a", model="big-m", response_model="big-m", usage={})
        assert ctx.fire("post_llm_call", session_id="a", assistant_response="42",
                        side_question_token="tok") == [None], name
        assert not ctx.side_prompts and rt.sizer.store.totals()["checks"] == 0, name
        rt.sizer.store.close()


def test_stored_text_is_removed_after_text_days_in_every_mode(tmp_path: Path) -> None:
    day = 86400.0
    rs = make(tmp_path)
    run_turn(rs, "old", "What is 3 * 7?", reply="21", good=True)
    rs.clock.t += 89 * day
    rs.expire_text(force=True)
    assert rs.store.turn("old:1")["text"] is not None  # younger than the limit: kept
    rs.clock.t += 2 * day
    run_turn(rs, "new", "What is 4 * 7?", reply="28")
    rs.expire_text(force=True)
    old, new = rs.store.turn("old:1"), rs.store.turn("new:1")
    assert old["text"] is None and old["reply"] is None and old["score"] == 1.0  # the outcome stays
    assert new["text"] is not None
    # a session that ends removes old text as well, also when Continuum is off
    rs.clock.t += 91 * day
    rs.store.close()
    off = make(tmp_path, mode="off")
    off.clock.t = rs.clock.t
    off.end_session("any")
    assert off.store.turn("new:1")["text"] is None
    off.store.close()
    # text_days: 0 keeps text; a value that is no number means the default, not "for ever"
    keep = make(tmp_path / "keep", text_days=0)
    run_turn(keep, "s", "What is 3 * 7?", reply="21")
    keep.clock.t += 400 * day
    keep.expire_text(force=True)
    assert keep.store.turn("s:1")["text"] is not None
    assert make(tmp_path / "bad", text_days="x").text_days() == 90 == make(tmp_path / "neg", text_days=-3).text_days()


def test_whole_file_commands_answer_only_to_the_local_user(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "a", "What is 3 * 7?", reply="21", good=True)
    before = (rs.store.text_turns(), rs.store.policy_log(50), rs.strictness())
    for platform in ("telegram", "discord", "api_server", ""):
        for sub in ("forget-text", "forget_text", "unblock", "export", "import x.jsonl", "tune", "audit"):
            text = wiring.command(rs, tmp_path, sub, session_id="chat-1", platform=platform)
            assert "only for the local user" in text and "hermes continuum" in text, (platform, sub)
    assert (rs.store.text_turns(), rs.store.policy_log(50), rs.strictness()) == before
    assert not (tmp_path / "exports").exists()
    # a chat user still has the commands about the user's own session
    assert "Choice" in wiring.command(rs, tmp_path, "why", session_id="a", platform="telegram")
    assert "marked bad" in wiring.command(rs, tmp_path, "bad", session_id="a", platform="telegram")
    assert "stays on your own model" in wiring.command(rs, tmp_path, "pin", session_id="a", platform="telegram")
    for platform in ("cli", "tui", "local"):
        assert "only for the local user" not in wiring.command(rs, tmp_path, "audit", platform=platform)
    assert "1 turns" in wiring.command(rs, tmp_path, "forget-text", platform="cli")


def test_lab_commands_store_nothing_unless_the_lab_is_on(plugin: Any, tmp_path: Path) -> None:
    for settings in ({}, {"mode": "off"}, {"mode": "off", "enable_lab_tools": True}):
        ctx = FakeCtx(small_model="small-m", **settings)
        plugin.register(ctx)
        seen = {p.name: p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()}
        for sub in ("lab", "lab step", "step", "omninode"):
            assert "stored nothing" in ctx.commands["continuum"](sub, platform="cli"), (settings, sub)
        assert {p.name: p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()} == seen, settings


def test_two_processes_on_one_record_file_keep_each_others_pins_and_blocks(tmp_path: Path) -> None:
    one, two = make(tmp_path), make(tmp_path)  # two Hermes processes, one record file
    one.pin("a", True)
    two.pin("b", True)
    assert set(one.pinned()) == set(two.pinned()) == {"a", "b"}
    assert one.start_turn("b", "What is 3 * 7?", model="big-m").mode == "pinned"  # a pin made by the other process
    two.pin("a", False)
    assert set(one.pinned()) == {"b"}
    assert one.start_turn("a", "What is 4 * 7?", model="big-m").mode != "pinned"


def test_a_turn_that_two_models_answered_is_kept_out_of_the_evidence(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    granted = [True]
    rs.switch_allowed = lambda: granted[0]
    rs.start_turn("m", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "m", "big-m")[0]["model"] == "small-m"
    rs.record_attempt("m", ok=True, model="small-m", finish="tool_calls")
    granted[0] = False  # from here the turn runs on the user's own model
    assert rs.shape_request({"model": "big-m", "messages": []}, "m", "big-m", api_call_count=2) is None
    rs.record_attempt("m", ok=True, model="big-m", finish="stop")
    rs.end_turn("m", "21")
    rs.feedback(True, "m")
    assert rs.store.turn("m:1")["mode"] == "mixed"
    assert all(r["turn_key"] != "m:1" for r in rs.store.scored_turns(100))
    assert all(r.get("turn_key") != "m:1" for r in rs.store.export_rows())
    assert rs.store.totals()["turns"] == 1  # the turn itself stays on the record


def test_a_failed_write_leaves_no_open_transaction(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", good=True)
    other = sqlite3.connect(tmp_path / "rightsize.db", timeout=0.1)
    other.execute("BEGIN IMMEDIATE")
    other.execute("UPDATE rs_turns SET reason = 'theirs'")
    with pytest.raises(sqlite3.OperationalError):
        rs.store.update_turn("s:1", reason="mine")
    assert rs.store.turn("s:1")["reason"] != "theirs"  # a read in between, before the other process commits
    other.commit()
    other.close()
    assert rs.store.turn("s:1")["reason"] == "theirs"  # and no old state is read after it
    rs.store.update_turn("s:1", reason="mine")
    assert rs.store.turn("s:1")["reason"] == "mine"


# ---- the fourth review: clean-up off the turn path, the record after a failed write, error text, the lab ----
def test_a_turn_only_clears_old_text_and_the_start_of_hermes_rewrites_the_file(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "old", "What is 3 * 7?", reply="21", good=True)
    rs.clock.t += 91 * 86400.0
    vacuums: list[str] = []
    rs.store._db.set_trace_callback(lambda sql: vacuums.append(sql) if "VACUUM" in sql.upper() else None)
    rs.start_turn("new", "What is 4 * 7?", model="big-m")  # the hourly check on the path of a turn
    assert rs.store.turn("old:1")["text"] is None  # the rows are cleared at once
    assert not vacuums and rs.store.compact_pending()  # the whole file is not rewritten while a user waits
    rs.end_session("new")
    assert not vacuums  # nor at the end of a session
    rs.store.close()
    again = make(tmp_path)  # the start of Hermes does it
    assert not again.store.compact_pending()
    again.store.close()


def test_forget_text_says_so_when_the_file_cannot_be_rewritten(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)
    reader = sqlite3.connect(tmp_path / "rightsize.db", timeout=0.1)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM rs_turns").fetchone()  # another process reads the file now
    text = wiring.command(rs, tmp_path, "forget-text", platform="cli")
    assert "Removed the stored Task and answer text of 1 turns" in text
    assert "could not be rewritten now" in text and rs.store.compact_pending()
    assert rs.store.turn("s:1")["text"] is None  # the row itself is cleared in any case
    reader.rollback()
    reader.close()
    text = wiring.command(rs, tmp_path, "forget-text", platform="cli")  # nothing left to clear, the rewrite is done
    assert "could not be rewritten" not in text and not rs.store.compact_pending()


def test_an_answer_of_the_users_own_model_is_never_credited_to_the_cheaper_model(tmp_path: Path) -> None:
    import sqlite3

    granted = [True]
    rs = make(tmp_path, explore_share=1.0)
    rs.switch_allowed = lambda: granted[0]
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.store.turn("s:1")["mode"] == "explore"
    granted[0] = False  # the consent goes before the first request; the write of that fact fails
    update = rs.store.update_turn
    failed: list[str] = []

    def held(turn_key: str, **fields: Any) -> None:
        if fields.get("mode") == "observe" and not failed:
            failed.append(turn_key)
            raise sqlite3.OperationalError("database is locked")
        update(turn_key, **fields)

    rs.store.update_turn = held  # type: ignore[method-assign]
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None  # goes out unchanged
    assert failed == ["s:1"] and rs.store.turn("s:1")["mode"] == "explore"  # the row is still wrong here
    rs.record_attempt("s", ok=True, model="big-m", finish="stop")
    rs.end_turn("s", "21")
    row = rs.store.turn("s:1")
    assert (row["mode"], row["model"]) == ("observe", "big-m")  # written again at the end of the turn
    rs.feedback(True, "s")
    assert all(r["rung"] != "small-m" for r in rs.store.scored_turns(100))


def test_a_call_that_cannot_be_recorded_still_returns_the_turn_to_the_users_own_model(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")[0]["model"] == "small-m"
    add = rs.store.add_attempt

    def held(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    rs.store.add_attempt = held  # type: ignore[method-assign]
    rs.record_attempt("s", ok=False, model="small-m", error="Timeout", retryable=True, reason="timeout")
    rs.store.add_attempt = add  # type: ignore[method-assign]
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None  # back on the own model
    rs.record_attempt("s", ok=True, model="big-m", finish="stop")
    rs.end_turn("s", "21")
    assert rs.store.turn("s:1")["escalated"] == 1


def test_the_end_of_a_session_still_audits_when_the_clean_up_fails(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)

    def held(*args: Any, **kwargs: Any) -> int:
        raise sqlite3.OperationalError("database is locked")

    rs.expire_text = held  # type: ignore[method-assign]
    assert "audit" in rs.end_session("s")


def test_no_provider_message_text_is_stored(plugin: Any) -> None:
    secret = "Invalid request: the prompt 'my bank code is 4471' is too long for account acct_99"
    ctx = FakeCtx(small_model="small-m", explore_share=1.0)
    rt = plugin.register(ctx)
    request = {"model": "big-m", "messages": []}
    for n, error in enumerate((RuntimeError(secret), secret, {"type": "BadRequestError", "message": secret})):
        sid = f"s{n}"
        ctx.fire("pre_llm_call", session_id=sid, user_message=f"Name {n + 3} colours.", model="big-m", platform="cli")
        assert ctx.middleware["llm_request"](request=request, session_id=sid, model="big-m")["request"]["model"] == "small-m"
        ctx.fire("api_request_error", session_id=sid, model="big-m", error=error, retryable=False, status_code=404,
                 reason="model_not_found")
        blocked = json.dumps(rt.sizer.store.get_policy("blocked_rungs", {}))
        assert "small-m" in blocked and "bank" not in blocked and "acct_99" not in blocked
        if n < 2:
            assert "small-m" in ctx.commands["continuum"]("unblock", platform="cli")
    store = rt.sizer.store
    kinds = [a["error"] for n in range(3) for a in store.attempts(f"s{n}:1")]
    assert kinds == ["RuntimeError model_not_found 404", "error model_not_found 404",
                     "BadRequestError model_not_found 404"]
    dump = json.dumps([store.get_policy("blocked_rungs", {}), rt.sizer.status_text(), kinds])
    assert "bank" not in dump and "acct_99" not in dump and "small-m" in dump
    store.close()


def test_lab_tools_are_hidden_in_off_mode_and_lab_commands_are_local(plugin: Any) -> None:
    lab = lambda ctx: next(v for k, v in ctx.tools.items() if k != wiring.RECALL_TOOL)["check_fn"]  # noqa: E731
    on = FakeCtx(small_model="small-m", enable_lab_tools=True)
    plugin.register(on)
    assert lab(on)() is True
    assert "only for the local user" in on.commands["continuum"]("lab step", platform="discord")
    off = FakeCtx(small_model="small-m", enable_lab_tools=True, mode="off")
    plugin.register(off)
    assert lab(off)() is False
    hidden = FakeCtx(small_model="small-m")
    plugin.register(hidden)
    assert lab(hidden)() is False
    bad = FakeCtx(small_model="small-m", enable_lab_tools=True, compute_headroom_ratio="a lot")
    plugin.register(bad)  # a wrong value of a lab setting never stops the plugin from loading
    assert lab(bad)() is True


def test_settings_typed_in_other_ways(tmp_path: Path) -> None:
    assert make(tmp_path / "a", store_text="").flag("store_text") is False  # an empty value is not a yes
    rs = make(tmp_path / "b", size_tools=True)  # `size_tools: on` without quotes arrives as a yes/no
    for n in range(4):
        run_turn(rs, f"s{n}", f"What is {n} * 7?", reply="ok", good=True)
    tools = [{"type": "function", "function": {"name": name}} for name in ("alpha", "beta", "gamma")]
    rs.start_turn("t", "What is 9 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, "t", "big-m")
    assert rs.store.turn("t:1")["tools_total"] == 3  # the tool list was looked at: the setting was read as "on"


def test_a_check_turn_is_not_checked_in_observe_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum_core.rightsize import chooser

    def always_check(fp: Any, text: str, rungs: Any, *args: Any, **kwargs: Any) -> Any:
        return chooser.Decision(rungs[-1].name, rungs[-1].model, "reference-check", "small-m qualifies")

    monkeypatch.setattr(chooser, "choose", always_check)
    auto = make(tmp_path / "auto", check_share=1.0)
    assert auto.start_turn("s", "What is 3 * 7?", model="big-m").mode == "reference-check"
    auto.record_attempt("s", ok=True, model="big-m", finish="stop")
    assert auto.end_turn("s", "21")["check"] is True
    for name, rs in (("observe", make(tmp_path / "observe", check_share=1.0, mode="observe")),
                     ("no grant", make(tmp_path / "nogrant", check_share=1.0))):
        if name == "no grant":
            rs.switch_allowed = lambda: False
        dec = rs.start_turn("s", "What is 3 * 7?", model="big-m")
        assert dec.mode == "observe" and "would have chosen" in dec.reason, name
        rs.record_attempt("s", ok=True, model="big-m", finish="stop")
        assert rs.end_turn("s", "21")["check"] is False, name


def test_the_dashboard_counts_a_mixed_turn_for_no_model(tmp_path: Path) -> None:
    from continuum_core.rightsize.store import read_snapshot

    rs = make(tmp_path, explore_share=1.0)
    run_turn(rs, "a", "What is 3 * 7?", reply="21", good=True)
    rs.store.update_turn("a:1", mode="mixed")
    snap = read_snapshot(tmp_path / "rightsize.db")
    assert snap["by_model"] == [] and snap["turns"] == 1


def test_the_terminal_command_runs_the_same_commands(plugin: Any, capsys: pytest.CaptureFixture[str]) -> None:
    class Ctx(FakeCtx):
        def register_cli_command(self, name: str, handler_fn: Any, setup_fn: Any = None, **_: Any) -> None:
            self.cli = (name, handler_fn, setup_fn)

    ctx = Ctx(small_model="small-m")
    plugin.register(ctx)
    name, handler, setup = ctx.cli
    import argparse

    parser = argparse.ArgumentParser()
    setup(parser)
    assert name == "continuum" and handler(parser.parse_args(["forget-text"])) == 0
    assert "Removed the stored Task and answer text" in capsys.readouterr().out  # a whole-file command, as cli
    assert handler(parser.parse_args(["pin"])) == 0
    assert "belong to one session" in capsys.readouterr().out
    assert handler(parser.parse_args([])) == 0
    assert "Continuum" in capsys.readouterr().out


# ---- the fifth review: the rewrite at start, a failed end-of-turn write, what a chat user sees, the lab ----
def test_a_large_record_file_is_not_rewritten_at_the_start_of_hermes(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "old", "What is 3 * 7?", reply="21", good=True)
    rs.clock.t += 91 * 86400.0
    rs.start_turn("new", "What is 4 * 7?", model="big-m")
    assert rs.store.turn("old:1")["text"] is None and rs.store.compact_pending()
    rs.store.close()
    from continuum_core.rightsize.engine import load_config

    cfg = load_config()
    cfg["settings"]["rewrite_at_start_max_mib"] = 0.000001  # the file is larger than the limit
    small_limit = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m"}, config=cfg, clock=Clock())
    assert small_limit.store.size_bytes() > 1 and small_limit.store.compact_pending()  # start stays short
    small_limit.store.close()
    again = make(tmp_path)  # the default limit: this small file is rewritten
    assert not again.store.compact_pending()
    again.store.close()


def test_the_rewrite_mark_is_written_with_the_removal(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)
    assert rs.store.forget_text(compact=False) == 1 and rs.store.compact_pending()
    assert rs.store.forget_text(compact=False) == 0 and rs.store.compact_pending()  # nothing new, the mark stays
    assert rs.store.forget_text() == 0 and not rs.store.compact_pending()  # the rewrite that was still owed
    assert not hasattr(rs.store, "compacted")


def test_the_rewrite_mark_does_not_need_a_second_write(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)

    def held(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    rs.store.set_policy = held  # type: ignore[method-assign]  # a write after the removal would fail
    assert rs.store.forget_text(compact=False) == 1
    assert rs.store.compact_pending()  # the mark went in with the removal itself


def test_the_rewrite_command_and_the_status_line(tmp_path: Path) -> None:
    from continuum_core.rightsize import hermes as hm

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)
    assert "needs no rewrite" in hm.command(rs, tmp_path, "compact", platform="cli")
    rs.store.forget_text(compact=False)
    assert "hermes continuum compact" in rs.status_text(detail=True)
    assert "hermes continuum compact" not in rs.status_text(detail=False)
    assert "runs only for the local user" in hm.command(rs, tmp_path, "compact", platform="discord")
    assert rs.store.compact_pending()
    assert "was rewritten" in hm.command(rs, tmp_path, "compact", platform="cli")
    assert not rs.store.compact_pending() and "hermes continuum compact" not in rs.status_text(detail=True)


def test_a_busy_file_is_tried_once_at_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum_core.rightsize.store import RecordStore

    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)
    rs.store.forget_text(compact=False)
    rs.store.close()
    tries: list[int] = []

    def busy(self: Any) -> bool:
        tries.append(1)
        return False

    monkeypatch.setattr(RecordStore, "compact", busy)
    again = make(tmp_path)
    assert tries == [1] and again.store.compact_pending()


def _hold_the_end_of_turn(rs: RightSizer, times: int) -> list[str]:
    import sqlite3

    update = rs.store.update_turn
    failed: list[str] = []

    def held(turn_key: str, **fields: Any) -> None:
        if fields.get("closed") == 2 and len(failed) < times:
            failed.append(turn_key)
            raise sqlite3.OperationalError("database is locked")
        update(turn_key, **fields)

    rs.store.update_turn = held  # type: ignore[method-assign]
    return failed


def test_a_failed_end_of_turn_write_does_not_blame_the_cheaper_model(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")[0]["model"] == "small-m"
    rs.record_attempt("s", ok=True, model="small-m", finish="stop")
    failed = _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "21")
    assert failed == ["s:1"] and rs.store.turn("s:1")["closed"] == 0  # the row looks like a turn with no reply
    rs.start_turn("s", "Write a haiku about rain.", model="big-m")  # the next message of the session
    row = rs.store.turn("s:1")
    assert row["reply"] == "21" and row["source"] == "moved_on"  # the end was written first; never "empty"
    assert failed == ["s:1"]


def test_a_failed_end_of_turn_write_on_the_own_model_is_written_later(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=0.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None  # the user's own model
    rs.record_attempt("s", ok=True, model="big-m", finish="stop", usage={"completion_tokens": 5})
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "21")
    rs.start_turn("s", "Write a haiku about rain.", model="big-m")
    row = rs.store.turn("s:1")
    assert (row["reply"], row["source"], row["model"]) == ("21", "moved_on", "big-m") and not rs._unwritten


def test_a_users_mark_is_kept_when_the_end_of_the_turn_is_written_late(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop")
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "22")
    assert rs.feedback(False, "s")["source"] == "explicit"  # /continuum bad
    rs.start_turn("s", "Write a haiku about rain.", model="big-m")  # the kept end is written now
    row = rs.store.turn("s:1")
    assert (row["closed"], row["source"], row["score"], row["reply"]) == (1, "explicit", 0.0, "22")
    assert not rs._unwritten


def test_a_turn_cut_short_does_not_close_a_turn_whose_end_is_kept(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop")
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "21")
    rs.drop_open("s", "interrupted")
    assert rs.store.turn("s:1")["closed"] == 0 and "s:1" in rs._unwritten
    rs.end_session("s")
    row = rs.store.turn("s:1")
    assert (row["closed"], row["source"], row["score"], row["reply"]) == (1, "unwritten", None, "21")


def test_a_failed_turn_never_becomes_a_pass_for_the_cheaper_model(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")[0]["model"] == "small-m"
    rs.record_attempt("s", ok=False, model="small-m", error="timeout")  # the cheaper call fails
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m") is None  # back on the own model
    rs.record_attempt("s", ok=True, model="big-m", finish="stop")
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "21")
    rs.start_turn("s", "thanks, that is great", model="big-m")
    row = rs.store.turn("s:1")
    assert row["source"] in ("api_error", "api_error_transient", "escalated") and row["source"] != "thanks"
    assert row["escalated"] == 1 and row["score"] < 1.0


def test_an_empty_reply_keeps_its_signal_when_the_write_fails(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="length")
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "The answer is")
    rs.start_turn("s", "thanks", model="big-m")
    assert rs.store.turn("s:1")["source"] == "cut_off"


def test_kept_ends_cost_one_write_while_the_file_is_held(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=0.0)
    update = rs.store.update_turn
    held = {"on": False, "tries": 0}

    def maybe(turn_key: str, **fields: Any) -> None:
        if held["on"]:
            held["tries"] += 1
            raise sqlite3.OperationalError("database is locked")
        update(turn_key, **fields)

    rs.store.update_turn = maybe  # type: ignore[method-assign]
    for i in range(5):
        rs.start_turn(f"s{i}", f"What is {i + 3} * 7?", model="big-m")
        rs.record_attempt(f"s{i}", ok=True, model="big-m", finish="stop")
        held["on"] = True  # the file stays held from here on, also at the next message
        with pytest.raises(sqlite3.Error):
            rs.end_turn(f"s{i}", "ok")
    assert len(rs._unwritten) == 5
    held["on"], held["tries"] = True, 0
    rs._write_kept()
    assert held["tries"] == 1 and len(rs._unwritten) == 5  # one write that failed, not five
    held["on"] = False
    rs._write_kept()
    assert not rs._unwritten and all(rs.store.turn(f"s{i}:1")["closed"] == 2 for i in range(5))


def test_a_turn_whose_end_was_lost_with_the_process_gives_no_outcome(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop", usage={"completion_tokens": 9})
    rs.start_turn("t", "What is 4 * 7?", model="big-m")  # a turn with no model output at all
    rs.store.close()  # Hermes stops before either end is written
    again = make(tmp_path, explore_share=1.0)
    again.start_turn("s", "Write a haiku about rain.", model="big-m")
    again.end_session("t")
    answered, silent = again.store.turn("s:1"), again.store.turn("t:1")
    assert (answered["closed"], answered["source"], answered["score"]) == (1, "unwritten", None)
    assert (silent["closed"], silent["source"]) == (1, "empty")  # no model output: no reply reached the user


def test_an_end_of_turn_that_can_never_be_written_gives_no_outcome(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop")
    _hold_the_end_of_turn(rs, 99)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "21")
    rs.end_session("s")
    row = rs.store.turn("s:1")
    assert (row["closed"], row["source"], row["score"]) == (1, "unwritten", None)
    assert not rs.store.scored_turns(100) and not rs._unwritten  # it says nothing for or against a model


def test_a_chat_user_sees_counts_and_no_provider_detail(plugin: Any) -> None:
    import re as _re

    ctx, rt = _wired(plugin)
    ctx.fire("pre_llm_call", session_id="home", user_message="What is 2 * 7?", model="big-m", platform="cli")
    out = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="home", model="big-m",
                                        provider="prov-a", base_url="https://name:sk-key" + "@" + "a.example/v1/?api_key=sk-key")
    ctx.fire("post_api_request", session_id="home", model="big-m", response_model=out["request"]["model"], usage={})
    home = rt.sizer.store.get_policy("rung_home", {})
    # scheme, host and port stay; the path, user name, password and query are kept only as a one-way mark
    assert _re.fullmatch(r"https://a\.example#p=[0-9a-f]{8}", home["base_url"]) and "sk-key" not in json.dumps(home)
    assert "prov-a" in ctx.commands["continuum"]("status", platform="cli")  # the local user sees the provider
    assert "prov-a" not in ctx.commands["continuum"]("status", platform="discord")
    for i in range(3):
        _hermes_refusal(ctx, f"s{i}", f"What is {i + 3} * 7?", "server said: no <b>go</b> https://x.example/?k=1")
    block = rt.sizer.blocked()["small-m"]
    assert block["reason"] == "other"  # a sentence is not a reason word: none of the provider's text is kept
    assert "sk-key" not in json.dumps(rt.sizer.store.get_policy("rung_home", {}))
    local = ctx.commands["continuum"]("status", platform="cli")
    chat = ctx.commands["continuum"]("status", platform="discord")
    assert "Blocked: small-m (the provider refused it: " + block["reason"] in local
    assert "Blocked: small-m." in chat and block["reason"] not in chat and "prov-a" not in chat
    assert "Records: 4 turns" in chat  # the counts stay
    assert ctx.commands["continuum"]("status") == chat  # no platform name: treated as a chat platform


def test_the_rung_settings_are_stored_as_a_hash(plugin: Any) -> None:
    ctx, rt = _wired(plugin)
    ctx.fire("pre_llm_call", session_id="a", user_message="What is 3 * 7?", model="big-m", platform="cli")
    out = ctx.middleware["llm_request"](request={"model": "big-m", "messages": []}, session_id="a", model="big-m",
                                        provider="prov-a", base_url="https://a.example/v1")
    ctx.fire("post_api_request", session_id="a", model="big-m", response_model=out["request"]["model"], usage={})
    home = rt.sizer.store.get_policy("rung_home", {})
    assert home["provider"] == "prov-a" and len(home["settings"]) == 16
    assert int(home["settings"], 16) >= 0 and "small-m" not in json.dumps(home["settings"])


def test_a_lab_tool_that_is_called_while_the_lab_is_off_runs_nothing(plugin: Any, tmp_path: Path) -> None:
    ctx = FakeCtx(small_model="small-m")  # enable_lab_tools is not set
    plugin.register(ctx)
    lab = [name for name in ctx.tools if name != "continuum_recall"]
    assert len(lab) == 18 and all(ctx.tools[name]["check_fn"]() is False for name in lab)
    before = sorted(p.name for p in tmp_path.iterdir())
    for name in lab:  # the host can still call a tool that it listed a moment ago
        answer = json.loads(ctx.tools[name]["handler"]({"goal": "x", "query": "x", "text": "x"}))
        assert "are off" in answer["error"], name
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # and nothing was stored


def test_forget_text_names_the_files_it_does_not_touch(tmp_path: Path) -> None:
    rs = make(tmp_path)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", good=True)
    assert "Not touched" not in wiring.command(rs, tmp_path, "forget-text", platform="cli")
    (tmp_path / "rightsize.unreadable-1700000000.db").write_bytes(b"not a record file")
    (tmp_path / "continuum.db").write_bytes(b"")
    text = wiring.command(rs, tmp_path, "forget-text", platform="cli")
    assert "Not touched by this command" in text
    assert "rightsize.unreadable-1700000000.db" in text and "continuum.db" in text



# ---------------- seventh review ----------------
def test_a_provider_address_keeps_host_and_port_and_only_a_mark_of_the_rest() -> None:
    from continuum_core.rightsize.engine import _clean_address as clean

    key = "sk-" + "abc123"
    with_key = clean("https://proxy.example:8443/" + key + "/v1")
    assert with_key.startswith("https://proxy.example:8443#p=") and key not in with_key  # a key in the path is not kept
    assert clean(with_key) == with_key  # cleaning a stored value changes nothing, so old and new values compare
    assert clean("https://proxy.example:8443/other/v1") != with_key  # another path is another address
    assert clean("HTTP://LocalHost:11434/") == "http://localhost:11434"
    assert clean("http://[::1]:8080/v1").startswith("http://[::1]:8080#p=")
    assert clean("https://u:pw" + "@" + "a.example/v1?k=1") == clean("https://a.example/v1/?k=2")  # dropped, not marked
    one, two = clean("localhost:11434/v1"), clean("otherhost:11434/v1")  # no scheme: only a mark is kept
    assert one.startswith("unreadable#p=") and one != two and clean(one) == one and "localhost" not in one
    assert clean("") == "" and clean(None) == ""


def test_a_turn_that_fell_back_keeps_its_failure_when_its_end_was_never_written(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")[0]["model"] == "small-m"
    rs.record_attempt("s", ok=True, model="small-m", finish="tool_calls", usage={"completion_tokens": 9})
    rs.record_attempt("s", ok=False, model="small-m", error="timeout")  # the turn falls back to the own model
    assert rs.store.turn("s:1")["escalated"] == 1
    again = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m"}, clock=Clock())  # a restart
    again.start_turn("s", "Write a haiku about rain.", model="big-m", platform="cli")
    row = again.store.turn("s:1")
    assert (row["source"], row["score"]) == ("escalated", 0.0)  # not "no outcome": the cheaper model did fail


def test_a_lost_end_with_a_call_that_worked_and_no_token_counts_blames_no_model(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop", usage={})  # the provider gave no counts
    again = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m"}, clock=Clock())
    again.start_turn("s", "Write a haiku about rain.", model="big-m", platform="cli")
    row = again.store.turn("s:1")
    assert (row["closed"], row["source"], row["score"]) == (1, "unwritten", None)


def test_tool_errors_keep_their_signal_when_the_end_of_turn_write_fails(tmp_path: Path) -> None:
    import sqlite3

    rs = make(tmp_path, explore_share=1.0)
    rs.start_turn("s", "List the files in the project folder.", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    for i in range(4):
        rs.record_tool("s", "terminal", "error: no such file" if i < 3 else "ok")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop", usage={"completion_tokens": 9})
    _hold_the_end_of_turn(rs, 1)
    with pytest.raises(sqlite3.Error):
        rs.end_turn("s", "Here are the files.")
    rs.start_turn("s", "thanks", model="big-m")  # the kept end is written now; "thanks" must not make it a pass
    row = rs.store.turn("s:1")
    assert (row["source"], row["score"]) == ("tool_errors", 0.0)


def test_a_good_mark_on_a_turn_that_fell_back_is_no_pass_for_the_cheaper_model(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    run_turn(rs, "s", "What is 3 * 7?", reply="21", fail_small=True)  # the own model gave the answer
    row = rs.feedback(True, "s")  # /continuum good
    assert row["escalated"] == 1 and (row["source"], row["score"]) != ("explicit", 1.0) and row["score"] == 0.0


def test_compact_names_the_reason_when_no_record_file_is_in_use(plugin: Any) -> None:
    ctx, rt = _wired(plugin)
    rt.sizer.store_note = "the record file is held by another process"
    real = rt.sizer.store.in_memory
    rt.sizer.store.in_memory = True  # as when the file could not be opened
    try:
        out = ctx.commands["continuum"]("compact", platform="cli")
    finally:
        rt.sizer.store.in_memory = real
    assert "held by another process" in out


# ---------------- trial turns that follow the evidence, and the cost of tuning ----------------
def test_trial_share_follows_checked_outcomes_near_the_task() -> None:
    fp, text = fingerprint("What is 99 * 7?", FP), "What is 99 * 7?"
    on = {**CFG["chooser"], "promising_boost": 5.0, "promising_max_share": 0.5}
    off = {**CFG["chooser"], "promising_boost": 1.0}
    passed = Pool(_records(2, "small-m", 1.0), FP, BALANCED)   # two checked passes: promising, not yet enough
    failed = Pool(_records(2, "small-m", 0.0), FP, BALANCED)   # two checked failures
    ask = lambda pool, cfg, u: choose(fp, text, RUNGS, pool, cfg, u=u, **STRICT)  # noqa: E731
    # With the plain share of 0.1, a turn at u = 0.3 stays on the user's own model.
    assert ask(passed, off, 0.3).mode == "reference"
    # Checked passes near the Task raise the share (0.1 * 5 = 0.5): the same turn is now a trial turn.
    dec = ask(passed, on, 0.3)
    assert dec.mode == "explore" and dec.model == "small-m" and "passed on similar Tasks" in dec.reason
    # The share never goes above the cap.
    assert ask(passed, {**on, "promising_boost": 50.0}, 0.6).mode == "reference"
    # Checked failures near the Task lower it (0.1 / 5 = 0.02): a turn that was a trial turn is not one any more.
    assert ask(failed, off, 0.05).mode == "explore"
    assert ask(failed, on, 0.05).mode == "reference"
    assert ask(failed, on, 0.01).mode == "explore"   # a rare trial turn stays, so a model that got better is seen
    # No record near the Task: the plain share, with or without the boost.
    empty = Pool([], FP, BALANCED)
    assert ask(empty, on, 0.05).mode == "explore" and ask(empty, on, 0.3).mode == "reference"


def test_trial_share_setting_is_off_in_the_shipped_config_unless_named() -> None:
    assert float(CFG["chooser"]["promising_boost"]) >= 1.0
    assert 0.0 < float(CFG["chooser"]["promising_max_share"]) <= 0.5


def test_tuning_pays_for_each_pair_once_and_gives_the_same_scores() -> None:
    from continuum_core.rightsize import chooser as chm
    rng = random.Random(11)
    fit = []
    stems = [("What is {} * 7?", 1.0), ("Refactor module number {} into services and explain the design.", 0.0)]
    for i in range(60):
        stem, score = stems[i % 2]
        rec = _records(1, "small-m", score, stem.replace("{}", str(rng.randrange(10_000))))[0]
        rec["turn_key"] = f"t{i}"
        fit.append(rec)
    for name, weights in FP["presets"].items():
        pool = Pool(fit, FP, weights)
        slow = chm._brier(pool.records, pool, CFG["chooser"], True)
        fast = chm._brier(pool.records, pool, CFG["chooser"], True, chm._pair_parts(pool.records, pool))
        assert slow == pytest.approx(fast), name
    calls = {"n": 0}
    real = fpm.level_similarities

    def counted(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real(*a, **k)

    fpm.level_similarities = counted  # type: ignore[assignment]
    try:
        rep = tune_presets(fit, [], FP, CFG["chooser"], FP["default_preset"], 0.01, max_queries=20)
    finally:
        fpm.level_similarities = real  # type: ignore[assignment]
    assert rep["scored_on"] == 20 and rep["pool"] == 60
    assert calls["n"] == 20 * 60   # one pass for all presets, on a bounded sample of the scored Tasks


def test_session_end_tunes_only_after_new_scored_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from continuum_core.rightsize import chooser as chm
    rs = make(tmp_path)
    runs = {"n": 0}
    real = chm.tune_presets

    def counted(*a: Any, **k: Any) -> Any:
        runs["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(chm, "tune_presets", counted)
    every, least = int(CFG["tune"]["every"]), int(CFG["tune"]["min_records"])
    for i in range(least + 3 * every):
        run_turn(rs, f"s{i}", f"What is {i + 2} * 7?", good=True)
        rs.end_session(f"s{i}")
    # One run per `every` new scored records at most, never one per session.
    assert 1 <= runs["n"] <= (least + 3 * every) // every + 1
    before = runs["n"]
    out = rs.end_session("nothing-new")
    assert runs["n"] == before and "waits for" in out["tune"]["skipped"]
    assert "fit_error" in rs.tune_now() and runs["n"] == before + 1   # the command runs it at once


# ---------------- eighth review and state-machine guards ----------------
def test_clean_address_lowercases_host_without_trailing_slash() -> None:
    from continuum_core.rightsize.engine import _clean_address as clean

    assert clean("http://LocalHost:11434") == "http://localhost:11434"
    assert clean("HTTPS://API.EXAMPLE.COM:8443") == "https://api.example.com:8443"


def test_feedback_and_why_respect_escalated_and_unscored_closed_turns(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    # 1. Escalated turn with transient error: /continuum good keeps the failure and says so; /continuum bad upgrades to explicit 1.0
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=False, model="small-m", error="RateLimit", retryable=True, status=429, reason="rate_limit")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="big-m", finish="stop", usage={"completion_tokens": 5})
    rs.end_turn("s", "21")
    assert rs.store.turn("s:1")["source"] == "api_error_transient" and rs.store.turn("s:1")["weight"] < 1.0
    good_msg = wiring.command(rs, tmp_path, "good", session_id="s", platform="cli")
    assert "keeps its failure record" in good_msg and rs.store.turn("s:1")["score"] == 0.0
    bad_msg = wiring.command(rs, tmp_path, "bad", session_id="s", platform="cli")
    assert "marked bad" in bad_msg and rs.store.turn("s:1")["source"] == "explicit" and rs.store.turn("s:1")["weight"] == 1.0
    # 2. Non-retryable provider refusal (host_error): stays unscored on /continuum good or bad, and why_text says not counted
    _refuse(rs, "r", "What is 4 * 7?", error="RateLimit", retryable=False, status=429, reason="rate_limit")
    assert rs.store.turn("r:1")["source"] == "host_error" and rs.store.turn("r:1")["score"] is None
    assert "not counted" in rs.why_text("r") and "open (" not in rs.why_text("r")
    ref_msg = wiring.command(rs, tmp_path, "bad", session_id="r", platform="cli")
    assert "not scored" in ref_msg and rs.store.turn("r:1")["score"] is None


def test_chat_status_and_unblock_messages_match_record_and_provider_state(tmp_path: Path) -> None:
    (tmp_path / "rightsize.db").write_bytes(b"not a database\n" * 100)
    rs = make(tmp_path, explore_share=1.0)
    assert "started a new record file" in rs.status_text(detail=False)
    assert "not using its record file" not in rs.status_text(detail=False)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m", provider="prov-a", base_url="https://a.example/v1")
    rs.record_attempt("s", ok=True, model="small-m", finish="stop")
    rs.end_turn("s", "21")
    assert rs.store.get_policy("rung_home", {}).get("provider") == "prov-a"
    msg = wiring.command(rs, tmp_path, "unblock", platform="cli")
    assert "cleared the remembered provider" in msg and rs.store.get_policy("rung_home", {}) == {}
    assert wiring.command(rs, tmp_path, "unblock", platform="cli") == "No cheaper model is blocked."


def test_forget_text_strips_pending_unwritten_reply_and_tool_miss_survives_session_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    rs = make(tmp_path, store_text=True, size_tools="on", explore_share=0.0)
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}}
             for n in ("terminal", "read_file", "web_search", "browser", "image_gen", "calendar")]
    for i in range(8):
        rs.start_turn(f"seed{i}", f"Search the web for the price of item {i + 100} and report it.", model="big-m")
        rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, f"seed{i}", "big-m")
        rs.record_tool(f"seed{i}", "web_search", "{}")
        rs.record_attempt(f"seed{i}", ok=True, model="big-m")
        rs.end_turn(f"seed{i}", "done")
        rs.feedback(True, f"seed{i}", source="check")

    rs.start_turn("s", "Search the web for the price of item 977 and report it.", model="big-m")
    shaped = rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, "s", "big-m")
    assert shaped is not None and len(shaped[0]["tools"]) < len(tools)
    rs.record_attempt("s", ok=True, model="big-m", finish="stop")

    # Simulate SQLite busy during end_turn while reply says a tool was missing
    real_update = rs.store.update_turn

    def busy_once(*a: Any, **k: Any) -> Any:
        monkeypatch.setattr(rs.store, "update_turn", real_update)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rs.store, "update_turn", busy_once)
    with pytest.raises(sqlite3.OperationalError):
        rs.end_turn("s", "I don't have a tool to open the calendar.")
    assert "reply" in rs._unwritten["s:1"]["fields"]

    # forget_text must strip the unwritten reply before _write_kept flushes it
    rs.forget_text()
    assert "reply" not in rs._unwritten["s:1"]["fields"]
    rs.end_session("s")
    assert rs.store.turn("s:1")["reply"] is None

    # Even after end_turn raised and end_session ran, the next turn in session "s" restores the full tool list
    rs.start_turn("s", "Search the web for the price of item 978 and report it.", model="big-m")
    assert rs.shape_request({"model": "big-m", "messages": [], "tools": tools}, "s", "big-m") is None




# ---------------- a cheaper model moves only on marked outcomes ----------------
# In two benchmark runs the user's own model, asked PASS or FAIL about the cheaper model's answers, passed 224 of
# 270 wrong answers and failed 5 of 208 right ones. With those PASSes counted as strong evidence, Continuum moved
# most Tasks to the cheaper model and lost about 21 points. These tests hold the rules that stop that.

def shipped(**chooser: Any) -> dict[str, Any]:
    """The config as it ships (the module copy was read before the test fixture turned the marks rule off)."""
    cfg = json.loads(json.dumps(CFG))
    cfg["chooser"].update(chooser)
    return cfg


def make_shipped(tmp_path: Path, config: dict[str, Any] | None = None, **settings: Any) -> RightSizer:
    return RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", **settings},
                      config=config or shipped(), clock=Clock())


def one_turn(rs: RightSizer, session: str, text: str, platform: str = "cli", sender_id: str = "",
             check: str | None = None, mark: bool | None = None) -> Any:
    dec = rs.start_turn(session, text, model="big-m", platform=platform, sender_id=sender_id)
    shaped = rs.shape_request({"model": "big-m", "messages": [{"role": "user", "content": text}]}, session, "big-m")
    sent = shaped[0]["model"] if shaped else "big-m"
    rs.record_attempt(session, ok=True, model=sent, usage={"prompt_tokens": 100, "completion_tokens": 20},
                      finish="stop")
    ended = rs.end_turn(session, "42")
    if check and ended and ended.get("check"):
        rs.check_result(ended["turn_key"], check)
    if mark is not None:
        rs.feedback(mark, session)
    rs.end_session(session)
    return dec, sent


def test_the_shipped_config_keeps_the_marks_rule_on() -> None:
    assert CFG["chooser"]["require_checked"] is True
    assert CFG["chooser"]["explore_needs_marks"] > 0
    assert CFG["outcomes"]["reference_check"]["pass_weight"] < CFG["chooser"]["strong_weight"]
    assert CFG["outcomes"]["reference_check"]["weight"] >= CFG["chooser"]["strong_weight"]


def test_no_trial_turn_before_the_user_marks_one_of_their_turns(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, explore_share=1.0)
    for i in range(6):
        dec, sent = one_turn(rs, f"s{i}", f"What is {i + 3} plus {i + 4}?")
        assert dec.mode == "reference" and sent == "big-m"
        assert "wait until you mark" in dec.reason
    one_turn(rs, "s-marked", "What is 9 plus 10?", mark=True)  # the user marks one answer good
    dec, sent = one_turn(rs, "s-after", "What is 11 plus 12?")
    assert dec.mode == "explore" and sent == "small-m"


def test_a_mark_of_one_sender_starts_no_trials_for_another_on_a_shared_platform(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, explore_share=1.0)
    one_turn(rs, "a1", "What is 2 plus 2?", platform="telegram", sender_id="alice", mark=True)
    dec, sent = one_turn(rs, "b1", "What is 3 plus 3?", platform="telegram", sender_id="bob")
    assert dec.mode == "reference" and sent == "big-m"
    dec, sent = one_turn(rs, "a2", "What is 4 plus 4?", platform="telegram", sender_id="alice")
    assert dec.mode == "explore" and sent == "small-m"


def test_trials_stop_when_the_last_mark_is_older_than_the_window(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, config=shipped(explore_needs_marks=5), explore_share=1.0)
    one_turn(rs, "m", "What is 1 plus 1?", mark=True)
    modes = [one_turn(rs, f"t{i}", f"What is {i + 2} times {i + 5}?")[0].mode for i in range(8)]
    assert modes[0] == "explore" and modes[-1] == "reference"


def test_check_passes_alone_never_move_a_task_to_the_cheaper_model(tmp_path: Path) -> None:
    # Trials allowed without marks, every cheaper answer checked, and every check says PASS.
    rs = make_shipped(tmp_path, config=shipped(explore_needs_marks=0), explore_share=0.5, check_share=1.0)
    modes = [one_turn(rs, f"s{i}", f"What is {i + 2} * 7?", check="PASS")[0].mode for i in range(80)]
    assert "explore" in modes and "exploit" not in modes
    rows = [r for r in rs.store.scored_turns(500) if r["source"] == "reference_check"]
    assert rows and all(r["weight"] < CFG["chooser"]["strong_weight"] for r in rows if r["score"] >= 0.5)


def test_a_check_fail_is_strong_evidence_against_the_cheaper_model(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, config=shipped(explore_needs_marks=0), explore_share=0.5, check_share=1.0)
    for i in range(40):
        one_turn(rs, f"s{i}", f"What is {i + 2} * 7?", check="FAIL")
    rows = [r for r in rs.store.scored_turns(500) if r["source"] == "reference_check"]
    assert rows and all(r["score"] == 0.0 and r["weight"] >= CFG["chooser"]["strong_weight"] for r in rows)
    dec = rs.start_turn("next", "What is 99 * 7?", model="big-m", platform="cli")
    assert dec.mode == "reference"


def test_marked_good_answers_still_move_a_kind_of_task(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, explore_share=0.5)
    one_turn(rs, "first", "What is 1 * 7?", mark=True)
    modes = []
    for i in range(60):
        dec, sent = one_turn(rs, f"s{i}", f"What is {i + 2} * 7?", mark=True)
        modes.append(dec.mode)
    assert "exploit" in modes[20:]


# ---------------- fixes after the November 10 check of the eighth-review edits ----------------

def test_off_mode_leaves_an_unreadable_record_file_as_it_is(tmp_path: Path) -> None:
    (tmp_path / "rightsize.db").write_bytes(b"this is not a record file\n" * 200)
    rs = make(tmp_path, mode="off")
    assert not list(tmp_path.glob("rightsize.unreadable-*"))  # not renamed
    assert (tmp_path / "rightsize.db").read_bytes().startswith(b"this is not a record file")  # not replaced
    assert rs.store.in_memory and "left as it is" in rs.store_note


def test_forget_text_reaches_answers_another_process_writes_later(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    clock = Clock()  # one wall clock for both processes, as on a real machine
    gateway = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    gateway.start_turn("g", "Summarise the meeting notes from Monday.", model="big-m", platform="cli")
    gateway.record_attempt("g", ok=True, model="big-m", finish="stop")
    real_update = gateway.store.update_turn

    def busy_once(*a: Any, **k: Any) -> Any:
        monkeypatch.setattr(gateway.store, "update_turn", real_update)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(gateway.store, "update_turn", busy_once)
    with pytest.raises(sqlite3.OperationalError):
        gateway.end_turn("g", "The meeting agreed on three points.")
    assert "reply" in gateway._unwritten["g:1"]["fields"]  # the gateway still holds the answer in memory
    shell = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True},
                       clock=clock)  # `hermes continuum forget-text` in a second process
    shell.forget_text()
    gateway._write_kept()  # the gateway writes its kept fields later
    row = gateway.store.turn("g:1")
    assert row["reply"] is None and row["text"] is None


def test_the_tool_miss_flags_stay_bounded(tmp_path: Path) -> None:
    rs = make(tmp_path)
    rs._tool_miss.update(f"s{i}" for i in range(1000))  # full: the next flag clears the old ones first
    rs.start_turn("last", "Open my calendar for tomorrow.", model="big-m")
    rs.shape_request({"model": "big-m", "messages": [], "tools": [{"type": "function", "function": {"name": "terminal"}}]},
                     "last", "big-m")
    rs.record_attempt("last", ok=True, model="big-m", finish="stop")
    live = rs._live["last"]
    live["tools_keep"] = ["terminal"]
    rs.end_turn("last", "I don't have a tool to open the calendar.")
    assert len(rs._tool_miss) <= 1000 and "last" in rs._tool_miss


def test_the_tune_note_tells_no_held_out_records_from_not_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rs = make(tmp_path)
    base = {"chosen": "sketch-first", "current": "sketch-first", "fit_error": {"sketch-first": 0.1}, "held_out": {}}
    monkeypatch.setattr(rs, "tune_now", lambda due_only=False: {**base, "holdout_records": 0})
    assert "no held-out records yet" in (wiring.command(rs, tmp_path, "tune", platform="cli") or "")
    monkeypatch.setattr(rs, "tune_now", lambda due_only=False: {**base, "holdout_records": 12})
    text = wiring.command(rs, tmp_path, "tune", platform="cli") or ""
    assert "not needed" in text and "no held-out records yet" not in text



# ---------------- ninth review: thank-yous, comparison turns, a running turn and old rows ----------------

def test_thank_yous_after_trial_turns_never_move_a_task(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, explore_share=1.0)
    one_turn(rs, "m", "What is 1 * 7?", mark=True)  # one mark opens trial turns
    modes = []
    for i in range(30):  # every trial answer is followed by "thanks" in the same session, and never marked
        dec = rs.start_turn(f"s{i}", f"What is {i + 2} * 7?", model="big-m", platform="cli")
        modes.append(dec.mode)
        shaped = rs.shape_request({"model": "big-m", "messages": []}, f"s{i}", "big-m")
        rs.record_attempt(f"s{i}", ok=True, model=shaped[0]["model"] if shaped else "big-m", finish="stop")
        rs.end_turn(f"s{i}", "The answer is right here.")
        rs.start_turn(f"s{i}", "thanks, great", model="big-m", platform="cli")
        rs.end_session(f"s{i}")
    assert "explore" in modes and "exploit" not in modes
    thanks = [r for r in rs.store.scored_turns(500) if r["source"] == "thanks"]
    assert thanks and all(r["weight"] < CFG["chooser"]["strong_weight"] for r in thanks)


def test_comparison_turns_go_on_while_trials_wait_for_a_mark(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path, config=shipped(explore_needs_marks=10), explore_share=1.0)
    for i in range(60):  # marked turns: the cheaper model qualifies for this kind of Task
        one_turn(rs, f"m{i}", f"What is {i + 2} * 7?", mark=True)
    for i in range(12):  # then no mark for longer than the window
        one_turn(rs, f"q{i}", f"Name a colour of the rainbow number {i}.")
    assert not rs._marks_recently(("cli", owner_of("cli", "")))
    modes = [one_turn(rs, f"t{i}", f"What is {i + 70} * 7?")[0].mode for i in range(8)]
    # Half of the turns of a qualified kind still go to the user's own model for the comparison (explore_share / 2),
    # and no trial turn is made. Unmarked turns then crowd the marked ones out of the nearest records, so the kind
    # returns to the user's own model by itself.
    assert "exploit" in modes and "reference-check" in modes and "explore" not in modes


def test_forget_text_reaches_a_turn_that_is_still_running(tmp_path: Path) -> None:
    clock = Clock()
    gateway = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    gateway.start_turn("g", "Summarise the notes of the Monday meeting.", model="big-m", platform="cli")
    gateway.record_attempt("g", ok=True, model="big-m", finish="stop")
    shell = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    shell.forget_text()  # runs while the gateway's turn is still open
    gateway.end_turn("g", "The meeting agreed on three points.")
    assert gateway.store.turn("g:1")["reply"] is None


def test_check_passes_and_thank_yous_in_older_rows_count_as_weak(tmp_path: Path) -> None:
    rs = make_shipped(tmp_path)
    for i in range(30):  # rows as an older version wrote them: a PASS at 0.8, a thank-you at 0.6
        rs.store.open_turn({"turn_key": f"old{i}:1", "session_id": f"old{i}", "seq": 1, "ts": 1_600_000_000.0 + i,
                            "platform": "cli", "owner": owner_of("cli", ""), "exact": f"e{i}", "template": "t",
                            "contract": "c", "fp": json.dumps(fpm.fingerprint(f"What is {i + 2} * 7?", FP)),
                            "base_model": "big-m", "rung": "small-m", "model": "small-m", "mode": "explore",
                            "reason": "", "split": "fit"})
        rs.store.update_turn(f"old{i}:1", score=1.0, weight=0.8 if i % 2 else 0.6,
                             source="reference_check" if i % 2 else "thanks", closed=1)
    rs._dirty()
    assert not any(r["strong"] for r in rs._pool("any").records)
    assert rs.start_turn("new", "What is 99 * 7?", model="big-m", platform="cli").mode != "exploit"


# ---------------- tenth review ----------------

def test_a_failed_read_of_the_forget_mark_keeps_the_end_of_the_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    rs = make(tmp_path, store_text=True, explore_share=1.0)
    rs.start_turn("s", "What is 3 * 7?", model="big-m")
    rs.shape_request({"model": "big-m", "messages": []}, "s", "big-m")
    rs.record_attempt("s", ok=True, model="small-m", finish="length")
    real = rs.store.get_policy

    def broken(key: str, default: Any = None) -> Any:
        if key == "text_forgotten_at":
            raise sqlite3.OperationalError("database is locked")
        return real(key, default)

    monkeypatch.setattr(rs.store, "get_policy", broken)
    rs.end_turn("s", "The answer is")
    row = rs.store.turn("s:1")
    assert row["source"] == "cut_off" and row["score"] == 0.0  # the failure signal of the cheaper model is kept
    assert row["reply"] is None  # an unreadable mark keeps no answer
    assert row["text"] == "What is 3 * 7?"  # and the message stays: forget-text did not run


def test_the_trial_note_is_not_shown_in_observe_mode_or_without_the_grant(tmp_path: Path) -> None:
    observe = make_shipped(tmp_path / "a", mode="observe", explore_share=1.0)
    dec, _ = one_turn(observe, "s", "What is 3 + 4?")
    assert "wait until you mark" not in dec.reason
    auto = make_shipped(tmp_path / "b", explore_share=1.0)
    dec, _ = one_turn(auto, "s", "What is 3 + 4?")
    assert "wait until you mark" in dec.reason
    auto._may_switch = lambda: False  # type: ignore[method-assign]  # as without the llm.model_override grant
    dec, _ = one_turn(auto, "t", "What is 5 + 4?")
    assert "wait until you mark" not in dec.reason



# ---------------- eleventh review ----------------

def test_the_audit_target_is_not_lowered_by_failures_of_the_own_model_alone() -> None:
    defaults = {"floor": 0.7, "min_evidence": 3.0}
    cheap = [{"rung": "small-m", "mode": "exploit", "score": 0.0, "weight": 0.8, "split": "holdout",
              "source": "explicit"}] * 5
    # the own model: 60 PASSes of an older build (0.8) and 5 FAILs, nobody marked anything
    own = ([{"rung": "reference", "mode": "reference", "score": 1.0, "weight": 0.8, "split": "holdout",
             "source": "reference_check"}] * 60
           + [{"rung": "reference", "mode": "reference", "score": 0.0, "weight": 0.8, "split": "holdout",
               "source": "reference_check"}] * 5)
    rep = audit(cheap + own, "reference", defaults, defaults, CFG["holdout"], 1.0)
    assert rep["views"]["all"]["reference_evidence"] == 0.0  # no marks on the own model: the default floor rules
    assert rep["action"] == "tighten"


def test_an_answer_written_just_before_forget_text_is_cleared_later(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    clock = Clock()
    gateway = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    gateway.start_turn("g", "Summarise the notes of the Monday meeting.", model="big-m", platform="cli")
    gateway.record_attempt("g", ok=True, model="big-m", finish="stop")
    shell = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    real = gateway.store.update_turn
    calls = {"n": 0}

    def racing(key: str, **fields: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:  # the end of the turn: forget-text runs in the shell just before the write lands
            shell.forget_text()
            real(key, **fields)
            return
        if calls["n"] == 2 and fields == {"reply": None}:  # the clearing write fails: the file is busy
            raise sqlite3.OperationalError("database is locked")
        real(key, **fields)

    monkeypatch.setattr(gateway.store, "update_turn", racing)
    with pytest.raises(sqlite3.OperationalError):
        gateway.end_turn("g", "The meeting agreed on three points.")
    assert gateway.store.turn("g:1")["reply"] == "The meeting agreed on three points."  # the race is real
    assert gateway._unwritten["g:1"]["fields"]["reply"] is None  # the kept end will clear it, not write it again
    gateway._write_kept()  # the next message of any session writes the kept end
    assert gateway.store.turn("g:1")["reply"] is None



def test_a_late_write_that_waits_behind_forget_text_writes_no_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    clock = Clock()
    gateway = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    gateway.start_turn("g", "Summarise the notes of the Monday meeting.", model="big-m", platform="cli")
    gateway.record_attempt("g", ok=True, model="big-m", finish="stop")
    real = gateway.store.update_turn

    def busy_once(*a: Any, **k: Any) -> Any:
        monkeypatch.setattr(gateway.store, "update_turn", real)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(gateway.store, "update_turn", busy_once)
    with pytest.raises(sqlite3.OperationalError):
        gateway.end_turn("g", "The meeting agreed on three points.")
    assert gateway._unwritten["g:1"]["fields"]["reply"]  # the answer waits to be written
    shell = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    calls = {"n": 0}

    def forget_first(key: str, **fields: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:  # the late write read the mark before forget-text, and lands after it
            shell.forget_text()
        real(key, **fields)

    monkeypatch.setattr(gateway.store, "get_policy", lambda key, default=None: 0 if key == "text_forgotten_at"
                        and calls["n"] == 0 else RecordStore.get_policy(gateway.store, key, default))
    monkeypatch.setattr(gateway.store, "update_turn", forget_first)
    gateway._write_kept()
    assert gateway.store.turn("g:1")["reply"] is None


def test_the_status_counts_a_saving_only_on_cheaper_turns_marked_good(tmp_path: Path) -> None:
    rs = make(tmp_path, explore_share=1.0)
    run_turn(rs, "a", "What is 3 * 7?", reply="21")
    rs.feedback(False, "a")  # the cheaper model's answer was wrong
    st = rs.status()
    assert st["small_turns"] == 1 and st["small_turns_marked_good"] == 0 and st["estimated_reference_tokens_saved"] == 0
    assert "estimated saving" not in rs.status_text()
    run_turn(rs, "b", "What is 4 * 7?", reply="28")
    rs.feedback(True, "b")
    st = rs.status()
    assert st["small_turns"] == 2 and st["small_turns_marked_good"] == 1 and st["estimated_reference_tokens_saved"] > 0
    assert "estimated saving on the turns marked good" in rs.status_text()



def test_forget_text_during_the_start_of_a_turn_clears_its_message_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    gateway = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    shell = RightSizer(tmp_path / "rightsize.db", settings={"small_model": "small-m", "store_text": True}, clock=clock)
    real_open = gateway.store.open_turn

    def forget_then_open(row: Any) -> None:  # the turn's insert waited behind forget-text in the shell
        shell.forget_text()
        real_open(row)

    monkeypatch.setattr(gateway.store, "open_turn", forget_then_open)
    gateway.start_turn("g", "Summarise the notes of the Monday meeting.", model="big-m", platform="cli")
    gateway.record_attempt("g", ok=True, model="big-m", finish="stop")
    gateway.end_turn("g", "The meeting agreed on three points.")
    row = gateway.store.turn("g:1")
    assert row["reply"] is None and row["text"] is None
