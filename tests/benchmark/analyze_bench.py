#!/usr/bin/env python3
"""Reads the records of one benchmark run (bench_hermes.py) and writes RESULTS.md and results.json beside them.

    analyze_bench.py --run NAME [--base A-own-model] [--price-ratio 0.2]

Everything in the report is counted from records.jsonl (answers, scored with the answer key), meter.jsonl (one
line per model call, written by the meter plugin in every arm) and, for the arms with Continuum, the plugin's own
record file. Intervals are 95% intervals from 2,000 resamples of the Tasks. No network, no model call.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if path.exists() else []


def interval(values: list[float], seed: int = 20261110, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def arm_summary(folder: Path, price_ratio: float) -> dict:
    spec = json.loads((folder / "spec.json").read_text())
    records = load(folder / "records.jsonl")
    meter = load(folder / "meter.jsonl")
    small, ref = spec["small_model"], spec["reference_model"]
    by_item: dict[str, list[dict]] = defaultdict(list)
    for m in meter:
        by_item[m["item"]].append(m)
    rows = []
    for r in records:
        calls = by_item.get(r["id"], [])
        turn_ok = [c for c in calls if c["kind"] == "turn" and c["ok"]]
        answered_by = turn_ok[-1]["model"] if turn_ok else ""
        tok = Counter()
        sec = Counter()
        for c in calls:
            u = c.get("usage") or {}
            tok[c["model"]] += int(u.get("prompt_tokens", u.get("input_tokens", 0))) + \
                int(u.get("completion_tokens", u.get("output_tokens", 0)))
            sec[c["model"]] += float(c.get("seconds") or 0.0)
        rows.append({"id": r["id"], "kind": r["kind"], "right": bool(r.get("right")), "error": bool(r.get("error")),
                     "small": answered_by == small, "answered_by": answered_by,
                     "tok_small": tok[small], "tok_ref": sum(v for k, v in tok.items() if k != small),
                     "sec": sum(sec.values()), "calls": len(calls),
                     "checks": sum(1 for c in calls if c["kind"] == "aux" and c.get("task") == "plugin_side_question"),
                     "largest_prompt": max([int((c.get("usage") or {}).get("prompt_tokens", 0)) for c in calls] or [0])})
    for row in rows:
        row["price"] = row["tok_ref"] + price_ratio * row["tok_small"]
    out = {"arm": folder.name, "about": spec["arms"][folder.name].get("about", ""), "rows": rows, "spec": spec}
    db = folder / "home" / "plugin-data" / "continuum" / "rightsize.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read only: an arm may still be running
        con.row_factory = sqlite3.Row
        turns = [dict(t) for t in con.execute(
            "SELECT session_id, text, mode, model, source, score, weight, escalated FROM rs_turns ORDER BY ts")]
        out["modes"] = Counter(t["mode"] for t in turns)
        out["sources"] = Counter(t["source"] or "open" for t in turns)
        gold = {r["id"]: bool(r.get("right")) for r in records}
        suffix = spec.get("prompt_suffix", "")
        by_text = {it["question"] + suffix: it["id"] for it in load(Path(spec["items"]))} if spec.get("items") else {}
        # (the check said PASS, the answer key says right), matched by the Hermes session of the Task (each Task
        # runs in its own session); by the Task text only for records that carry no session id
        by_session = {r["session_id"]: bool(r.get("right")) for r in records if r.get("session_id")}
        verdicts = []
        for t in turns:
            if t["source"] != "reference_check" or t["score"] is None:
                continue
            if t["session_id"] in by_session:
                verdicts.append((t["score"] >= 0.5, by_session[t["session_id"]]))
            elif by_text.get(t["text"] or "") in gold:
                verdicts.append((t["score"] >= 0.5, gold[by_text[t["text"]]]))
        out["verdicts"] = verdicts
        policy = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM rs_policy")}
        out["policy"] = policy
    return out


def stats(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {"n": len(rows), "right": sum(r["right"] for r in rows) / n, "small_share": sum(r["small"] for r in rows) / n,
            "sec": sum(r["sec"] for r in rows), "tok_ref": sum(r["tok_ref"] for r in rows),
            "tok_small": sum(r["tok_small"] for r in rows), "price": sum(r["price"] for r in rows),
            "checks": sum(r["checks"] for r in rows), "errors": sum(r["error"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="A-own-model")
    ap.add_argument("--price-ratio", type=float, default=0.2)
    ap.add_argument("--partial", action="store_true", help="also report arms that have not finished; every arm is then cut to the shortest")
    a = ap.parse_args()
    run = HERE / "runs" / a.run

    def is_arm(d: Path) -> bool:
        if not (d.is_dir() and (d / "records.jsonl").exists() and (d / "spec.json").exists()):
            return False
        return d.name in json.loads((d / "spec.json").read_text()).get("arms", {})  # kept-aside folders are not arms

    arms = {d.name: arm_summary(d, a.price_ratio) for d in sorted(run.iterdir()) if is_arm(d)}
    if not arms:
        print("no records")
        return 1
    longest = max(len(s["rows"]) for s in arms.values())
    unfinished = {n: len(s["rows"]) for n, s in arms.items() if len(s["rows"]) < longest}
    if unfinished and not a.partial:
        arms = {n: s for n, s in arms.items() if n not in unfinished}
    base = arms.get(a.base)
    n_all = min(len(s["rows"]) for s in arms.values())
    half = n_all // 2
    spec = next(iter(arms.values()))["spec"]
    L: list[str] = [f"# Continuum in Hermes: run `{a.run}`", ""]
    phases = spec.get("phases") or []
    tools = ", ".join(spec["toolsets"]) or "none (see the limits below)"
    first = next(iter(arms.values()))["rows"][:n_all]
    L += ["## Conditions", "",
          f"- Tasks: {n_all} {spec.get('tasks_are', 'public checkable Tasks')} in one fixed mixed order, each one a new Hermes session. Kinds: "
          + ", ".join(f"{k} ({v})" for k, v in sorted(Counter(r['kind'] for r in first).items())) + "."]
    L += [f"- {ph['title']} (Tasks {ph['from'] + 1} to {min(ph['to'], n_all)}): {ph['about']}" for ph in phases]
    L += [f"- Hermes: its own agent loop (`AIAgent`, as in Hermes's batch runner), Hermes's own system prompt, "
          f"toolsets: {tools}, at most {spec['max_iterations']} model calls a Task.",
          f"- Models: the user's own model is `{spec['reference_model']}`, the small model is `{spec['small_model']}`; both "
          f"are open-weight models on one 12 GB graphics card on one machine, with a real context window of "
          f"{spec.get('real_context')} tokens"
          + (f" and temperature {spec['temperature_in_model_alias']}" if spec.get("temperature_in_model_alias") is not None else "")
          + f". Largest prompt in the run: {max(r['largest_prompt'] for s in arms.values() for r in s['rows'])} tokens.",
          "- Every Task ends with the same request in every arm: " + json.dumps(spec.get("prompt_suffix", "").strip()) + ".",
          f"- Scoring: the last `Answer:` line of the reply against {spec.get('answer_key_is', 'the answer key of the public set')}.",
          "- Cost: seconds of model time on that card and tokens, from Hermes's own per-call hooks, the same in every arm. "
          "Checks by the user's own model are counted in the arm that makes them.",
          "- One pass, one order, one seed. Intervals: 95%, from resampling the Tasks.", ""]
    L += ["## Arms", ""] + [f"- **{name}**: {s['about']}" for name, s in arms.items()] + [""]
    if unfinished:
        L += [("Not finished and " + ("cut to the shortest arm" if a.partial else "left out") + ": "
               + ", ".join(f"{n} ({k} Tasks so far)" for n, k in unfinished.items()) + "."), ""]

    result: dict = {"run": a.run, "n": n_all, "arms": {}}
    sections = ([(ph["title"], int(ph["from"]), min(int(ph["to"]), n_all)) for ph in phases if int(ph["from"]) < n_all]
                or [("All Tasks", 0, n_all), (f"Second half (Tasks {half + 1} to {n_all})", half, n_all)])
    for title, lo, hi in sections:
        L += [f"## {title}", "",
              "| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |",
              "|---|---|---|---|---|---|---|"]
        for name, s in arms.items():
            st = stats(s["rows"][lo:hi])
            lo_i, hi_i = interval([float(r["right"]) for r in s["rows"][lo:hi]])
            L.append(f"| {name} | {pct(st['right'])} ({pct(lo_i)} to {pct(hi_i)}) | {pct(st['small_share'])} | "
                     f"{st['sec']:.0f} | {st['tok_ref']:,} | {st['tok_small']:,} | {st['checks']} |")
            result["arms"].setdefault(name, {})[title] = {**st, "right_interval": [lo_i, hi_i]}
        L.append("")
        if base:
            L += [f"Against **{a.base}** on the same Tasks:", "",
                  "| Arm | Right, difference in points | Model seconds | Tokens on the user's model | "
                  f"Price at {a.price_ratio:g} to 1 (an assumed price ratio) |", "|---|---|---|---|---|"]
            b_rows = {r["id"]: r for r in base["rows"][lo:hi]}
            b = stats(list(b_rows.values()))
            for name, s in arms.items():
                if name == a.base:
                    continue
                pairs = [(r, b_rows[r["id"]]) for r in s["rows"][lo:hi] if r["id"] in b_rows]
                diff = [float(x["right"]) - float(y["right"]) for x, y in pairs]
                d = sum(diff) / (len(diff) or 1)
                lo_i, hi_i = interval(diff)
                st = stats([x for x, _ in pairs])
                ratio = lambda key: f"{100 * st[key] / b[key]:.0f}% of {a.base}" if b[key] else "n/a"
                L.append(f"| {name} | {100 * d:+.1f} ({100 * lo_i:+.1f} to {100 * hi_i:+.1f}) | {ratio('sec')} | "
                         f"{ratio('tok_ref')} | {ratio('price')} |")
                result["arms"][name][title + " vs base"] = {
                    "right_diff": d, "interval": [lo_i, hi_i], "sec_ratio": st["sec"] / b["sec"] if b["sec"] else None,
                    "tok_ref_ratio": st["tok_ref"] / b["tok_ref"] if b["tok_ref"] else None,
                    "price_ratio": st["price"] / b["price"] if b["price"] else None}
            L.append("")

    k_title, k_lo, k_hi = sections[-1] if phases else ("all Tasks", 0, n_all)
    L += [f"## By kind of Task ({k_title})", "", "| Arm | " + " | ".join(sorted({r['kind'] for s in arms.values() for r in s['rows']})) + " |",
          "|---|" + "---|" * len({r['kind'] for s in arms.values() for r in s['rows']})]
    kinds = sorted({r["kind"] for s in arms.values() for r in s["rows"]})
    for name, s in arms.items():
        cells = []
        for k in kinds:
            rows = [r for r in s["rows"][k_lo:k_hi] if r["kind"] == k]
            st = stats(rows)
            cells.append(f"{pct(st['right'])}, small {pct(st['small_share'])}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    with_plugin = {n: s for n, s in arms.items() if "modes" in s}
    if with_plugin:
        L += ["## What Continuum did", ""]
        for name, s in with_plugin.items():
            L.append(f"- **{name}**: choices " + ", ".join(f"{k} {v}" for k, v in s["modes"].most_common())
                     + "; outcome sources " + ", ".join(f"{k} {v}" for k, v in s["sources"].most_common())
                     + "; strictness now " + ", ".join(f"{k} {v}" for k, v in sorted(s["policy"].items())
                                                         if k in ("floor", "min_evidence", "preset")) + ".")
            v = s.get("verdicts") or []
            if v:
                agree = sum(1 for said, was in v if said == was)
                missed = sum(1 for said, was in v if said and not was)
                wrong = sum(1 for _, was in v if not was)
                L.append(f"  - Checks by the user's own model: {len(v)}. They agreed with the answer key on {agree} "
                         f"({pct(agree / len(v))}). Of the {wrong} wrong answers checked, the check passed {missed}.")
                result["arms"][name]["checks"] = {"n": len(v), "agree": agree, "wrong_checked": wrong, "wrong_passed": missed}
        L.append("")
    errors = {n: sum(r["error"] for r in s["rows"]) for n, s in arms.items()}
    L += ["## Faults", "", "Tasks that ended with a harness error (counted as wrong): "
          + ", ".join(f"{n} {e}" for n, e in errors.items()) + ".", ""]
    snap = run / "plugin-snapshot.json"
    frozen = json.loads(snap.read_text()) if snap.exists() else {}
    L += ["## Limits", "",
          "- One pass in one order with one seed on one machine. A second pass can differ by a few points; read the intervals.",
          "- No tools. Hermes refuses tool use on a local endpoint with a window under 64,000 tokens, and this card "
          f"cannot hold both models at that size. The Hermes config declares {spec.get('declared_context')} tokens; "
          f"the models really serve {spec.get('real_context')}. No prompt in the run came near that.",
          "- Two small open-weight models (4-bit). The gap between a user's own model and a cheaper one is different "
          "on other models, and so is the share of Tasks that can move.",
          "- Every Task is a new session with one turn. Outcome signals that need a second message (a thank-you, a "
          "correction) do not occur here.",
          "- In the marked arms the harness marks every answer from the answer key. That stands for a user who marks "
          "every turn; few users do. The self-checked arm uses no answer key while it runs.",
          f"- The price column uses an assumed price ratio of {a.price_ratio:g} to 1 between the two models. The measured "
          "units are seconds and tokens.",
          "- Seconds are model seconds on a shared 12 GB card, where the two models take turns in memory. A model "
          "switch costs load time here that a hosted provider does not have."]
    if frozen:
        mark = str(frozen.get("sha256_of_paths_and_bytes") or "")[:8]
        L += [f"- Plugin code: a frozen copy taken {frozen.get('taken')} ({frozen.get('files')} files, mark "
              f"`{mark}`, {frozen.get('tests_at_this_state')}; full record in `results.json`). Later changes to "
              "the plugin are not in this run."]
    L.append("")
    (run / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    (run / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
