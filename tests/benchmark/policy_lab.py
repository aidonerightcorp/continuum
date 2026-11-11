#!/usr/bin/env python3
"""Policy lab: replay a finished benchmark run through Continuum's own engine, with no model call.

A benchmark run has two no-plugin arms: every Task answered by the user's own model (arm A) and every Task
answered by the small model (arm B). So for every Task the outcome of both models is on record. This script
feeds the same Tasks, in the same order, to the plugin's real engine (the same code that runs inside Hermes),
lets it choose a model per Task, and looks the outcome up in the arm of the model it chose. With `--feedback key`
every turn is marked from the answer key (as live arm C); with `--feedback check` nobody marks and the user's own
model's PASS or FAIL is drawn at the rates measured in the live D arms (P(PASS | right) 203/208, P(PASS | wrong)
224/270), independently of the kind of Task; with `--feedback none` there are no marks and no checks.

What a replay is good for: comparing settings of the chooser on many orders and seeds in seconds, before a
live run is spent on them. What it is not: a measured result of Hermes with the plugin. A live run draws a
new answer from the model, a replay reuses one recorded answer. Figures from this script are named "replay"
wherever they are shown, and a setting chosen here is confirmed on Tasks the replay never read.

Usage:
  policy_lab.py --run two-phase-v1 --items items/hermes-two-phase.jsonl --grid grid.json --seeds 8 --out lab/x.json
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent / "continuum"
REF, SMALL = "own-model", "small-model"


def load_arm(run: Path, arm: str) -> tuple[dict[str, bool], dict[str, tuple[int, int, float]]]:
    right = {}
    for line in open(run / arm / "records.jsonl", encoding="utf-8"):
        r = json.loads(line)
        right[r["id"]] = bool(r.get("right"))
    use: dict[str, list[float]] = {}
    for line in open(run / arm / "meter.jsonl", encoding="utf-8"):
        m = json.loads(line)
        if m.get("kind") != "turn" or not m.get("item"):
            continue
        u = m.get("usage") or {}
        acc = use.setdefault(m["item"], [0, 0, 0.0])
        acc[0] += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        acc[1] += int(u.get("output_tokens") or u.get("completion_tokens") or 0)
        acc[2] += float(m.get("seconds") or 0.0)
    return right, {k: (int(v[0]), int(v[1]), float(v[2])) for k, v in use.items()}


def replay(job: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, job["plugin"])
    from continuum_core.rightsize import RightSizer, load_config  # the plugin's own engine

    cfg = load_config()
    for section, values in (job.get("config") or {}).items():
        cfg.setdefault(section, {}).update(values)
    items, phases = job["items"], job["phases"]
    order = list(range(len(items)))
    if job.get("shuffle"):
        rnd = random.Random(job["seed"])
        order = []
        for lo, hi in phases:
            part = list(range(lo, hi))
            rnd.shuffle(part)
            order += part
    a_right, a_use, b_right, b_use = job["a_right"], job["a_use"], job["b_right"], job["b_use"]
    settings = {"small_model": SMALL, "check_share": 0.0, **(job.get("settings") or {})}
    stats = [dict(n=0, right=0, small=0, small_right=0, own_tok=0, small_tok=0, a_tok=0, a_right=0, b_right=0,
                  secs=0.0, a_secs=0.0, modes={}) for _ in phases]
    kinds: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        rs = RightSizer(Path(tmp) / "rightsize.db", settings=settings, config=cfg)
        for pos, idx in enumerate(order):
            it = items[idx]
            ph = next(i for i, (lo, hi) in enumerate(phases) if lo <= idx < hi)
            sess = f"lab-{job['seed']}-{pos}"
            dec = rs.start_turn(sess, it["text"], model=REF, platform="batch")
            shaped = rs.shape_request({"model": REF, "messages": [{"role": "user", "content": it["text"]}]}, sess, REF)
            sent = shaped[0]["model"] if shaped else REF
            on_small = sent == SMALL
            use = (b_use if on_small else a_use).get(it["id"], (0, 0, 0.0))
            ok = bool((b_right if on_small else a_right).get(it["id"], False))
            rs.record_attempt(sess, ok=True, model=sent, finish="stop",
                              usage={"prompt_tokens": use[0], "completion_tokens": use[1]})
            ended = rs.end_turn(sess, "recorded answer")
            rs.audit_if_new()
            if job.get("feedback", "key") == "key":  # a user who marks every turn (live arm C)
                rs.feedback(ok, sess, source="explicit")
            elif job.get("feedback") == "check" and ended and ended.get("check"):
                # Nobody marks; the user's own model is asked PASS or FAIL (live arm D). Its verdict is drawn
                # from the rates measured in the two live D arms: P(PASS | right), P(PASS | wrong).
                p_right, p_wrong = job.get("judge") or (0.976, 0.830)
                draw = random.Random(f"{job['seed']}:{it['id']}:judge").random()
                rs.check_result(ended["turn_key"], "PASS" if draw < (p_right if ok else p_wrong) else "FAIL")
            rs.end_session(sess)
            st = stats[ph]
            st["n"] += 1
            st["right"] += ok
            st["small"] += on_small
            st["small_right"] += ok and on_small
            st["small_tok" if on_small else "own_tok"] += use[0] + use[1]
            st["secs"] += use[2]
            au = a_use.get(it["id"], (0, 0, 0.0))
            st["a_tok"] += au[0] + au[1]
            st["a_secs"] += au[2]
            st["a_right"] += bool(a_right.get(it["id"], False))
            st["b_right"] += bool(b_right.get(it["id"], False))
            st["modes"][dec.mode] = st["modes"].get(dec.mode, 0) + 1
            if ph == len(phases) - 1:
                k = kinds.setdefault(it["kind"], dict(n=0, small=0, right=0, a_right=0, b_right=0))
                k["n"] += 1
                k["small"] += on_small
                k["right"] += ok
                k["a_right"] += bool(a_right.get(it["id"], False))
                k["b_right"] += bool(b_right.get(it["id"], False))
        strict = rs.strictness()
    return {"name": job["name"], "seed": job["seed"], "phases": stats, "kinds": kinds, "strictness": strict}


def summarise(rows: list[dict[str, Any]], ratio: float) -> dict[str, Any]:
    out: dict[str, Any] = {"seeds": len(rows), "phases": []}
    for ph in range(len(rows[0]["phases"])):
        acc, a_acc, share, own, price, secs, small_acc = [], [], [], [], [], [], []
        for r in rows:
            st = r["phases"][ph]
            n = max(1, st["n"])
            acc.append(st["right"] / n)
            a_acc.append(st["a_right"] / n)
            share.append(st["small"] / n)
            own.append(st["own_tok"] / max(1, st["a_tok"]))
            price.append((st["own_tok"] + ratio * st["small_tok"]) / max(1, st["a_tok"]))
            secs.append(st["secs"] / max(1e-9, st["a_secs"]))
            if st["small"]:
                small_acc.append(st["small_right"] / st["small"])
        diff = [x - y for x, y in zip(acc, a_acc)]
        mean = statistics.fmean
        out["phases"].append({
            "accuracy": round(mean(acc), 4), "own_model_accuracy": round(mean(a_acc), 4),
            "difference": round(mean(diff), 4), "difference_min": round(min(diff), 4),
            "difference_max": round(max(diff), 4), "small_share": round(mean(share), 4),
            "small_share_min": round(min(share), 4), "small_share_max": round(max(share), 4),
            "small_model_accuracy_on_its_turns": round(mean(small_acc), 4) if small_acc else None,
            "own_model_tokens_vs_A": round(mean(own), 4), "price_vs_A": round(mean(price), 4),
            "model_seconds_vs_A": round(mean(secs), 4)})
    kinds: dict[str, dict[str, float]] = {}
    for r in rows:
        for k, v in r["kinds"].items():
            acc = kinds.setdefault(k, dict(n=0, small=0, right=0, a_right=0, b_right=0))
            for f in acc:
                acc[f] += v[f]
    out["test_kinds"] = {k: {"small_share": round(v["small"] / max(1, v["n"]), 3),
                             "accuracy": round(v["right"] / max(1, v["n"]), 3),
                             "own_model": round(v["a_right"] / max(1, v["n"]), 3),
                             "small_model": round(v["b_right"] / max(1, v["n"]), 3)} for k, v in sorted(kinds.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--spec", default="bench_spec.json")
    ap.add_argument("--grid", required=True, help="JSON: {name: {settings: {...}, config: {section: {...}}}}")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--shuffle", action="store_true", help="shuffle the order inside each phase per seed")
    ap.add_argument("--plugin", default=str(PLUGIN))
    ap.add_argument("--price-ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feedback", choices=("key", "check", "none"), default="key",
                    help="key: every turn marked from the answer key (as live arm C); check: nobody marks, the own "
                         "model's PASS/FAIL drawn at the measured rates (as live arm D); none: no marks, no checks")
    ap.add_argument("--judge", default="0.976,0.830", help="P(PASS | right),P(PASS | wrong) for --feedback check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    spec = json.loads((HERE / a.spec).read_text())
    suffix = spec.get("prompt_suffix", "")
    items = []
    for line in open(HERE / a.items if not Path(a.items).is_absolute() else a.items, encoding="utf-8"):
        it = json.loads(line)
        items.append({"id": it["id"], "kind": it["kind"], "text": it["question"] + suffix})
    phases = [(int(p["from"]), int(p["to"])) for p in spec["phases"]]
    a_right, a_use = load_arm(run, "A-own-model")
    b_right, b_use = load_arm(run, "B-small-model")
    missing = [it["id"] for it in items if it["id"] not in a_right or it["id"] not in b_right]
    if missing:
        print(f"{len(missing)} Tasks have no record in arm A or B; the replay needs both arms complete.")
        return 2
    grid = json.loads(Path(a.grid).read_text())
    jobs = [dict(name=name, seed=seed, shuffle=a.shuffle, plugin=a.plugin, items=items, phases=phases,
                 a_right=a_right, a_use=a_use, b_right=b_right, b_use=b_use,
                 settings=g.get("settings"), config=copy.deepcopy(g.get("config")), feedback=a.feedback,
                 judge=[float(x) for x in a.judge.split(",")])
            for name, g in grid.items() for seed in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        rows = pool.map(replay, jobs, chunksize=1)
    result = {"what": "REPLAY of recorded answers through the plugin's engine; not a live run", "run": a.run,
              "items": a.items, "seeds": a.seeds, "shuffle": a.shuffle, "price_ratio": a.price_ratio,
              "plugin": a.plugin, "feedback": a.feedback, "judge": a.judge, "seconds": round(time.time() - t0, 1),
              "grid": grid, "settings": {}}
    for name in grid:
        result["settings"][name] = summarise([r for r in rows if r["name"] == name], a.price_ratio)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    last = len(phases) - 1
    print(f"replay of {a.run}: {len(jobs)} replays in {result['seconds']} s -> {out}")
    print(f"{'setting':34s} {'learn: small':>12s} {'diff':>7s} | {'test: small':>11s} {'acc':>6s} {'diff':>7s} "
          f"{'(min..max)':>15s} {'own tok':>8s} {'price':>6s}")
    for name, s in result["settings"].items():
        l, t = s["phases"][0], s["phases"][last]
        print(f"{name:34s} {l['small_share']:12.1%} {l['difference']:+7.1%} | {t['small_share']:11.1%} "
              f"{t['accuracy']:6.1%} {t['difference']:+7.1%} {t['difference_min']:+7.1%}..{t['difference_max']:+6.1%} "
              f"{t['own_model_tokens_vs_A']:8.1%} {t['price_vs_A']:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
