#!/usr/bin/env python3
"""On/off benchmark of the Continuum plugin inside real Hermes.

One process runs one arm. Every Task is one real Hermes session: the harness builds Hermes's own agent
(`AIAgent`, the class Hermes's batch runner uses), gives it the Task as the user message, and reads the reply.
Continuum, when the arm loads it, is discovered and loaded by Hermes from the arm's own Hermes home, exactly as
for a user. A second small plugin (`bench_meter`) is loaded in every arm and writes one line per model call, so
arms with and without Continuum are measured the same way.

    bench_hermes.py --run NAME --arm ARM --items FILE [--limit N] [--spec FILE] [--endpoint URL]
                    [--reference MODEL] [--small MODEL]

Files written, all under runs/NAME/ARM/:
    home/            the arm's Hermes home (config.yaml, the plugin's own data folder)
    records.jsonl    one line per Task: reply, parsed answer, right or wrong, seconds
    meter.jsonl      one line per model call: model, tokens, seconds
    spec.json        the settings the arm ran with

A run can be stopped and started again; Tasks that already have a record are skipped. The order of the Tasks is
the order of the items file, and it matters: Continuum learns from the earlier Tasks.

The harness makes no network call of its own. Hermes calls only the endpoint named in the spec.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Where Hermes and the plugin are. HERMES_AGENT_DIR is a checkout of Hermes Agent; CONTINUUM_PLUGIN_DIR is the
# plugin folder (found from the folders above this file when the harness sits inside the plugin repository).
FORK = Path(os.environ.get("HERMES_AGENT_DIR") or ROOT.parent / "work-2026-11-09-hermes-catalog-prs" / "hermes-agent-fork")
PLUGIN = Path(os.environ.get("CONTINUUM_PLUGIN_DIR")
              or next((p for p in HERE.parents if (p / "plugin.yaml").exists()), ROOT / "continuum"))

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
OPTION = re.compile(r"\(?\b([A-H])\b\)?")


def wants(question: str) -> str:
    """A Task with lettered options wants a letter; any other Task here wants a number."""
    return "choice" if re.search(r"\n\s*\(A\)", question) else "number"


def parse_answer(text: str, kind: str):
    """Read the answer from the reply: the last 'Answer:' line when there is one, else the end of the reply."""
    text = (text or "").strip()
    marks = [m.end() for m in re.finditer(r"answer\s*(?:is)?\s*[:=]", text, re.I)]
    tail = text[marks[-1]:] if marks else text[-200:]
    tail = tail.replace("**", "").replace("$", "").strip()
    if kind == "choice":
        m = OPTION.search(tail[:40]) if marks else None
        if not m:
            found = re.findall(r"\(([A-H])\)", tail)
            return f"({found[-1]})" if found else None
        return f"({m.group(1)})"
    line = tail.split("\n")[0].split("=")[-1] if marks else tail  # "Answer: 3 + 4 = 7" means 7
    found = NUMBER.findall(line)
    if not found:
        return None
    pick = found[0] if marks else found[-1]
    try:
        return float(pick.replace(",", ""))
    except ValueError:
        return None


def right(pred, gold, tol: float = 1e-3) -> bool:
    if pred is None:
        return False
    if isinstance(gold, str) or isinstance(pred, str):
        return pred == gold
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))  # a JSON string is a valid YAML string


def write_home(home: Path, spec: dict, arm: dict, model: str) -> None:
    """The arm's Hermes home: its config file and the plugins Hermes will find there."""
    (home / "plugins").mkdir(parents=True, exist_ok=True)
    enabled = ["bench_meter"] + (["continuum"] if arm.get("plugin") else [])
    for name, src in (("bench_meter", HERE / "bench_meter"), ("continuum", PLUGIN)):
        link = home / "plugins" / name
        if name in enabled and not link.exists():
            link.symlink_to(src, target_is_directory=True)
    lines = ["model:", f"  default: {yaml_scalar(model)}", "  provider: custom",
             f"  base_url: {yaml_scalar(spec['endpoint'])}", f"  context_length: {int(spec['declared_context'])}",
             "plugins:", "  enabled:"] + [f"    - {n}" for n in enabled]
    if arm.get("plugin"):
        settings = {"small_model": spec["small_model"], **(arm.get("settings") or {})}
        lines += ["  entries:", "    continuum:", "      granted_capabilities:"]
        lines += [f"        - {g}" for g in arm.get("grants") or []] or ["        []"]
        lines += ["      settings:"] + [f"        {k}: {yaml_scalar(v)}" for k, v in settings.items()]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spec", default=str(HERE / "bench_spec.json"))
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--small", default="")
    a = ap.parse_args()

    spec = json.loads(Path(a.spec).read_text())
    for key, value in (("endpoint", a.endpoint), ("reference_model", a.reference), ("small_model", a.small)):
        if value:
            spec[key] = value
    arm = spec["arms"][a.arm]
    model = spec["reference_model"] if arm["model"] == "reference" else spec["small_model"]
    out = HERE / "runs" / a.run / a.arm
    home = out / "home"
    out.mkdir(parents=True, exist_ok=True)
    write_home(home, spec, arm, model)
    (out / "spec.json").write_text(json.dumps({**spec, "arms": {a.arm: arm}, "model": model,
                                                "items": str(Path(a.items).resolve())}, indent=2))

    items = [json.loads(line) for line in open(a.items, encoding="utf-8") if line.strip()]
    if a.limit:
        items = items[:a.limit]
    records = out / "records.jsonl"
    done = {json.loads(line)["id"] for line in open(records, encoding="utf-8")} if records.exists() else set()

    if not (FORK / "run_agent.py").exists():
        print(f"Hermes Agent was not found at {FORK}. Set HERMES_AGENT_DIR to a checkout of Hermes Agent.")
        return 2
    # Hermes reads its home from the environment, so set it before Hermes is imported.
    os.environ["HERMES_HOME"] = str(home)
    os.environ["BENCH_METER_FILE"] = str(out / "meter.jsonl")
    os.environ.setdefault("HERMES_QUIET", "1")
    sys.path.insert(0, str(FORK))
    os.chdir(out)
    from run_agent import AIAgent  # noqa: E402  (Hermes)
    from hermes_cli.plugins import discover_plugins, get_plugin_command_handler, invoke_hook  # noqa: E402
    from hermes_cli import version_info  # noqa: E402

    # A plugin arm without the plugin is a wrong arm, so check before the first Task. Hermes reads its own
    # version from git with a 3 second limit; on a busy machine that call can time out, Hermes then reports an
    # older release, and a plugin with `requires_hermes` is skipped. Ask again a few times before giving up.
    hermes = version_info.get_version_info().base_version
    if arm.get("plugin"):
        for attempt in range(8):
            discover_plugins(force=attempt > 0)
            if get_plugin_command_handler("continuum") is not None:
                break
            time.sleep(4)
            version_info._reset_version_info_cache()
            hermes = version_info.get_version_info().base_version
        else:
            print(f"Continuum did not load in Hermes {hermes}; arm {a.arm} was not run.", flush=True)
            return 3
    print(f"{time.strftime('%H:%M:%S')} {a.arm}: Hermes {hermes}, Continuum "
          f"{'loaded' if arm.get('plugin') else 'not used'}", flush=True)
    (out / "spec.json").write_text(json.dumps({**spec, "arms": {a.arm: arm}, "model": model, "hermes_version": hermes,
                                                "items": str(Path(a.items).resolve())}, indent=2))

    started = time.time()
    for n, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        os.environ["BENCH_METER_ITEM"] = item["id"]
        kind = wants(item["question"])
        prompt = item["question"] + spec.get("prompt_suffix", "")
        t0 = time.time()
        row = {"id": item["id"], "kind": item["kind"], "arm": a.arm, "n": n}
        agent = None
        try:
            agent = AIAgent(model=model, base_url=spec["endpoint"], api_key="local", provider="custom",
                            max_iterations=int(spec["max_iterations"]), enabled_toolsets=list(spec["toolsets"]),
                            quiet_mode=True, save_trajectories=False, skip_context_files=True, skip_memory=True,
                            skip_background_review=True, platform="batch")
            result = agent.run_conversation(prompt, task_id=f"{a.run}-{a.arm}-{n}")
            reply = result.get("final_response") or ""
            pred = parse_answer(reply, kind)
            row.update(reply=reply[-1500:], pred=pred, right=right(pred, item["gold"]), gold=item["gold"],
                       api_calls=result.get("api_calls"), completed=bool(result.get("completed")),
                       session_id=agent.session_id)
            # a reference check runs on its own thread; let it finish before the session closes
            for t in threading.enumerate():
                if t.name == "continuum-check":
                    t.join(90)
            if arm.get("feedback") == "answer_key":
                handler = get_plugin_command_handler("continuum")
                row["marked"] = handler("good" if row["right"] else "bad", session_id=agent.session_id)[:80]
            invoke_hook("on_session_finalize", session_id=agent.session_id, platform="batch")  # as the CLI at exit
        except Exception as exc:  # a failed Task is a record, not a stop
            row.update(error=f"{type(exc).__name__}: {exc}"[:300], right=False, gold=item["gold"])
        finally:
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass
        row["seconds"] = round(time.time() - t0, 2)
        with open(records, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if n % 10 == 0 or n == len(items):
            rows = [json.loads(line) for line in open(records, encoding="utf-8")]
            ok = sum(1 for r in rows if r.get("right"))
            print(f"{time.strftime('%H:%M:%S')} {a.arm}: {len(rows)}/{len(items)} Tasks, {ok} right, "
                  f"{time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
