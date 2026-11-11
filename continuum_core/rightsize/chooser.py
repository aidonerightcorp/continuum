"""The chooser: from the records of the Tasks that look most like a new Task, pick the cheapest rung
that should still pass, and say why.

A rung is one way to run a turn: a model on the user's own endpoint, with optional request settings.
Rungs are ordered from cheapest to the reference rung, which is the model the user configured.

Rules, all with their numbers in config/rightsize.json:

  * a cheaper rung is used only when the lower bound of its pass rate on similar recorded Tasks is at
    least the floor, with enough evidence, and its estimate is within the tolerance of the reference;
  * a small, bounded share of turns tries a cheaper rung that is not yet known, or checks the reference;
    checked passes near a Task raise that share there and checked failures lower it (never above
    `promising_max_share`), so the trial turns go where they tell the most;
  * Tasks that match the never-downsize pattern always get the reference rung;
  * a share of Task templates is held out: their records are never read by the chooser, so they show
    how the policy does on Tasks it has no record of. The audit tightens the policy when it does worse
    there, and relaxes it step by step when it does well.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import fingerprint as fpm


@dataclass
class Rung:
    name: str
    model: str = ""          # "" means the model the user configured (the reference)
    cost: float = 1.0        # relative price of one token
    params: dict[str, Any] = field(default_factory=dict)
    max_context: int = 0     # 0 = unknown

    @property
    def is_reference(self) -> bool:
        return not self.model


@dataclass
class Decision:
    rung: str
    model: str
    mode: str                # exploit | explore | reference | reference-check | pinned | observe
    reason: str
    floor: float = 0.0
    best_similarity: float = 0.0
    neighbours: int = 0
    estimates: dict[str, dict[str, float]] = field(default_factory=dict)
    tools_keep: list[str] | None = None
    recall_key: str | None = None
    recall_similarity: float = 0.0
    trials_waiting: bool = False  # a trial turn would have been possible, but trials wait for a recent mark

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# Outcomes that come through the engine's `feedback`: a user's mark (/continuum good|bad) and a trusted check that a
# caller reports (for example a test run). They are the only strong outcomes that can be a pass.
MARK_SOURCES = ("explicit", "check")


def is_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("source") or "").split(":")[-1] in MARK_SOURCES


def is_strong(row: Mapping[str, Any], strong_weight: float) -> bool:
    """A checked outcome: weight at or above `strong_weight`, except a PASS of the user's own model and a thank-you,
    which are weak whatever weight a row carries (rows written before November 11, and imported files, carry 0.8 and
    0.6 for them)."""
    src = str(row.get("source") or "").split(":")[-1]
    if float(row.get("score") or 0.0) >= 0.5 and src in ("reference_check", "thanks"):
        return False
    return float(row.get("weight") or 0.0) >= strong_weight


def wilson(p: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score interval for a pass rate p seen on n (possibly fractional) units of evidence."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def split_of(template: str, share: float, salt: str = "continuum") -> str:
    """Held-out or fit, by Task template, so that near copies of one Task stay on one side."""
    h = int(hashlib.sha256((salt + ":" + template).encode()).hexdigest()[:8], 16) / float(1 << 32)
    return "holdout" if h < share else "fit"


def uniform(key: str, salt: str = "explore") -> float:
    """A repeatable number in [0, 1) for one turn."""
    return int(hashlib.sha256((salt + ":" + key).encode()).hexdigest()[:12], 16) / float(1 << 48)


class Pool:
    """The recorded Tasks the chooser may read (the fit split), with their fingerprints ready."""

    def __init__(self, records: Sequence[Mapping[str, Any]], fp_cfg: Mapping[str, Any], weights: Mapping[str, float]):
        self.records = list(records)
        self.fp_cfg = fp_cfg
        self.weights = dict(weights)
        self.records = [r for r in self.records if isinstance(r.get("fp"), Mapping)
                        and isinstance(r["fp"].get("profile"), (list, tuple))]
        self.scales = fpm.profile_scales([r["fp"]["profile"] for r in self.records])
        self.rates: dict[str, tuple[float, float]] = {}
        self.strong_rates: dict[str, tuple[float, float]] = {}  # checked outcomes only (a mark, a check, a fault)
        for only_strong, rates in ((False, self.rates), (True, self.strong_rates)):
            acc: dict[str, list[float]] = {}
            for r in self.records:
                if only_strong and not r.get("strong", True):
                    continue
                a = acc.setdefault(r["rung"], [0.0, 0.0])
                a[0] += r["score"] * r["weight"]
                a[1] += r["weight"]
            for rung, (passes, n) in acc.items():
                rates[rung] = ((passes / n) if n else 0.5, n)

    def rank(self, fp: Mapping[str, Any], limit: int = 0, min_similarity: float = 0.0,
             skip_template: str | None = None) -> list[tuple[float, Mapping[str, Any]]]:
        out = []
        for r in self.records:
            if skip_template is not None and r["template"] == skip_template:
                continue
            try:
                sim = fpm.fuse(fpm.level_similarities(fp, r["fp"], self.scales), self.fp_cfg, self.weights)
            except (KeyError, TypeError, ValueError):
                continue  # one unreadable record never stops the choice
            if sim >= min_similarity:
                out.append((sim, r))
        out.sort(key=lambda x: -x[0])
        return out[:limit] if limit else out

    def estimate(self, ranked: Sequence[tuple[float, Mapping[str, Any]]], rung: str, cfg: Mapping[str, Any],
                 floor_z: float | None = None) -> dict[str, float]:
        """Pass rate of one rung on the ranked neighbours, shrunk toward the rung's rate over the pool.

        An outcome that nobody checked ("the user moved on", "the session ended") counts as a weak pass. Once
        checked outcomes near the Task reach `strong_takeover` units, the estimate reads the checked ones alone,
        so a pile of unchecked turns cannot hide failures that a check or a mark found."""
        # sums: [own passes, own n, imported passes, imported n], for all outcomes and for checked ones
        sums = {False: [0.0, 0.0, 0.0, 0.0], True: [0.0, 0.0, 0.0, 0.0]}
        for sim, r in ranked:
            if r["rung"] == rung:
                at = 2 if r.get("imported") else 0
                for only_strong in (False, True):
                    if only_strong and not r.get("strong", True):
                        continue
                    sums[only_strong][at] += sim * r["weight"] * r["score"]
                    sums[only_strong][at + 1] += sim * r["weight"]
        # Imported records help, and never decide: their evidence is capped below what a choice needs.
        cap = float(cfg.get("import_max_evidence", 2.0))
        for acc in sums.values():
            if acc[3] > cap:
                acc[2], acc[3] = acc[2] * cap / acc[3], cap
        passes, n = sums[False][0] + sums[False][2], sums[False][1] + sums[False][3]
        s_passes, s_n = sums[True][0] + sums[True][2], sums[True][1] + sums[True][3]
        strong = s_n >= float(cfg.get("strong_takeover", 1.0))
        rates = self.rates
        if strong:
            passes, n = s_passes, s_n
            if self.strong_rates.get(rung, (0.5, 0.0))[1] > 0:
                rates = self.strong_rates
        centre = rates.get(rung, (0.5, 0.0))[0]
        strength = float(cfg.get("prior_strength", 1.0))
        total = n + strength
        p = (passes + strength * centre) / total if total else centre
        lb, ub = wilson(p, total, float(floor_z if floor_z is not None else cfg.get("z", 1.0)))
        return {"p": round(p, 4), "lb": round(lb, 4), "ub": round(ub, 4), "n": round(n, 3), "checked": float(strong)}


def choose(fp: Mapping[str, Any], text: str, rungs: Sequence[Rung], pool: Pool, cfg: Mapping[str, Any],
           *, floor: float, min_evidence: float, explore_share: float, u: float,
           request_tokens: int = 0, trials: bool = True) -> Decision:
    """Pick a rung for one Task. `rungs` is ordered cheapest first; the last one is the reference.

    `trials=False` turns off trial turns only (the user has no recent mark, see the engine). The comparison turns
    that keep the reference rate current where a cheaper model already qualifies (`explore_share / 2`) stay."""
    ref = rungs[-1]
    k = int(cfg.get("neighbours", 12))
    ranked_all = pool.rank(fp, min_similarity=float(cfg.get("min_similarity", 0.35)))
    estimates: dict[str, dict[str, float]] = {}
    for rung in rungs:
        top = [x for x in ranked_all if x[1]["rung"] == rung.name][:k]
        estimates[rung.name] = pool.estimate(top, rung.name, cfg)
    best = ranked_all[0][0] if ranked_all else 0.0
    base = Decision(ref.name, ref.model, "reference", "", floor=floor, best_similarity=round(best, 4),
                    neighbours=len(ranked_all), estimates=estimates)
    if len(rungs) < 2:
        base.mode, base.reason = "observe", "one rung only: recording, not choosing"
        return base
    never = cfg.get("never_downsize") or ""
    if never and re.search(never, text or "", re.I):
        base.mode, base.reason = "pinned", "the Task matches the never-downsize pattern"
        return base
    tol = float(cfg.get("tolerance", 0.05))
    ref_est = estimates[ref.name]
    ref_known = bool(ref_est.get("checked"))  # unchecked turns of the reference say nothing to compare against
    fill = float(cfg.get("context_fill", 0.8))
    unknown: list[Rung] = []
    for rung in rungs[:-1]:
        est = estimates[rung.name]
        if rung.max_context and request_tokens and request_tokens > fill * rung.max_context:
            continue
        behind = ref_known and bool(est.get("checked")) and est["p"] < ref_est["p"] - tol
        # A cheaper model moves only on checked outcomes (a user's mark, a failed check, a fault). Weak signals
        # ("the user moved on", "the session ended", a PASS from the user's own model) say little about
        # whether an answer was right: in our runs the own-model check passed 83% of the wrong answers.
        checked_enough = bool(est.get("checked")) or not cfg.get("require_checked", True)
        if est["n"] >= min_evidence and est["lb"] >= floor and not behind and checked_enough:
            if u < explore_share / 2.0:
                base.mode = "reference-check"
                base.reason = (f"{rung.name} qualifies (lower bound {est['lb']:.2f} >= floor {floor:.2f}); "
                               f"this turn checks {ref.name} to keep its rate current")
                return base
            return Decision(rung.name, rung.model, "exploit",
                            f"{rung.name}: lower bound {est['lb']:.2f} >= floor {floor:.2f} on {est['n']:.1f} units of "
                            f"evidence from similar Tasks", floor=floor, best_similarity=round(best, 4),
                            neighbours=len(ranked_all), estimates=estimates)
        ruled_out = est["n"] >= min_evidence and (est["ub"] < floor or behind)
        if not ruled_out:
            unknown.append(rung)
    if unknown and not trials:
        base.trials_waiting = True
    if unknown and trials:
        # Trial turns go where they tell the most. Checked passes near this Task raise the share of trial
        # turns here, so a cheaper model that does well on a kind of Task is proven (or disproven) quickly;
        # checked failures near this Task lower it, so few turns are spent on a kind it cannot do. With no
        # checked outcome near the Task, the plain share applies. All of it scales with `explore_share`.
        rung = unknown[0]
        est = estimates[rung.name]
        share, why = explore_share, f"only {est['n']:.1f} units of evidence near this Task"
        boost = float(cfg.get("promising_boost") or 1.0)
        if boost > 1.0 and est.get("checked") and est["n"] >= float(cfg.get("promising_evidence") or 1.0):
            if est["p"] >= floor:
                share = min(float(cfg.get("promising_max_share") or 0.5), explore_share * boost)
                why = (f"it passed on similar Tasks so far (rate {est['p']:.2f} on {est['n']:.1f} units); "
                       f"more trial turns here until the evidence is enough")
            elif est["p"] < floor - tol:
                share = explore_share / boost
                why = (f"it mostly failed on similar Tasks so far (rate {est['p']:.2f} on {est['n']:.1f} units); "
                       f"a rare trial turn keeps that current")
        if u < share:
            return Decision(rung.name, rung.model, "explore", f"trying {rung.name}: {why}",
                            floor=floor, best_similarity=round(best, 4), neighbours=len(ranked_all),
                            estimates=estimates)
    cheapest = estimates[rungs[0].name]
    if cfg.get("require_checked", True) and not cheapest.get("checked") and cheapest["n"] >= min_evidence:
        base.reason = (f"{ref.name}: {rungs[0].name} has {cheapest['n']:.1f} units of evidence, none of it checked; "
                       f"a cheaper model moves only on marked outcomes (/continuum good|bad)")
        return base
    base.reason = (f"{ref.name}: {rungs[0].name} has lower bound {cheapest['lb']:.2f} on {cheapest['n']:.1f} units of "
                   f"evidence (needs {floor:.2f} on {min_evidence:.1f})")
    return base


def choose_tools(fp: Mapping[str, Any], text: str, pool_any: Pool, tool_names: Sequence[str],
                 cfg: Mapping[str, Any]) -> list[str] | None:
    """The tools to keep for one Task, or None to keep all. A tool is kept when the user names it, when
    it is on the always-keep list, or when a similar Task that passed used it."""
    ranked = [x for x in pool_any.rank(fp, min_similarity=float(cfg.get("min_similarity", 0.5)))
              if x[1]["score"] >= 0.5 and not x[1].get("imported")]  # the user's own records only
    if len(ranked) < int(cfg.get("min_neighbours", 5)):
        return None
    keep = set(cfg.get("always_keep", []))
    for _, r in ranked:
        keep.update(r.get("tools_used") or [])
    low = (text or "").lower()
    for name in tool_names:
        if name.lower() in low or name.lower().replace("_", " ") in low:
            keep.add(name)
    kept = [n for n in tool_names if n in keep]
    return kept if len(kept) < len(tool_names) else None


def in_scope(record: Mapping[str, Any], scope: tuple[str, str] | None) -> bool:
    """A record may be shown to a turn only when it comes from the same platform and the same sender."""
    return scope is None or (not record.get("imported") and (record.get("platform") or "") == scope[0]
                             and (record.get("owner") or "") == scope[1])


def find_recall(fp: Mapping[str, Any], pool_any: Pool, cfg: Mapping[str, Any],
                scope: tuple[str, str] | None = None) -> tuple[str | None, float]:
    """The record of a very similar Task with a strong good outcome and a stored reply, if any."""
    for sim, r in pool_any.rank(fp, min_similarity=float(cfg.get("min_similarity", 0.92)))[:25]:
        if in_scope(r, scope) and r["score"] >= 0.99 and r["weight"] >= float(cfg.get("min_weight", 0.9)) and r.get("has_reply"):
            return r["turn_key"], sim
    return None, 0.0


# ---------- learning safeguards ----------
def _rate(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    n = sum(r["weight"] for r in rows)
    return ((sum(r["score"] * r["weight"] for r in rows) / n) if n else 0.0, n)


def audit(turns: Sequence[Mapping[str, Any]], reference: str, state: Mapping[str, float],
          defaults: Mapping[str, float], cfg: Mapping[str, Any], z: float = 1.0) -> dict[str, Any]:
    """Check the policy on what it could not have read: each turn's outcome came after its choice, and the
    held-out templates never fed the chooser. Returns the new strictness and the evidence for it."""
    new = {"floor": float(state["floor"]), "min_evidence": float(state["min_evidence"])}
    report: dict[str, Any] = {"action": "none", "views": {}}
    need = float(cfg.get("audit_min_evidence", 4.0))
    margin = float(cfg.get("audit_margin", 0.1))
    worse = better = False
    for view, rows in (("held-out", [t for t in turns if t["split"] == "holdout"]), ("all", list(turns))):
        cheap = [t for t in rows if t["mode"] == "exploit" and t["rung"] != reference]
        refs = [t for t in rows if t["rung"] == reference]
        strong = float(cfg.get("strong_weight", 0.5))
        # Both models are rated on marks, the one kind of outcome that holds passes and failures alike (the other
        # strong outcomes are failures only). Too few marks on the cheaper model: all of its outcomes are read.
        # Too few marks on the user's own model: the target is the default floor.
        p_c, n_c = _rate([t for t in cheap if is_mark(t)])
        if n_c < need:
            p_c, n_c = _rate(cheap)
        p_r, n_r = _rate([t for t in refs if is_mark(t)])
        lb, ub = wilson(p_c, n_c, z)
        target = (p_r if n_r >= need else float(defaults["floor"])) - margin
        report["views"][view] = {"cheap_rate": round(p_c, 4), "cheap_evidence": round(n_c, 2),
                                 "reference_rate": round(p_r, 4), "reference_evidence": round(n_r, 2),
                                 "target": round(target, 4), "lower": round(lb, 4), "upper": round(ub, 4)}
        if n_c >= need:
            worse = worse or ub < target
            better = better or lb >= target
    if worse:
        new["floor"] = min(float(cfg.get("max_floor", 0.95)), new["floor"] + float(cfg.get("step_floor", 0.05)))
        new["min_evidence"] = min(float(cfg.get("max_evidence", 12.0)),
                                  new["min_evidence"] + float(cfg.get("step_evidence", 1.0)))
        report["action"] = "tighten"
    elif better and (new["floor"] > defaults["floor"] or new["min_evidence"] > defaults["min_evidence"]):
        new["floor"] = max(float(defaults["floor"]), new["floor"] - float(cfg.get("step_floor", 0.05)))
        new["min_evidence"] = max(float(defaults["min_evidence"]),
                                  new["min_evidence"] - float(cfg.get("step_evidence", 1.0)))
        report["action"] = "relax"
    report["state"] = {k: round(v, 4) for k, v in new.items()}
    return report


def _pair_parts(queries: Sequence[Mapping[str, Any]], pool: Pool) -> list[list[tuple[dict[str, float], Mapping[str, Any]]]]:
    """Raw similarity, level by level, of every query against every pool record. The raw levels do not
    depend on the preset weights, so one pass serves every preset."""
    out = []
    for q in queries:
        row = []
        for r in pool.records:
            try:
                row.append((fpm.level_similarities(q["fp"], r["fp"], pool.scales), r))
            except (KeyError, TypeError, ValueError):
                continue  # one unreadable record never stops the tuning
        out.append(row)
    return out


def _brier(queries: Sequence[Mapping[str, Any]], pool: Pool, cfg: Mapping[str, Any], leave_template: bool,
           parts: Sequence[Sequence[tuple[Mapping[str, float], Mapping[str, Any]]]] | None = None) -> tuple[float, float]:
    k = int(cfg.get("neighbours", 12))
    least = float(cfg.get("min_similarity", 0.35))
    err = n = 0.0
    for i, q in enumerate(queries):
        if parts is None:
            ranked = pool.rank(q["fp"], min_similarity=least, skip_template=q["template"] if leave_template else None)
        else:
            ranked = []
            for levels, r in parts[i]:
                if leave_template and r["template"] == q["template"]:
                    continue
                sim = fpm.fuse(levels, pool.fp_cfg, pool.weights)
                if sim >= least:
                    ranked.append((sim, r))
            ranked.sort(key=lambda x: -x[0])
        ranked = [x for x in ranked if x[1]["rung"] == q["rung"]][:k]
        est = pool.estimate(ranked, q["rung"], cfg)
        err += q["weight"] * (est["p"] - q["score"]) ** 2
        n += q["weight"]
    return ((err / n) if n else 1.0, n)


def _spread(rows: Sequence[Mapping[str, Any]], most: int) -> list[Mapping[str, Any]]:
    """At most `most` rows, taken evenly over the whole list (0 keeps all)."""
    rows = list(rows)
    if most <= 0 or len(rows) <= most:
        return rows
    step = len(rows) / float(most)
    return [rows[int(i * step)] for i in range(most)]


def tune_presets(fit: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], fp_cfg: Mapping[str, Any],
                 chooser_cfg: Mapping[str, Any], current: str, margin: float, max_queries: int = 0) -> dict[str, Any]:
    """Pick the similarity weights that predict recorded outcomes best. Each preset is scored on the fit
    records with every Task's own template left out, and the winner replaces the current preset only when
    it is not worse on the held-out records by more than the margin.

    The cost is one similarity per pair of (scored Task, pool record), paid once for all presets. With
    `max_queries` the scored Tasks are a spread sample of that size, so the work stays bounded when the
    record file grows."""
    presets = fp_cfg["presets"]
    pools = {name: Pool(fit, fp_cfg, weights) for name, weights in presets.items()}
    first = next(iter(pools.values()))  # every pool holds the same records and the same profile scales
    queries = _spread(first.records, max_queries)
    parts = _pair_parts(queries, first)
    scores = {name: round(_brier(queries, pool, chooser_cfg, True, parts)[0], 5) for name, pool in pools.items()}
    best = min(scores, key=lambda n: (scores[n], n != current))
    out = {"fit_error": scores, "candidate": best, "current": current, "chosen": current, "held_out": {},
           "scored_on": len(queries), "pool": len(first.records)}
    if best != current and holdout and current in pools:
        held = _spread(holdout, max_queries)
        held_parts = _pair_parts(held, first)
        for name in (current, best):
            out["held_out"][name] = round(_brier(held, pools[name], chooser_cfg, False, held_parts)[0], 5)
        if out["held_out"][best] <= out["held_out"][current] + margin:
            out["chosen"] = best
    return out
