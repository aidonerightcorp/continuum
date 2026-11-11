"""The Task fingerprint, at several levels, from exact match to continuous similarity.

A fingerprint is what Continuum keeps about a Task in order to find Tasks that look like it later.
It never authorizes anything by itself: it only chooses which past records are read.

    exact      same text after case and space folding                      hash
    template   same text after every number is masked                      hash
    contract   same input form and output form (text -> number, ...)       label read from the text by rules
    lsh        shares one whole band of the MinHash sketch                 reported; not part of the fused score
    sketch     estimated Jaccard overlap of words and word pairs           32 integers
    words      cosine of hashed word and word-pair counts                  sparse vector
    profile    distance between shape measurements (length, numbers, ...)  16 numbers

Everything is plain Python with no model call and no dependency. Scales, powers, weights and the
contract rules come from config/rightsize.json.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Iterable, Mapping, Sequence

WORD = re.compile(r"[\w']+")  # any script; on plain ASCII text this equals [a-z0-9_']+
LETTERS = re.compile(r"(?:[^\W\d_]|')+")  # runs of letters of any script
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
URL = re.compile(r"https?://\S+")
PATH = re.compile(r"(?:[\w.-]+/){1,}[\w.-]+|\b[\w-]+\.(?:py|js|ts|md|json|yaml|yml|txt|csv|html|css|sh|rs|go|c|h|java|sql)\b")
OPTION = re.compile(r"\n\s*\(?[A-Za-z][\).]\s")
SENTENCE = re.compile(r"[.?!](?:\s|$)")
_MERSENNE = (1 << 61) - 1

PROFILE_NAMES = (
    "log_words", "numbers", "distinct_numbers", "digit_share", "operator_share", "options", "lines",
    "sentences", "commas", "capital_words", "word_length", "question_mark", "code_share", "paths",
    "urls", "log_chars",
)


def normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def tokens(text: str) -> list[str]:
    return WORD.findall(str(text or "").lower())


def _h64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _features(words: Sequence[str]) -> list[str]:
    """Words and neighbouring word pairs."""
    return list(words) + [f"{a} {b}" for a, b in zip(words, words[1:])]


def _permutations(size: int) -> list[tuple[int, int]]:
    rng = random.Random(0x7E1E10)  # fixed, so a sketch made on one machine compares with one made on another
    return [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE)) for _ in range(size)]


_PERMS: dict[int, list[tuple[int, int]]] = {}


def minhash(features: Iterable[str], size: int = 32) -> list[int]:
    """MinHash sketch. A Task with no word gets an empty sketch, which matches nothing."""
    hashed = {_h64(f) % _MERSENNE for f in features}
    if not hashed:
        return []
    perms = _PERMS.setdefault(size, _permutations(size))
    return [min((a * h + b) % _MERSENNE for h in hashed) for a, b in perms]


def word_vector(features: Iterable[str], buckets: int = 512) -> dict[str, float]:
    """Hashed counts with a damped count, unit length. Keys are strings so the vector is plain JSON."""
    counts: dict[int, int] = {}
    for f in features:
        k = _h64("v:" + f) % buckets
        counts[k] = counts.get(k, 0) + 1
    weights = {k: 1.0 + math.log(c) for k, c in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return {str(k): round(w / norm, 5) for k, w in weights.items()}


def template_of(text: str) -> str:
    """The text with every number masked: two Tasks that differ only in their numbers share it."""
    masked = NUMBER.sub("#", URL.sub("<url>", normalized(text)))
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()[:24]


def contract_of(text: str, rules: Mapping[str, Any]) -> str:
    """The form of the Task: what goes in and what must come out. Read from the text by ordered rules."""
    raw = str(text or "")
    low = raw.lower().strip()
    out = "open"
    for name, pattern in rules.get("contract_out", []):
        if re.search(pattern, raw if name == "choice" else low, re.S if name == "code" else 0):
            out = name
            break
    inp = "text"
    for name, pattern in rules.get("contract_in", []):
        if re.search(pattern, raw if name == "code" else low):
            inp = name
            break
    return f"in:{inp}|out:{out}"


def profile_of(text: str) -> list[float]:
    q = str(text or "")
    words = LETTERS.findall(q)
    nums = NUMBER.findall(q)
    n = max(1, len(q))
    return [round(v, 5) for v in (
        math.log1p(len(words)),
        float(len(nums)),
        float(len(set(nums))),
        sum(c.isdigit() for c in q) / n,
        sum(c in "+-*/=()" for c in q) / n,
        float(len(OPTION.findall(q))),
        float(q.count("\n") + 1),
        float(len(SENTENCE.findall(q))),
        float(q.count(",")),
        float(sum(w[0].isupper() for w in words)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        float("?" in q),
        sum(c in "{}[];<>`|\\" for c in q) / n,
        float(len(PATH.findall(q))),
        float(len(URL.findall(q))),
        math.log1p(len(q)),
    )]


def fingerprint(text: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Every level of one Task. The result is plain JSON and holds no text of the Task."""
    words = tokens(text)
    feats = _features(words)
    return {
        "exact": hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()[:32],
        "template": template_of(text),
        "contract": contract_of(text, cfg),
        "sketch": minhash(feats, int(cfg.get("minhash_size", 32))),
        "words": word_vector(feats, int(cfg.get("word_buckets", 512))),
        "profile": profile_of(text),
        "n_words": len(words),
    }


# ---------- similarity ----------
def sketch_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def shares_band(a: Sequence[int], b: Sequence[int], band_count: int = 4) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    rows = len(a) // band_count
    return any(a[i * rows:(i + 1) * rows] == b[i * rows:(i + 1) * rows] for i in range(band_count))


def words_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return max(0.0, min(1.0, sum(w * b.get(k, 0.0) for k, w in a.items())))


def profile_scales(profiles: Sequence[Sequence[float]]) -> list[float]:
    """Spread of each measurement over a pool of Tasks, so that no one measurement rules the distance."""
    if not profiles:
        return [1.0] * len(PROFILE_NAMES)
    out = []
    for j in range(len(profiles[0])):
        col = [p[j] for p in profiles if len(p) > j]
        mean = sum(col) / len(col)
        sd = math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
        out.append(max(sd, 0.05 * (abs(mean) + 1.0)))
    return out


def profile_similarity(a: Sequence[float], b: Sequence[float], scales: Sequence[float]) -> float:
    n = min(len(a), len(b), len(scales))
    if not n:
        return 0.0
    d2 = sum(((a[j] - b[j]) / scales[j]) ** 2 for j in range(n)) / n
    return math.exp(-d2 / 2.0)


def level_similarities(a: Mapping[str, Any], b: Mapping[str, Any], scales: Sequence[float]) -> dict[str, float]:
    """Raw similarity of two fingerprints at every level, each between 0 and 1."""
    return {
        "exact": float(a["exact"] == b["exact"]),
        "template": float(a["template"] == b["template"]),
        "contract": float(a["contract"] == b["contract"]),
        "lsh": float(shares_band(a["sketch"], b["sketch"])),  # shown by /continuum why; fuse() does not read it
        "sketch": sketch_similarity(a["sketch"], b["sketch"]),
        "words": words_similarity(a["words"], b["words"]),
        "profile": profile_similarity(a["profile"], b["profile"], scales),
    }


def fuse(parts: Mapping[str, float], cfg: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """One number from the levels. Exact match is 1. A shared template sets a floor. A different
    contract scales the result down. In between, the continuous levels are averaged with the weights."""
    if parts["exact"]:
        return 1.0
    total = wsum = 0.0
    for name, w in weights.items():
        if w <= 0:
            continue
        level = cfg["levels"][name]
        total += w * min(1.0, parts[name] / level["scale"]) ** level["power"]
        wsum += w
    score = total / wsum if wsum else 0.0
    if parts["template"]:
        score = max(score, float(cfg.get("template_floor", 0.9)))
    if not parts["contract"]:
        score *= float(cfg.get("contract_mismatch", 0.5))
    return score
