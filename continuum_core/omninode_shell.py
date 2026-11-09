"""OmniNode Shell & Algebraic Shape Composition (Stratum 8: Engines 29 & 30 + grid/1 Recorder)."""

from __future__ import annotations

import ast
import copy
import datetime
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from continuum_core.config_loader import (
    ManifoldConfigSpec,
    OmniNodeGridSpec,
    get_manifold_spec,
    get_omninode_grid_spec,
)
from continuum_core.linalg_backend import l2_normalize, vec_dot, vec_norm

if TYPE_CHECKING:
    from continuum_core.storage import ContinuumStore
    from continuum_core.weight_bus import CyberneticWeightBus

CARD = {
    "name": "omninode_shell",
    "role": "director",
    "stratum": "omninode_shell_and_shape_composition",
    "contract": "omninode.shell/v1",
    "describes": "Encases every Continuum engine, tool, and Hermes surface as a self-describing OmniNode CARD with CAS lockfiles, >> // | @ composition, and grid/1 receipts.",
    "keywords": [
        "omninode",
        "shell",
        "card",
        "lockfile",
        "cas",
        "shape",
        "grid1",
        "compose",
    ],
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_SHAPE_TOKEN_RE = re.compile(r"\s*(>>|//|\||@|\(|\)|[a-zA-Z][a-zA-Z0-9_-]*)")

TOOL_NAMES_CANONICAL: tuple[str, ...] = (
    "continuum_fission_dag",
    "continuum_claim_and_fence",
    "continuum_verify_record",
    "continuum_compile_cartridge",
    "continuum_analyze_spectrum",
    "continuum_surgical_edit",
    "continuum_coupling_step",
    "continuum_morphism_transfer",
    "continuum_calibrate_effort",
    "continuum_manifold_status",
    "continuum_context_grep",
    "continuum_attractor_recall",
    "continuum_sheaf_consistency_audit",
    "continuum_conformal_calibrate",
    "continuum_hyperlora_modulate",
    "continuum_probe_latent_state",
    "continuum_steer_nullspace_vector",
    "continuum_compose_omninode",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_obj(obj: Any) -> str:
    return _sha256_text(_canonical_json(obj))


def _words(text: Any) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


class OmniNodeFailure(Exception):
    """Typed failure inside an OmniNode pipeline so waterfall/quorum directors react deterministically."""

    def __init__(self, why: str, kind: str = "capability") -> None:
        super().__init__(why)
        self.kind = kind


class ShapeSyntaxError(OmniNodeFailure):
    """Raised when a declarative shape expression cannot be tokenized or parsed."""

    def __init__(self, why: str) -> None:
        super().__init__(why, "bad_input")


class StaticAstCardInspector:
    """Sub-Engine 29.1 — Reads module CARD descriptors via static Python AST without executing code."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()

    @staticmethod
    def inspect_source(source_text: str, source_file: str = "<memory>") -> dict[str, Any] | None:
        """Extract top-level CARD literal from Python source text without importing or running it."""
        tree = ast.parse(source_text, filename=source_file)
        src_hash = _sha256_text(source_text)
        for node in tree.body:
            raw_card: Any = None
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CARD" for t in node.targets
            ):
                raw_card = ast.literal_eval(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "CARD"
                and node.value is not None
            ):
                raw_card = ast.literal_eval(node.value)
            if isinstance(raw_card, dict):
                card_doc = {
                    "name": str(raw_card.get("name", Path(source_file).stem)),
                    "role": str(raw_card.get("role", "module")),
                    "stratum": str(
                        raw_card.get("stratum", "omninode_shell_and_shape_composition")
                    ),
                    "describes": str(raw_card.get("describes", "")),
                    "keywords": sorted(str(k) for k in raw_card.get("keywords", [])),
                    "contract": str(raw_card.get("contract", "continuum.module/v1")),
                    "source_file": source_file,
                    "source_sha256": src_hash,
                }
                card_doc["sha256"] = _sha256_obj(card_doc)
                return card_doc
        return None

    def inspect_python_file(self, py_path: Path) -> dict[str, Any] | None:
        """Extract top-level CARD literal from py_path via static AST without importing or running it."""
        return self.inspect_source(py_path.read_text(encoding="utf-8"), str(py_path))

    def inspect_repository_modules(self, repo_root: Path) -> dict[str, dict[str, Any]]:
        """Scan Python modules in repo_root (excluding tests) and return static AST CARDs."""
        cards: dict[str, dict[str, Any]] = {}
        skip_parts = {".git", "__pycache__", ".pytest_cache", ".omni", "tests"}
        for py_path in sorted(repo_root.rglob("*.py")):
            if any(part in skip_parts for part in py_path.parts):
                continue
            rel = py_path.relative_to(repo_root).as_posix()
            text = py_path.read_text(encoding="utf-8")
            found = self.inspect_source(text, rel)
            if found is not None:
                cards[found["name"]] = found
        return cards

    def build_full_catalog(
        self,
        repo_root: Path | None = None,
        tool_descriptions: Mapping[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build self-describing CARD descriptors for all 30 engines, 18 tools, 5 surfaces, ops, and modules."""
        root = repo_root or Path(__file__).resolve().parent.parent
        module_cards = self.inspect_repository_modules(root)
        catalog: dict[str, dict[str, Any]] = {}

        for m_name, m_card in module_cards.items():
            catalog[f"module::{m_name}"] = m_card

        # 1. All 30 Functional Cybernetic Engines (112 Sub-Engines)
        for eng in self._spec.section("engines"):
            eid = str(eng["engine_id"])
            axis_names = [str(p["name"]) for p in eng["node_grid"]["parameters"]]
            sub_list = [str(s) for s in eng["sub_engines"]]
            kw = sorted(
                set(_words(eid) + _words(eng["stratum"]) + sub_list + axis_names)
            )
            doc = {
                "name": eid,
                "role": "engine",
                "stratum": str(eng["stratum"]),
                "class_name": str(eng["class_name"]),
                "sub_engines": sub_list,
                "parameter_axes": axis_names,
                "describes": str(eng["description"]),
                "keywords": kw,
                "contract": "continuum.engine/v1",
            }
            doc["sha256"] = _sha256_obj(doc)
            catalog[eid] = doc

        # 2. All 18 continuum_* Agent Tools
        desc_lookup = dict(tool_descriptions or {})
        for t_name in TOOL_NAMES_CANONICAL:
            desc = desc_lookup.get(
                t_name,
                f"Native Hermes agent tool {t_name} encased as a deterministic OmniNode.",
            )
            doc = {
                "name": t_name,
                "role": "tool",
                "stratum": "omninode_shell_and_shape_composition",
                "describes": str(desc),
                "keywords": sorted(set(_words(t_name) + _words(desc))),
                "contract": "continuum.tool/v1",
            }
            doc["sha256"] = _sha256_obj(doc)
            catalog[t_name] = doc

        # 3. All 5 Native Hermes Extension Surfaces
        for surf in self._omni_spec.section("hermes_surfaces"):
            s_name = str(surf["name"])
            rel_file = str(surf["source_file"])
            src_path = root / rel_file
            src_sha = (
                _sha256_text(src_path.read_text(encoding="utf-8"))
                if src_path.is_file()
                else _sha256_text(s_name)
            )
            doc = {
                "name": s_name,
                "role": "surface",
                "stratum": str(surf["stratum"]),
                "source_file": rel_file,
                "source_sha256": src_sha,
                "describes": str(surf["describes"]),
                "keywords": sorted(str(k) for k in surf["keywords"]),
                "contract": str(surf["contract"]),
            }
            doc["sha256"] = _sha256_obj(doc)
            catalog[s_name] = doc

        # 4. Composable Shape Stage Primitives
        for op in self._omni_spec.section("shape_stage_ops"):
            op_name = str(op["name"])
            doc = {
                "name": op_name,
                "role": str(op["role"]),
                "stratum": str(op["stratum"]),
                "backing_engine": str(op["backing_engine"]),
                "describes": str(op["describes"]),
                "keywords": sorted(str(k) for k in op["keywords"]),
                "contract": str(op["contract"]),
            }
            doc["sha256"] = _sha256_obj(doc)
            catalog[op_name] = doc

        return catalog


class ContentAddressedCasLockfile:
    """Sub-Engine 29.2 — Content-addressed .omni/cas/<sha256>.json store, append-only facts, and lockfile verifier."""

    def __init__(
        self,
        state_root: Path | None = None,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        o_cfg = self._spec.section("omninode_shell")
        short_n = int(
            self._omni_spec.section("grid1_numeric_bounds")["cas_short_hash_chars"]
        )
        ver_tag = _sha256_text(
            f"{self._spec.section('version')}:{Path(__file__).stat().st_mtime_ns}"
        )[:short_n]
        base = (
            Path(state_root)
            if state_root is not None
            else Path(tempfile.gettempdir()) / f"continuum_omni_{ver_tag}"
        )
        self.omni_dir = base / str(o_cfg["state_subdir"])
        self.cas_dir = self.omni_dir / str(o_cfg["cas_subdir"])
        self.lockfile_path = self.omni_dir / str(o_cfg["lockfile_name"])
        self.facts_path = self.omni_dir / str(o_cfg["facts_log_name"])
        self._io_lock = threading.RLock()
        self.cas_dir.mkdir(parents=True, exist_ok=True)

    def record_fact(self, what: str, **fields: Any) -> dict[str, Any]:
        """Append one immutable fact record to .omni/facts.jsonl."""
        precision = int(
            self._omni_spec.section("grid1_numeric_bounds")[
                "default_float_precision_digits"
            ]
        )
        rec = dict(fields)
        rec["fact"] = what
        rec["t"] = round(time.time(), precision)
        line = _canonical_json(rec) + "\n"
        with self._io_lock:
            self.omni_dir.mkdir(parents=True, exist_ok=True)
            with open(self.facts_path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        return rec

    def read_facts(self) -> list[dict[str, Any]]:
        if not self.facts_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.facts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def pin_catalog(self, catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Store every CARD in .omni/cas/<sha256>.json and writeomninode.lock.json if not yet created."""
        with self._io_lock:
            self.cas_dir.mkdir(parents=True, exist_ok=True)
            nodes_pin: dict[str, dict[str, Any]] = {}
            for name in sorted(catalog.keys()):
                card = dict(catalog[name])
                digest = str(card["sha256"])
                cas_file = self.cas_dir / f"{digest}.json"
                serialized = _canonical_json(card)
                if not cas_file.is_file():
                    tmp_file = cas_file.with_suffix(f".{os.getpid()}.tmp")
                    tmp_file.write_text(serialized, encoding="utf-8")
                    os.replace(tmp_file, cas_file)
                nodes_pin[name] = {
                    "name": name,
                    "role": str(card.get("role", "worker")),
                    "stratum": str(card.get("stratum", "")),
                    "contract": str(card.get("contract", "omninode/v1")),
                    "sha256": digest,
                    "cas_file": f"cas/{digest}.json",
                }
            lock_doc = {
                "schema": "omninode.lock/v1",
                "version": str(self._spec.section("version")),
                "nodes": nodes_pin,
            }
            lock_doc["lockfile_sha256"] = _sha256_obj(nodes_pin)
            if not self.lockfile_path.is_file():
                tmp_lock = self.lockfile_path.with_suffix(f".{os.getpid()}.tmp")
                tmp_lock.write_text(
                    json.dumps(lock_doc, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp_lock, self.lockfile_path)
                self.record_fact(
                    "lockfile_initialized",
                    lockfile_sha256=lock_doc["lockfile_sha256"],
                    node_count=len(nodes_pin),
                )
            return self.read_lockfile()

    def read_lockfile(self) -> dict[str, Any]:
        if not self.lockfile_path.is_file():
            return {}
        try:
            return json.loads(self.lockfile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OmniNodeFailure(
                f"corrupted OmniNode lockfile: {exc}", "contract"
            ) from exc

    def verify_integrity(
        self, live_catalog: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Verify lockfile pins against both CAS blobs and live CARD digests; fail closed on mismatch."""
        lock_doc = self.read_lockfile()
        if not lock_doc or "nodes" not in lock_doc:
            raise OmniNodeFailure("missing OmniNode lockfile nodes table", "contract")
        pinned_nodes: Mapping[str, Any] = lock_doc["nodes"]
        expected_lock_sha = _sha256_obj(pinned_nodes)
        if str(lock_doc.get("lockfile_sha256", "")) != expected_lock_sha:
            self.record_fact(
                "hash_mismatch",
                component="omninode.lock.json",
                kind="contract",
                why="lockfile_sha256 header mismatch",
            )
            raise OmniNodeFailure(
                "omninode.lock.json integrity digest mismatch", "contract"
            )

        verified_pins: dict[str, str] = {}
        for name, pin in pinned_nodes.items():
            pin_sha = str(pin.get("sha256", ""))
            if name not in live_catalog:
                self.record_fact(
                    "hash_mismatch",
                    component=name,
                    kind="contract",
                    why="pinned component missing from live catalog",
                )
                raise OmniNodeFailure(
                    f"pinned component '{name}' missing from live catalog", "contract"
                )
            live_card = dict(live_catalog[name])
            if str(live_card["sha256"]) != pin_sha:
                self.record_fact(
                    "hash_mismatch",
                    component=name,
                    expected=pin_sha,
                    actual=str(live_card["sha256"]),
                    kind="contract",
                )
                raise OmniNodeFailure(
                    f"hash_mismatch for '{name}': lockfile pins {pin_sha} but live CARD is {live_card['sha256']}",
                    "contract",
                )
            cas_file = self.cas_dir / f"{pin_sha}.json"
            if not cas_file.is_file():
                self.record_fact(
                    "hash_mismatch",
                    component=name,
                    kind="contract",
                    why="missing CAS entry",
                )
                raise OmniNodeFailure(
                    f"missing CAS object for '{name}' ({pin_sha})", "contract"
                )
            try:
                cas_card = json.loads(cas_file.read_text(encoding="utf-8"))
            except ValueError as exc:
                self.record_fact(
                    "hash_mismatch",
                    component=name,
                    kind="contract",
                    why="unparseable CAS JSON",
                )
                raise OmniNodeFailure(
                    f"corrupted CAS JSON for '{name}'", "contract"
                ) from exc
            cas_recomputed = _sha256_obj(
                {k: v for k, v in cas_card.items() if k != "sha256"}
            )
            if str(cas_card.get("sha256", "")) != pin_sha or cas_recomputed != pin_sha:
                self.record_fact(
                    "hash_mismatch",
                    component=name,
                    kind="contract",
                    why="tampered CAS content",
                )
                raise OmniNodeFailure(
                    f"hash_mismatch in CAS blob for '{name}' ({pin_sha})", "contract"
                )
            verified_pins[name] = pin_sha

        return {
            "verified": True,
            "lockfile_sha256": expected_lock_sha,
            "total_cards": len(verified_pins),
            "pinned_components": verified_pins,
        }


class TypedFailureWaterfallRouter:
    """Sub-Engine 29.3 — Classifies exceptions into FAILURE_KINDS and governs // waterfall routing."""

    def __init__(self, omni_spec: OmniNodeGridSpec | None = None) -> None:
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        self.failure_kinds: tuple[str, ...] = tuple(
            str(k) for k in self._omni_spec.section("failure_kinds")
        )

    def classify(self, err: BaseException) -> str:
        if isinstance(err, OmniNodeFailure) and err.kind in self.failure_kinds:
            return err.kind
        if isinstance(err, TimeoutError):
            return "transient"
        if isinstance(err, (FileNotFoundError, KeyError, StopIteration, ValueError)):
            if isinstance(err, (KeyError, FileNotFoundError, StopIteration)):
                return "data"
            return "bad_input"
        if isinstance(err, TypeError):
            return "bad_input"
        return "capability"

    def can_waterfall(self, failure_kind: str, policy: str) -> bool:
        """Contract violations never waterfall; other kinds follow the configured NodeGrid policy."""
        if failure_kind == "contract":
            return False
        if policy == "strict_fail_closed":
            return failure_kind in ("transient", "capability")
        if policy == "waterfall_all_non_contract":
            return True
        # Default "retry_transient_then_waterfall": allow transient, capability, data, budget
        return failure_kind in ("transient", "capability", "data", "budget")


class CapabilityGapSynthesizer:
    """Sub-Engine 29.4 — IDF lexical + 4-space Matryoshka CARD search and capability-gap synthesizer."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()

    def search_cards(
        self,
        catalog: Mapping[str, Mapping[str, Any]],
        query: str,
        *,
        bus: CyberneticWeightBus | None = None,
        slice_dim: int | None = None,
        exclude: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        q_words = set(_words(query))
        excluded = set(exclude)
        searchable = [
            c
            for k, c in catalog.items()
            if not k.startswith("module::") and c["name"] not in excluded
        ]
        if not searchable:
            return []

        df: dict[str, int] = {}
        for card in searchable:
            vocab = (
                set(_words(card.get("name", "")))
                | set(_words(card.get("describes", "")))
                | set(str(w).lower() for w in card.get("keywords", []))
            )
            for token in vocab:
                df[token] = df.get(token, 0) + 1

        use_dim = (
            int(slice_dim)
            if slice_dim is not None
            else int(self._spec.section("matryoshka_manifold")["fine_dim"])
        )
        q_vec = bus.manifold.embed_task(query) if bus is not None else None
        precision = int(
            self._omni_spec.section("grid1_numeric_bounds")[
                "default_float_precision_digits"
            ]
        )
        half = self._spec.c_float("half")

        ranked: list[dict[str, Any]] = []
        total_n = float(len(searchable))
        for card in searchable:
            vocab = (
                set(_words(card.get("name", "")))
                | set(_words(card.get("describes", "")))
                | set(str(w).lower() for w in card.get("keywords", []))
            )
            hits = q_words & vocab
            lex_score = sum(
                math.log(1.0 + total_n / float(df[w])) for w in hits
            )
            emb_bonus = 0.0
            if q_vec is not None and bus is not None:
                c_text = f"{card.get('name', '')} {card.get('describes', '')}"
                c_vec = bus.manifold.embed_task(c_text, stratum=str(card.get("stratum", "")))
                cos_val = max(0.0, q_vec.cosine(c_vec, use_dim))
                emb_bonus = half * cos_val if hits else 0.0
            score = lex_score + emb_bonus
            ranked.append(
                {
                    "component": str(card["name"]),
                    "role": str(card.get("role", "worker")),
                    "score": round(score, precision),
                    "matched": sorted(hits),
                    "sha256": str(card["sha256"]),
                }
            )
        ranked.sort(key=lambda r: (-float(r["score"]), str(r["component"])))
        return ranked

    def synthesize_gap(
        self,
        step_name: str,
        ranked_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        max_alt = int(
            self._omni_spec.section("grid1_numeric_bounds")["max_search_alternatives"]
        )
        nearest = list(ranked_candidates[:max_alt])
        fallback_chain = (
            f"orient >> {nearest[0]['component']} // fallback_compress"
            if nearest
            else "orient >> fallback_compress"
        )
        return {
            "missing_capability": step_name,
            "required_contract": "omninode.op/v1",
            "nearest_cards": nearest,
            "proposed_composite_shape": fallback_chain,
        }


class SequentialPipeOperator:
    """Sub-Engine 30.1 — Recursive-descent parser and balanced sequential pipe >> operator."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        self.precedence: tuple[str, ...] = tuple(
            str(op) for op in self._omni_spec.section("operator_precedence")
        )

    def tokenize(self, shape_text: str) -> list[str]:
        cleaned = (shape_text or "").strip()
        if not cleaned:
            raise ShapeSyntaxError("shape expression must not be empty")
        tokens: list[str] = []
        pos = 0
        end = len(shape_text.rstrip())
        while pos < end:
            match = _SHAPE_TOKEN_RE.match(shape_text, pos)
            if not match:
                raise ShapeSyntaxError(
                    f"invalid token near {shape_text[pos:pos + self._spec.c_int('sixteen')]!r}"
                )
            tokens.append(match.group(1))
            pos = match.end()
        return tokens

    def parse(self, shape_text: str) -> Any:
        tokens = self.tokenize(shape_text)
        cursor = [0]
        ops = set(self.precedence)

        def peek() -> str | None:
            return tokens[cursor[0]] if cursor[0] < len(tokens) else None

        def take() -> str:
            tok = tokens[cursor[0]]
            cursor[0] += 1
            return tok

        def parse_atom() -> Any:
            tok = take() if peek() is not None else None
            if tok == "(":
                inner = parse_level(0)
                if peek() != ")":
                    raise ShapeSyntaxError("unclosed '(' in shape expression")
                take()
                return inner
            if tok is None or tok in ops or tok == ")":
                raise ShapeSyntaxError(f"expected step identifier, found {tok!r}")
            return tok

        def parse_level(idx: int) -> Any:
            if idx == len(self.precedence):
                return parse_atom()
            current_op = self.precedence[idx]
            left = parse_level(idx + 1)
            while peek() == current_op:
                take()
                right = parse_level(idx + 1)
                left = (current_op, left, right)
            return left

        tree = parse_level(0)
        if peek() is not None:
            raise ShapeSyntaxError(f"unexpected trailing token {peek()!r}")
        return tree

    def balance(self, tree: Any) -> Any:
        if isinstance(tree, str):
            return tree
        op, left, right = tree
        if op != ">>":
            return (op, self.balance(left), self.balance(right))
        chain: list[Any] = []

        def collect(node: Any) -> None:
            if isinstance(node, tuple):
                op_child, left_child, right_child = node
                if op_child == ">>":
                    collect(left_child)
                    collect(right_child)
                    return
            chain.append(self.balance(node))

        collect(tree)
        two = self._spec.c_int("two")

        def fold(items: list[Any]) -> Any:
            if len(items) == 1:
                return items[0]
            mid = len(items) // two
            return (">>", fold(items[:mid]), fold(items[mid:]))

        return fold(chain)

    @staticmethod
    def leaves(tree: Any) -> list[str]:
        if isinstance(tree, str):
            return [tree]
        _, left_child, right_child = tree
        return [
            *SequentialPipeOperator.leaves(left_child),
            *SequentialPipeOperator.leaves(right_child),
        ]

    @staticmethod
    def serialize_tree(tree: Any) -> Any:
        if isinstance(tree, str):
            return tree
        op_tok, left_child, right_child = tree
        return [
            op_tok,
            SequentialPipeOperator.serialize_tree(left_child),
            SequentialPipeOperator.serialize_tree(right_child),
        ]


class ParallelSheafFanoutOperator:
    """Sub-Engine 30.2 — Parallel | fanout operator with Cellular Sheaf coboundary consistency auditing."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    @staticmethod
    def _branch_label(branch_node: Any, index: int) -> str:
        return branch_node if isinstance(branch_node, str) else f"branch_{index}"

    @staticmethod
    def _extract_latent_stalk(branch_out: Mapping[str, Any]) -> list[float] | None:
        for key in (
            "erased_vector",
            "steered_residual_vector",
            "steering_vector",
            "compressed_vector",
            "latent_vector",
        ):
            val = branch_out.get(key)
            if isinstance(val, list) and val and all(isinstance(x, (int, float)) for x in val):
                return [float(x) for x in val]
        return None

    def merge_branches(
        self,
        base_env: Mapping[str, Any],
        branch_results: Sequence[tuple[str, bool, dict[str, Any]]],
        *,
        bus: CyberneticWeightBus,
        sheaf_audit_enabled: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        merged = dict(base_env)
        branches_map: dict[str, Any] = {}
        oks: list[bool] = []
        claims: list[dict[str, Any]] = []

        for label, ok, out in branch_results:
            oks.append(ok)
            branches_map[label] = out
            for k, v in out.items():
                if k != "parallel_branches":
                    merged[k] = v
            stalk = self._extract_latent_stalk(out)
            if stalk is not None:
                claims.append({"agent_id": label, "state_vector": stalk})
            else:
                claims.append({"agent_id": label, "text": _canonical_json(out)})

        merged["parallel_branches"] = branches_map
        if sheaf_audit_enabled and len(claims) >= self._spec.c_int("two"):
            sheaf = bus.sheaf_consistency_audit(claims)
            merged["parallel_sheaf_audit"] = {
                "coboundary_energy": sheaf["global_obstruction_energy"],
                "post_consensus_energy": sheaf["post_repair_obstruction_energy"],
                "is_globally_consistent": sheaf["consistent"],
                "harmonic_consensus_stalks": sheaf["harmonic_consensus_stalks"],
            }
        return all(oks), merged


class IndependentQuorumGateOperator:
    """Sub-Engine 30.3 — Independent @ quorum check enforcing maker_id != checker_id."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def verify_handoff(
        self,
        maker_output: Mapping[str, Any],
        *,
        maker_id: str,
        checker_id: str,
        min_quality_floor: float,
        enforcement_mode: str = "strict_distinct_actors",
        store: ContinuumStore | None = None,
        shape_text: str = "",
    ) -> dict[str, Any]:
        m_clean = (maker_id or "").strip()
        c_clean = (checker_id or "").strip()
        if not m_clean or not c_clean:
            raise OmniNodeFailure(
                "independent @ quorum check requires non-empty maker_id and checker_id",
                "contract",
            )
        if m_clean.lower() == c_clean.lower():
            raise OmniNodeFailure(
                f"independent @ quorum check rejected identical maker and checker '{m_clean}'",
                "contract",
            )

        drift = float(maker_output.get("nullspace_anchor_drift", 0.0) or 0.0)
        tol = self._spec.c_float("anchor_drift_zero_tol")
        drift_ok = drift <= tol

        sheaf_ok = True
        if enforcement_mode == "sheaf_and_distinct_actors":
            sheaf_info = maker_output.get("parallel_sheaf_audit")
            if isinstance(sheaf_info, dict):
                sheaf_ok = bool(sheaf_info.get("is_globally_consistent", True))

        raw_quality = float(maker_output.get("quality_score", 1.0) or 1.0)
        half = self._spec.c_float("half")
        quality = raw_quality if drift_ok else half * raw_quality
        passed = bool(drift_ok and sheaf_ok and quality >= min_quality_floor)

        persisted_record: dict[str, Any] | None = None
        if store is not None:
            dag = store.fission_dag(
                f"omninode_shape::{shape_text or 'pipeline'}",
                subtasks=[
                    {
                        "title": "Execute composed OmniNode pipeline",
                        "contract": "omninode.check/v1",
                    }
                ],
            )
            task_id = str(dag["tasks"][0]["task_id"])
            store.claim_and_fence(task_id, m_clean)
            persisted_record = store.verify_record(
                task_id,
                maker_id=m_clean,
                checker_id=c_clean,
                passed=passed,
                quality_score=quality,
                output_summary=f"OmniNode @ quorum check ({shape_text})",
                tier_id="verify_tier",
            )

        if not passed:
            raise OmniNodeFailure(
                f"quorum gate failed (drift_ok={drift_ok}, sheaf_ok={sheaf_ok}, quality={quality})",
                "contract",
            )

        out = dict(maker_output)
        out["check_state"] = "verified_independent_quorum"
        out["quorum_verdict"] = {
            "passed": passed,
            "maker_id": m_clean,
            "checker_id": c_clean,
            "quality_score": quality,
            "anchor_drift_zero": drift_ok,
            "enforcement_mode": enforcement_mode,
            "verification_id": (
                persisted_record.get("verification_id")
                if isinstance(persisted_record, dict)
                else None
            ),
        }
        return out


class Grid1CanonicalCellRecorder:
    """Sub-Engine 30.4 — Normalizes full run configs and appends format='grid/1' SHA-256 cell rows."""

    def __init__(
        self,
        state_root: Path | None = None,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        o_cfg = self._spec.section("omninode_shell")
        short_n = int(
            self._omni_spec.section("grid1_numeric_bounds")["cas_short_hash_chars"]
        )
        ver_tag = _sha256_text(
            f"{self._spec.section('version')}:{Path(__file__).stat().st_mtime_ns}"
        )[:short_n]
        base = (
            Path(state_root)
            if state_root is not None
            else Path(tempfile.gettempdir()) / f"continuum_omni_{ver_tag}"
        )
        self.omni_dir = base / str(o_cfg["state_subdir"])
        self.records_path = self.omni_dir / str(o_cfg["grid_records_name"])
        self._io_lock = threading.RLock()

        self.format_tag: str = str(self._omni_spec.section("format"))
        self.unknown_sentinel: str = str(self._omni_spec.section("unknown_sentinel"))
        self.axes_spec: dict[str, dict[str, Any]] = dict(
            self._omni_spec.section("grid1_axes")
        )
        self.applies_spec: dict[str, str] = dict(
            self._omni_spec.section("grid1_applies")
        )
        self.allowed_enums: dict[str, list[str]] = dict(
            self._omni_spec.section("grid1_allowed_enums")
        )
        self.outcome_fields: tuple[str, ...] = tuple(
            str(f) for f in self._omni_spec.section("grid1_outcome_fields")
        )
        self.bounds: dict[str, Any] = dict(
            self._omni_spec.section("grid1_numeric_bounds")
        )

    def _norm_value(self, axis_name: str, val: Any, precision_digits: int) -> Any:
        if val is None or val == self.unknown_sentinel:
            return val
        typ = str(self.axes_spec[axis_name]["type"])
        if typ == "num":
            return round(float(val), precision_digits)
        if typ == "int":
            return int(val)
        if typ == "bool":
            return bool(val)
        if typ == "list":
            seq = val if isinstance(val, (list, tuple, set)) else [val]
            return sorted(str(x) for x in seq)
        if typ == "map":
            if isinstance(val, Mapping):
                return {str(k): val[k] for k in sorted(val.keys())}
            return val
        return str(val)

    def _norm_extra(self, val: Any, precision_digits: int) -> Any:
        if isinstance(val, bool) or val is None:
            return val
        if isinstance(val, int):
            return int(val)
        if isinstance(val, float):
            return round(float(val), precision_digits)
        if isinstance(val, Mapping):
            return {
                str(k): self._norm_extra(val[k], precision_digits)
                for k in sorted(val.keys())
            }
        if isinstance(val, (list, tuple)):
            return [self._norm_extra(x, precision_digits) for x in val]
        return str(val)

    def normalize(
        self,
        config: Mapping[str, Any] | None,
        unknown: Sequence[str] = (),
        *,
        precision_digits: int | None = None,
    ) -> dict[str, Any]:
        """Normalize every grid/1 axis, apply conditional defaults, preserve unknowns, and sort extra keys."""
        prec = (
            int(precision_digits)
            if precision_digits is not None
            else int(self.bounds["default_float_precision_digits"])
        )
        raw = dict(config or {})
        unknown_set = set(unknown or ())
        bad_unknown = sorted(k for k in unknown_set if k not in self.axes_spec)
        if bad_unknown:
            raise OmniNodeFailure(
                f"unknown names that are not registered grid/1 axes: {bad_unknown}",
                "bad_input",
            )

        out: dict[str, Any] = {}
        for axis_name in self.axes_spec:
            if axis_name in unknown_set and raw.get(axis_name) in (
                None,
                self.unknown_sentinel,
            ):
                out[axis_name] = self.unknown_sentinel
            else:
                out[axis_name] = self._norm_value(axis_name, raw.get(axis_name), prec)

        for axis_name, dep_axis in self.applies_spec.items():
            if (
                out[axis_name] is None
                and out.get(dep_axis) is not None
                and axis_name not in unknown_set
            ):
                out[axis_name] = self.axes_spec[axis_name]["default"]

        extra = {
            str(k): self._norm_extra(raw[k], prec)
            for k in sorted(raw.keys())
            if k not in self.axes_spec
        }
        if extra:
            out["extra"] = extra
        return out

    def cell_id(
        self,
        config: Mapping[str, Any] | None,
        unknown: Sequence[str] = (),
        *,
        precision_digits: int | None = None,
    ) -> str:
        normed = self.normalize(
            config, unknown=unknown, precision_digits=precision_digits
        )
        return _sha256_obj(normed)

    def check_invalid(
        self,
        config: Mapping[str, Any] | None,
        unknown: Sequence[str] = (),
        *,
        precision_digits: int | None = None,
    ) -> list[dict[str, str]]:
        c = self.normalize(config, unknown=unknown, precision_digits=precision_digits)

        def is_set(v: Any) -> bool:
            return v not in (None, self.unknown_sentinel)

        bad: list[dict[str, str]] = []
        if not is_set(c.get("model")) and any(
            is_set(c.get(k)) for k in ("max_output", "context_window", "model_digest")
        ):
            bad.append(
                {
                    "rule": "model-settings-need-model",
                    "why": "model settings provided without a model",
                }
            )
        if is_set(c.get("temperature")):
            t_val = float(c["temperature"])
            if not (
                float(self.bounds["temperature_min"])
                <= t_val
                <= float(self.bounds["temperature_max"])
            ):
                bad.append(
                    {
                        "rule": "temperature-range",
                        "why": "temperature outside allowed range",
                    }
                )
        if is_set(c.get("top_k")) and c.get("retrieval_method") in (None, "none"):
            bad.append(
                {
                    "rule": "top-k-needs-retrieval",
                    "why": "top_k specified without retrieval_method",
                }
            )
        if (
            c.get("retrieval_method") in ("embedding", "hybrid")
            and c.get("embedding_model") is None
        ):
            bad.append(
                {
                    "rule": "embedding-retrieval-needs-model",
                    "why": "embedding retrieval requires embedding_model",
                }
            )
        if not is_set(c.get("browser_name")) and any(
            is_set(c.get(k)) for k in ("browser_version", "viewport")
        ):
            bad.append(
                {
                    "rule": "browser-settings-need-browser",
                    "why": "browser settings provided without browser_name",
                }
            )
        for enum_axis, allowed in self.allowed_enums.items():
            if is_set(c.get(enum_axis)) and c[enum_axis] not in allowed:
                bad.append(
                    {
                        "rule": f"{enum_axis}-values",
                        "why": f"{enum_axis} must be one of {allowed}",
                    }
                )
        if is_set(c.get("sparsity")):
            s_val = float(c["sparsity"])
            if not (
                float(self.bounds["sparsity_min"])
                <= s_val
                <= float(self.bounds["sparsity_max"])
            ):
                bad.append(
                    {"rule": "sparsity-range", "why": "sparsity outside [0, 1]"}
                )
        return bad

    def record_run(
        self,
        *,
        kind: str,
        source: str,
        config: Mapping[str, Any],
        outcome: Mapping[str, Any],
        unknown: Sequence[str] = (),
        question: str | None = None,
        design: str = "omninode-shape",
        extra: Mapping[str, Any] | None = None,
        precision_digits: int | None = None,
    ) -> dict[str, Any]:
        run_hex = int(self.bounds["run_id_hex_chars"])
        unknown_sorted = sorted(set(unknown or ()))
        normed = self.normalize(
            config, unknown=unknown_sorted, precision_digits=precision_digits
        )
        cell = _sha256_obj(normed)
        inv = self.check_invalid(
            config, unknown=unknown_sorted, precision_digits=precision_digits
        )
        oc = {field: outcome.get(field) for field in self.outcome_fields}
        row_doc = {
            "format": self.format_tag,
            "run_id": "run-" + uuid.uuid4().hex[:run_hex],
            "kind": kind,
            "source": source,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "cell": cell,
            "config": normed,
            "unknown": unknown_sorted,
            "invalid": inv,
            "outcome": oc,
            "question": question,
            "question_class": None,
            "repeat": None,
            "group": None,
            "design": design,
            "selection_p": 1.0,
            "imported_from": None,
            "extra": dict(extra or {}),
        }
        with self._io_lock:
            self.omni_dir.mkdir(parents=True, exist_ok=True)
            with open(self.records_path, "a", encoding="utf-8") as fh:
                fh.write(_canonical_json(row_doc) + "\n")
                fh.flush()
        return row_doc

    def count_records(self) -> int:
        if not self.records_path.is_file():
            return 0
        return sum(
            1
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


class OmniNodeShellEncaser:
    """Engine 29 — Encases all 30 engines, 18 tools, and 5 Hermes surfaces as self-describing OmniNodes."""

    def __init__(
        self,
        *,
        bus: CyberneticWeightBus,
        store: ContinuumStore | None = None,
        state_root: Path | None = None,
        repo_root: Path | None = None,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        self.bus = bus
        self.store = store
        resolved_state = (
            Path(state_root)
            if state_root is not None
            else (store.db_path.parent if store is not None else None)
        )
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent

        # 4 Sub-Engines of Engine 29
        self.card_inspector = StaticAstCardInspector(self._spec, self._omni_spec)
        self.cas_lockfile = ContentAddressedCasLockfile(
            resolved_state, self._spec, self._omni_spec
        )
        self.waterfall_router = TypedFailureWaterfallRouter(self._omni_spec)
        self.gap_synthesizer = CapabilityGapSynthesizer(self._spec, self._omni_spec)

        self.catalog: dict[str, dict[str, Any]] = self.card_inspector.build_full_catalog(
            self.repo_root
        )
        self.lock_doc: dict[str, Any] = self.cas_lockfile.pin_catalog(self.catalog)

    def refresh_catalog(
        self, tool_descriptions: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        self.catalog = self.card_inspector.build_full_catalog(
            self.repo_root, tool_descriptions=tool_descriptions
        )
        # Ensure any newly added descriptive fields exist in CAS; if lockfile not yet created, pin it
        return self.cas_lockfile.pin_catalog(self.catalog)

    def resolve_step_card(
        self,
        step_token: str,
        *,
        min_search_score: float,
        slice_dim: int,
    ) -> tuple[dict[str, Any], str]:
        """Resolve a step token to a pinned OmniNode CARD by exact ID, alias, or capability search."""
        aliases = {
            "probe_latent": "sae_probe",
            "steer_nullspace": "nullspace_steer",
            "ib_compress": "fallback_compress",
            "context_compress": "fallback_compress",
            "conformal_calibrate": "conformal_gate",
            "hyperlora_modulate": "compile_cartridge",
            "check": "independent_quorum_check",
            "quorum_check": "independent_quorum_check",
        }
        canonical = aliases.get(step_token, step_token)
        if canonical in self.catalog:
            card = self.catalog[canonical]
            self.cas_lockfile.record_fact(
                "chosen",
                component=card["name"],
                sha256=card["sha256"],
                how="named_in_shape",
            )
            return card, "named_in_shape"

        # Explicit typed fault-injection probes for testing waterfall & contract routing
        if step_token.startswith("fail_"):
            kind = step_token[len("fail_") :]
            if kind in self.waterfall_router.failure_kinds:
                probe_card = {
                    "name": step_token,
                    "role": "fault_probe",
                    "stratum": "omninode_shell_and_shape_composition",
                    "describes": f"Deterministic typed {kind} fault probe for waterfall verification.",
                    "keywords": ["fault", "probe", kind],
                    "contract": "omninode.probe/v1",
                }
                probe_card["sha256"] = _sha256_obj(probe_card)
                return probe_card, "fault_probe"

        ranked = self.gap_synthesizer.search_cards(
            self.catalog,
            step_token.replace("_", " "),
            bus=self.bus,
            slice_dim=slice_dim,
        )
        if ranked and float(ranked[0]["score"]) >= min_search_score:
            chosen_name = str(ranked[0]["component"])
            card = self.catalog[chosen_name]
            self.cas_lockfile.record_fact(
                "chosen",
                component=chosen_name,
                sha256=card["sha256"],
                how="capability_search",
                query=step_token,
            )
            return card, "capability_search"

        gap = self.gap_synthesizer.synthesize_gap(step_token, ranked)
        self.cas_lockfile.record_fact(
            "capability_gap",
            step=step_token,
            proposed=gap["proposed_composite_shape"],
        )
        raise OmniNodeFailure(
            f"capability gap for step '{step_token}'; proposed={gap['proposed_composite_shape']}",
            "capability",
        )


class AlgebraicShapeAndGrid1Director:
    """Engine 30 — Executes >> // | @ OmniNode compositions and records canonical grid/1 SHA-256 cells."""

    def __init__(
        self,
        encaser: OmniNodeShellEncaser,
        *,
        spec: ManifoldConfigSpec | None = None,
        omni_spec: OmniNodeGridSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._omni_spec = omni_spec or get_omninode_grid_spec()
        self.encaser = encaser
        state_parent = encaser.cas_lockfile.omni_dir.parent

        # 4 Sub-Engines of Engine 30
        self.pipe_op = SequentialPipeOperator(self._spec, self._omni_spec)
        self.parallel_op = ParallelSheafFanoutOperator(self._spec)
        self.quorum_op = IndependentQuorumGateOperator(self._spec)
        self.grid1_recorder = Grid1CanonicalCellRecorder(
            state_parent, self._spec, self._omni_spec
        )
        self._tool_handlers: dict[str, Callable[[dict[str, Any]], str]] = {}
        self.last_grid1_cell: str = self._compute_baseline_cell()

    def register_tool_handlers(
        self, handlers: Mapping[str, Callable[[dict[str, Any]], str]]
    ) -> None:
        self._tool_handlers.update(handlers)

    def _compute_baseline_cell(self) -> str:
        pins = {
            k: v["sha256"]
            for k, v in self.encaser.lock_doc.get("nodes", {}).items()
            if not k.startswith("module::")
        }
        cfg = {
            "harness": "hermes-continuum-omninode",
            "harness_version": str(self._spec.section("version")),
            "place": "local",
            "hardware": "cpu",
            "retrieval_method": "hybrid",
            "embedding_model": "matryoshka-4space-128d",
            "tools": list(TOOL_NAMES_CANONICAL),
            "skills": ["continuum"],
            "components": pins,
        }
        return self.grid1_recorder.cell_id(cfg)

    def apply_grid_overrides(
        self, overrides: Mapping[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        """Validate and clamp user grid_overrides against ParameterBounds and NodeGrid constraints."""
        clamped_by_engine: dict[str, dict[str, Any]] = {}
        if not overrides:
            return clamped_by_engine
        dotted_aliases: dict[str, tuple[str, str]] = {
            "sae.topk": ("sparse_feature_dictionary", "topk_active_latents"),
            "sae.jumprelu": ("sparse_feature_dictionary", "jumprelu_theta"),
            "steer.strength": (
                "nullspace_activation_steerer",
                "steering_strength_alpha",
            ),
            "conformal.alpha": (
                "conformal_calibration_engine",
                "target_miscoverage_alpha",
            ),
            "compile.target": (
                "portable_target_lowering_engine",
                "default_target_format",
            ),
        }
        for key, val in overrides.items():
            if key in dotted_aliases:
                eid_alias, param_alias = dotted_aliases[key]
                if eid_alias in self.encaser.bus.hypergrid.grids:
                    grid = self.encaser.bus.hypergrid.grids[eid_alias]
                    if param_alias in grid.parameters:
                        merged = dict(clamped_by_engine.get(eid_alias, {}))
                        merged[param_alias] = val
                        cand = grid.clamp_values(merged)
                        if not grid.check_constraints(cand):
                            raise OmniNodeFailure(
                                f"override '{key}' violated constraints on '{eid_alias}'",
                                "bad_input",
                            )
                        clamped_by_engine[eid_alias] = cand
                        continue
            if isinstance(val, Mapping) and key in self.encaser.bus.hypergrid.grids:
                grid = self.encaser.bus.hypergrid.grids[key]
                cand = grid.clamp_values(dict(val))
                if not grid.check_constraints(cand):
                    raise OmniNodeFailure(
                        f"grid_overrides violated constraints for engine '{key}'",
                        "bad_input",
                    )
                clamped_by_engine[key] = cand
            else:
                for eid, grid in self.encaser.bus.hypergrid.grids.items():
                    if key in grid.parameters:
                        merged = dict(clamped_by_engine.get(eid, {}))
                        merged[key] = val
                        cand = grid.clamp_values(merged)
                        if not grid.check_constraints(cand):
                            raise OmniNodeFailure(
                                f"override '{key}' violated constraints on '{eid}'",
                                "bad_input",
                            )
                        clamped_by_engine[eid] = cand
        return clamped_by_engine

    def _execute_leaf_step(
        self,
        step_token: str,
        env: dict[str, Any],
        *,
        enc_params: Mapping[str, Any],
        trace_log: list[dict[str, Any]],
        used_pins: dict[str, str],
        clamped_overrides: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        t0 = time.time()
        thousand = float(self._spec.c_int("one_thousand"))
        card, how_chosen = self.encaser.resolve_step_card(
            step_token,
            min_search_score=float(enc_params["capability_search_min_score"]),
            slice_dim=int(enc_params["matryoshka_slice_dim"]),
        )
        cname = str(card["name"])
        csha = str(card["sha256"])
        used_pins[cname] = csha

        if bool(enc_params.get("cas_verify_on_dispatch", True)) and cname in self.encaser.catalog:
            cas_path = self.encaser.cas_lockfile.cas_dir / f"{csha}.json"
            if not cas_path.is_file():
                raise OmniNodeFailure(
                    f"CAS blob missing on dispatch for '{cname}'", "contract"
                )

        if card.get("role") == "fault_probe":
            f_kind = cname[len("fail_") :]
            raise OmniNodeFailure(f"injected {f_kind} fault at {cname}", f_kind)

        bus = self.encaser.bus
        goal_text = str(env.get("goal", "") or "omninode pipeline execution")
        out: dict[str, Any] = {}

        try:
            if cname == "orient":
                effort = bus.calibrate_effort(goal_text)
                t_vec = bus.manifold.embed_task(goal_text)
                slice_d = int(effort["recommended_matryoshka_dim"])
                out = {
                    "oriented": True,
                    "effort_calibration": effort,
                    "task2vec_slice": t_vec.slice_prefix(slice_d),
                    "latent_vector": t_vec.slice_prefix(
                        int(self._spec.section("latent_interpretability")["default_latent_dim"])
                    ),
                }
            elif cname == "sae_probe":
                act_in = env.get("activation_vector") or env.get("latent_vector")
                probe = bus.probe_latent_state(
                    activation_vector=act_in if isinstance(act_in, list) else None,
                    trace_text=goal_text,
                    intermediate_vector=env.get("intermediate_vector")
                    if isinstance(env.get("intermediate_vector"), list)
                    else None,
                    concept_name=str(env.get("concept_name", "truthfulness")),
                )
                # Recover unit probe direction from state prober for downstream LEACE erasure
                h_used = (
                    [float(x) for x in act_in]
                    if isinstance(act_in, list) and act_in
                    else bus.manifold.embed_trace(goal_text).slice_prefix(
                        int(probe["latent_dim"])
                    )
                )
                p_state = bus.state_prober_engine.probe_state(
                    h_used, concept_name=str(env.get("concept_name", "truthfulness"))
                )
                out = {
                    "sae_probe": probe,
                    "latent_vector": h_used,
                    "probe_direction": p_state["probe_direction"],
                    "elk_divergence": probe["elk_divergence"],
                    "conformal_truthfulness_set": probe["conformal_truthfulness_set"],
                }
            elif cname == "leace_erase":
                h_in = env.get("latent_vector") or env.get("activation_vector")
                if not isinstance(h_in, list) or not h_in:
                    d_lat = int(
                        self._spec.section("latent_interpretability")["default_latent_dim"]
                    )
                    h_vec = [
                        float(v)
                        for v in bus.manifold.embed_trace(goal_text).slice_prefix(d_lat)
                    ]
                else:
                    h_vec = [float(v) for v in h_in]
                raw_dir = env.get("probe_direction")
                if not isinstance(raw_dir, list) or len(raw_dir) != len(h_vec):
                    p_state = bus.state_prober_engine.probe_state(h_vec)
                    u_hat = l2_normalize([float(v) for v in p_state["probe_direction"]])
                else:
                    u_hat = l2_normalize([float(v) for v in raw_dir])
                proj_coeff = vec_dot(h_vec, u_hat) if vec_norm(u_hat) > 0.0 else 0.0
                erased = [h_vec[i] - proj_coeff * u_hat[i] for i in range(len(h_vec))]
                post_dot = abs(vec_dot(erased, u_hat)) if vec_norm(u_hat) > 0.0 else 0.0
                tol = self._spec.c_float("anchor_drift_zero_tol")
                out = {
                    "erased_vector": erased,
                    "erased_concept_projection": 0.0 if post_dot <= tol else float(post_dot),
                    "leace_removed_magnitude": abs(float(proj_coeff)),
                }
            elif cname == "nullspace_steer":
                steer_ovr = clamped_overrides.get("nullspace_activation_steerer", {})
                res_in = env.get("latent_vector") or env.get("activation_vector")
                steer = bus.steer_nullspace_vector(
                    positive_activations=env.get("positive_activations")
                    if isinstance(env.get("positive_activations"), list)
                    else None,
                    negative_activations=env.get("negative_activations")
                    if isinstance(env.get("negative_activations"), list)
                    else None,
                    positive_prompt=str(
                        env.get("positive_prompt", "") or f"verified {goal_text}"
                    ),
                    negative_prompt=str(
                        env.get("negative_prompt", "") or "drifted unverified output"
                    ),
                    anchor_vectors=env.get("anchor_vectors")
                    if isinstance(env.get("anchor_vectors"), list)
                    else None,
                    residual_vector=res_in if isinstance(res_in, list) else None,
                    steering_strength=float(steer_ovr["steering_strength_alpha"])
                    if "steering_strength_alpha" in steer_ovr
                    else None,
                    target=str(env.get("target", "rust") or "rust"),
                )
                out = {
                    "nullspace_steer": {
                        "cartridge_id": steer["cartridge_id"],
                        "target": steer["target"],
                        "nullspace_anchor_drift": steer["nullspace_anchor_drift"],
                        "raw_anchor_drift": steer["raw_anchor_drift"],
                        "retained_steering_norm_ratio": steer[
                            "retained_steering_norm_ratio"
                        ],
                    },
                    "nullspace_anchor_drift": steer["nullspace_anchor_drift"],
                    "steered_residual_vector": steer["steered_residual_vector"],
                    "portable_artifact": steer["artifact"],
                }
            elif cname == "fallback_compress":
                d_lat = int(
                    self._spec.section("latent_interpretability")["default_latent_dim"]
                )
                full_v = bus.manifold.embed_trace(goal_text).slice_prefix(
                    int(self._spec.section("matryoshka_manifold")["coarse_dim"])
                )
                ib_res = bus.ib_engine.compress_representation([full_v])
                comp_mat = ib_res.get("compressed_matrix")
                comp_vec = (
                    comp_mat[0][:d_lat]
                    if isinstance(comp_mat, list) and comp_mat
                    else full_v[:d_lat]
                )
                retained_ratio = float(ib_res["retained_modes"]) / float(
                    max(1, int(ib_res["total_modes"]))
                )
                out = {
                    "fallback_activated": True,
                    "compressed_vector": comp_vec,
                    "latent_vector": comp_vec,
                    "ib_retained_energy_ratio": retained_ratio,
                }
            elif cname == "sheaf_audit":
                claims = env.get("claims")
                if not isinstance(claims, list) or len(claims) < self._spec.c_int("two"):
                    v_a = env.get("erased_vector") or env.get("latent_vector") or [1.0, 0.0]
                    v_b = env.get("steered_residual_vector") or v_a
                    claims = [
                        {"agent_id": "branch_primary", "state_vector": v_a},
                        {"agent_id": "branch_secondary", "state_vector": v_b},
                    ]
                sheaf = bus.sheaf_consistency_audit(claims)
                out = {"sheaf_audit": sheaf}
            elif cname == "conformal_gate":
                cal = env.get("calibration_scores") or [
                    self._spec.c_float("tenth"),
                    self._spec.c_float("quarter"),
                    self._spec.c_float("half"),
                ]
                cand = env.get("candidate_scores") or [
                    self._spec.c_float("tenth"),
                    self._spec.c_float("default_threshold_percent"),
                ]
                conf = bus.conformal_calibrate(cal, cand, context_query=goal_text)
                out = {"conformal_gate": conf}
            elif cname == "compile_cartridge":
                target_ovr = clamped_overrides.get(
                    "portable_target_lowering_engine", {}
                ).get("default_target_format")
                target = str(
                    target_ovr or env.get("target", "rust") or "rust"
                )
                mod = bus.hyperlora_modulate(goal_text, target=target)
                out = {
                    "compiled_cartridge": {
                        "cartridge_id": mod["cartridge_id"],
                        "target": target,
                        "anchor_drift": mod["anchor_drift"],
                    },
                    "nullspace_anchor_drift": mod["anchor_drift"],
                    "portable_artifact": mod["artifact"],
                    "compiled_artifact_preview": mod["artifact"],
                }
            elif cname in self._tool_handlers and cname != "continuum_compose_omninode":
                raw_json = self._tool_handlers[cname](dict(env))
                out = {"tool_result": json.loads(raw_json)}
            elif cname in bus.engines:
                spec_info = bus.analyze_spectrum(engine_id=cname)
                q_vec = bus.manifold.embed_task(goal_text, stratum=str(card.get("stratum", "")))
                fluid = bus.hypergrid.interpolate_fluid_config(cname, q_vec)
                params = dict(fluid["parameters"])
                if cname in clamped_overrides:
                    params.update(clamped_overrides[cname])
                out = {
                    "engine_execution": {
                        "engine_id": cname,
                        "active_parameters": params,
                        "effective_rank": spec_info["effective_rank"],
                        "signal_rank": spec_info["signal_components_above_noise"],
                    }
                }
            else:
                out = {"executed_component": cname, "contract": card.get("contract")}
        except Exception as exc:
            ms_err = int((time.time() - t0) * thousand)
            f_kind = self.encaser.waterfall_router.classify(exc)
            trace_log.append(
                {
                    "step": step_token,
                    "component": cname,
                    "card_sha256": csha,
                    "how_chosen": how_chosen,
                    "ok": False,
                    "failure": f_kind,
                    "error": str(exc),
                    "ms": ms_err,
                }
            )
            self.encaser.cas_lockfile.record_fact(
                "observed",
                component=cname,
                sha256=csha,
                ok=False,
                kind=f_kind,
                ms=ms_err,
            )
            if isinstance(exc, OmniNodeFailure):
                raise
            raise OmniNodeFailure(str(exc), f_kind) from exc

        ms_ok = int((time.time() - t0) * thousand)
        trace_log.append(
            {
                "step": step_token,
                "component": cname,
                "card_sha256": csha,
                "how_chosen": how_chosen,
                "ok": True,
                "ms": ms_ok,
            }
        )
        self.encaser.cas_lockfile.record_fact(
            "observed",
            component=cname,
            sha256=csha,
            ok=True,
            ms=ms_ok,
        )
        return out

    def _eval_tree(
        self,
        node: Any,
        env: dict[str, Any],
        *,
        depth: int,
        max_depth: int,
        enc_params: Mapping[str, Any],
        dir_params: Mapping[str, Any],
        maker_id: str,
        checker_id: str,
        shape_text: str,
        trace_log: list[dict[str, Any]],
        waterfall_log: list[dict[str, Any]],
        used_pins: dict[str, str],
        clamped_overrides: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        if depth > max_depth:
            raise OmniNodeFailure(
                f"shape nesting depth {depth} exceeds max_nesting_depth {max_depth}",
                "bad_input",
            )

        if isinstance(node, str):
            step_out = self._execute_leaf_step(
                node,
                env,
                enc_params=enc_params,
                trace_log=trace_log,
                used_pins=used_pins,
                clamped_overrides=clamped_overrides,
            )
            return True, {**env, **step_out}

        op, left, right = node
        if op == ">>":
            ok_l, env_l = self._eval_tree(
                left,
                env,
                depth=depth,
                max_depth=max_depth,
                enc_params=enc_params,
                dir_params=dir_params,
                maker_id=maker_id,
                checker_id=checker_id,
                shape_text=shape_text,
                trace_log=trace_log,
                waterfall_log=waterfall_log,
                used_pins=used_pins,
                clamped_overrides=clamped_overrides,
            )
            if not ok_l:
                return False, env_l
            return self._eval_tree(
                right,
                env_l,
                depth=depth,
                max_depth=max_depth,
                enc_params=enc_params,
                dir_params=dir_params,
                maker_id=maker_id,
                checker_id=checker_id,
                shape_text=shape_text,
                trace_log=trace_log,
                waterfall_log=waterfall_log,
                used_pins=used_pins,
                clamped_overrides=clamped_overrides,
            )

        if op == "//":
            try:
                ok_l, env_l = self._eval_tree(
                    left,
                    copy.deepcopy(env),
                    depth=depth + 1,
                    max_depth=max_depth,
                    enc_params=enc_params,
                    dir_params=dir_params,
                    maker_id=maker_id,
                    checker_id=checker_id,
                    shape_text=shape_text,
                    trace_log=trace_log,
                    waterfall_log=waterfall_log,
                    used_pins=used_pins,
                    clamped_overrides=clamped_overrides,
                )
                if ok_l:
                    return True, env_l
            except OmniNodeFailure as err:
                policy = str(enc_params["waterfall_failure_policy"])
                if not self.encaser.waterfall_router.can_waterfall(err.kind, policy):
                    raise
                waterfall_log.append(
                    {
                        "failed_branch": self.pipe_op.serialize_tree(left),
                        "failure_kind": err.kind,
                        "reason": str(err),
                        "fallback_branch": self.pipe_op.serialize_tree(right),
                    }
                )
                return self._eval_tree(
                    right,
                    copy.deepcopy(env),
                    depth=depth + 1,
                    max_depth=max_depth,
                    enc_params=enc_params,
                    dir_params=dir_params,
                    maker_id=maker_id,
                    checker_id=checker_id,
                    shape_text=shape_text,
                    trace_log=trace_log,
                    waterfall_log=waterfall_log,
                    used_pins=used_pins,
                    clamped_overrides=clamped_overrides,
                )
            return False, env

        if op == "|":
            branch_results: list[tuple[str, bool, dict[str, Any]]] = []
            for idx, sub_branch in enumerate((left, right)):
                label = self.parallel_op._branch_label(sub_branch, idx)
                ok_b, out_b = self._eval_tree(
                    sub_branch,
                    copy.deepcopy(env),
                    depth=depth + 1,
                    max_depth=max_depth,
                    enc_params=enc_params,
                    dir_params=dir_params,
                    maker_id=maker_id,
                    checker_id=checker_id,
                    shape_text=shape_text,
                    trace_log=trace_log,
                    waterfall_log=waterfall_log,
                    used_pins=used_pins,
                    clamped_overrides=clamped_overrides,
                )
                delta_b = {
                    k: v for k, v in out_b.items() if k not in env or out_b[k] != env[k]
                }
                branch_results.append((label, ok_b, delta_b))
            return self.parallel_op.merge_branches(
                env,
                branch_results,
                bus=self.encaser.bus,
                sheaf_audit_enabled=bool(dir_params["parallel_sheaf_audit_enabled"]),
            )

        if op == "@":
            ok_m, maker_out = self._eval_tree(
                left,
                env,
                depth=depth,
                max_depth=max_depth,
                enc_params=enc_params,
                dir_params=dir_params,
                maker_id=maker_id,
                checker_id=checker_id,
                shape_text=shape_text,
                trace_log=trace_log,
                waterfall_log=waterfall_log,
                used_pins=used_pins,
                clamped_overrides=clamped_overrides,
            )
            if not ok_m:
                return False, maker_out
            t0_q = time.time()
            thousand = float(self._spec.c_int("one_thousand"))
            q_label = right if isinstance(right, str) else "independent_quorum_check"
            q_card, q_how = self.encaser.resolve_step_card(
                q_label,
                min_search_score=float(enc_params["capability_search_min_score"]),
                slice_dim=int(enc_params["matryoshka_slice_dim"]),
            )
            used_pins[str(q_card["name"])] = str(q_card["sha256"])
            try:
                verified_env = self.quorum_op.verify_handoff(
                    maker_out,
                    maker_id=maker_id,
                    checker_id=checker_id,
                    min_quality_floor=float(dir_params["quorum_min_quality_floor"]),
                    enforcement_mode=str(dir_params["quorum_enforcement_mode"]),
                    store=self.encaser.store,
                    shape_text=shape_text,
                )
            except OmniNodeFailure as q_err:
                ms_q = int((time.time() - t0_q) * thousand)
                trace_log.append(
                    {
                        "step": q_label,
                        "component": str(q_card["name"]),
                        "card_sha256": str(q_card["sha256"]),
                        "how_chosen": q_how,
                        "ok": False,
                        "failure": q_err.kind,
                        "error": str(q_err),
                        "ms": ms_q,
                    }
                )
                raise
            ms_q_ok = int((time.time() - t0_q) * thousand)
            trace_log.append(
                {
                    "step": q_label,
                    "component": str(q_card["name"]),
                    "card_sha256": str(q_card["sha256"]),
                    "how_chosen": q_how,
                    "ok": True,
                    "ms": ms_q_ok,
                }
            )
            return True, verified_env

        raise ShapeSyntaxError(f"unsupported shape operator {op!r}")

    def compose_and_execute(
        self,
        shape: str | None = None,
        *,
        goal: str = "",
        given: Mapping[str, Any] | None = None,
        grid_overrides: Mapping[str, Any] | None = None,
        maker_id: str | None = None,
        checker_id: str | None = None,
        verify_lockfile: bool = True,
        unknown_axes: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Compile and run an algebraic OmniNode shape pipeline and emit a canonical grid/1 receipt."""
        t_start = time.time()
        thousand = float(self._spec.c_int("one_thousand"))
        o_cfg = self._spec.section("omninode_shell")
        shape_expr = str(
            shape or self._omni_spec.section("default_canonical_shape")
        ).strip()
        use_maker = (
            str(maker_id)
            if maker_id is not None
            else str(o_cfg["default_maker_id"])
        )
        use_checker = (
            str(checker_id)
            if checker_id is not None
            else str(o_cfg["default_checker_id"])
        )

        q_vec = self.encaser.bus.manifold.embed_task(
            f"{shape_expr} {goal}",
            stratum="omninode_shell_and_shape_composition",
        )
        enc_params = dict(
            self.encaser.bus.hypergrid.interpolate_fluid_config(
                "omninode_shell_encaser", q_vec
            )["parameters"]
        )
        dir_params = dict(
            self.encaser.bus.hypergrid.interpolate_fluid_config(
                "algebraic_shape_and_grid1_director", q_vec
            )["parameters"]
        )

        trace_log: list[dict[str, Any]] = []
        waterfall_log: list[dict[str, Any]] = []
        used_pins: dict[str, str] = {}
        lock_report: dict[str, Any] = {
            "verified": False,
            "lockfile_sha256": str(
                self.encaser.lock_doc.get("lockfile_sha256", "")
            ),
            "total_cards": len(self.encaser.catalog),
            "pinned_components": {},
        }

        ok = False
        failure_kind: str | None = None
        error_msg: str | None = None
        final_env: dict[str, Any] = {}
        parsed_tree: Any = shape_expr
        leaf_list: list[str] = []
        clamped_overrides: dict[str, dict[str, Any]] = {}

        try:
            clamped_overrides = self.apply_grid_overrides(grid_overrides)
            if "omninode_shell_encaser" in clamped_overrides:
                enc_params.update(clamped_overrides["omninode_shell_encaser"])
            if "algebraic_shape_and_grid1_director" in clamped_overrides:
                dir_params.update(
                    clamped_overrides["algebraic_shape_and_grid1_director"]
                )

            if verify_lockfile or bool(enc_params.get("strict_lockfile_pinning", True)):
                if verify_lockfile:
                    lock_report = self.encaser.cas_lockfile.verify_integrity(
                        self.encaser.catalog
                    )

            raw_tree = self.pipe_op.parse(shape_expr)
            parsed_tree = (
                self.pipe_op.balance(raw_tree)
                if bool(dir_params["balance_sequential_chains"])
                else raw_tree
            )
            leaf_list = self.pipe_op.leaves(parsed_tree)

            max_depth = (
                int(enc_params["max_nesting_depth"])
                + self._spec.c_int("four")
            )
            init_env: dict[str, Any] = dict(given or {})
            if goal:
                init_env["goal"] = goal
            elif "goal" not in init_env:
                init_env["goal"] = shape_expr

            ok, final_env = self._eval_tree(
                parsed_tree,
                init_env,
                depth=0,
                max_depth=max_depth,
                enc_params=enc_params,
                dir_params=dir_params,
                maker_id=use_maker,
                checker_id=use_checker,
                shape_text=shape_expr,
                trace_log=trace_log,
                waterfall_log=waterfall_log,
                used_pins=used_pins,
                clamped_overrides=clamped_overrides,
            )
        except Exception as exc:
            ok = False
            failure_kind = self.encaser.waterfall_router.classify(exc)
            error_msg = str(exc)

        total_ms = int((time.time() - t_start) * thousand)
        prec_digits = int(dir_params["grid1_float_precision_digits"])

        grid1_cfg: dict[str, Any] = {
            "harness": "hermes-continuum-omninode",
            "harness_version": str(self._spec.section("version")),
            "prompt_version": _sha256_text(shape_expr),
            "skills": ["continuum"],
            "tools": list(TOOL_NAMES_CANONICAL),
            "retrieval_method": "hybrid",
            "top_k": int(enc_params["matryoshka_slice_dim"]),
            "embedding_model": "matryoshka-4space-128d",
            "director_weights": {
                "omninode_shell_encaser": enc_params,
                "algebraic_shape_and_grid1_director": dir_params,
            },
            "fallback_order": [
                str(w.get("fallback_branch", "")) for w in waterfall_log
            ]
            or ["fallback_compress"],
            "place": "local",
            "hardware": "cpu",
            "code_commit": str(self._spec.section("version")),
            "components": dict(sorted(used_pins.items()))
            if used_pins
            else {
                k: v["sha256"]
                for k, v in self.encaser.lock_doc.get("nodes", {}).items()
                if not k.startswith("module::")
            },
            "shape_expression": shape_expr,
        }
        if grid_overrides:
            for k_o, v_o in grid_overrides.items():
                if k_o in self.grid1_recorder.axes_spec:
                    grid1_cfg[k_o] = v_o
                else:
                    grid1_cfg[f"override_{k_o}"] = v_o

        score_val = (
            float(
                final_env.get("quorum_verdict", {}).get("quality_score", 1.0)
                if isinstance(final_env.get("quorum_verdict"), dict)
                else 1.0
            )
            if ok
            else 0.0
        )
        grid1_receipt = self.grid1_recorder.record_run(
            kind="omninode-shape",
            source="continuum",
            config=grid1_cfg,
            outcome={
                "passed": ok,
                "score": round(score_val, prec_digits),
                "cost": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": total_ms,
                "failure": failure_kind,
            },
            unknown=unknown_axes,
            question=goal or shape_expr,
            precision_digits=prec_digits,
        )
        self.last_grid1_cell = str(grid1_receipt["cell"])

        # Remove bulky internal echoes from output while keeping all rich results
        clean_output = {
            k: v
            for k, v in final_env.items()
            if k not in ("probe_direction",)
        }

        return {
            "ok": ok,
            "failure": failure_kind,
            "error": error_msg,
            "shape": shape_expr,
            "ast_tree": self.pipe_op.serialize_tree(parsed_tree),
            "leaf_steps": leaf_list,
            "maker_id": use_maker,
            "checker_id": use_checker,
            "clamped_grid_overrides": clamped_overrides,
            "waterfall_recoveries": waterfall_log,
            "output": clean_output,
            "trace": trace_log,
            "lockfile": {
                "verified": bool(lock_report["verified"]),
                "lockfile_sha256": str(lock_report["lockfile_sha256"]),
                "total_cards": len(self.encaser.catalog),
                "used_component_pins": dict(sorted(used_pins.items())),
            },
            "grid1_receipt": grid1_receipt,
            "cost_usd": 0.0,
            "extra_llm_tokens": 0,
        }

    def status_summary(self) -> dict[str, Any]:
        """Telemetry block included in manifold_status, /continuum CLI, Desktop SDK, and Dashboard."""
        try:
            ver = self.encaser.cas_lockfile.verify_integrity(self.encaser.catalog)
            lock_ok = bool(ver["verified"])
            lock_sha = str(ver["lockfile_sha256"])
        except OmniNodeFailure:
            lock_ok = False
            lock_sha = str(self.encaser.lock_doc.get("lockfile_sha256", ""))

        role_counts: dict[str, int] = {}
        for key, card in self.encaser.catalog.items():
            if key.startswith("module::"):
                role_counts["module"] = role_counts.get("module", 0) + 1
            else:
                role = str(card.get("role", "worker"))
                role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "encased": True,
            "lockfile_verified": lock_ok,
            "lockfile_sha256": lock_sha,
            "total_cards": len(self.encaser.catalog),
            "cards_by_role": role_counts,
            "card_counts": {
                "engines": role_counts.get("engine", 0),
                "tools": role_counts.get("tool", 0),
                "surfaces": role_counts.get("surface", 0),
                "stages": role_counts.get("op", 0)
                + role_counts.get("check", 0)
                + role_counts.get("worker", 0)
                + role_counts.get("checker", 0),
                "modules": role_counts.get("module", 0),
            },
            "grid1_format": self.grid1_recorder.format_tag,
            "active_grid1_cell": self.last_grid1_cell,
            "grid1_records_count": self.grid1_recorder.count_records(),
            "supported_operators": list(self.pipe_op.precedence),
            "default_canonical_shape": str(
                self._omni_spec.section("default_canonical_shape")
            ),
        }
