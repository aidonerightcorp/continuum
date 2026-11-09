"""Comprehensive v1.1.0 tests across 5 Hermes surfaces, 24 cybernetic engines, 160D TT-Cross grid, Sheaf, Conformal, HyperLoRA, and 9 portable runtimes."""

from __future__ import annotations

import base64
import concurrent.futures
import ctypes
import importlib.util
import json
import math
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as continuum_plugin  # noqa: E402
from continuum_core.context_engine import ContinuumContextEngine  # noqa: E402
from continuum_core.cybernetic_engines import TensorTrainCross  # noqa: E402
from continuum_core.delegation_provider import (  # noqa: E402
    ContinuumDelegationProvider,
    DelegationRouteError,
)
from continuum_core.linalg_backend import (  # noqa: E402
    add_scaled,
    identity_matrix,
    outer_product,
    safetensors_loads,
    vec_norm,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold  # noqa: E402
from continuum_core.memory_provider import ContinuumMemoryProvider  # noqa: E402
from continuum_core.storage import ContinuumStore  # noqa: E402
from continuum_core.weight_bus import CyberneticWeightBus  # noqa: E402
from dashboard import plugin_api as dashboard_api  # noqa: E402


def _resolve_rustc() -> str | None:
    found = shutil.which("rustc")
    if found:
        return found
    local_rustc = Path.home() / ".local" / "bin" / "rustc"
    if local_rustc.is_file():
        return str(local_rustc)
    return None


class _HarnessContext:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.hooks: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.context_engine = None
        self.memory_provider = None
        self.delegation_provider = None

    def get_config(self, key: str, default=None):
        return default

    def register_context_engine(self, eng) -> None:
        self.context_engine = eng

    def register_memory_provider(self, prov) -> None:
        self.memory_provider = prov

    def register_delegation_provider(self, prov) -> None:
        self.delegation_provider = prov

    def register_tool(self, name: str, handler, **kwargs) -> None:
        self.tools[name] = handler

    def register_hook(self, hook_name: str, callback, **kwargs) -> None:
        self.hooks[hook_name] = callback

    def register_command(self, name: str, handler, **kwargs) -> None:
        self.commands[name] = handler

    def register_skill(self, name: str, path: Path, **kwargs) -> None:
        pass


# ─── 1. 4-Space Nested Embedding Manifold Tests ───────────────────────────────


def test_matryoshka_four_spaces_and_elastic_prefix_unit_norms() -> None:
    m = MatryoshkaManifold()
    v_task = m.embed_task("compile high precision matrix operator", stratum="compile")
    v_cfg = m.embed_config({"ridge_lambda": 0.01, "layers": 3}, stratum="compile")
    v_wt = m.embed_weights([[[1.0, 0.2], [0.1, 0.8]]], stratum="compile")
    v_tr = m.embed_trace("verified matrix operator trace", stratum="compile")

    for vec in (v_task, v_cfg, v_wt, v_tr):
        assert len(vec.values) == 128
        for dim in (16, 32, 64, 128):
            sliced = vec.slice_prefix(dim)
            assert len(sliced) == dim
            assert abs(vec_norm(sliced) - 1.0) < 1e-6

    assert v_task.cosine(v_cfg, 16) > 0.0
    assert v_task.cosine(v_wt, 16) > -1.0


def test_matryoshka_progressive_cascade_search_saves_macs() -> None:
    m = MatryoshkaManifold()
    candidates = [
        (
            f"item_{i}",
            m.embed_trace(f"document topic number {i} spectral matrix"),
            {"idx": i},
        )
        for i in range(40)
    ]
    target_vec = m.embed_trace("document topic number 17 spectral matrix")
    res = m.progressive_cascade_search(target_vec, candidates, top_k=3)
    assert len(res["matches"]) == 3
    assert res["matches"][0]["id"] == "item_17"
    assert res["mac_operations_used"] < res["exhaustive_mac_baseline"]
    assert res["compute_savings_ratio"] > 0.25


# ─── 2. Fluid 160D TT-Cross Adaptive Grid & Zero-Token Exploration Tests ──────


def test_fluid_hypergrid_soft_kernel_interpolation_and_clamping() -> None:
    bus = CyberneticWeightBus()
    q_reflex = bus.manifold.embed_task(
        "ultra fast reflex token filter", stratum="ingestion_and_memory"
    )
    q_deep = bus.manifold.embed_task(
        "deep exhaustive verification spectrum", stratum="adaptation_and_experimentation"
    )

    cfg_a = bus.hypergrid.interpolate_fluid_config(
        "quotient_context_dag_engine", q_reflex, temperature=0.15
    )
    cfg_b = bus.hypergrid.interpolate_fluid_config(
        "quotient_context_dag_engine", q_deep, temperature=0.15
    )

    assert abs(sum(cfg_a["kernel_weights"].values()) - 1.0) < 1e-6
    assert 0.35 <= cfg_a["parameters"]["compression_ratio_target"] <= 0.88
    assert cfg_a["interpolation_entropy"] >= 0.0
    assert cfg_b["config2vec"].space == "config2vec"


def test_woodbury_bilinear_surrogate_reduces_epistemic_uncertainty() -> None:
    bus = CyberneticWeightBus()
    t_vec = bus.manifold.embed_task("novel algebraic compilation task", stratum="compile")
    c_vec = bus.manifold.embed_config(
        {"rank_cap": 8, "ridge": 0.02}, stratum="compile"
    )

    before = bus.hypergrid.surrogate.score(t_vec, c_vec)["epistemic_std"]
    bus.hypergrid.surrogate.observe(t_vec, c_vec, reward=0.92)
    after = bus.hypergrid.surrogate.score(t_vec, c_vec)["epistemic_std"]
    assert after < before


def test_opportunistic_sha_and_tt_cross_exploration_zero_tokens() -> None:
    bus = CyberneticWeightBus()
    skipped = bus.hypergrid.opportunistic_explore(
        "fluid_hypergrid_interpolator_engine", compute_headroom_ratio=0.0
    )
    assert skipped["skipped_low_headroom"] is True
    assert skipped["candidates_generated"] == 0

    active = bus.hypergrid.opportunistic_explore(
        "fluid_hypergrid_interpolator_engine",
        compute_headroom_ratio=0.85,
        seed=42,
    )
    assert active["skipped_low_headroom"] is False
    assert active["candidates_generated"] >= 10
    assert active["covering_array_rows"] >= 1
    assert active["tt_cross_evaluations"] >= 10
    assert active["tt_cross_compression_factor"] > 1.0
    assert len(active["rung_counts"]) == 3
    assert active["rung_counts"][0] >= active["rung_counts"][1] >= active["rung_counts"][2]
    assert active["extra_llm_tokens"] == 0
    assert active["winner_cell_id"] is not None


def test_tt_cross_epistatic_search_finds_non_additive_optimum() -> None:
    tt = TensorTrainCross()

    # Non-additive epistatic objective over 6 axes x 3 levels (3^6 = 729 combinations)
    def epistatic_obj(idx: list[int]) -> float:
        coupled_pair_1 = 2.5 if (idx[0] == 2 and idx[1] == 2) else -(idx[0] + idx[1]) * 0.2
        coupled_pair_2 = 2.0 if (idx[2] == 1 and idx[3] == 2) else 0.0
        tail = 0.5 * float(idx[4] == 2) + 0.5 * float(idx[5] == 1)
        return coupled_pair_1 + coupled_pair_2 + tail

    res = tt.fit_and_search(epistatic_obj, num_axes=6, num_levels=3, tt_rank=3, sweeps=2)
    assert res["best_score"] >= 4.5
    assert res["evaluations_used"] < 200
    assert res["compression_factor_vs_grid"] > 3.5
    assert len(res["tt_core_frobenius_norms"]) == 6


def test_hypergrid_boundary_auto_expansion() -> None:
    bus = CyberneticWeightBus()
    grid = bus.hypergrid.grids["ridge_polytope_compiler_engine"]
    old_high = grid.parameters["rank_cap"].bounds.high
    old_gen = grid.generation
    expansions = grid.maybe_expand_boundaries({"rank_cap": old_high})
    assert len(expansions) == 1
    assert grid.parameters["rank_cap"].bounds.high > old_high
    assert grid.generation == old_gen + 1


# ─── 3. Surface 1: ContextEngine Tests ────────────────────────────────────────


def test_context_engine_preserves_exact_message_reference_for_prompt_cache(
    tmp_path: Path,
) -> None:
    bus = CyberneticWeightBus()
    store = ContinuumStore(db_path=tmp_path / "ctx.db", manifold=bus.manifold)
    eng = ContinuumContextEngine(bus=bus, store=store)

    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Short prompt within budget"},
    ]
    selected = eng.select_context(messages, budget_tokens=4096)
    assert selected is messages
    assert bus.cost_ledger.prompt_cache_identity_hits == 1


def test_context_engine_quotient_compression_and_grep(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = ContinuumStore(db_path=tmp_path / "ctx_comp.db", manifold=bus.manifold)
    eng = ContinuumContextEngine(
        bus=bus, store=store, context_length=400, threshold_percent=0.2
    )

    msgs = [{"role": "system", "content": "Core rules"}]
    for i in range(12):
        msgs.append(
            {"role": "user", "content": "duplicate filler note about routine check"}
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Unique spectral eigenvalue observation alpha_{i} = 1.732",
            }
        )
    msgs.append({"role": "user", "content": "Final active question"})

    compacted = eng.compress(msgs, force=True, focus_topic="spectral eigenvalue")
    assert len(compacted) < len(msgs)
    assert eng.compression_count == 1

    grep_raw = eng.handle_tool_call(
        "continuum_context_grep", {"query": "alpha_5", "limit": 5}
    )
    grep_data = json.loads(grep_raw)
    assert grep_data["count"] >= 1
    assert any("alpha_5" in m["snippet"] for m in grep_data["matches"])


def test_context_engine_clone_and_pooled_thread_safety(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = ContinuumStore(db_path=tmp_path / "ctx_pool.db", manifold=bus.manifold)
    parent = ContinuumContextEngine(bus=bus, store=store)
    child = parent.clone_for_agent()
    child.update_model("fast-tier", context_length=8192)
    assert parent.context_length != 8192

    def _worker(tid: int) -> int:
        clone = parent.clone_for_agent()
        clone.on_session_start(session_id=f"sess_{tid}")
        turns = [
            {"role": "user", "content": f"thread {tid} message {k}"} for k in range(10)
        ]
        out = clone.compress(turns, force=True)
        return len(out)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(8)))
    assert all(r >= 2 for r in results)


# ─── 4. Surface 2: MemoryProvider Tests ───────────────────────────────────────


def test_memory_provider_profile_isolation_and_hopfield_recall(tmp_path: Path) -> None:
    home_a = tmp_path / "profile_alpha"
    home_b = tmp_path / "profile_beta"

    prov_a = ContinuumMemoryProvider()
    prov_b = ContinuumMemoryProvider()
    assert prov_a.is_available() is True

    prov_a.initialize("s1", hermes_home=str(home_a), agent_context="primary")
    prov_b.initialize("s2", hermes_home=str(home_b), agent_context="primary")

    prov_a.sync_turn(
        "What is the Frobenius norm target?", "Keep anchor drift strictly at zero."
    )
    rec_a = json.loads(
        prov_a.handle_tool_call(
            "continuum_attractor_recall", {"query": "Frobenius anchor drift"}
        )
    )
    rec_b = json.loads(
        prov_b.handle_tool_call(
            "continuum_attractor_recall", {"query": "Frobenius anchor drift"}
        )
    )

    assert len(rec_a["matches"]) == 1
    assert len(rec_b["matches"]) == 0


def test_memory_provider_suppresses_writes_in_cron_and_subagent(tmp_path: Path) -> None:
    for ctx_kind in ("cron", "subagent", "flush"):
        home = tmp_path / f"home_{ctx_kind}"
        prov = ContinuumMemoryProvider()
        prov.initialize("s_bg", hermes_home=str(home), agent_context=ctx_kind)
        prov.sync_turn("background prompt", "background completion")
        prov.on_pre_compress([{"role": "user", "content": "transient subagent note"}])
        recalled = json.loads(
            prov.handle_tool_call("continuum_attractor_recall", {"query": "background"})
        )
        assert recalled["matches"] == []


def test_memory_provider_pre_compress_consolidates_safetensors_cartridge(
    tmp_path: Path,
) -> None:
    prov = ContinuumMemoryProvider()
    prov.initialize(
        "s_main", hermes_home=str(tmp_path / "home_main"), agent_context="primary"
    )
    summary = prov.on_pre_compress(
        [
            {
                "role": "user",
                "content": "Configure Woodburys inverse covariance solver",
            },
            {"role": "assistant", "content": "Verified O(d^2) rank-one update active"},
        ]
    )
    assert "Woodbury" in summary
    prefetched = prov.prefetch("Woodbury inverse covariance")
    assert "hopfield=" in prefetched
    assert "Continuum Memory Recall" in prov.system_prompt_block()
    prov.shutdown()
    assert prov.system_prompt_block() == ""


# ─── 5. Surface 3: DelegationProvider Tests ───────────────────────────────────


def test_delegation_provider_linucb_routing_and_maker_checker_enforcement(
    tmp_path: Path,
) -> None:
    bus = CyberneticWeightBus()
    store = ContinuumStore(db_path=tmp_path / "del.db", manifold=bus.manifold)
    delegator = ContinuumDelegationProvider(bus=bus, store=store)

    params = delegator.task_parameters({})
    assert {"engine_stratum", "verification_role", "maker_id"} <= set(params.keys())

    maker_route = delegator.route_task(
        {
            "goal": "compile C99 operator kernel",
            "engine_stratum": "compile",
            "verification_role": "maker",
        },
        {},
    )
    assert "label" in maker_route
    maker_label = maker_route["label"]

    checker_route = delegator.route_task(
        {
            "goal": "audit C99 operator kernel",
            "engine_stratum": "verify",
            "verification_role": "checker",
            "maker_id": maker_label,
        },
        {},
    )
    assert checker_route["label"] != maker_label

    with pytest.raises(DelegationRouteError):
        delegator.route_task(
            {
                "goal": "audit with colliding tier",
                "verification_role": "checker",
                "maker_id": "solo_tier",
            },
            {
                "tiers": [
                    {
                        "tier_id": "solo_tier",
                        "model": "solo-model",
                        "reasoning_effort": "high",
                    }
                ]
            },
        )


# ─── 6. 9 Portable Cartridge Runtimes & Real Native Execution Parity ──────────


@pytest.mark.parametrize(
    "target",
    [
        "rust",
        "glsl",
        "js",
        "python",
        "sql",
        "c99",
        "wasm",
        "wat",
        "wgsl",
        "safetensors",
    ],
)
def test_compile_cartridge_all_nine_portable_targets_plus_safetensors(
    target: str, tmp_path: Path
) -> None:
    bus = CyberneticWeightBus()
    rules = [
        {"input": [1.0, 0.0, 0.5, -0.5], "target": [0.8, -0.2, 0.1, 0.4]},
        {"input": [0.0, 1.0, -0.5, 0.5], "target": [-0.3, 0.9, 0.4, -0.1]},
        {"input": [0.5, 0.5, 0.5, 0.5], "target": [0.25, 0.35, 0.25, 0.15]},
    ]
    cart = bus.compiler.compile_stack(
        "test_cart", rules, num_layers=3, activation="gelu"
    )
    artifact = bus.compiler.lower_to_target(cart, target)
    assert isinstance(artifact, str) and len(artifact) > 20

    if target == "rust":
        assert "#![no_std]" in artifact
        assert 'pub unsafe extern "C" fn infer' in artifact
    elif target == "glsl":
        assert "#version 300 es" in artifact
        assert "fragColor = vec4(" in artifact
    elif target == "js":
        assert "function infer(" in artifact
        assert "module.exports" in artifact
    elif target == "safetensors":
        raw = base64.b64decode(artifact)
        tensors, meta = safetensors_loads(raw)
        assert "readout_w" in tensors
        assert "layer_0.film_gamma" in tensors
        assert meta["cartridge_id"] == "test_cart"
    elif target == "wasm":
        wasm_bytes = base64.b64decode(artifact)
        assert list(wasm_bytes[:8]) == [0, 97, 115, 109, 1, 0, 0, 0]
        assert list(REPO_ROOT.rglob("*.wasm")) == []
    elif target == "python":
        mod_file = tmp_path / "generated_cartridge.py"
        mod_file.write_text(artifact, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("generated_cartridge", mod_file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.infer([1.0, 0.0, 0.5, -0.5], [0.4])
        assert len(out) == 4 and all(math.isfinite(v) for v in out)


def test_cross_runtime_execution_parity_rust_c99_wasm_js_sql_python() -> None:
    """Compile and execute the same cartridge across rustc (#![no_std]), gcc (C99), node (JS & WASM), sqlite3, and python."""
    bus = CyberneticWeightBus()
    rules = [
        {"input": [1.0, 0.2, -0.3, 0.4], "target": [0.9, -0.5, 0.3, 0.7]},
        {"input": [-0.4, 1.1, 0.6, -0.2], "target": [-0.2, 0.8, 0.5, -0.4]},
        {"input": [0.3, -0.5, 1.2, 0.1], "target": [0.4, 0.1, -0.7, 0.6]},
        {"input": [0.2, 0.3, -0.2, 1.0], "target": [0.1, -0.3, 0.2, 0.9]},
    ]
    cart = bus.compiler.compile_stack("parity_cart", rules, num_layers=3)
    x_vec = [0.75, -0.40, 0.25, 0.60]
    z_vec = [0.35]

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tdir = Path(tmp_dir_str)

        # 1. Standalone Python reference
        py_src = bus.compiler.lower_to_target(cart, "python")
        py_path = tdir / "cart_mod.py"
        py_path.write_text(py_src, encoding="utf-8")
        py_spec = importlib.util.spec_from_file_location("cart_mod", py_path)
        assert py_spec is not None and py_spec.loader is not None
        py_mod = importlib.util.module_from_spec(py_spec)
        py_spec.loader.exec_module(py_mod)
        y_py = py_mod.infer(x_vec, z_vec)
        y_py_z0 = py_mod.infer(x_vec, [0.0])

        # 2. Rust #![no_std] compiled via rustc -O --crate-type cdylib -C panic=abort
        rustc_bin = _resolve_rustc()
        assert rustc_bin is not None, "rustc must be available for native #![no_std] verification"
        rs_path = tdir / "cart.rs"
        rs_so = tdir / "libcart_rs.so"
        rs_path.write_text(bus.compiler.lower_to_target(cart, "rust"), encoding="utf-8")
        subprocess.run(
            [
                rustc_bin,
                "-O",
                "--crate-type",
                "cdylib",
                "-C",
                "panic=abort",
                str(rs_path),
                "-o",
                str(rs_so),
            ],
            check=True,
        )
        rs_lib = ctypes.CDLL(str(rs_so))
        rs_lib.infer.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        rs_lib.infer.restype = None
        c_x = (ctypes.c_float * 4)(*x_vec)
        c_z = (ctypes.c_float * 1)(*z_vec)
        c_y_rs = (ctypes.c_float * 4)()
        rs_lib.infer(c_x, c_z, c_y_rs)
        y_rs = [float(c_y_rs[i]) for i in range(4)]

        # 3. ANSI C99 compiled via gcc -O2 -shared -fPIC
        gcc_bin = shutil.which("gcc")
        assert gcc_bin is not None, "gcc must be available for C99 verification"
        c_path = tdir / "cart.c"
        c_so = tdir / "libcart_c.so"
        c_path.write_text(bus.compiler.lower_to_target(cart, "c99"), encoding="utf-8")
        subprocess.run(
            [gcc_bin, "-O2", "-shared", "-fPIC", str(c_path), "-lm", "-o", str(c_so)],
            check=True,
        )
        c_lib = ctypes.CDLL(str(c_so))
        c_lib.infer.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        c_lib.infer.restype = None
        c_y_c99 = (ctypes.c_float * 4)()
        c_lib.infer(c_x, c_z, c_y_c99)
        y_c99 = [float(c_y_c99[i]) for i in range(4)]

        # 4. JavaScript executed via node
        node_bin = shutil.which("node")
        assert node_bin is not None, "node must be available for JS/WASM verification"
        js_path = tdir / "cart.js"
        js_path.write_text(bus.compiler.lower_to_target(cart, "js"), encoding="utf-8")
        js_out = subprocess.check_output(
            [
                node_bin,
                "-e",
                f"const m = require({json.dumps(str(js_path))}); console.log(JSON.stringify(m.infer({json.dumps(x_vec)}, {json.dumps(z_vec)})));",
            ],
            text=True,
        )
        y_js = [float(v) for v in json.loads(js_out.strip())]

        # 5. WebAssembly IEEE-754 binary instantiated in V8 (node WebAssembly.instantiate)
        wasm_b64 = bus.compiler.lower_to_target(cart, "wasm")
        wasm_out = subprocess.check_output(
            [
                node_bin,
                "-e",
                (
                    f"const buf = Buffer.from({json.dumps(wasm_b64)}, 'base64');"
                    "WebAssembly.instantiate(buf).then(({instance}) => {"
                    f"  const y0 = instance.exports.infer({x_vec[0]}, {x_vec[1]}, {x_vec[2]}, {x_vec[3]}, {z_vec[0]});"
                    "  console.log(JSON.stringify(y0));"
                    "});"
                ),
            ],
            text=True,
        )
        y0_wasm = float(json.loads(wasm_out.strip()))

        # 6. ANSI SQL CTE executed in Python sqlite3 in-memory engine (z0 = 0 -> gamma = 1.0)
        sql_query = bus.compiler.lower_to_target(cart, "sql")
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE input_vector (in_idx INTEGER, val REAL);")
            conn.executemany(
                "INSERT INTO input_vector (in_idx, val) VALUES (?, ?);",
                list(enumerate(x_vec)),
            )
            rows = conn.execute(sql_query).fetchall()
            y_sql = [float(r[1]) for r in rows]
        finally:
            conn.close()

        # Verify tight numerical agreement across all runtimes
        for idx in range(4):
            assert abs(y_rs[idx] - y_py[idx]) < 1e-5, f"Rust mismatch at {idx}: {y_rs} vs {y_py}"
            assert abs(y_c99[idx] - y_py[idx]) < 1e-5, f"C99 mismatch at {idx}: {y_c99} vs {y_py}"
            assert abs(y_js[idx] - y_py[idx]) < 1e-6, f"JS mismatch at {idx}: {y_js} vs {y_py}"
            assert abs(y_sql[idx] - y_py_z0[idx]) < 1e-6, f"SQL mismatch at {idx}: {y_sql} vs {y_py_z0}"
        assert abs(y0_wasm - y_py[0]) < 1e-5, f"WASM mismatch: {y0_wasm} vs {y_py[0]}"


def test_multilayer_film_conditioning_and_early_exit() -> None:
    bus = CyberneticWeightBus()
    rules = [{"input": [1.0, 0.0, 0.0, 0.0], "target": [4.0, -2.0, -2.0, -2.0]}]
    cart = bus.compiler.compile_stack(
        "film_stack", rules, num_layers=5, early_exit_margin=0.15
    )
    res_fast = cart.forward(
        [1.0, 0.0, 0.0, 0.0], cond_vec=[0.9] * 16, allow_early_exit=True
    )
    res_full = cart.forward(
        [1.0, 0.0, 0.0, 0.0], cond_vec=[-0.9] * 16, allow_early_exit=False
    )

    assert res_fast["layers_executed"] < 5
    assert res_fast["layers_saved"] >= 1
    assert res_fast["output"] != res_full["output"]


# ─── 7. Cybernetic v1.1.0 Engines: Sheaf, Conformal, HyperLoRA, Causal, VCG ───


def test_sheaf_consistency_audit_detects_contradictions_and_repairs_consensus() -> None:
    bus = CyberneticWeightBus()

    # Case 1: Globally consistent stalks across all agents -> 0.0 obstruction energy
    consistent_claims = [
        {"agent_id": "planner", "state_vector": [0.8, -0.4, 0.2, 0.5]},
        {"agent_id": "builder", "state_vector": [0.8, -0.4, 0.2, 0.5]},
        {"agent_id": "auditor", "state_vector": [0.8, -0.4, 0.2, 0.5]},
    ]
    ok_res = bus.sheaf_consistency_audit(consistent_claims, stalk_dim=4)
    assert ok_res["consistent"] is True
    assert ok_res["global_obstruction_energy"] < 1e-12
    assert ok_res["contradictory_edges"] == []

    # Case 2: Auditor strongly contradicts builder & planner -> non-zero cohomological obstruction
    conflicting_claims = [
        {"agent_id": "planner", "state_vector": [1.0, 0.5, 0.0, 0.2]},
        {"agent_id": "builder", "state_vector": [1.0, 0.5, 0.0, 0.2]},
        {"agent_id": "rogue_node", "state_vector": [-1.0, -0.8, 0.9, -0.7]},
    ]
    bad_res = bus.sheaf_consistency_audit(
        conflicting_claims, stalk_dim=4, harmonic_eta=0.8
    )
    assert bad_res["consistent"] is False
    assert bad_res["global_obstruction_energy"] > 1.0
    assert len(bad_res["contradictory_edges"]) >= 2
    assert all("rogue_node" in (e["source"], e["target"]) for e in bad_res["contradictory_edges"])
    assert bad_res["post_repair_obstruction_energy"] < bad_res["global_obstruction_energy"]
    assert bad_res["energy_reduction_ratio"] > 0.5


def test_conformal_calibration_marginal_coverage_aci_and_abstention() -> None:
    bus = CyberneticWeightBus()
    rng = random.Random(2026)

    # Empirical finite-sample 1 - alpha coverage verification over exchangeable draws
    alpha = 0.10
    cal_scores = [rng.uniform(0.0, 1.0) for _ in range(199)]
    covered = 0
    trials = 250
    for _ in range(trials):
        true_score = rng.uniform(0.0, 1.0)
        res = bus.conformal_calibrate(
            cal_scores, [true_score, 0.99], alpha=alpha, abstention_max_set_size=2
        )
        if 0 in res["prediction_set_indices"]:
            covered += 1
    empirical_cov = covered / float(trials)
    assert empirical_cov >= 1.0 - alpha - 0.04

    # Online ACI shift adaptation: consecutive misses lower alpha_t (widening coverage)
    aci_tight = bus.conformal_calibrate(
        cal_scores,
        [0.12, 0.65, 0.91],
        alpha=0.10,
        current_alpha_t=0.10,
        recent_misses=[False, False, False, False],
    )
    aci_wide = bus.conformal_calibrate(
        cal_scores,
        [0.12, 0.65, 0.91],
        alpha=0.10,
        current_alpha_t=0.10,
        recent_misses=[True, True, True, True],
    )
    assert aci_wide["adaptive_alpha_t"] < aci_tight["adaptive_alpha_t"]
    assert aci_wide["conformal_quantile_q_hat"] >= aci_tight["conformal_quantile_q_hat"]

    # Ambiguous candidate set triggers selective abstention to full LLM deliberation
    ambiguous = bus.conformal_calibrate(
        cal_scores,
        [0.10, 0.15, 0.20, 0.25],
        alpha=0.10,
        abstention_max_set_size=2,
    )
    assert ambiguous["abstain"] is True
    assert ambiguous["selective_routing_tier"] == "full_llm_deliberation"


def test_hyperlora_kronecker_modulation_zero_anchor_drift_and_rust_compilation() -> None:
    bus = CyberneticWeightBus()
    probe = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mod_a = bus.hyperlora_modulate(
        "symbolic rust parser synthesis",
        engine_id="multilayer_film_molora_stack_engine",
        target="rust",
        export_dim=4,
        counterexample_probes=[probe],
    )
    mod_b = bus.hyperlora_modulate(
        "webgl2 fragment shader rasterizer",
        engine_id="multilayer_film_molora_stack_engine",
        target="glsl",
        export_dim=4,
        persist_modulation=False,
    )

    assert mod_a["anchor_drift"] == 0.0
    assert mod_a["raw_numerical_drift"] < 1e-12
    assert mod_a["delta_frobenius_norm"] > 0.0
    assert "#![no_std]" in mod_a["artifact"]
    assert "#version 300 es" in mod_b["artifact"]
    assert mod_a["weight2vec_16d"] != mod_b["weight2vec_16d"]
    assert mod_a["weight2vec_64d"] != mod_b["weight2vec_64d"]

    # Compile and execute the HyperLoRA-modulated #![no_std] Rust kernel natively
    rustc_bin = _resolve_rustc()
    assert rustc_bin is not None
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tdir = Path(tmp_dir_str)
        rs_file = tdir / "hyperlora.rs"
        so_file = tdir / "libhyperlora.so"
        rs_file.write_text(mod_a["artifact"], encoding="utf-8")
        subprocess.run(
            [
                rustc_bin,
                "-O",
                "--crate-type",
                "cdylib",
                "-C",
                "panic=abort",
                str(rs_file),
                "-o",
                str(so_file),
            ],
            check=True,
        )
        lib = ctypes.CDLL(str(so_file))
        lib.infer.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        c_x = (ctypes.c_float * 4)(1.0, 0.0, 0.5, -0.5)
        c_z = (ctypes.c_float * 1)(0.0)
        c_y = (ctypes.c_float * 4)()
        lib.infer(c_x, c_z, c_y)
        assert all(math.isfinite(float(c_y[i])) for i in range(4))


@pytest.mark.parametrize(
    "target",
    ["rust", "glsl", "js", "python", "sql", "c99", "wasm", "wat", "wgsl"],
)
def test_hyperlora_modulate_exports_all_nine_targets_with_zero_drift(target: str) -> None:
    bus = CyberneticWeightBus()
    res = bus.hyperlora_modulate(
        f"input conditioned modulation for {target}",
        target=target,
        export_dim=4,
    )
    assert res["anchor_drift"] == 0.0
    assert res["target"] == target
    assert isinstance(res["artifact"], str) and len(res["artifact"]) > 20


def test_causal_scm_vcg_mechanism_invariants_ib_oed_and_self_healing() -> None:
    bus = CyberneticWeightBus()

    # 1. Pearl 3-step Causal SCM: X0 -> X1 (weight 2.0) -> X2 (weight 0.5)
    # Observed: u0=1, u1=0, u2=0 => X0=1, X1=2, X2=1. Intervention do(X1 = 6) => X0 stays 1, X2 becomes 3!
    b_scm = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]
    cf = bus.causal_engine.intervene_and_counterfactual(
        [1.0, 2.0, 1.0], b_scm, {1: 6.0}
    )
    assert abs(cf["counterfactual_state"][0] - 1.0) < 1e-5
    assert abs(cf["counterfactual_state"][1] - 6.0) < 1e-5
    assert abs(cf["counterfactual_state"][2] - 3.0) < 1e-5

    # 2. VCG + Strictly Proper Brier Mechanism: truthful report dominates misreport
    mech = bus.mechanism_engine.evaluate_mechanism(
        [
            {"agent_id": "truthful_checker", "confidence": 0.98, "value": 1.0},
            {"agent_id": "noisy_checker", "confidence": 0.15, "value": 1.0},
        ],
        outcome_truth=1.0,
    )
    payoffs = {p["agent_id"]: p["vcg_incentive_payoff"] for p in mech["agent_payoffs"]}
    assert payoffs["truthful_checker"] > payoffs["noisy_checker"]

    # 3. Formal Octagonal Invariant Synthesis
    traces = [[0.2, 0.3, 0.5], [0.25, 0.35, 0.55], [0.3, 0.4, 0.6]]
    inv_ok = bus.invariant_engine.synthesize_and_verify(traces, [0.26, 0.36, 0.56])
    inv_bad = bus.invariant_engine.synthesize_and_verify(traces, [0.95, 0.10, 0.56])
    assert inv_ok["holds"] is True
    assert inv_bad["holds"] is False and len(inv_bad["violations"]) >= 2

    # 4. D-Optimal Active Experimental Design
    oed = bus.experimental_design_engine.select_optimal_probe(
        [[0.1, 0.0, 0.0], [0.0, 2.5, 0.0]],
        identity_matrix(3),
    )
    assert oed["selected_index"] == 1
    assert oed["expected_information_gain_nats"] > 0.5

    # 5. Information Bottleneck spectral filter
    ib = bus.ib_engine.compress_representation(
        bus.engines["information_bottleneck_compression_engine"].weights, beta=2.0
    )
    assert 0 <= ib["retained_modes"] <= ib["total_modes"]

    # 6. Autonomic Self-Healing restores exact 0.0 anchor drift after injected fault
    eng = bus.engines["autonomic_self_healing_engine"]
    corrupted = add_scaled(
        eng.weights,
        outer_product([0.5] * 8, eng.anchor_inputs[0], scale=1.0),
        scale=1.0,
    )
    healed = bus.self_healing_engine.heal_if_drifted(
        corrupted,
        eng.baseline_weights,
        eng.anchor_inputs,
        bus._project_onto_anchor_nullspace,
    )
    assert healed["healed"] is True
    assert healed["drift_before"] > 0.1
    assert healed["drift_after"] == 0.0


# ─── 8. Surgical Edits, Morphism, Coupling, and 15 Registered Tools ───────────


def test_fission_claim_fence_verify_and_cache_hit_pipeline(tmp_path: Path) -> None:
    rt = continuum_plugin.ContinuumRuntime(db_path=tmp_path / "pipe.db")
    dag = rt.store.fission_dag("Design parser. Build evaluator. Verify invariants.")
    assert dag["task_count"] == 3
    first_id = dag["tasks"][0]["task_id"]

    lease_1 = rt.store.claim_and_fence(
        first_id, "agent_maker_1", lease_ttl_seconds=300.0
    )
    assert lease_1["acquired"] is True
    assert lease_1["fencing_stamp"].startswith("fence_")

    lease_2 = rt.store.claim_and_fence(
        first_id, "agent_intruder", lease_ttl_seconds=300.0
    )
    assert lease_2["acquired"] is False

    with pytest.raises(ValueError, match="Independent verification"):
        rt.store.verify_record(
            first_id, maker_id="agent_maker_1", checker_id="agent_maker_1", passed=True
        )

    ver = rt.store.verify_record(
        first_id,
        maker_id="agent_maker_1",
        checker_id="agent_checker_2",
        passed=True,
        quality_score=0.95,
        output_summary="Verified parser AST contract",
    )
    assert ver["passed"] is True
    assert ver["dawid_skene_posterior_truth"] > 0.5
    assert "agent_maker_1" in ver["eigentrust_scores"]

    cached = rt.store.claim_and_fence(first_id, "agent_later")
    assert cached["cache_hit"] is True
    assert cached["output_summary"] == "Verified parser AST contract"


@pytest.mark.parametrize(
    "mode",
    [
        "svd_nullspace",
        "lora_xs",
        "recursive_woodbury",
        "procrustes_align",
        "hopfield_write",
    ],
)
def test_surgical_edits_preserve_exact_zero_anchor_drift_and_cegar(mode: str) -> None:
    bus = CyberneticWeightBus()
    probe = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    eng = bus.engines["svd_nullspace_surgical_editor_engine"]
    probe_before = [
        sum(eng.weights[r][c] * probe[c] for c in range(8)) for r in range(8)
    ]

    res = bus.surgical_edit(
        "svd_nullspace_surgical_editor_engine",
        mode=mode,
        input_vec=[0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.8],
        counterexample_probes=[probe],
    )
    assert res["anchor_drift"] == 0.0

    probe_after = [
        sum(eng.weights[r][c] * probe[c] for c in range(8)) for r in range(8)
    ]
    probe_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(probe_before, probe_after)))
    assert probe_diff < 1e-9


def test_spectral_diagnostics_morphism_coupling_and_evsi() -> None:
    bus = CyberneticWeightBus()
    spec_diag = bus.analyze_spectrum(
        engine_id="rmt_marchenko_pastur_spectral_engine",
        sample_inputs=[[1.0, 2.0, 0.5, 1.0, 0.3, 0.8, 1.2, 0.4]],
    )
    assert spec_diag["effective_rank"] > 1.0
    assert spec_diag["marchenko_pastur_bulk_edge"] > 0.0
    assert len(spec_diag["wanda_row_saliency"]) == 8

    dag_a = [
        {"id": "n1", "depends_on": []},
        {"id": "n2", "depends_on": ["n1"]},
        {"id": "n3", "depends_on": ["n2"]},
    ]
    dag_b = [
        {"id": "m1", "depends_on": []},
        {"id": "m2", "depends_on": ["m1"]},
        {"id": "m3", "depends_on": ["m2"]},
    ]
    morph = bus.morphism_transfer(dag_a, dag_b)
    assert abs(morph["dag_spectral_isomorphism"] - 1.0) < 1e-6
    assert morph["procrustes_orthogonality_error"] < 1e-6
    assert morph["target_anchor_drift"] == 0.0

    step = bus.coupling_step(
        impulse_engine_id="wanda_saliency_pruning_engine",
        impulse_delta=0.05,
        compute_headroom_ratio=0.7,
    )
    assert step["coupling_step"] == 1
    assert step["opportunistic_exploration"]["extra_llm_tokens"] == 0

    evsi_low = bus.calibrate_effort("routine ping", stakes=0.01)
    evsi_high = bus.calibrate_effort(
        "critical distributed protocol verification", stakes=5.0
    )
    assert evsi_low["reasoning_effort"] == "low"
    assert (
        evsi_high["recommended_matryoshka_dim"]
        >= evsi_low["recommended_matryoshka_dim"]
    )


def test_all_fifteen_registered_tools_hooks_command_and_dashboard_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import continuum_core.storage as storage_mod

    test_db = tmp_path / "registered_tools.db"
    monkeypatch.setattr(
        storage_mod, "resolve_continuum_db_path", lambda hermes_home=None: test_db
    )
    monkeypatch.setattr(
        dashboard_api, "resolve_continuum_db_path", lambda hermes_home=None: test_db
    )
    monkeypatch.setattr(dashboard_api, "_BASELINE_BUS", None)

    ctx = _HarnessContext()
    continuum_plugin.register(ctx)
    assert len(ctx.tools) == 15

    ctx.hooks["on_session_start"](session_id="test_sess")
    ctx.hooks["pre_tool_call"]()
    status_txt = ctx.commands["continuum"]("status")
    step_txt = ctx.commands["continuum"]("step")
    assert "24 engines (88 sub-engines)" in status_txt
    assert "160 tunable params" in status_txt
    assert "9 targets" in status_txt
    assert "adaptation step" in step_txt

    fission = json.loads(ctx.tools["continuum_fission_dag"]({"goal": "Alpha. Beta."}))
    tid = fission["tasks"][0]["task_id"]
    claim = json.loads(
        ctx.tools["continuum_claim_and_fence"]({"task_id": tid, "actor_id": "m1"})
    )
    assert claim["acquired"] is True
    ver = json.loads(
        ctx.tools["continuum_verify_record"](
            {"task_id": tid, "maker_id": "m1", "checker_id": "c1", "passed": True}
        )
    )
    assert ver["passed"] is True

    cart = json.loads(
        ctx.tools["continuum_compile_cartridge"](
            {
                "rules": [{"condition": "high load", "action": "shed noncritical"}],
                "target": "rust",
                "probe_input": [0.5, -0.2, 0.1, 0.8],
            }
        )
    )
    assert "#![no_std]" in cart["artifact"]
    assert cart["probe_evaluation"] is not None

    sheaf = json.loads(
        ctx.tools["continuum_sheaf_consistency_audit"](
            {
                "claims": [
                    {"agent_id": "a1", "state_vector": [1.0, 0.0, 0.0, 0.0]},
                    {"agent_id": "a2", "state_vector": [-1.0, 0.0, 0.0, 0.0]},
                ]
            }
        )
    )
    assert sheaf["consistent"] is False
    assert len(sheaf["contradictory_edges"]) == 1

    conf = json.loads(
        ctx.tools["continuum_conformal_calibrate"](
            {
                "calibration_scores": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "candidate_scores": [0.08, 0.92],
                "alpha": 0.2,
            }
        )
    )
    assert conf["prediction_set_indices"] == [0]

    hlora = json.loads(
        ctx.tools["continuum_hyperlora_modulate"](
            {"task_description": "embedded sensor filter", "target": "glsl"}
        )
    )
    assert hlora["anchor_drift"] == 0.0
    assert "#version 300 es" in hlora["artifact"]

    m_status = json.loads(
        ctx.tools["continuum_manifold_status"](
            {"engine_id": "fluid_hypergrid_interpolator_engine"}
        )
    )
    assert "interpolated_engine_config" in m_status

    ctx.hooks["post_tool_call"]()
    ctx.hooks["on_session_end"]()

    dash_status = dashboard_api.get_continuum_status()
    dash_engines = dashboard_api.get_continuum_engines()
    assert dash_status["engine_count"] == 24
    assert len(dash_engines["engines"]) == 24
    assert len(dash_engines["supported_targets"]) == 9
