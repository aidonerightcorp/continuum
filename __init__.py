"""Continuum — Unified Self-Improving Agent Engine for Hermes Agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = str(Path(__file__).resolve().parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from continuum_core.config_loader import get_manifold_spec  # noqa: E402
from continuum_core.context_engine import (  # noqa: E402
    CONTEXT_GREP_SCHEMA,
    ContinuumContextEngine,
)
from continuum_core.delegation_provider import (  # noqa: E402
    ContinuumDelegationProvider,
    DelegationRouteError,
)
from continuum_core.memory_provider import (  # noqa: E402
    ATTRACTOR_RECALL_SCHEMA,
    ContinuumMemoryProvider,
)
from continuum_core.storage import ContinuumStore  # noqa: E402
from continuum_core.weight_bus import CyberneticWeightBus  # noqa: E402


class ContinuumRuntime:
    """Per-registration runtime container coordinating all five Hermes extension surfaces."""

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self.spec = get_manifold_spec()
        self.bus = CyberneticWeightBus(self.spec)
        self.store = ContinuumStore(
            db_path=db_path, spec=self.spec, manifold=self.bus.manifold
        )
        self.context_engine = ContinuumContextEngine(
            bus=self.bus, store=self.store, spec=self.spec
        )
        self.memory_provider = ContinuumMemoryProvider(
            bus=self.bus, spec=self.spec
        )
        self.delegation_provider = ContinuumDelegationProvider(
            bus=self.bus, store=self.store, spec=self.spec
        )
        self.tool_invocations: int = 0
        self.persist_status_snapshot()

    def persist_status_snapshot(self) -> dict[str, Any]:
        status = self.bus.manifold_status()
        status["tool_invocations"] = self.tool_invocations
        status["eigentrust"] = self.store.compute_eigentrust()
        self.store.save_snapshot("latest_status", status)
        return status


def _build_tool_definitions(
    rt: ContinuumRuntime,
) -> list[tuple[str, str, dict[str, Any], Any]]:
    def _handle_fission_dag(args: dict[str, Any], **_: Any) -> str:
        goal = str(args.get("goal", "") or "")
        subtasks = args.get("subtasks")
        max_fanout = args.get("max_fanout")
        out = rt.store.fission_dag(
            goal,
            subtasks=subtasks if isinstance(subtasks, list) else None,
            max_fanout=int(max_fanout) if max_fanout is not None else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_claim_and_fence(args: dict[str, Any], **_: Any) -> str:
        task_id = str(args["task_id"])
        actor_id = str(args.get("actor_id", "primary_maker") or "primary_maker")
        ttl = args.get("lease_ttl_seconds")
        out = rt.store.claim_and_fence(
            task_id,
            actor_id,
            lease_ttl_seconds=float(ttl) if ttl is not None else None,
        )
        return json.dumps(out)

    def _handle_verify_record(args: dict[str, Any], **_: Any) -> str:
        out = rt.store.verify_record(
            str(args["task_id"]),
            maker_id=str(args["maker_id"]),
            checker_id=str(args["checker_id"]),
            passed=bool(args.get("passed", True)),
            quality_score=float(args.get("quality_score", 1.0)),
            output_summary=str(args.get("output_summary", "") or ""),
            tier_id=str(args.get("tier_id", "verify_tier") or "verify_tier"),
            notes=str(args.get("notes", "") or ""),
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_compile_cartridge(args: dict[str, Any], **_: Any) -> str:
        cid = str(args.get("cartridge_id", "compiled_cartridge") or "compiled_cartridge")
        rules = args.get("rules", [])
        target = str(args.get("target", "rust") or "rust")
        num_layers = args.get("num_layers")
        activation = str(args.get("activation", "gelu") or "gelu")
        early_margin = args.get("early_exit_margin")

        cart = rt.bus.compiler.compile_stack(
            cid,
            rules if isinstance(rules, list) else [],
            num_layers=int(num_layers) if num_layers is not None else None,
            activation=activation,
            early_exit_margin=float(early_margin) if early_margin is not None else None,
        )
        artifact = rt.bus.compiler.lower_to_target(cart, target)
        probe_input = args.get("probe_input")
        cond_vec = args.get("condition_embedding")
        eval_info = None
        if isinstance(probe_input, list):
            eval_info = cart.forward(probe_input, cond_vec=cond_vec)
            rt.bus.cost_ledger.cartridge_evaluations += 1
            rt.bus.cost_ledger.layers_saved_by_early_exit += int(
                eval_info["layers_saved"]
            )
        rt.persist_status_snapshot()
        return json.dumps(
            {
                "cartridge_id": cart.cartridge_id,
                "target": target,
                "supported_targets": rt.bus.compiler.supported_targets,
                "layers": len(cart.layers),
                "early_exit_margin": cart.early_exit_margin,
                "fit_mse": cart.fit_mse,
                "weight2vec_16d": cart.weight2vec.slice_prefix(
                    rt.spec.section("matryoshka_manifold")["coarse_dim"]
                ),
                "artifact": artifact,
                "probe_evaluation": eval_info,
            }
        )

    def _handle_analyze_spectrum(args: dict[str, Any], **_: Any) -> str:
        eid = args.get("engine_id")
        mat = args.get("matrix")
        samples = args.get("sample_inputs")
        out = rt.bus.analyze_spectrum(
            engine_id=str(eid) if eid else None,
            matrix=mat if isinstance(mat, list) else None,
            sample_inputs=samples if isinstance(samples, list) else None,
        )
        return json.dumps(out)

    def _handle_surgical_edit(args: dict[str, Any], **_: Any) -> str:
        eid = str(
            args.get("engine_id", rt.bus.engine_order[0]) or rt.bus.engine_order[0]
        )
        mode = str(args.get("mode", "svd_nullspace") or "svd_nullspace")
        out = rt.bus.surgical_edit(
            eid,
            mode=mode,
            input_vec=args.get("input_vec"),
            target_vec=args.get("target_vec"),
            counterexample_probes=args.get("counterexample_probes"),
            step_gain=float(args["step_gain"])
            if "step_gain" in args and args["step_gain"] is not None
            else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_coupling_step(args: dict[str, Any], **_: Any) -> str:
        headroom = args.get("compute_headroom_ratio")
        replays = rt.store.recent_verified_records()
        out = rt.bus.coupling_step(
            impulse_engine_id=str(args["impulse_engine_id"])
            if args.get("impulse_engine_id")
            else None,
            impulse_delta=float(args["impulse_delta"])
            if args.get("impulse_delta") is not None
            else None,
            compute_headroom_ratio=float(headroom) if headroom is not None else None,
            replay_records=replays,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_morphism_transfer(args: dict[str, Any], **_: Any) -> str:
        src_dag = args.get("source_dag", [])
        dst_dag = args.get("target_dag", [])
        src_eng = str(
            args.get("source_engine_id", "ridge_polytope_compiler_engine")
            or "ridge_polytope_compiler_engine"
        )
        dst_eng = str(
            args.get("target_engine_id", "multilayer_film_molora_stack_engine")
            or "multilayer_film_molora_stack_engine"
        )
        out = rt.bus.morphism_transfer(
            src_dag if isinstance(src_dag, list) else [],
            dst_dag if isinstance(dst_dag, list) else [],
            source_engine_id=src_eng,
            target_engine_id=dst_eng,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_calibrate_effort(args: dict[str, Any], **_: Any) -> str:
        goal = str(args.get("goal", "") or "")
        stakes = args.get("stakes")
        stratum = str(args.get("stratum", "compile") or "compile")
        out = rt.bus.calibrate_effort(
            goal,
            stakes=float(stakes) if stakes is not None else None,
            stratum=stratum,
        )
        return json.dumps(out)

    def _handle_manifold_status(args: dict[str, Any], **_: Any) -> str:
        status = rt.persist_status_snapshot()
        engine_filter = args.get("engine_id")
        if engine_filter:
            eid = str(engine_filter)
            q_vec = rt.bus.manifold.embed_task(eid)
            fluid = rt.bus.hypergrid.interpolate_fluid_config(eid, q_vec)
            fluid_clean = {k: v for k, v in fluid.items() if k != "config2vec"}
            status["interpolated_engine_config"] = fluid_clean
        return json.dumps(status)

    def _handle_context_grep(args: dict[str, Any], **kw: Any) -> str:
        return rt.context_engine.handle_tool_call("continuum_context_grep", args, **kw)

    def _handle_attractor_recall(args: dict[str, Any], **kw: Any) -> str:
        return rt.memory_provider.handle_tool_call("continuum_attractor_recall", args, **kw)

    def _handle_sheaf_consistency_audit(args: dict[str, Any], **_: Any) -> str:
        claims = args.get("claims", [])
        edges = args.get("edges")
        stalk_dim = args.get("stalk_dim")
        eta = args.get("harmonic_eta")
        edge_th = args.get("edge_threshold")
        out = rt.bus.sheaf_consistency_audit(
            claims if isinstance(claims, list) else [],
            edges if isinstance(edges, list) else None,
            stalk_dim=int(stalk_dim) if stalk_dim is not None else None,
            harmonic_eta=float(eta) if eta is not None else None,
            edge_threshold=float(edge_th) if edge_th is not None else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_conformal_calibrate(args: dict[str, Any], **_: Any) -> str:
        cal_scores = args.get("calibration_scores", [])
        cand_scores = args.get("candidate_scores", [])
        alpha = args.get("alpha")
        alpha_t = args.get("current_alpha_t")
        misses = args.get("recent_misses")
        is_prob = bool(args.get("scores_are_probabilities", False))
        max_set = args.get("abstention_max_set_size")
        ctx_q = str(args.get("context_query", "") or "conformal selective calibration")
        out = rt.bus.conformal_calibrate(
            cal_scores if isinstance(cal_scores, list) else [],
            cand_scores if isinstance(cand_scores, list) else [],
            alpha=float(alpha) if alpha is not None else None,
            current_alpha_t=float(alpha_t) if alpha_t is not None else None,
            recent_misses=misses if isinstance(misses, list) else None,
            scores_are_probabilities=is_prob,
            abstention_max_set_size=int(max_set) if max_set is not None else None,
            context_query=ctx_q,
        )
        return json.dumps(out)

    def _handle_hyperlora_modulate(args: dict[str, Any], **_: Any) -> str:
        desc = str(args.get("task_description", "") or "")
        eid = str(
            args.get("engine_id", "multilayer_film_molora_stack_engine")
            or "multilayer_film_molora_stack_engine"
        )
        target = str(args.get("target", "rust") or "rust")
        emb = args.get("task_embedding")
        ranks = args.get("rank_components")
        gain = args.get("modulation_gain")
        exp_dim = args.get("export_dim")
        persist = bool(args.get("persist_modulation", True))
        probes = args.get("counterexample_probes")
        out = rt.bus.hyperlora_modulate(
            desc,
            engine_id=eid,
            task_embedding=emb if isinstance(emb, list) else None,
            target=target,
            rank_components=int(ranks) if ranks is not None else None,
            modulation_gain=float(gain) if gain is not None else None,
            export_dim=int(exp_dim) if exp_dim is not None else None,
            persist_modulation=persist,
            counterexample_probes=probes if isinstance(probes, list) else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    all_targets = [
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
        "wasm_base64",
    ]

    return [
        (
            "continuum_fission_dag",
            "Break a complex goal into ordered subtasks with clear interface contracts and clarifying questions.",
            {
                "name": "continuum_fission_dag",
                "description": "Break a complex goal into ordered subtasks stored in local SQLite WAL with interface contracts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Objective to decompose into subtasks."},
                        "subtasks": {"type": "array", "items": {"type": "object"}},
                        "max_fanout": {"type": "integer"},
                    },
                    "required": ["goal"],
                },
            },
            _handle_fission_dag,
        ),
        (
            "continuum_claim_and_fence",
            "Claim an exclusive subtask lease or reuse an already-verified result from cache.",
            {
                "name": "continuum_claim_and_fence",
                "description": "Atomically claim a subtask lease with a fencing stamp or return a verified cached output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "actor_id": {"type": "string"},
                        "lease_ttl_seconds": {"type": "number"},
                    },
                    "required": ["task_id"],
                },
            },
            _handle_claim_and_fence,
        ),
        (
            "continuum_verify_record",
            "Record an independent maker != checker review and update routing, consensus, and reputation models.",
            {
                "name": "continuum_verify_record",
                "description": "Record an independent verification where the reviewer differs from the author.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "maker_id": {"type": "string"},
                        "checker_id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "quality_score": {"type": "number"},
                        "output_summary": {"type": "string"},
                        "tier_id": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["task_id", "maker_id", "checker_id", "passed"],
                },
            },
            _handle_verify_record,
        ),
        (
            "continuum_compile_cartridge",
            "Compile repeated rules or examples into fast multi-layer modules across 9 zero-dependency runtimes.",
            {
                "name": "continuum_compile_cartridge",
                "description": "Compile decision rules or input-target examples into rust (#![no_std]), glsl, js, python, sql, c99, wasm, wat, wgsl, or safetensors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cartridge_id": {"type": "string"},
                        "rules": {"type": "array", "items": {"type": "object"}},
                        "target": {
                            "type": "string",
                            "enum": all_targets,
                        },
                        "num_layers": {"type": "integer"},
                        "activation": {"type": "string", "enum": ["gelu", "silu", "relu", "identity"]},
                        "early_exit_margin": {"type": "number"},
                        "probe_input": {"type": "array", "items": {"type": "number"}},
                        "condition_embedding": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["rules", "target"],
                },
            },
            _handle_compile_cartridge,
        ),
        (
            "continuum_analyze_spectrum",
            "Inspect matrix rank, signal-to-noise threshold, and input feature importance for any engine.",
            {
                "name": "continuum_analyze_spectrum",
                "description": "Analyze singular value health, effective rank, and input sensitivity for an engine or matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "sample_inputs": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                },
            },
            _handle_analyze_spectrum,
        ),
        (
            "continuum_surgical_edit",
            "Apply targeted weight updates that preserve protected baseline behaviors with zero drift.",
            {
                "name": "continuum_surgical_edit",
                "description": "Update an engine's weights while guaranteeing zero drift on protected baseline test cases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["svd_nullspace", "lora_xs", "recursive_woodbury", "procrustes_align", "hopfield_write"],
                        },
                        "input_vec": {"type": "array", "items": {"type": "number"}},
                        "target_vec": {"type": "array", "items": {"type": "number"}},
                        "counterexample_probes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "step_gain": {"type": "number"},
                    },
                },
            },
            _handle_surgical_edit,
        ),
        (
            "continuum_coupling_step",
            "Coordinate cross-engine updates and test new TT-Cross parameter candidates against saved history using zero LLM tokens.",
            {
                "name": "continuum_coupling_step",
                "description": "Run a stability-checked engine coordination step and explore epistatic TT-Cross configurations offline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "impulse_engine_id": {"type": "string"},
                        "impulse_delta": {"type": "number"},
                        "compute_headroom_ratio": {"type": "number"},
                    },
                },
            },
            _handle_coupling_step,
        ),
        (
            "continuum_morphism_transfer",
            "Compare two task workflows structurally and adapt learned weights from one workflow to another.",
            {
                "name": "continuum_morphism_transfer",
                "description": "Measure structural similarity between two task graphs and transfer aligned weights safely.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_dag": {"type": "array", "items": {"type": "object"}},
                        "target_dag": {"type": "array", "items": {"type": "object"}},
                        "source_engine_id": {"type": "string"},
                        "target_engine_id": {"type": "string"},
                    },
                    "required": ["source_dag", "target_dag"],
                },
            },
            _handle_morphism_transfer,
        ),
        (
            "continuum_calibrate_effort",
            "Choose reasoning effort, embedding depth, and whether to ask the user a clarifying question.",
            {
                "name": "continuum_calibrate_effort",
                "description": "Estimate uncertainty and stakes to pick low/medium/high effort and whether to escalate to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "stakes": {"type": "number"},
                        "stratum": {"type": "string"},
                    },
                    "required": ["goal"],
                },
            },
            _handle_calibrate_effort,
        ),
        (
            "continuum_manifold_status",
            "Inspect live status, TT-Cross configuration tuning progress, and token/compute savings across all 24 engines.",
            {
                "name": "continuum_manifold_status",
                "description": "Return health, verified configuration counts, and efficiency metrics for Continuum's 24 engines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                    },
                },
            },
            _handle_manifold_status,
        ),
        (
            "continuum_context_grep",
            CONTEXT_GREP_SCHEMA["description"],
            CONTEXT_GREP_SCHEMA,
            _handle_context_grep,
        ),
        (
            "continuum_attractor_recall",
            ATTRACTOR_RECALL_SCHEMA["description"],
            ATTRACTOR_RECALL_SCHEMA,
            _handle_attractor_recall,
        ),
        (
            "continuum_sheaf_consistency_audit",
            "Audit multi-agent claims via Cellular Sheaf Laplacian coboundary energy, flag contradictions, and compute harmonic consensus.",
            {
                "name": "continuum_sheaf_consistency_audit",
                "description": "Build a Cellular Sheaf Laplacian over agent claims and restriction maps to localize contradictions and compute harmonic consensus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Agent claim nodes, each with agent_id and either state_vector or text.",
                        },
                        "edges": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional communication edges with source, target, weight, and restriction maps.",
                        },
                        "stalk_dim": {"type": "integer"},
                        "harmonic_eta": {"type": "number"},
                        "edge_threshold": {"type": "number"},
                    },
                    "required": ["claims"],
                },
            },
            _handle_sheaf_consistency_audit,
        ),
        (
            "continuum_conformal_calibrate",
            "Build distribution-free Split Conformal + Online ACI 1-alpha prediction sets with selective abstention routing.",
            {
                "name": "continuum_conformal_calibrate",
                "description": "Compute finite-sample 1 - alpha conformal prediction sets, online ACI alpha adaptation, and selective cartridge/LLM routing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calibration_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Historical calibration nonconformity scores (or probabilities if scores_are_probabilities=true).",
                        },
                        "candidate_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Candidate action or output scores to filter into a guaranteed prediction set.",
                        },
                        "alpha": {"type": "number"},
                        "current_alpha_t": {"type": "number"},
                        "recent_misses": {"type": "array", "items": {"type": "boolean"}},
                        "scores_are_probabilities": {"type": "boolean"},
                        "abstention_max_set_size": {"type": "integer"},
                        "context_query": {"type": "string"},
                    },
                    "required": ["calibration_scores", "candidate_scores"],
                },
            },
            _handle_conformal_calibrate,
        ),
        (
            "continuum_hyperlora_modulate",
            "Synthesize input-conditioned Kronecker HyperLoRA weight modulations in the safety-anchor nullspace (0.0 drift) across 9 targets.",
            {
                "name": "continuum_hyperlora_modulate",
                "description": "Generate an input-conditioned Kronecker weight modulation projected strictly onto the anchor nullspace and lower to any portable target.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string"},
                        "engine_id": {"type": "string"},
                        "task_embedding": {"type": "array", "items": {"type": "number"}},
                        "target": {"type": "string", "enum": all_targets},
                        "rank_components": {"type": "integer"},
                        "modulation_gain": {"type": "number"},
                        "export_dim": {"type": "integer"},
                        "persist_modulation": {"type": "boolean"},
                        "counterexample_probes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                },
            },
            _handle_hyperlora_modulate,
        ),
    ]


def register(ctx: Any) -> ContinuumRuntime:
    """Hermes v0.23.0+ entry point registering native providers, 15 tools, 4 hooks, /continuum, and skill."""
    rt = ContinuumRuntime()
    default_headroom = float(
        ctx.get_config(
            "compute_headroom_ratio",
            rt.spec.section("hypergrid_fluid_interpolation")[
                "opportunistic_exploration"
            ]["default_compute_headroom"],
        )
    )
    idle_explore = bool(ctx.get_config("enable_idle_exploration", True))

    # Surface 1, 2, 3 — Native ContextEngine, MemoryProvider, DelegationProvider
    ctx.register_context_engine(rt.context_engine)
    ctx.register_memory_provider(rt.memory_provider)
    ctx.register_delegation_provider(rt.delegation_provider)

    # Surface 4a — 15 Agent Tools
    for tool_name, desc, schema, handler in _build_tool_definitions(rt):
        ctx.register_tool(
            name=tool_name,
            toolset="continuum",
            schema=schema,
            handler=handler,
            description=desc,
        )

    # Surface 4b — 4 Lifecycle Hooks
    def _on_session_start(*args: Any, **kwargs: Any) -> None:
        sid = str(kwargs.get("session_id", "default") or "default")
        rt.context_engine.on_session_start(session_id=sid)

    def _pre_tool_call(*args: Any, **kwargs: Any) -> None:
        rt.tool_invocations += 1

    def _post_tool_call(*args: Any, **kwargs: Any) -> None:
        rt.persist_status_snapshot()

    def _on_session_end(*args: Any, **kwargs: Any) -> None:
        if idle_explore:
            replays = rt.store.recent_verified_records()
            rt.bus.coupling_step(
                compute_headroom_ratio=default_headroom,
                replay_records=replays,
            )
        rt.persist_status_snapshot()

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)

    # Surface 4c — /continuum Slash Command
    def _slash_command_handler(raw_args: str = "", **_: Any) -> str:
        arg = (raw_args or "").strip().lower()
        if arg.startswith("step"):
            step_res = rt.bus.coupling_step(
                compute_headroom_ratio=default_headroom,
                replay_records=rt.store.recent_verified_records(),
            )
            rt.persist_status_snapshot()
            return (
                f"Continuum adaptation step #{step_res['coupling_step']} complete | "
                f"stability delta={step_res['lyapunov_delta']:.6f} | "
                f"configs tested (0 tokens)={step_res['opportunistic_exploration']['candidates_generated']}"
            )
        snap = rt.persist_status_snapshot()
        ledger = snap["grid_ledger_counts"]
        cost = snap["cost_ledger"]
        targets_n = len(snap.get("supported_targets", []))
        return (
            f"Continuum v{snap['version']} | "
            f"{snap['engine_count']} engines ({snap['sub_engine_count']} sub-engines) | "
            f"{snap['hypergrid_total_dimensions']} tunable params | "
            f"{targets_n} targets | "
            f"embedding slices={snap['matryoshka_prefix_dims']} | "
            f"verified/promoted={ledger.get('verified', 0)}/{ledger.get('promoted', 0)} | "
            f"zero-token config probes={cost['sha_zero_token_probes']}"
        )

    ctx.register_command(
        "continuum",
        _slash_command_handler,
        description="View Continuum engine status or run '/continuum step' to tune configurations.",
        args_hint="[status|step]",
    )

    # Surface 4d — Bundled Continuum Skill
    skill_path = (
        Path(__file__).resolve().parent / "skills" / "continuum" / "SKILL.md"
    )
    if skill_path.is_file() and hasattr(ctx, "register_skill"):
        ctx.register_skill(
            "continuum",
            skill_path,
            description="Workflow guide for Continuum task planning, Sheaf & Conformal verification, and 9-target skill compilation.",
        )

    return rt


__all__ = [
    "ContinuumContextEngine",
    "ContinuumDelegationProvider",
    "ContinuumMemoryProvider",
    "ContinuumRuntime",
    "DelegationRouteError",
    "register",
]
