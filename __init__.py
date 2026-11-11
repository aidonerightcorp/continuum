"""Continuum: Hermes sizes itself to each Task, and learns from its own records."""

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

CARD = {
    "name": "continuum_plugin_root",
    "role": "surface",
    "stratum": "omninode_shell_and_shape_composition",
    "contract": "hermes.tools_hooks_cli/v1",
    "describes": "Hermes plugin entry point: the right-sizing loop (hooks, request middleware, continuum_recall, /continuum) and, behind a setting, the experimental engines.",
    "keywords": ["continuum", "hermes", "plugin", "omninode", "tools", "hooks", "cli"],
}


class ContinuumRuntime:
    """Per-registration runtime container coordinating all five OmniNode-encased Hermes extension surfaces."""

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self.spec = get_manifold_spec()
        self.bus = CyberneticWeightBus(self.spec)
        self.store = ContinuumStore(
            db_path=db_path, spec=self.spec, manifold=self.bus.manifold
        )
        self.bus.bind_omninode_store(self.store)
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
        tool_defs = _build_tool_definitions(self)
        self.bus.omninode_director.register_tool_handlers(
            {t_name: handler for t_name, _, _, handler in tool_defs}
        )
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

    def _handle_probe_latent_state(args: dict[str, Any], **_: Any) -> str:
        act_vec = args.get("activation_vector")
        trace_txt = str(args.get("trace_text", "") or "")
        inter_vec = args.get("intermediate_vector")
        layer_acts = args.get("layer_activations")
        pos_ex = args.get("positive_examples")
        neg_ex = args.get("negative_examples")
        cal_sc = args.get("calibration_scores")
        concept = str(args.get("concept_name", "truthfulness") or "truthfulness")
        topk = args.get("topk")
        alpha = args.get("alpha")
        exp_target = args.get("export_target")
        exp_dim = args.get("export_dim")
        out = rt.bus.probe_latent_state(
            activation_vector=act_vec if isinstance(act_vec, list) else None,
            trace_text=trace_txt,
            intermediate_vector=inter_vec if isinstance(inter_vec, list) else None,
            layer_activations=layer_acts if isinstance(layer_acts, list) else None,
            positive_examples=pos_ex if isinstance(pos_ex, list) else None,
            negative_examples=neg_ex if isinstance(neg_ex, list) else None,
            calibration_scores=cal_sc if isinstance(cal_sc, list) else None,
            concept_name=concept,
            topk=int(topk) if topk is not None else None,
            alpha=float(alpha) if alpha is not None else None,
            export_target=str(exp_target) if exp_target else None,
            export_dim=int(exp_dim) if exp_dim is not None else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_steer_nullspace_vector(args: dict[str, Any], **_: Any) -> str:
        pos_acts = args.get("positive_activations")
        neg_acts = args.get("negative_activations")
        pos_prompt = str(args.get("positive_prompt", "") or "")
        neg_prompt = str(args.get("negative_prompt", "") or "")
        anchors = args.get("anchor_vectors")
        res_vec = args.get("residual_vector")
        eid = str(
            args.get("engine_id", "nullspace_activation_steerer")
            or "nullspace_activation_steerer"
        )
        strength = args.get("steering_strength")
        feat_idx = args.get("feature_index")
        dir_src = args.get("direction_source")
        target = str(args.get("target", "rust") or "rust")
        exp_dim = args.get("export_dim")
        persist = bool(args.get("persist_weight_fold", False))
        out = rt.bus.steer_nullspace_vector(
            positive_activations=pos_acts if isinstance(pos_acts, list) else None,
            negative_activations=neg_acts if isinstance(neg_acts, list) else None,
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            anchor_vectors=anchors if isinstance(anchors, list) else None,
            residual_vector=res_vec if isinstance(res_vec, list) else None,
            engine_id=eid,
            steering_strength=float(strength) if strength is not None else None,
            feature_index=int(feat_idx) if feat_idx is not None else None,
            direction_source=str(dir_src) if dir_src else None,
            target=target,
            export_dim=int(exp_dim) if exp_dim is not None else None,
            persist_weight_fold=persist,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_compose_omninode(args: dict[str, Any], **_: Any) -> str:
        shape_str = args.get("shape")
        goal_str = str(args.get("goal", "") or "")
        given_obj = args.get("given")
        overrides_obj = args.get("grid_overrides")
        maker_id = args.get("maker_id")
        checker_id = args.get("checker_id")
        verify_lock = bool(args.get("verify_lockfile", True))
        unk_axes = args.get("unknown_axes")
        out = rt.bus.compose_omninode(
            str(shape_str) if shape_str is not None else None,
            goal=goal_str,
            given=given_obj if isinstance(given_obj, dict) else None,
            grid_overrides=overrides_obj if isinstance(overrides_obj, dict) else None,
            maker_id=str(maker_id) if maker_id is not None else None,
            checker_id=str(checker_id) if checker_id is not None else None,
            verify_lockfile=verify_lock,
            unknown_axes=unk_axes if isinstance(unk_axes, list) else (),
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
            "Inspect live status, OmniNode CARD lockfile, grid/1 cell hash, and TT-Cross tuning progress across all 30 engines.",
            {
                "name": "continuum_manifold_status",
                "description": "Return health, OmniNode CARD lockfile status, grid/1 active cell SHA-256, and verified configuration counts for Continuum's 30 engines.",
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
        (
            "continuum_probe_latent_state",
            "Probe residual activations or traces via Matryoshka Top-K SAE features, Crosscoder circuits, ELK layer divergence, and Split-Conformal truthfulness.",
            {
                "name": "continuum_probe_latent_state",
                "description": "Run Matryoshka Top-K sparse feature decomposition, Crosscoder circuit tracing, internal concept/ELK layer-divergence probing, and Split-Conformal truthfulness calibration, with optional 9-runtime probe export.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "activation_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Surface or residual activation vector to decompose and probe.",
                        },
                        "trace_text": {
                            "type": "string",
                            "description": "Optional trace or statement text embedded locally when activation_vector is omitted.",
                        },
                        "intermediate_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Optional intermediate belief-layer activation vector for ELK layer-divergence detection.",
                        },
                        "layer_activations": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Optional multi-layer residual activations for sparse crosscoder circuit tracing.",
                        },
                        "positive_examples": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "negative_examples": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "calibration_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "concept_name": {"type": "string"},
                        "topk": {"type": "integer"},
                        "alpha": {"type": "number"},
                        "export_target": {"type": "string", "enum": all_targets},
                        "export_dim": {"type": "integer"},
                    },
                },
            },
            _handle_probe_latent_state,
        ),
        (
            "continuum_steer_nullspace_vector",
            "Compute a Contrastive or SAE activation steering vector projected into Null(A_anchor) (0.0 anchor drift), fold rank-1 LoRA-XS weights, and export across 9 runtimes.",
            {
                "name": "continuum_steer_nullspace_vector",
                "description": "Extract a contrastive or monosemantic SAE steering vector projected strictly onto Null(A_anchor) with zero anchor subspace drift, zero-hook LoRA-XS weight folding, and 9-runtime portable export.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positive_activations": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "negative_activations": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "positive_prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                        "anchor_vectors": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Protected anchor directions whose responses must experience 0.0 drift.",
                        },
                        "residual_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "engine_id": {"type": "string"},
                        "steering_strength": {"type": "number"},
                        "feature_index": {"type": "integer"},
                        "direction_source": {
                            "type": "string",
                            "enum": [
                                "hybrid_sae_contrastive",
                                "contrastive_mass_mean",
                                "sae_decoder_column",
                            ],
                        },
                        "target": {"type": "string", "enum": all_targets},
                        "export_dim": {"type": "integer"},
                        "persist_weight_fold": {"type": "boolean"},
                    },
                },
            },
            _handle_steer_nullspace_vector,
        ),
        (
            "continuum_compose_omninode",
            "Compose and execute a declarative OmniNode >> // | @ pipeline with CAS lockfile verification and a canonical grid/1 SHA-256 cell receipt.",
            {
                "name": "continuum_compose_omninode",
                "description": "Execute a declarative OmniNode shape expression (>> sequential pipe, | parallel sheaf fanout, // waterfall fallback, @ independent maker != checker quorum gate) with ParameterBounds clamping, CAS lockfile integrity verification, and a format='grid/1' cell receipt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {
                            "type": "string",
                            "description": "Algebraic shape expression, e.g. 'orient >> sae_probe >> (leace_erase | nullspace_steer) // fallback_compress @ independent_quorum_check'.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "Optional task or prompt context passed into the pipeline envelope.",
                        },
                        "given": {
                            "type": "object",
                            "description": "Optional initial structured envelope inputs (activations, anchors, target runtime).",
                        },
                        "grid_overrides": {
                            "type": "object",
                            "description": "Optional NodeGrid or grid/1 axis overrides clamped by ParameterBounds.",
                        },
                        "maker_id": {"type": "string"},
                        "checker_id": {"type": "string"},
                        "verify_lockfile": {"type": "boolean"},
                        "unknown_axes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["shape"],
                },
            },
            _handle_compose_omninode,
        ),
    ]


RIGHTSIZE_SETTINGS = ("mode", "small_model", "small_model_cost_ratio", "small_model_max_context", "rungs",
                      "size_tools", "recall", "store_text", "text_days", "explore_share", "self_tune", "check_share")


class _LabRuntime:
    """Builds the experimental engines only when something asks for them."""

    def __init__(self) -> None:
        self._rt: ContinuumRuntime | None = None

    def get(self) -> ContinuumRuntime:
        if self._rt is None:
            self._rt = ContinuumRuntime()
        return self._rt

    def __getattr__(self, name: str) -> Any:
        return getattr(self.get(), name)


def register(ctx: Any) -> Any:
    """Hermes entry point.

    Always on: the right-sizing loop (7 hooks, 1 request middleware, the continuum_recall tool, /continuum).
    Behind the setting enable_lab_tools: the 18 tools of the experimental engines of version 1. The plugin
    registers nothing that its manifest does not declare: the context, memory and delegation providers of
    version 1 stay in the package as research code and are not registered with Hermes.
    """
    from continuum_core import storage as _storage
    from continuum_core.rightsize import RightSizer
    from continuum_core.rightsize import hermes as _wiring
    from continuum_core.rightsize.engine import _flag

    # A yes/no setting typed as text ("false", "no", "off") is a no; a value that is neither is a no too.
    lab_on = _flag(ctx.get_config("enable_lab_tools", False), False)
    rt = _LabRuntime()

    # ---- the right-sizing loop ----
    data_dir = Path(_storage.resolve_continuum_db_path()).parent
    settings = {key: ctx.get_config(key, None) for key in RIGHTSIZE_SETTINGS}
    no_home = data_dir.name == "continuum-standalone"
    if no_home:  # Hermes gave no data folder: the shared temp folder is no place for a user's messages
        settings["mode"] = "off"
    # With no data folder no file is opened at all, also not one that an earlier run left in the temp folder.
    sizer = RightSizer(":memory:" if no_home else data_dir / "rightsize.db", settings=settings)
    if no_home:
        sizer.store_note = "Hermes gave no plugin data folder, so Continuum is off and stores nothing"
        try:
            data_dir.rmdir()  # the lookup made this temp folder; it is removed again when it is empty
        except OSError:
            pass
    _wiring.install(ctx, sizer, data_dir)
    rt.sizer = sizer

    # ---- the experimental engines (hidden from the model unless enable_lab_tools is on) ----
    def _lab_ok() -> bool:
        # The lab keeps its own file: its tools exist for the model only when the user turned the lab on,
        # Continuum is not off, and Hermes gave the plugin a data folder.
        return bool(lab_on and not no_home and sizer.mode != "off")

    def _gated(handler: Any) -> Any:
        # The host can call a tool that it listed a moment ago, so each handler asks again before it runs.
        def run(args: dict[str, Any], **kw: Any) -> str:
            if not _lab_ok():
                return json.dumps({"error": "The experimental engines of Continuum version 1 are off. "
                                            "Nothing was run and nothing was stored."})
            return handler(args, **kw)
        return run

    for tool_name, desc, schema, handler in _build_tool_definitions(rt):  # type: ignore[arg-type]
        ctx.register_tool(
            name=tool_name,
            toolset="continuum",
            schema=schema,
            handler=_gated(handler),
            description=desc,
            check_fn=_lab_ok,
        )

    try:  # a setting typed by a person: a wrong value falls back to the spec's own default, it never stops the load
        default_headroom = float(ctx.get_config("compute_headroom_ratio", 0.0) or 0.0) if lab_on else 0.0
    except (TypeError, ValueError):
        default_headroom = 0.0
    if not 0.0 <= default_headroom <= 1.0:
        default_headroom = 0.0

    # ---- /continuum ----
    def _lab_command(arg: str) -> str:
        short_n = rt.spec.c_int("uid_digest_chars")
        headroom = default_headroom or float(
            rt.spec.section("hypergrid_fluid_interpolation")["opportunistic_exploration"]["default_compute_headroom"])
        if arg.startswith("step"):
            step_res = rt.bus.coupling_step(
                compute_headroom_ratio=headroom,
                replay_records=rt.store.recent_verified_records(),
            )
            rt.persist_status_snapshot()
            return (
                f"Continuum lab: adaptation step #{step_res['coupling_step']} complete | "
                f"stability delta={step_res['lyapunov_delta']:.6f} | "
                f"configs tested (no model call)={step_res['opportunistic_exploration']['candidates_generated']}"
            )
        if arg.startswith("omninode"):
            comp_res = rt.bus.compose_omninode(goal="slash command omninode check")
            rt.persist_status_snapshot()
            cell_short = str(comp_res["grid1_receipt"]["cell"])[:short_n]
            lock_short = str(comp_res["lockfile"]["lockfile_sha256"])[:short_n]
            return (
                f"Continuum lab: OmniNode run ok={comp_res['ok']} | "
                f"shape='{comp_res['shape']}' | "
                f"lockfile={lock_short} (matches={comp_res['lockfile']['verified']}) | "
                f"grid/1 cell={cell_short}"
            )
        snap = rt.persist_status_snapshot()
        omni = snap.get("omninode", {})
        return (
            f"Continuum lab v{snap['version']} (experimental engines; "
            f"{'on' if lab_on else 'off: set enable_lab_tools to show its tools to the model'}) | "
            f"{snap['engine_count']} engine modules | 18 lab tools | "
            f"{snap['hypergrid_total_dimensions']} settings | "
            f"lockfile={str(omni.get('lockfile_sha256', ''))[:short_n]} | "
            f"grid/1 cell={str(omni.get('active_grid1_cell', ''))[:short_n]}"
        )

    def _slash_command_handler(raw_args: str = "", session_id: str = "", platform: Any = "", **_: Any) -> str:
        try:
            platform = str(getattr(platform, "value", platform) or "")
            text = _wiring.command(sizer, data_dir, raw_args, session_id=str(session_id or ""), platform=platform)
            if text is not None:
                return text
            arg = (raw_args or "").strip().lower()
            if arg.startswith(("lab", "step", "omninode")):
                # The experimental engines keep their own file. They run only when the user turned them on,
                # Continuum is not off, and Hermes gave the plugin a data folder.
                if not lab_on or no_home or sizer.mode == "off":
                    return ("The experimental engines of version 1 are off, and this command stored nothing. "
                            "To use them, set enable_lab_tools: true (with mode auto or observe) and start "
                            "Hermes again.")
                if platform not in sizer.cfg["recall"].get("single_user_platforms", []):
                    return ("The lab commands run only for the local user. On the machine that runs Hermes, "
                            "type: hermes continuum " + arg)
                return _lab_command(arg.removeprefix("lab").strip())
            return _wiring.HELP
        except Exception as exc:  # a command must always answer
            return f"Continuum could not run that command ({type(exc).__name__}: {exc})."

    ctx.register_command(
        "continuum",
        _slash_command_handler,
        description="Right-sizing status, the reason for the last choice, and outcome feedback.",
        args_hint="[status|why|good|bad|pin|unpin|unblock|audit|tune|export|import FILE|forget-text|compact|lab|help]",
    )

    # ---- `hermes continuum …` in a terminal: the same commands, with no session open ----
    def _cli_setup(parser: Any) -> None:
        parser.add_argument("words", nargs="*", metavar="COMMAND",
                            help="status, why, good, bad, unblock, audit, tune, export, import FILE, forget-text, compact, help")

    def _cli_handler(args: Any) -> int:
        words = [str(w) for w in (getattr(args, "words", None) or [])]
        if words and words[0].lower() in ("pin", "unpin"):
            print("pin and unpin belong to one session: type /continuum pin inside that session.")
            return 0
        print(_slash_command_handler(" ".join(words), session_id="", platform="cli"))
        return 0

    if hasattr(ctx, "register_cli_command"):
        ctx.register_cli_command(
            name="continuum",
            help="Continuum right-sizing: status, the reason for the last choice, audit, export",
            setup_fn=_cli_setup,
            handler_fn=_cli_handler,
            description="The /continuum commands from a terminal. `why`, `good` and `bad` act on the newest turn.",
        )

    # ---- bundled skill ----
    skill_path = Path(__file__).resolve().parent / "skills" / "continuum" / "SKILL.md"
    if skill_path.is_file() and hasattr(ctx, "register_skill"):
        ctx.register_skill(
            "continuum",
            skill_path,
            description="How to read Continuum's right-sizing records and give it outcome feedback.",
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
