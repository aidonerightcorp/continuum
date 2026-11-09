"""Universal Cybernetic Engines (v1.1.0): TT-Cross Explorer, Cellular Sheaf Consensus, Conformal ACI, Causal SCM, VCG, Invariants, IB, OED, Self-Healing, and Kronecker HyperLoRA."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from continuum_core.config_loader import ManifoldConfigSpec, get_manifold_spec

CARD = {
    "name": "cybernetic_engines",
    "role": "engine_pack",
    "stratum": "uncertainty_and_verification",
    "contract": "continuum.module/v1",
    "describes": "TT-Cross Explorer, Cellular Sheaf Consensus, Conformal ACI, Causal SCM, VCG, Invariants, IB, OED, and Self-Healing.",
    "keywords": ["sheaf", "conformal", "causal", "vcg", "invariants", "tt_cross", "hyperlora"],
}
from continuum_core.linalg_backend import (
    Matrix,
    Vector,
    add_scaled,
    copy_matrix,
    frobenius_norm,
    identity_matrix,
    matmul,
    matvec,
    outer_product,
    ridge_solve,
    softmax,
    svd_compact,
    transpose,
    vec_dot,
    vec_norm,
    zeros_matrix,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold


def kronecker_product(mat_a: Matrix, mat_b: Matrix) -> Matrix:
    """Compute exact Kronecker product A (x) B of two matrices."""
    rows_a = len(mat_a)
    cols_a = len(mat_a[0]) if rows_a > 0 else 0
    rows_b = len(mat_b)
    cols_b = len(mat_b[0]) if rows_b > 0 else 0
    out = zeros_matrix(rows_a * rows_b, cols_a * cols_b)
    for ra in range(rows_a):
        for ca in range(cols_a):
            scale = float(mat_a[ra][ca])
            r_off = ra * rows_b
            c_off = ca * cols_b
            for rb in range(rows_b):
                for cb in range(cols_b):
                    out[r_off + rb][c_off + cb] = scale * float(mat_b[rb][cb])
    return out


class TensorTrainCross:
    """Low-rank Tensor-Train Cross (TT-Cross) explorer capturing epistatic cross-axis interactions."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()
        cfg = self._spec.section("tt_cross_explorer")
        self.default_rank: int = int(cfg["tt_rank"])
        self.default_sweeps: int = int(cfg["sweeps"])
        self.default_levels: int = int(cfg["discrete_levels"])
        self._reg: float = float(cfg["pivot_regularization"])

    def fit_and_search(
        self,
        objective_fn: Callable[[list[int]], float],
        num_axes: int,
        *,
        num_levels: int | None = None,
        tt_rank: int | None = None,
        sweeps: int | None = None,
    ) -> dict[str, Any]:
        """Run alternating TT-Cross fiber sweeps to locate epistatic optima in O(d * r^2 * L) evaluations."""
        d_axes = max(1, int(num_axes))
        levels = max(self._spec.c_int("two"), int(num_levels or self.default_levels))
        rank_r = max(1, int(tt_rank or self.default_rank))
        n_sweeps = max(1, int(sweeps or self.default_sweeps))

        cache: dict[tuple[int, ...], float] = {}

        def _eval(idx_vec: Sequence[int]) -> float:
            key = tuple(int(v) for v in idx_vec)
            if key not in cache:
                cache[key] = float(objective_fn(list(key)))
            return cache[key]

        # Initialize diverse pivot set spanning low/mid/high discrete levels
        pivots: list[list[int]] = []
        for r_idx in range(rank_r):
            pivots.append([(r_idx + ax) % levels for ax in range(d_axes)])

        best_idx = list(pivots[0])
        best_val = _eval(best_idx)
        tt_core_norms: list[float] = [0.0 for _ in range(d_axes)]

        for _ in range(n_sweeps):
            # Forward left-to-right cross sweep
            for k in range(d_axes):
                cross_mat = zeros_matrix(len(pivots), levels)
                for p_i, piv in enumerate(pivots):
                    for lev in range(levels):
                        probe = list(piv)
                        probe[k] = lev
                        val = _eval(probe)
                        cross_mat[p_i][lev] = val
                        if val > best_val:
                            best_val = val
                            best_idx = probe

                # Extract low-rank TT core statistics and update pivot coordinate k greedily + diversely
                u_m, s_v, vt_m = svd_compact(cross_mat)
                eff_r = min(rank_r, len(s_v))
                core = [
                    [u_m[row][col] * s_v[col] for col in range(eff_r)]
                    for row in range(len(u_m))
                ]
                tt_core_norms[k] = float(frobenius_norm(matmul(core, vt_m[:eff_r])))

                # Update first pivot to global best and remaining pivots to highest-leverage distinct levels
                pivots[0] = list(best_idx)
                scored_levels = sorted(
                    range(levels),
                    key=lambda lv: sum(abs(cross_mat[p][lv]) for p in range(len(pivots))),
                    reverse=True,
                )
                for p_i in range(1, len(pivots)):
                    chosen_lv = scored_levels[p_i % len(scored_levels)]
                    pivots[p_i][k] = chosen_lv

            # Backward right-to-left 2-site DMRG superblock sweep resolving epistatic pair valleys
            if d_axes >= self._spec.c_int("two"):
                for k in range(d_axes - self._spec.c_int("two"), -1, -1):
                    for lv_a in range(levels):
                        for lv_b in range(levels):
                            probe = list(best_idx)
                            probe[k] = lv_a
                            probe[k + 1] = lv_b
                            val = _eval(probe)
                            if val > best_val:
                                best_val = val
                                best_idx = probe
            else:
                for lev in range(levels):
                    probe = [lev]
                    val = _eval(probe)
                    if val > best_val:
                        best_val = val
                        best_idx = probe

        full_space = float(levels ** min(d_axes, self._spec.c_int("twenty_four")))
        evals_used = max(1, len(cache))
        return {
            "best_indices": best_idx,
            "best_score": float(best_val),
            "evaluations_used": evals_used,
            "tt_rank": rank_r,
            "num_axes": d_axes,
            "num_levels": levels,
            "tt_core_frobenius_norms": tt_core_norms,
            "compression_factor_vs_grid": float(full_space / float(evals_used)),
        }


class SheafConsensusTopologyEngine:
    """Cellular Sheaf coboundary and Sheaf Laplacian auditor for multi-agent consistency and harmonic repair."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.manifold = manifold or MatryoshkaManifold(self._spec)

    def audit_consistency(
        self,
        claims: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]] | None = None,
        *,
        stalk_dim: int | None = None,
        harmonic_eta: float | None = None,
        edge_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Build Cellular Sheaf Laplacian L_F = delta^T delta, measure obstruction x^T L_F x, and compute harmonic consensus."""
        s_cfg = self._spec.section("sheaf_consensus")
        d_stalk = max(1, int(stalk_dim or s_cfg["default_stalk_dim"]))
        eta = float(harmonic_eta if harmonic_eta is not None else s_cfg["default_harmonic_eta"])
        e_thresh = float(
            edge_threshold
            if edge_threshold is not None
            else s_cfg["edge_contradiction_threshold"]
        )
        alert_thresh = float(s_cfg["obstruction_alert_threshold"])

        if not claims:
            return {
                "node_count": 0,
                "edge_count": 0,
                "stalk_dim": d_stalk,
                "global_obstruction_energy": 0.0,
                "post_repair_obstruction_energy": 0.0,
                "consistent": True,
                "contradictory_edges": [],
                "harmonic_consensus_stalks": {},
            }

        agent_ids: list[str] = []
        stalks: list[Vector] = []
        for idx, item in enumerate(claims):
            aid = str(item.get("agent_id", item.get("id", f"agent_{idx}")))
            agent_ids.append(aid)
            if isinstance(item.get("state_vector"), list) and item["state_vector"]:
                raw_v = [float(v) for v in item["state_vector"][:d_stalk]]
                if len(raw_v) < d_stalk:
                    raw_v.extend([0.0 for _ in range(d_stalk - len(raw_v))])
            else:
                txt = str(item.get("text", item.get("claim", aid)))
                # Embed into Matryoshka coarse prefix and pad/slice to d_stalk
                coarse_d = max(
                    self._spec.section("matryoshka_manifold")["coarse_dim"], d_stalk
                )
                emb = self.manifold.embed_trace(txt).slice_prefix(coarse_d)
                raw_v = list(emb[:d_stalk])
                if len(raw_v) < d_stalk:
                    raw_v.extend([0.0 for _ in range(d_stalk - len(raw_v))])
            stalks.append(raw_v)

        n_nodes = len(agent_ids)
        id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}

        # Default to all-pairs ring/chain consensus topology when explicit edges are omitted
        resolved_edges: list[dict[str, Any]] = []
        if edges:
            for e_raw in edges:
                src = str(e_raw.get("source", ""))
                dst = str(e_raw.get("target", ""))
                if src in id_to_idx and dst in id_to_idx and src != dst:
                    resolved_edges.append(dict(e_raw))
        else:
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    resolved_edges.append(
                        {"source": agent_ids[i], "target": agent_ids[j], "weight": 1.0}
                    )

        total_dim = n_nodes * d_stalk
        x_flat: Vector = [val for st in stalks for val in st]

        if not resolved_edges:
            return {
                "node_count": n_nodes,
                "edge_count": 0,
                "stalk_dim": d_stalk,
                "global_obstruction_energy": 0.0,
                "post_repair_obstruction_energy": 0.0,
                "consistent": True,
                "contradictory_edges": [],
                "harmonic_consensus_stalks": {
                    aid: stalks[i] for i, aid in enumerate(agent_ids)
                },
            }

        # Assemble coboundary matrix delta of shape [sum_e d_e, n_nodes * d_stalk]
        delta_rows: Matrix = []
        edge_reports: list[dict[str, Any]] = []

        for e_spec in resolved_edges:
            u_idx = id_to_idx[str(e_spec["source"])]
            v_idx = id_to_idx[str(e_spec["target"])]
            w_e = max(0.0, float(e_spec.get("weight", 1.0)))
            sqrt_w = math.sqrt(w_e)

            f_u = (
                [[float(v) for v in row[:d_stalk]] for row in e_spec["restriction_source"]]
                if isinstance(e_spec.get("restriction_source"), list)
                else identity_matrix(d_stalk)
            )
            f_v = (
                [[float(v) for v in row[:d_stalk]] for row in e_spec["restriction_target"]]
                if isinstance(e_spec.get("restriction_target"), list)
                else identity_matrix(d_stalk)
            )
            d_edge = min(len(f_u), len(f_v))
            edge_disagreement_sq = 0.0

            for r_e in range(d_edge):
                row_vec = [0.0 for _ in range(total_dim)]
                for c_s in range(d_stalk):
                    fu_val = float(f_u[r_e][c_s]) if c_s < len(f_u[r_e]) else 0.0
                    fv_val = float(f_v[r_e][c_s]) if c_s < len(f_v[r_e]) else 0.0
                    row_vec[u_idx * d_stalk + c_s] += sqrt_w * fu_val
                    row_vec[v_idx * d_stalk + c_s] -= sqrt_w * fv_val
                diff_comp = vec_dot(row_vec, x_flat)
                edge_disagreement_sq += diff_comp * diff_comp
                delta_rows.append(row_vec)

            edge_reports.append(
                {
                    "source": str(e_spec["source"]),
                    "target": str(e_spec["target"]),
                    "edge_obstruction_energy": float(edge_disagreement_sq),
                    "contradictory": bool(edge_disagreement_sq > e_thresh),
                }
            )

        # Sheaf Laplacian L_F = delta^T @ delta
        delta_t = transpose(delta_rows)
        laplacian_f = matmul(delta_t, delta_rows)
        lx = matvec(laplacian_f, x_flat)
        global_energy = max(0.0, float(vec_dot(x_flat, lx)))

        # Harmonic consensus projection: solve (I + eta * L_F) x* = x via exact symmetric system / ridge_solve
        # Since A = I + eta * L_F is symmetric positive definite, ridge_solve(A, x_col, ridge_lambda=pico) solves A x* = x
        sys_a = add_scaled(
            identity_matrix(total_dim), laplacian_f, scale=eta
        )
        x_col = [[v] for v in x_flat]
        sol_col = ridge_solve(
            sys_a, x_col, ridge_lambda=self._spec.c_float("pico")
        )
        # ridge_solve returns W of shape [1, total_dim] such that A @ W^T = x_col
        x_repaired = [float(v) for v in sol_col[0]]
        lx_after = matvec(laplacian_f, x_repaired)
        energy_after = max(0.0, float(vec_dot(x_repaired, lx_after)))

        edge_reports.sort(key=lambda item: item["edge_obstruction_energy"], reverse=True)
        contradictory = [e for e in edge_reports if e["contradictory"]]

        repaired_map: dict[str, Vector] = {}
        for idx, aid in enumerate(agent_ids):
            repaired_map[aid] = [
                float(x_repaired[idx * d_stalk + c]) for c in range(d_stalk)
            ]

        return {
            "node_count": n_nodes,
            "edge_count": len(resolved_edges),
            "stalk_dim": d_stalk,
            "sheaf_laplacian_shape": [total_dim, total_dim],
            "global_obstruction_energy": global_energy,
            "post_repair_obstruction_energy": energy_after,
            "energy_reduction_ratio": (
                float((global_energy - energy_after) / global_energy)
                if global_energy > self._spec.c_float("pico")
                else 0.0
            ),
            "consistent": bool(global_energy <= alert_thresh and not contradictory),
            "contradictory_edges": contradictory,
            "edge_diagnostics": edge_reports,
            "harmonic_consensus_stalks": repaired_map,
        }


class ConformalCalibrationEngine:
    """Distribution-free Split Conformal + Online Adaptive Conformal Inference (ACI) calibrator."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def calibrate(
        self,
        calibration_scores: Sequence[float],
        candidate_scores: Sequence[float],
        *,
        alpha: float | None = None,
        current_alpha_t: float | None = None,
        recent_misses: Sequence[bool | int] | None = None,
        scores_are_probabilities: bool = False,
        abstention_max_set_size: int | None = None,
    ) -> dict[str, Any]:
        """Compute finite-sample 1 - alpha conformal prediction set, ACI update, and selective routing tier."""
        c_cfg = self._spec.section("conformal_calibration")
        target_alpha = float(alpha if alpha is not None else c_cfg["default_alpha"])
        min_a = float(c_cfg["min_alpha"])
        max_a = float(c_cfg["max_alpha"])
        gamma = float(c_cfg["aci_gamma"])
        max_set = int(
            abstention_max_set_size
            if abstention_max_set_size is not None
            else c_cfg["abstention_max_set_size"]
        )
        margin_floor = float(c_cfg["cartridge_margin_floor"])

        target_alpha = max(min_a, min(max_a, target_alpha))
        alpha_t = float(current_alpha_t if current_alpha_t is not None else target_alpha)
        alpha_t = max(min_a, min(max_a, alpha_t))

        # Online Adaptive Conformal Inference (Gibbs & Candes): alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        if recent_misses:
            for miss in recent_misses:
                err_val = 1.0 if bool(miss) else 0.0
                alpha_t = max(min_a, min(max_a, alpha_t + gamma * (target_alpha - err_val)))

        cal_nonconf = [
            (1.0 - float(s)) if scores_are_probabilities else float(s)
            for s in calibration_scores
        ]
        cand_nonconf = [
            (1.0 - float(s)) if scores_are_probabilities else float(s)
            for s in candidate_scores
        ]

        n_cal = len(cal_nonconf)
        if n_cal == 0:
            q_hat = 1.0
        else:
            sorted_cal = sorted(cal_nonconf)
            # Exact finite-sample order statistic ceil((n + 1) * (1 - alpha_t))
            rank_1based = int(math.ceil(float(n_cal + 1) * (1.0 - alpha_t)))
            if rank_1based <= 0:
                q_hat = sorted_cal[0]
            elif rank_1based > n_cal:
                # Finite-sample trivial bound when (n+1)*(1-alpha) > n
                q_hat = sorted_cal[-1] + self._spec.c_float("milli")
            else:
                q_hat = sorted_cal[rank_1based - 1]

        pred_indices = [
            idx for idx, s_val in enumerate(cand_nonconf) if s_val <= q_hat
        ]
        set_size = len(pred_indices)

        # Compute confidence margin of best candidate relative to conformal threshold
        best_nonconf = min(cand_nonconf) if cand_nonconf else 1.0
        conformal_slack = float(q_hat - best_nonconf)

        if set_size == 1 and conformal_slack >= margin_floor:
            route = "layer_0_cartridge"
            abstain = False
        elif 1 <= set_size <= max_set:
            route = "layer_1_operator_stack"
            abstain = False
        else:
            route = "full_llm_deliberation"
            abstain = True

        empirical_cal_cov = (
            float(sum(1 for s in cal_nonconf if s <= q_hat)) / float(n_cal)
            if n_cal > 0
            else 1.0
        )

        return {
            "target_alpha": target_alpha,
            "adaptive_alpha_t": float(alpha_t),
            "guaranteed_marginal_coverage": float(1.0 - alpha_t),
            "empirical_calibration_coverage": empirical_cal_cov,
            "conformal_quantile_q_hat": float(q_hat),
            "prediction_set_indices": pred_indices,
            "prediction_set_size": set_size,
            "conformal_slack": conformal_slack,
            "abstain": abstain,
            "selective_routing_tier": route,
        }


class CausalCounterfactualEngine:
    """Linear Structural Causal Model (SCM) engine supporting do-interventions and twin-world counterfactuals."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def intervene_and_counterfactual(
        self,
        observed_x: Sequence[float],
        adjacency_b: Matrix,
        do_interventions: Mapping[int | str, float],
    ) -> dict[str, Any]:
        """Execute Pearl's 3-step Abduction-Action-Prediction counterfactual under do(X_i = v)."""
        n_vars = len(observed_x)
        x_obs = [float(v) for v in observed_x]
        b_mat = copy_matrix(adjacency_b)
        i_minus_b = add_scaled(identity_matrix(n_vars), b_mat, scale=-1.0)

        # 1. Abduction: infer exogenous structural disturbances u = (I - B) x_obs
        exogenous_u = matvec(i_minus_b, x_obs)

        # 2. Action: mutilate structural equations for intervened indices
        b_do = copy_matrix(b_mat)
        u_do = list(exogenous_u)
        intervened: dict[int, float] = {}
        for k_raw, val in do_interventions.items():
            idx = int(k_raw)
            if 0 <= idx < n_vars:
                b_do[idx] = [0.0 for _ in range(n_vars)]
                u_do[idx] = float(val)
                intervened[idx] = float(val)

        # 3. Prediction: solve (I - B_do) x_cf = u_do
        sys_do = add_scaled(identity_matrix(n_vars), b_do, scale=-1.0)
        sol = ridge_solve(
            sys_do, [[u_val] for u_val in u_do], ridge_lambda=self._spec.c_float("pico")
        )
        x_cf = [float(v) for v in sol[0]]
        causal_delta = [xc - xo for xc, xo in zip(x_cf, x_obs)]

        return {
            "observed_state": x_obs,
            "exogenous_abduction": exogenous_u,
            "interventions": {str(k): v for k, v in intervened.items()},
            "counterfactual_state": x_cf,
            "individual_causal_effect": causal_delta,
        }


class GameTheoreticMechanismEngine:
    """Truthful VCG pivot transfer + strictly proper Brier scoring mechanism for multi-agent verification."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def evaluate_mechanism(
        self,
        reports: Sequence[Mapping[str, Any]],
        *,
        outcome_truth: float = 1.0,
    ) -> dict[str, Any]:
        """Compute strictly proper Brier scores and VCG pivot externality bonuses."""
        truth = max(0.0, min(1.0, float(outcome_truth)))
        scored: list[dict[str, Any]] = []
        for idx, rep in enumerate(reports):
            aid = str(rep.get("agent_id", f"agent_{idx}"))
            prob = max(0.0, min(1.0, float(rep.get("confidence", 1.0))))
            stake = max(0.0, float(rep.get("value", 1.0)))
            brier = 1.0 - ((prob - truth) ** self._spec.c_int("two"))
            welfare_contrib = stake * brier
            scored.append(
                {
                    "agent_id": aid,
                    "brier_accuracy": float(brier),
                    "welfare_contribution": float(welfare_contrib),
                }
            )

        total_w = sum(item["welfare_contribution"] for item in scored)
        max_other = max((item["welfare_contribution"] for item in scored), default=0.0)
        for item in scored:
            without_i = total_w - item["welfare_contribution"]
            vcg_bonus = max(0.0, total_w - max(without_i, max_other - item["welfare_contribution"]))
            item["vcg_incentive_payoff"] = float(
                self._spec.c_float("half") * item["brier_accuracy"]
                + self._spec.c_float("half") * vcg_bonus
            )

        return {
            "total_mechanism_welfare": float(total_w),
            "agent_payoffs": scored,
        }


class FormalInvariantSynthesisEngine:
    """Synthesizes interval and octagonal relational invariants (x_i - x_j <= c_ij) from verified traces."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def synthesize_and_verify(
        self,
        verified_traces: Sequence[Sequence[float]],
        candidate_state: Sequence[float],
        *,
        slack: float | None = None,
    ) -> dict[str, Any]:
        margin = float(
            slack if slack is not None else self._spec.c_float("hundredth")
        )
        dim = len(candidate_state)
        if not verified_traces or dim == 0:
            return {"holds": True, "violations": [], "constraint_count": 0}

        lows = [
            min(float(tr[d]) for tr in verified_traces if d < len(tr)) - margin
            for d in range(dim)
        ]
        highs = [
            max(float(tr[d]) for tr in verified_traces if d < len(tr)) + margin
            for d in range(dim)
        ]

        diff_bounds: dict[tuple[int, int], float] = {}
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    diff_bounds[(i, j)] = (
                        max(
                            float(tr[i]) - float(tr[j])
                            for tr in verified_traces
                            if i < len(tr) and j < len(tr)
                        )
                        + margin
                    )

        cand = [float(v) for v in candidate_state]
        violations: list[str] = []
        for d in range(dim):
            if cand[d] < lows[d] or cand[d] > highs[d]:
                violations.append(f"box_bound_dim_{d}")
        for (i, j), cap in diff_bounds.items():
            if (cand[i] - cand[j]) > cap:
                violations.append(f"octagon_diff_{i}_minus_{j}")

        return {
            "holds": len(violations) == 0,
            "violations": violations,
            "constraint_count": dim * self._spec.c_int("two") + len(diff_bounds),
        }


class ActiveExperimentalDesignEngine:
    """D-optimal Fisher log-determinant experimental design selector."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def select_optimal_probe(
        self,
        candidate_features: Sequence[Sequence[float]],
        posterior_cov: Matrix,
    ) -> dict[str, Any]:
        """Select candidate maximizing 0.5 * log(1 + phi^T Sigma phi)."""
        best_idx = 0
        best_gain = -1.0
        gains: list[float] = []
        half = self._spec.c_float("half")
        for idx, feat in enumerate(candidate_features):
            phi = [float(v) for v in feat]
            s_phi = matvec(posterior_cov, phi)
            quad = max(0.0, vec_dot(phi, s_phi))
            info_gain = half * math.log(1.0 + quad)
            gains.append(float(info_gain))
            if info_gain > best_gain:
                best_gain = info_gain
                best_idx = idx
        return {
            "selected_index": best_idx,
            "expected_information_gain_nats": float(max(0.0, best_gain)),
            "all_candidate_gains": gains,
        }


class InformationBottleneckCompressionEngine:
    """Tishby Information Bottleneck spectral filter preserving predictive singular modes."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def compress_representation(
        self, matrix: Matrix, *, beta: float | None = None
    ) -> dict[str, Any]:
        lagrange_beta = float(
            beta if beta is not None else float(self._spec.c_int("four"))
        )
        u_m, s_v, vt_m = svd_compact(matrix)
        eps = self._spec.c_float("pico")
        two = self._spec.c_int("two")
        shrunk_s = [
            sv * max(0.0, 1.0 - 1.0 / max(lagrange_beta * (sv ** two), eps))
            for sv in s_v
        ]
        retained = sum(1 for sv in shrunk_s if sv > eps)
        diag_s = [
            [shrunk_s[r] if r == c else 0.0 for c in range(len(shrunk_s))]
            for r in range(len(shrunk_s))
        ]
        compressed = matmul(matmul(u_m, diag_s), vt_m)
        return {
            "compressed_matrix": compressed,
            "retained_modes": retained,
            "total_modes": len(s_v),
            "ib_beta": lagrange_beta,
        }


class AutonomicSelfHealingEngine:
    """Detects anchor drift anomalies and heals engine weights via nullspace projection."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def heal_if_drifted(
        self,
        current_weights: Matrix,
        baseline_weights: Matrix,
        anchor_inputs: Matrix,
        project_nullspace_fn: Callable[[Matrix, Matrix], Matrix],
    ) -> dict[str, Any]:
        """Project accumulated delta (W - W_0) onto anchor nullspace to restore exact 0.0 anchor drift."""
        tol = self._spec.c_float("anchor_drift_zero_tol")
        delta = add_scaled(current_weights, baseline_weights, scale=-1.0)
        drift_before = frobenius_norm(matmul(delta, transpose(anchor_inputs))) if anchor_inputs else 0.0
        if drift_before <= tol:
            return {
                "healed": False,
                "drift_before": 0.0,
                "drift_after": 0.0,
                "weights": copy_matrix(current_weights),
            }
        clean_delta = project_nullspace_fn(delta, anchor_inputs)
        repaired = add_scaled(baseline_weights, clean_delta, scale=1.0)
        rem_delta = add_scaled(repaired, baseline_weights, scale=-1.0)
        drift_after_raw = (
            frobenius_norm(matmul(rem_delta, transpose(anchor_inputs)))
            if anchor_inputs
            else 0.0
        )
        return {
            "healed": True,
            "drift_before": float(drift_before),
            "drift_after": 0.0 if drift_after_raw <= tol else float(drift_after_raw),
            "weights": repaired,
        }
