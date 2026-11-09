"""Mechanistic Interpretability & Latent Steering Engines (v1.2.0): Matryoshka Top-K SAE, Crosscoder Circuit Tracer, Internal/ELK State Prober, and Nullspace Activation Steerer."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from continuum_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from continuum_core.cybernetic_engines import ConformalCalibrationEngine
from continuum_core.linalg_backend import (
    Matrix,
    Vector,
    add_scaled,
    copy_matrix,
    frobenius_norm,
    identity_matrix,
    l2_normalize,
    matmul,
    matvec,
    orthogonal_procrustes,
    outer_product,
    ridge_solve,
    transpose,
    vec_dot,
    vec_norm,
    zeros_matrix,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold


def _pad_or_trim(vec: Sequence[float], dim: int) -> Vector:
    out = [float(v) for v in vec[:dim]]
    if len(out) < dim:
        out.extend([0.0 for _ in range(dim - len(out))])
    return out


def _orthonormal_basis(rows: Sequence[Sequence[float]], dim: int, eps: float) -> list[Vector]:
    basis: list[Vector] = []
    for raw_row in rows:
        v = _pad_or_trim(raw_row, dim)
        for b_vec in basis:
            proj = vec_dot(v, b_vec)
            v = [vi - proj * bi for vi, bi in zip(v, b_vec)]
        nrm = vec_norm(v)
        if nrm > eps:
            basis.append([vi / nrm for vi in v])
    return basis


class SparseFeatureDictionaryEngine:
    """Engine 25: Matryoshka Top-K JumpReLU Sparse Autoencoder (SAE) with AuxK revival and selectivity scoring."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()
        self._firing_counts: dict[tuple[int, int], list[int]] = {}

    def build_dictionary(
        self,
        latent_dim: int,
        expansion_factor: int,
        *,
        normalize_columns: bool = True,
    ) -> Matrix:
        """Construct deterministic overcomplete dictionary D of shape [latent_dim, dict_size] with low mutual coherence."""
        d_in = max(1, int(latent_dim))
        exp_f = max(1, int(expansion_factor))
        dict_size = d_in * exp_f
        half = self._spec.c_float("half")
        tenth = self._spec.c_float("tenth")
        two = float(self._spec.c_int("two"))

        # Columns first: canonical identity frame followed by harmonic Gabor-like overcomplete atoms
        cols: list[Vector] = []
        for m_idx in range(dict_size):
            if m_idx < d_in:
                col = [1.0 if r == m_idx else 0.0 for r in range(d_in)]
                # Add slight structured harmonic ripple if normalized, keeping near-orthonormal core
                col = [
                    col[r] + tenth * half * math.sin(float((m_idx + 1) * (r + 1)))
                    for r in range(d_in)
                ]
            else:
                freq = float(m_idx - d_in + 1)
                phase = float(m_idx) * half
                col = [
                    math.cos((two * math.pi * freq * float(r + 1)) / float(max(1, d_in)) + phase)
                    for r in range(d_in)
                ]
            if normalize_columns:
                col = l2_normalize(col)
            cols.append(col)
        # Transpose [dict_size, d_in] -> [d_in, dict_size]
        return transpose(cols)

    def encode_and_score(
        self,
        activation_vec: Sequence[float],
        *,
        dictionary: Matrix | None = None,
        topk: int | None = None,
        jumprelu_theta: float | None = None,
        auxk_gain: float | None = None,
        expansion_factor: int | None = None,
        gating_mode: str = "matryoshka_topk",
        normalize_columns: bool = True,
    ) -> dict[str, Any]:
        """Decompose activation h into sparse monosemantic latents + AuxK dormant revival."""
        l_cfg = self._spec.section("latent_interpretability")
        d_in = max(1, len(activation_vec))
        h_vec = [float(v) for v in activation_vec]
        exp_f = int(
            expansion_factor
            if expansion_factor is not None
            else l_cfg["dictionary_expansion_factor"]
        )
        k_act = max(1, int(topk if topk is not None else l_cfg["default_topk"]))
        theta = float(
            jumprelu_theta if jumprelu_theta is not None else l_cfg["jumprelu_theta"]
        )
        rev_gain = float(
            auxk_gain if auxk_gain is not None else l_cfg["auxk_revival_gain"]
        )
        aux_k = max(1, int(l_cfg["auxk_topk"]))
        eps = self._spec.c_float("pico")
        two = self._spec.c_int("two")

        if dictionary is not None and len(dictionary) == d_in and len(dictionary[0]) > 0:
            dict_cols_raw = transpose(dictionary)
            dict_cols = [
                l2_normalize(col) if normalize_columns else [float(x) for x in col]
                for col in dict_cols_raw
            ]
        else:
            dict_mat = self.build_dictionary(
                d_in, exp_f, normalize_columns=normalize_columns
            )
            dict_cols = transpose(dict_mat)

        dict_size = len(dict_cols)
        k_act = min(k_act, dict_size)
        shape_key = (d_in, dict_size)
        if shape_key not in self._firing_counts:
            self._firing_counts[shape_key] = [0 for _ in range(dict_size)]
        counts = self._firing_counts[shape_key]

        coeffs = [0.0 for _ in range(dict_size)]
        feature_tiers: dict[int, int] = {}

        if gating_mode == "matryoshka_topk":
            # Progressive nested dictionary prefixes (M/4 -> M/2 -> M) fitting coarse-to-fine residuals
            divisors = [int(dv) for dv in l_cfg["matryoshka_tier_divisors"]]
            residual = list(h_vec)
            budget_left = k_act
            n_tiers = max(1, len(divisors))
            for tier_idx, div in enumerate(divisors):
                if budget_left <= 0:
                    break
                prefix_end = max(1, dict_size // max(1, div))
                tier_k = (
                    budget_left
                    if tier_idx == n_tiers - 1
                    else max(1, budget_left // max(1, n_tiers - tier_idx))
                )
                scored_tier: list[tuple[float, int]] = []
                for j_idx in range(prefix_end):
                    if coeffs[j_idx] > 0.0:
                        continue
                    pre_z = vec_dot(dict_cols[j_idx], residual)
                    gated_z = pre_z if pre_z > theta else 0.0
                    if gated_z > 0.0:
                        scored_tier.append((gated_z, j_idx))
                scored_tier.sort(key=lambda pair: pair[0], reverse=True)
                chosen = scored_tier[:tier_k]
                for act_val, j_idx in chosen:
                    coeffs[j_idx] = act_val
                    feature_tiers[j_idx] = prefix_end
                    residual = [
                        rv - act_val * dv
                        for rv, dv in zip(residual, dict_cols[j_idx])
                    ]
                    budget_left -= 1
        else:
            pre_acts = [vec_dot(col, h_vec) for col in dict_cols]
            if gating_mode == "soft_threshold":
                gated = [max(0.0, z_val - theta) for z_val in pre_acts]
            else:
                # jumprelu_topk
                gated = [z_val if z_val > theta else 0.0 for z_val in pre_acts]
            ranked = sorted(
                ((val, idx) for idx, val in enumerate(gated) if val > 0.0),
                reverse=True,
            )[:k_act]
            for val, idx in ranked:
                coeffs[idx] = val
                feature_tiers[idx] = dict_size

        # Guarantee at least top-1 positive projection if signal is non-zero and theta gated everything
        if all(c == 0.0 for c in coeffs) and vec_norm(h_vec) > eps:
            pos_proj = [(vec_dot(col, h_vec), idx) for idx, col in enumerate(dict_cols)]
            best_val, best_j = max(pos_proj, key=lambda item: abs(item[0]))
            if abs(best_val) > eps:
                coeffs[best_j] = abs(best_val)
                feature_tiers[best_j] = dict_size

        # Orthogonal least-squares support debiasing over selected Top-K active atoms to prevent overcomplete double-counting
        prelim_active = [idx for idx, c_val in enumerate(coeffs) if c_val > 0.0]
        if len(prelim_active) >= two:
            # Solve min_{a >= 0} || D_S a - h ||_2^2 via ridge_solve on D_S^T (shape [|S|, d_in] -> ridge_solve(D_S, h_col))
            d_sub = [
                [dict_cols[idx][r] for idx in prelim_active] for r in range(d_in)
            ]
            debiased = ridge_solve(
                d_sub,
                [[hv] for hv in h_vec],
                ridge_lambda=self._spec.c_float("micro"),
            )[0]
            for pos_i, atom_idx in enumerate(prelim_active):
                coeffs[atom_idx] = max(0.0, float(debiased[pos_i]))

        recon = [0.0 for _ in range(d_in)]
        active_indices: list[int] = []
        for idx, c_val in enumerate(coeffs):
            if c_val > 0.0:
                active_indices.append(idx)
                counts[idx] += 1
                recon = [rv + c_val * dv for rv, dv in zip(recon, dict_cols[idx])]

        residual_vec = [hv - rv for hv, rv in zip(h_vec, recon)]
        mse = sum(rv ** two for rv in residual_vec) / float(d_in)
        h_energy = sum(hv ** two for hv in h_vec)
        res_energy = sum(rv ** two for rv in residual_vec)
        r2 = max(0.0, 1.0 - (res_energy / max(h_energy, eps))) if h_energy > eps else 1.0

        # Sub-engine 3: AuxK dead-latent reviver over dormant dictionary atoms
        dead_indices = [idx for idx, cnt in enumerate(counts) if cnt == 0]
        aux_candidates: list[tuple[float, int]] = []
        for d_idx in dead_indices:
            align = abs(vec_dot(dict_cols[d_idx], residual_vec))
            if align > eps:
                aux_candidates.append((align, d_idx))
        aux_candidates.sort(key=lambda item: item[0], reverse=True)
        revived = aux_candidates[: min(aux_k, len(aux_candidates))]
        aux_residual = list(residual_vec)
        revived_ids: list[int] = []
        for align_val, d_idx in revived:
            step = rev_gain * align_val
            sign = 1.0 if vec_dot(dict_cols[d_idx], aux_residual) >= 0.0 else -1.0
            aux_residual = [
                ar - sign * step * dv for ar, dv in zip(aux_residual, dict_cols[d_idx])
            ]
            revived_ids.append(d_idx)
        aux_mse_after = sum(ar ** two for ar in aux_residual) / float(d_in)

        # Sub-engine 4: Monosemantic selectivity scorer
        total_act = sum(coeffs[idx] for idx in active_indices)
        active_features: list[dict[str, Any]] = []
        for idx in active_indices:
            other_coherence = max(
                (
                    abs(vec_dot(dict_cols[idx], dict_cols[other]))
                    for other in active_indices
                    if other != idx
                ),
                default=0.0,
            )
            concentration = coeffs[idx] / max(total_act, eps)
            selectivity = float(concentration * max(0.0, 1.0 - other_coherence))
            active_features.append(
                {
                    "feature_index": idx,
                    "feature_id": f"sae_latent_{idx}",
                    "activation": float(coeffs[idx]),
                    "matryoshka_prefix_tier": int(feature_tiers.get(idx, dict_size)),
                    "monosemantic_selectivity": selectivity,
                    "max_active_coherence": float(other_coherence),
                    "decoder_direction": [float(v) for v in dict_cols[idx]],
                }
            )
        active_features.sort(key=lambda item: item["activation"], reverse=True)

        return {
            "latent_dim": d_in,
            "dictionary_size": dict_size,
            "gating_mode": gating_mode,
            "l0_active_count": len(active_features),
            "active_features": active_features,
            "reconstruction": recon,
            "reconstruction_mse": float(mse),
            "explained_variance_r2": float(r2),
            "auxk_revival": {
                "dormant_latent_count": len(dead_indices),
                "revived_latent_indices": revived_ids,
                "post_auxk_mse": float(aux_mse_after),
            },
        }


class CrosslayerCircuitTracerEngine:
    """Engine 26: Multi-layer sparse crosscoder, MLP transcoder, Procrustes aligner, and causal patching attributor."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def trace_circuit(
        self,
        layer_activations: Sequence[Sequence[float]],
        *,
        shared_latents: int | None = None,
        attribution_threshold: float | None = None,
        l1_penalty: float | None = None,
        align_procrustes: bool = True,
        target_direction: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Trace shared cross-layer latents, transcoder edges, Procrustes basis drift, and causal attributions."""
        if not layer_activations or not layer_activations[0]:
            return {
                "layer_count": 0,
                "shared_crosscoder_features": [],
                "transcoder_edges": [],
                "procrustes_orthogonality_error": 0.0,
                "rotational_alignment_cosine": 1.0,
                "aligned_first_layer": [],
            }

        d_dim = len(layer_activations[0])
        layers: list[Vector] = [_pad_or_trim(lay, d_dim) for lay in layer_activations]
        n_layers = len(layers)
        eps = self._spec.c_float("pico")
        half = self._spec.c_float("half")
        tenth = self._spec.c_float("tenth")
        two = float(self._spec.c_int("two"))

        m_shared = max(
            1,
            int(
                shared_latents
                if shared_latents is not None
                else self._spec.c_int("twelve")
            ),
        )
        attr_th = float(
            attribution_threshold if attribution_threshold is not None else tenth * half
        )
        l1_pen = float(l1_penalty if l1_penalty is not None else tenth * tenth)

        # Sub-engine 3: Procrustes manifold alignment between earliest and final residual layer
        first_layer = layers[0]
        final_layer = layers[-1]
        if align_procrustes and n_layers >= self._spec.c_int("two"):
            # Build rank-stabilized local frames from layer vectors + small canonical basis perturbation
            src_frame = [
                [
                    first_layer[c] + (tenth if r == c else 0.0)
                    for c in range(d_dim)
                ]
                for r in range(d_dim)
            ]
            dst_frame = [
                [
                    final_layer[c] + (tenth if r == c else 0.0)
                    for c in range(d_dim)
                ]
                for r in range(d_dim)
            ]
            omega = orthogonal_procrustes(src_frame, dst_frame)
            aligned_first = matvec(omega, first_layer)
            orth_err = frobenius_norm(
                add_scaled(
                    matmul(transpose(omega), omega),
                    identity_matrix(d_dim),
                    scale=-1.0,
                )
            )
        else:
            aligned_first = list(first_layer)
            orth_err = 0.0

        rot_cos = vec_dot(l2_normalize(aligned_first), l2_normalize(final_layer))

        # Sub-engine 1 & 4: Multi-layer sparse crosscoder + causal patching attribution
        readout_dir = (
            l2_normalize(_pad_or_trim(target_direction, d_dim))
            if target_direction is not None and len(target_direction) > 0
            else l2_normalize(final_layer)
        )

        shared_features: list[dict[str, Any]] = []
        for f_idx in range(m_shared):
            layer_decoders: list[Vector] = []
            joint_pre = 0.0
            for l_idx, h_lay in enumerate(layers):
                phase = float((f_idx + 1) * (l_idx + 1)) * half
                atom = l2_normalize(
                    [
                        math.cos(
                            (two * math.pi * float(f_idx + 1) * float(c + 1))
                            / float(max(1, d_dim))
                            + phase
                        )
                        for c in range(d_dim)
                    ]
                )
                layer_decoders.append(atom)
                joint_pre += vec_dot(atom, h_lay) / float(n_layers)

            gated_z = max(0.0, joint_pre - l1_pen)
            if gated_z > 0.0:
                per_layer_contrib = [
                    abs(gated_z * vec_dot(layer_decoders[l_i], layers[l_i]))
                    for l_i in range(n_layers)
                ]
                active_layer_span = sum(
                    1 for val in per_layer_contrib if val > attr_th * half
                )
                causal_attr = float(gated_z * vec_dot(layer_decoders[-1], readout_dir))
                if abs(causal_attr) >= attr_th or gated_z >= attr_th:
                    shared_features.append(
                        {
                            "crosscoder_latent_id": f"cc_latent_{f_idx}",
                            "joint_activation": float(gated_z),
                            "layer_profile": [float(v) for v in per_layer_contrib],
                            "active_layer_span": int(active_layer_span),
                            "persistent_across_depth": bool(
                                active_layer_span >= min(n_layers, self._spec.c_int("two"))
                            ),
                            "causal_patching_attribution": causal_attr,
                        }
                    )

        shared_features.sort(
            key=lambda item: abs(item["causal_patching_attribution"]), reverse=True
        )

        # Sub-engine 2: MLP sparse transcoder transition edges across adjacent layers
        transcoder_edges: list[dict[str, Any]] = []
        for l_idx in range(n_layers - 1):
            delta_h = [
                layers[l_idx + 1][c] - layers[l_idx][c] for c in range(d_dim)
            ]
            delta_norm = vec_norm(delta_h)
            in_norm = max(vec_norm(layers[l_idx]), eps)
            gain_ratio = delta_norm / in_norm
            if gain_ratio >= attr_th:
                transcoder_edges.append(
                    {
                        "from_layer": l_idx,
                        "to_layer": l_idx + 1,
                        "mlp_residual_gain": float(gain_ratio),
                        "cosine_continuity": float(
                            vec_dot(
                                l2_normalize(layers[l_idx]),
                                l2_normalize(layers[l_idx + 1]),
                            )
                        ),
                    }
                )

        return {
            "layer_count": n_layers,
            "latent_dim": d_dim,
            "shared_crosscoder_features": shared_features,
            "transcoder_edges": transcoder_edges,
            "procrustes_orthogonality_error": float(orth_err),
            "rotational_alignment_cosine": float(rot_cos),
            "aligned_first_layer": [float(v) for v in aligned_first],
        }


class InternalStateProberEngine:
    """Engine 27: Mass-Mean & Shrinkage Fisher LDA concept prober, ELK layer-divergence detector, and Conformal truthfulness calibrator."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
        conformal: ConformalCalibrationEngine | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.manifold = manifold or MatryoshkaManifold(self._spec)
        self.conformal = conformal or ConformalCalibrationEngine(self._spec)

    def _mean_vec(self, rows: Sequence[Sequence[float]], dim: int) -> Vector:
        if not rows:
            return [0.0 for _ in range(dim)]
        acc = [0.0 for _ in range(dim)]
        for row in rows:
            padded = _pad_or_trim(row, dim)
            for idx in range(dim):
                acc[idx] += padded[idx]
        inv_n = 1.0 / float(len(rows))
        return [v * inv_n for v in acc]

    def probe_state(
        self,
        activation_vec: Sequence[float],
        *,
        intermediate_vec: Sequence[float] | None = None,
        positive_examples: Sequence[Sequence[float]] | None = None,
        negative_examples: Sequence[Sequence[float]] | None = None,
        calibration_scores: Sequence[float] | None = None,
        concept_name: str = "truthfulness",
        fisher_shrinkage: float | None = None,
        elk_alert_floor: float | None = None,
        alpha: float | None = None,
        estimator: str = "hybrid_elk",
    ) -> dict[str, Any]:
        """Probe internal concept probability, detect ELK belief-vs-output divergence, and build conformal truthfulness set."""
        l_cfg = self._spec.section("latent_interpretability")
        d_dim = max(1, len(activation_vec))
        h_surface = _pad_or_trim(activation_vec, d_dim)
        shrink = float(
            fisher_shrinkage
            if fisher_shrinkage is not None
            else l_cfg["fisher_shrinkage_lambda"]
        )
        elk_floor = float(
            elk_alert_floor
            if elk_alert_floor is not None
            else l_cfg["elk_divergence_alert_floor"]
        )
        temp = max(
            self._spec.c_float("hundredth"), float(l_cfg["probe_temperature"])
        )
        half = self._spec.c_float("half")
        eps = self._spec.c_float("pico")

        # Synthesize deterministic contrastive prototypes when explicit positive/negative clouds are omitted
        if positive_examples and negative_examples:
            pos_rows = [l2_normalize(_pad_or_trim(r, d_dim)) for r in positive_examples]
            neg_rows = [l2_normalize(_pad_or_trim(r, d_dim)) for r in negative_examples]
        else:
            pos_proto = self.manifold.embed_trace(
                f"verified truthful consistent {concept_name}",
                stratum="latent_interpretability_and_steering",
            ).slice_prefix(max(d_dim, self._spec.c_int("sixteen")))[:d_dim]
            neg_proto = self.manifold.embed_trace(
                f"fabricated deceptive contradictory {concept_name}",
                stratum="latent_interpretability_and_steering",
            ).slice_prefix(max(d_dim, self._spec.c_int("sixteen")))[:d_dim]
            pos_rows = [l2_normalize(pos_proto)]
            neg_rows = [l2_normalize(neg_proto)]

        # Sub-engine 1: Mass-Mean concept prober (mu_+ - mu_-)
        mu_pos = self._mean_vec(pos_rows, d_dim)
        mu_neg = self._mean_vec(neg_rows, d_dim)
        midpoint = [half * (mp + mn) for mp, mn in zip(mu_pos, mu_neg)]
        mass_mean_dir = [mp - mn for mp, mn in zip(mu_pos, mu_neg)]
        mass_mean_unit = l2_normalize(mass_mean_dir)

        # Sub-engine 2: Shrinkage Fisher Linear Discriminant prober ((1 - lambda) Sigma_W + lambda I)^{-1} (mu_+ - mu_-)
        cov_w = zeros_matrix(d_dim, d_dim)
        total_samples = max(1, len(pos_rows) + len(neg_rows))
        for row in pos_rows:
            diff = [rv - mp for rv, mp in zip(row, mu_pos)]
            cov_w = add_scaled(cov_w, outer_product(diff, diff), scale=1.0 / float(total_samples))
        for row in neg_rows:
            diff = [rv - mn for rv, mn in zip(row, mu_neg)]
            cov_w = add_scaled(cov_w, outer_product(diff, diff), scale=1.0 / float(total_samples))

        shrink_clamped = max(self._spec.c_float("milli"), min(1.0, shrink))
        reg_cov = add_scaled(
            add_scaled(zeros_matrix(d_dim, d_dim), cov_w, scale=1.0 - shrink_clamped),
            identity_matrix(d_dim),
            scale=shrink_clamped,
        )
        fisher_sol = ridge_solve(
            reg_cov,
            [[v] for v in mass_mean_dir],
            ridge_lambda=self._spec.c_float("micro"),
        )
        fisher_unit = l2_normalize([float(v) for v in fisher_sol[0]])

        if estimator == "mass_mean":
            probe_dir = mass_mean_unit
        elif estimator == "fisher_lda":
            probe_dir = fisher_unit
        else:
            probe_dir = l2_normalize(
                [half * (mm + fl) for mm, fl in zip(mass_mean_unit, fisher_unit)]
            )

        def _concept_prob(vec: Vector) -> tuple[float, float]:
            u_vec = l2_normalize(vec)
            centered = [vi - mi for vi, mi in zip(u_vec, midpoint)]
            margin = vec_dot(probe_dir, centered)
            clamped_logit = max(
                -float(self._spec.c_int("twenty")),
                min(float(self._spec.c_int("twenty")), margin / temp),
            )
            prob = 1.0 / (1.0 + math.exp(-clamped_logit))
            return float(prob), float(margin)

        p_surface, margin_surface = _concept_prob(h_surface)

        # Sub-engine 3: ELK layer-divergence detector (internal belief vs surface output)
        if intermediate_vec is not None and len(intermediate_vec) > 0:
            h_mid = _pad_or_trim(intermediate_vec, d_dim)
            p_mid, margin_mid = _concept_prob(h_mid)
            signed_gap = float(p_mid - p_surface)
            abs_gap = abs(signed_gap)
            cos_layers = vec_dot(l2_normalize(h_mid), l2_normalize(h_surface))
            dir_div = float(max(0.0, half * (1.0 - cos_layers)))
            elk_score = float(max(abs_gap, half * dir_div))
            elk_alert = bool(elk_score >= elk_floor)
        else:
            p_mid = p_surface
            margin_mid = margin_surface
            signed_gap = 0.0
            dir_div = 0.0
            elk_score = 0.0
            elk_alert = False

        # Sub-engine 4: Split-Conformal truthfulness calibration
        if calibration_scores is not None and len(calibration_scores) > 0:
            cal_nonconf = [float(s) for s in calibration_scores]
            scores_are_probs = False
            cand_scores = [1.0 - p_surface, p_surface]
        else:
            # Build calibration nonconformity scores (1 - true_class_prob) + finite-sample jackknife slack
            jackknife_slack = self._spec.c_float("hundredth") / float(max(1, total_samples))
            cal_nonconf = []
            for r_pos in pos_rows:
                pr_p, _ = _concept_prob(r_pos)
                cal_nonconf.append(min(1.0, (1.0 - pr_p) + jackknife_slack))
            for r_neg in neg_rows:
                pr_n, _ = _concept_prob(r_neg)
                cal_nonconf.append(min(1.0, pr_n + jackknife_slack))
            scores_are_probs = False
            cand_scores = [1.0 - p_surface, p_surface]

        conf_res = self.conformal.calibrate(
            cal_nonconf,
            cand_scores,
            alpha=alpha,
            scores_are_probabilities=scores_are_probs,
            abstention_max_set_size=1,
        )
        label_names = [concept_name, f"non_{concept_name}"]
        conformal_labels = [
            label_names[idx]
            for idx in conf_res["prediction_set_indices"]
            if 0 <= idx < len(label_names)
        ]

        return {
            "concept_name": concept_name,
            "estimator": estimator,
            "probe_direction": [float(v) for v in probe_dir],
            "probe_midpoint": [float(v) for v in midpoint],
            "surface_margin": margin_surface,
            "concept_probabilities": {
                concept_name: p_surface,
                f"non_{concept_name}": float(1.0 - p_surface),
            },
            "elk_divergence": {
                "mid_layer_belief_prob": float(p_mid),
                "mid_layer_margin": float(margin_mid),
                "surface_output_prob": float(p_surface),
                "signed_belief_minus_surface": signed_gap,
                "directional_divergence": dir_div,
                "elk_contradiction_score": elk_score,
                "elk_alert_floor": elk_floor,
                "elk_contradiction_detected": elk_alert,
            },
            "conformal_truthfulness": {
                "prediction_set": conformal_labels,
                "conformal_quantile_q_hat": conf_res["conformal_quantile_q_hat"],
                "guaranteed_coverage": conf_res["guaranteed_marginal_coverage"],
                "abstain": conf_res["abstain"] or elk_alert,
            },
            "fisher_vs_mass_mean_cosine": float(
                vec_dot(fisher_unit, mass_mean_unit) if vec_norm(fisher_unit) > eps else 1.0
            ),
        }


class NullspaceActivationSteererEngine:
    """Engine 28: Contrastive/SAE control vector extraction, Null(A_anchor) steering, and zero-hook weight folding."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()

    def compute_nullspace_steering(
        self,
        *,
        positive_activations: Sequence[Sequence[float]],
        negative_activations: Sequence[Sequence[float]],
        anchor_vectors: Sequence[Sequence[float]],
        base_weights: Matrix,
        project_matrix_nullspace_fn: Callable[[Matrix, Matrix], Matrix],
        sae_decoder_direction: Sequence[float] | None = None,
        residual_vector: Sequence[float] | None = None,
        steering_strength: float | None = None,
        norm_clamp_ratio: float | None = None,
        direction_source: str = "hybrid_sae_contrastive",
        enforce_nullspace: bool = True,
    ) -> dict[str, Any]:
        """Project steering vector strictly into Null(A_anchor) (0.0 drift) and fold into rank-1 LoRA-XS + bias patch."""
        l_cfg = self._spec.section("latent_interpretability")
        rows = len(base_weights)
        cols = len(base_weights[0]) if rows > 0 else 1
        alpha = float(
            steering_strength
            if steering_strength is not None
            else l_cfg["default_steering_alpha"]
        )
        clamp_r = float(
            norm_clamp_ratio
            if norm_clamp_ratio is not None
            else l_cfg["steering_norm_clamp_ratio"]
        )
        tol = self._spec.c_float("anchor_drift_zero_tol")
        eps = self._spec.c_float("nano")
        half = self._spec.c_float("half")

        # Sub-engine 1: Contrastive / SAE direction extraction
        pos_mean = [0.0 for _ in range(cols)]
        for p_row in positive_activations:
            pv = _pad_or_trim(p_row, cols)
            pos_mean = [a + b / float(max(1, len(positive_activations))) for a, b in zip(pos_mean, pv)]

        neg_mean = [0.0 for _ in range(cols)]
        for n_row in negative_activations:
            nv = _pad_or_trim(n_row, cols)
            neg_mean = [a + b / float(max(1, len(negative_activations))) for a, b in zip(neg_mean, nv)]

        contrastive_dir = l2_normalize([pm - nm for pm, nm in zip(pos_mean, neg_mean)])
        if sae_decoder_direction is not None and len(sae_decoder_direction) > 0:
            sae_dir = l2_normalize(_pad_or_trim(sae_decoder_direction, cols))
        else:
            sae_dir = list(contrastive_dir)

        if direction_source == "contrastive_mass_mean":
            unit_raw = contrastive_dir
        elif direction_source == "sae_decoder_column":
            unit_raw = sae_dir
        else:
            # Align SAE sign with contrastive direction before blending so they reinforce coherently
            sign = 1.0 if vec_dot(sae_dir, contrastive_dir) >= 0.0 else -1.0
            unit_raw = l2_normalize(
                [half * (cd + sign * sd) for cd, sd in zip(contrastive_dir, sae_dir)]
            )

        eff_scale = min(abs(alpha), max(eps, clamp_r)) * (1.0 if alpha >= 0.0 else -1.0)
        v_raw = [eff_scale * uv for uv in unit_raw]

        # Measure anchor disturbance of naive unprojected steering vector: ||A_anchor @ v_raw||_2
        padded_anchors: Matrix = [_pad_or_trim(ar, cols) for ar in anchor_vectors]
        raw_anchor_drift = (
            vec_norm(matvec(padded_anchors, v_raw)) if padded_anchors else 0.0
        )

        # Sub-engine 2: Anchor nullspace projection v_perp = (I - B B^T) v_raw in Null(A_anchor)
        if enforce_nullspace and padded_anchors:
            basis = _orthonormal_basis(padded_anchors, cols, eps)
            v_clean = list(v_raw)
            for b_vec in basis:
                proj = vec_dot(v_clean, b_vec)
                v_clean = [vc - proj * bi for vc, bi in zip(v_clean, b_vec)]
            # Fallback to canonical nullspace coordinate if v_raw lay 100% inside anchor span
            if vec_norm(v_clean) <= eps and len(basis) < cols:
                for c_idx in range(cols - 1, -1, -1):
                    cand = [eff_scale if j == c_idx else 0.0 for j in range(cols)]
                    for b_vec in basis:
                        pr = vec_dot(cand, b_vec)
                        cand = [cv - pr * bi for cv, bi in zip(cand, b_vec)]
                    if vec_norm(cand) > eps:
                        cand_unit = l2_normalize(cand)
                        v_clean = [eff_scale * cu for cu in cand_unit]
                        break
        else:
            v_clean = list(v_raw)

        post_drift_raw = (
            vec_norm(matvec(padded_anchors, v_clean)) if padded_anchors else 0.0
        )
        verified_vec_drift = 0.0 if post_drift_raw <= tol else float(post_drift_raw)
        retained_ratio = vec_norm(v_clean) / max(vec_norm(v_raw), eps)

        # Sub-engine 3: Zero-hook weight folder (rank-1 LoRA-XS DeltaW + bias patch)
        u_out = l2_normalize(matvec(base_weights, v_clean))
        if vec_norm(u_out) <= eps:
            u_out = [1.0 if r == 0 else 0.0 for r in range(rows)]
        v_norm = max(vec_norm(v_clean), eps)
        v_unit = [vc / v_norm for vc in v_clean]
        raw_delta_w = outer_product(u_out, v_unit, scale=vec_norm(v_clean) * self._spec.c_float("quarter"))
        folded_delta_w = (
            project_matrix_nullspace_fn(raw_delta_w, padded_anchors)
            if enforce_nullspace and padded_anchors
            else raw_delta_w
        )
        steered_weights = add_scaled(base_weights, folded_delta_w, scale=1.0)
        folded_bias = matvec(base_weights, v_clean)

        weight_drift_raw = (
            frobenius_norm(matmul(folded_delta_w, transpose(padded_anchors)))
            if padded_anchors
            else 0.0
        )
        verified_weight_drift = (
            0.0 if weight_drift_raw <= tol else float(weight_drift_raw)
        )

        if residual_vector is not None and len(residual_vector) > 0:
            h_in = _pad_or_trim(residual_vector, cols)
            steered_h: Vector | None = [hi + vi for hi, vi in zip(h_in, v_clean)]
        else:
            steered_h = None

        return {
            "direction_source": direction_source,
            "steering_strength": float(eff_scale),
            "raw_steering_vector": [float(v) for v in v_raw],
            "steering_vector": [float(v) for v in v_clean],
            "raw_anchor_drift": float(raw_anchor_drift),
            "nullspace_anchor_drift": verified_vec_drift,
            "raw_numerical_nullspace_drift": float(post_drift_raw),
            "retained_steering_norm_ratio": float(retained_ratio),
            "steered_residual_vector": steered_h,
            "folded_lora_xs_patch": {
                "rank1_u": [float(v) for v in u_out],
                "rank1_v": [float(v) for v in v_unit],
                "folded_bias_patch": [float(v) for v in folded_bias],
                "delta_w_frobenius_norm": float(frobenius_norm(folded_delta_w)),
                "engine_weight_anchor_drift": verified_weight_drift,
            },
            "steered_weights": steered_weights,
        }
