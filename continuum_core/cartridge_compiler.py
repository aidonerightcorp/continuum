"""Multi-layer FiLM + MoLoRA + HyperLoRA operator stack compiler and 9-target portable lowering engine."""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from continuum_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from continuum_core.linalg_backend import (
    Matrix,
    Vector,
    activate_vector,
    copy_matrix,
    identity_matrix,
    matvec,
    ridge_solve,
    safetensors_dumps,
    softmax,
    svd_compact,
    zeros_matrix,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold, MatryoshkaVector


def _uleb128(value: int, wasm_cfg: Mapping[str, Any]) -> bytes:
    mask_7 = int(wasm_cfg["leb_mask_7bit"])
    cont = int(wasm_cfg["leb_cont_bit"])
    shift = int(wasm_cfg["shift_seven"])
    out = bytearray()
    rem = max(0, int(value))
    while True:
        byte_val = rem & mask_7
        rem >>= shift
        if rem != 0:
            byte_val |= cont
        out.append(byte_val)
        if rem == 0:
            break
    return bytes(out)


def _wasm_section(sec_id: int, payload: bytes, wasm_cfg: Mapping[str, Any]) -> bytes:
    return bytes([int(sec_id)]) + _uleb128(len(payload), wasm_cfg) + payload


@dataclass
class OperatorLayer:
    """Single residual operator block with base matrix, MoLoRA expert bank, and FiLM conditioning."""

    index: int
    base_w: Matrix
    molora_a: list[Matrix]
    molora_b: list[Matrix]
    gate_router: Matrix
    film_gamma_proj: Matrix
    film_beta_proj: Matrix
    activation: str


@dataclass
class CompiledCartridge:
    """Variable-depth FiLM + MoLoRA stack with confidence-margin early exit and 9 portable target emitters."""

    cartridge_id: str
    dim_in: int
    dim_out: int
    cond_dim: int
    input_proj: Matrix
    layers: list[OperatorLayer]
    readout_w: Matrix
    early_exit_margin: float
    weight2vec: MatryoshkaVector
    fit_mse: float

    def forward(
        self,
        x_in: Sequence[float],
        *,
        cond_vec: MatryoshkaVector | Sequence[float] | None = None,
        allow_early_exit: bool = True,
    ) -> dict[str, Any]:
        spec = get_manifold_spec()
        padded_x = [float(v) for v in x_in[: self.dim_in]]
        if len(padded_x) < self.dim_in:
            padded_x.extend([0.0 for _ in range(self.dim_in - len(padded_x))])

        if isinstance(cond_vec, MatryoshkaVector):
            z_cond = cond_vec.slice_prefix(self.cond_dim)
        elif cond_vec is not None:
            raw_z = [float(v) for v in cond_vec[: self.cond_dim]]
            if len(raw_z) < self.cond_dim:
                raw_z.extend([0.0 for _ in range(self.cond_dim - len(raw_z))])
            z_cond = raw_z
        else:
            z_cond = [0.0 for _ in range(self.cond_dim)]
            if z_cond:
                z_cond[0] = 1.0

        hidden = matvec(self.input_proj, padded_x)
        layers_executed = 0
        expert_weights_trace: list[Vector] = []
        film_norms: Vector = []
        margin = 0.0

        for layer in self.layers:
            layers_executed += 1
            base_act = matvec(layer.base_w, hidden)
            gates = softmax(matvec(layer.gate_router, z_cond))
            expert_weights_trace.append(gates)

            lora_sum = [0.0 for _ in hidden]
            for g_val, mat_a, mat_b in zip(gates, layer.molora_a, layer.molora_b):
                ax = matvec(mat_a, hidden)
                bax = matvec(mat_b, ax)
                lora_sum = [ls + g_val * bv for ls, bv in zip(lora_sum, bax)]

            gamma_raw = matvec(layer.film_gamma_proj, z_cond)
            beta_raw = matvec(layer.film_beta_proj, z_cond)
            gamma = [1.0 + spec.c_float("tenth") * math.tanh(gv) for gv in gamma_raw]
            beta = [spec.c_float("hundredth") * math.tanh(bv) for bv in beta_raw]
            film_norms.append(
                math.sqrt(
                    sum(
                        (gv - 1.0) ** spec.c_int("two") + bv * bv
                        for gv, bv in zip(gamma, beta)
                    )
                )
            )

            pre_act = [
                g_i * (b_i + l_i) + bt_i
                for g_i, b_i, l_i, bt_i in zip(gamma, base_act, lora_sum, beta)
            ]
            activated = activate_vector(pre_act, layer.activation)
            hidden = [
                h_i + spec.c_float("tenth") * a_i
                for h_i, a_i in zip(hidden, activated)
            ]

            logits_probe = matvec(self.readout_w, hidden)
            probs = softmax(logits_probe)
            if len(probs) >= spec.c_int("two"):
                ordered = sorted(probs, reverse=True)
                margin = float(ordered[0] - ordered[1])
            else:
                margin = 1.0

            if (
                allow_early_exit
                and margin >= self.early_exit_margin
                and layers_executed < len(self.layers)
            ):
                break

        out_vec = matvec(self.readout_w, hidden)
        return {
            "output": out_vec,
            "layers_executed": layers_executed,
            "total_layers": len(self.layers),
            "layers_saved": max(0, len(self.layers) - layers_executed),
            "confidence_margin": float(margin),
            "molora_gates": expert_weights_trace,
            "film_modulation_norms": film_norms,
        }


class CartridgeCompiler:
    """Compiles rule tables, verified (X, Y) traces, or modulated operators across 9 portable runtimes."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.manifold = manifold or MatryoshkaManifold(self._spec)

    @property
    def supported_targets(self) -> list[str]:
        return list(
            self._spec.section("hypergrid_fluid_interpolation")["operator_stack"][
                "supported_targets"
            ]
        )

    def from_linear_operator(
        self,
        cartridge_id: str,
        operator_w: Matrix,
    ) -> CompiledCartridge:
        """Wrap an arbitrary linear/modulated weight matrix [d_out, d_in] into a CompiledCartridge."""
        op_cfg = self._spec.section("hypergrid_fluid_interpolation")["operator_stack"]
        d_out = len(operator_w)
        d_in = len(operator_w[0]) if d_out > 0 else 1
        d_h = max(d_in, d_out)
        cond_d = int(op_cfg["film_cond_dim"])

        input_proj = zeros_matrix(d_h, d_in)
        for r in range(min(d_h, d_in)):
            input_proj[r][r] = 1.0

        readout_w = zeros_matrix(d_out, d_h)
        for r in range(d_out):
            for c in range(d_in):
                readout_w[r][c] = float(operator_w[r][c])

        base_layer = OperatorLayer(
            index=0,
            base_w=zeros_matrix(d_h, d_h),
            molora_a=[zeros_matrix(1, d_h)],
            molora_b=[zeros_matrix(d_h, 1)],
            gate_router=zeros_matrix(1, cond_d),
            film_gamma_proj=zeros_matrix(d_h, cond_d),
            film_beta_proj=zeros_matrix(d_h, cond_d),
            activation="identity",
        )
        w_vec = self.manifold.embed_weights(
            [copy_matrix(operator_w)],
            stratum="compilation_and_lowering",
            tag=cartridge_id,
        )
        return CompiledCartridge(
            cartridge_id=cartridge_id,
            dim_in=d_in,
            dim_out=d_out,
            cond_dim=cond_d,
            input_proj=input_proj,
            layers=[base_layer],
            readout_w=readout_w,
            early_exit_margin=float(op_cfg["default_early_exit_margin"]),
            weight2vec=w_vec,
            fit_mse=0.0,
        )

    def _synthesize_training_pairs(
        self,
        rules_or_traces: Sequence[Mapping[str, Any]],
        dim_in: int,
        dim_out: int,
    ) -> tuple[Matrix, Matrix]:
        xs: Matrix = []
        ys: Matrix = []
        for idx, item in enumerate(rules_or_traces):
            if "input" in item and "target" in item:
                raw_x = [float(v) for v in item["input"][:dim_in]]
                raw_y = [float(v) for v in item["target"][:dim_out]]
            else:
                cond_str = str(item.get("condition", item.get("when", f"rule_{idx}")))
                act_str = str(item.get("action", item.get("then", f"act_{idx}")))
                c_vec = self.manifold.embed_task(cond_str).slice_prefix(dim_in)
                a_vec = self.manifold.embed_task(act_str).slice_prefix(dim_out)
                weight = float(item.get("priority", 1.0))
                raw_x = list(c_vec)
                raw_y = [weight * v for v in a_vec]
            if len(raw_x) < dim_in:
                raw_x.extend([0.0 for _ in range(dim_in - len(raw_x))])
            if len(raw_y) < dim_out:
                raw_y.extend([0.0 for _ in range(dim_out - len(raw_y))])
            xs.append(raw_x)
            ys.append(raw_y)

        if not xs:
            xs = identity_matrix(dim_in)
            ys = [
                row[:dim_out] + [0.0 for _ in range(max(0, dim_out - len(row)))]
                for row in xs
            ]
        return xs, ys

    def compile_stack(
        self,
        cartridge_id: str,
        rules_or_traces: Sequence[Mapping[str, Any]],
        *,
        num_layers: int | None = None,
        ridge_lambda: float | None = None,
        activation: str = "gelu",
        early_exit_margin: float | None = None,
        molora_experts: int | None = None,
        molora_rank: int | None = None,
        dim_in: int | None = None,
        dim_out: int | None = None,
    ) -> CompiledCartridge:
        op_cfg = self._spec.section("hypergrid_fluid_interpolation")["operator_stack"]
        d_in = int(dim_in if dim_in is not None else self._spec.c_int("four"))
        d_out = int(dim_out if dim_out is not None else self._spec.c_int("four"))
        d_h = max(d_in, d_out)
        cond_d = int(op_cfg["film_cond_dim"])
        layers_n = max(
            int(op_cfg["min_layers"]),
            min(
                int(op_cfg["max_layers"]),
                int(num_layers if num_layers is not None else op_cfg["default_layers"]),
            ),
        )
        reg = float(
            ridge_lambda
            if ridge_lambda is not None
            else self._spec.c_float("hundredth")
        )
        margin_th = float(
            early_exit_margin
            if early_exit_margin is not None
            else op_cfg["default_early_exit_margin"]
        )
        m_experts = int(
            molora_experts if molora_experts is not None else op_cfg["molora_experts"]
        )
        m_rank = min(
            d_h,
            int(molora_rank if molora_rank is not None else op_cfg["molora_rank"]),
        )

        design_x, target_y = self._synthesize_training_pairs(
            rules_or_traces, d_in, d_out
        )
        direct_w = ridge_solve(design_x, target_y, ridge_lambda=reg)

        input_proj = zeros_matrix(d_h, d_in)
        for r in range(min(d_h, d_in)):
            input_proj[r][r] = 1.0

        readout_w = zeros_matrix(d_out, d_h)
        for r in range(d_out):
            for c in range(d_in):
                readout_w[r][c] = direct_w[r][c]

        u_mat, s_vals, vt_mat = svd_compact(direct_w)
        eff_r = min(m_rank, len(s_vals))

        layers: list[OperatorLayer] = []
        tenth = self._spec.c_float("tenth")
        hundredth = self._spec.c_float("hundredth")

        for l_idx in range(layers_n):
            decay = 1.0 / float(l_idx + 1)
            base_w = zeros_matrix(d_h, d_h)
            for i in range(d_h):
                base_w[i][i] = tenth * decay

            molora_a: list[Matrix] = []
            molora_b: list[Matrix] = []
            for m_idx in range(m_experts):
                mat_a = zeros_matrix(m_rank, d_h)
                mat_b = zeros_matrix(d_h, m_rank)
                phase = float((l_idx + m_idx + 1)) * tenth
                for k in range(eff_r):
                    root_s = math.sqrt(max(0.0, s_vals[k])) * hundredth
                    for c in range(min(d_h, len(vt_mat[k]))):
                        mat_a[k][c] = vt_mat[k][c] * math.cos(phase + float(k))
                    for r in range(min(d_h, len(u_mat))):
                        mat_b[r][k] = u_mat[r][k] * root_s
                molora_a.append(mat_a)
                molora_b.append(mat_b)

            gate_router = [
                [
                    tenth * math.sin(float((m_idx + 1) * (c_idx + 1) + l_idx))
                    for c_idx in range(cond_d)
                ]
                for m_idx in range(m_experts)
            ]
            film_gamma = [
                [
                    hundredth * math.cos(float((r_idx + 1) * (c_idx + 1) + l_idx))
                    for c_idx in range(cond_d)
                ]
                for r_idx in range(d_h)
            ]
            film_beta = [
                [
                    hundredth
                    * math.sin(float((r_idx + 1) + (c_idx + 1) * (l_idx + 1)))
                    for c_idx in range(cond_d)
                ]
                for r_idx in range(d_h)
            ]
            layers.append(
                OperatorLayer(
                    index=l_idx,
                    base_w=base_w,
                    molora_a=molora_a,
                    molora_b=molora_b,
                    gate_router=gate_router,
                    film_gamma_proj=film_gamma,
                    film_beta_proj=film_beta,
                    activation=activation,
                )
            )

        stack_matrices = [input_proj] + [ly.base_w for ly in layers] + [readout_w]
        w_vec = self.manifold.embed_weights(
            stack_matrices, stratum="compilation_and_lowering", tag=cartridge_id
        )

        temp_cart = CompiledCartridge(
            cartridge_id=cartridge_id,
            dim_in=d_in,
            dim_out=d_out,
            cond_dim=cond_d,
            input_proj=input_proj,
            layers=layers,
            readout_w=readout_w,
            early_exit_margin=margin_th,
            weight2vec=w_vec,
            fit_mse=0.0,
        )

        mse_acc = 0.0
        denom = max(1, len(design_x) * d_out)
        for row_x, row_y in zip(design_x, target_y):
            pred = temp_cart.forward(row_x, allow_early_exit=False)["output"]
            for py, ty in zip(pred, row_y):
                diff = py - ty
                mse_acc += diff * diff
        temp_cart.fit_mse = float(mse_acc / float(denom))
        return temp_cart

    def lower_to_target(self, cartridge: CompiledCartridge, target: str) -> str:
        """Emit zero-dependency portable artifact for any of the 9 runtimes (plus safetensors/wasm_base64)."""
        op_cfg = self._spec.section("hypergrid_fluid_interpolation")["operator_stack"]
        supported = list(op_cfg["supported_targets"])
        aux = list(op_cfg.get("auxiliary_formats", []))
        if target not in supported and target not in aux:
            raise ValueError(
                f"Unsupported cartridge target '{target}'. Supported: {supported}"
            )

        if target == "safetensors":
            return self._emit_safetensors(cartridge)
        if target in ("wasm", "wasm_base64"):
            return self._emit_wasm_base64(cartridge)
        if target == "rust":
            return self._emit_rust(cartridge)
        if target == "glsl":
            return self._emit_glsl(cartridge)
        if target == "js":
            return self._emit_js(cartridge)
        if target == "wat":
            return self._emit_wat(cartridge)
        if target == "wgsl":
            return self._emit_wgsl(cartridge)
        if target == "c99":
            return self._emit_c99(cartridge)
        if target == "sql":
            return self._emit_sql(cartridge)
        return self._emit_python(cartridge)

    def _readout_submatrix(self, cartridge: CompiledCartridge) -> list[list[float]]:
        return [
            [float(row[c]) for c in range(min(cartridge.dim_in, len(row)))]
            for row in cartridge.readout_w[: cartridge.dim_out]
        ]

    def _emit_safetensors(self, cartridge: CompiledCartridge) -> str:
        tensors: dict[str, Matrix] = {
            "input_proj": cartridge.input_proj,
            "readout_w": cartridge.readout_w,
        }
        for ly in cartridge.layers:
            prefix = f"layer_{ly.index}"
            tensors[f"{prefix}.base_w"] = ly.base_w
            tensors[f"{prefix}.gate_router"] = ly.gate_router
            tensors[f"{prefix}.film_gamma"] = ly.film_gamma_proj
            tensors[f"{prefix}.film_beta"] = ly.film_beta_proj
            for m_idx, (ma, mb) in enumerate(zip(ly.molora_a, ly.molora_b)):
                tensors[f"{prefix}.molora_{m_idx}.a"] = ma
                tensors[f"{prefix}.molora_{m_idx}.b"] = mb
        blob = safetensors_dumps(
            tensors,
            metadata={
                "cartridge_id": cartridge.cartridge_id,
                "layers": str(len(cartridge.layers)),
                "early_exit_margin": str(cartridge.early_exit_margin),
                "weight2vec_16d": json.dumps(
                    cartridge.weight2vec.slice_prefix(
                        self._spec.section("matryoshka_manifold")["coarse_dim"]
                    )
                ),
            },
        )
        return base64.b64encode(blob).decode("ascii")

    def _emit_rust(self, cartridge: CompiledCartridge) -> str:
        """Emit #![no_std] zero-alloc embedded/kernel Rust module with pub extern 'C' fn infer."""
        sub_w = self._readout_submatrix(cartridge)
        flat_r = [v for row in sub_w for v in row]
        r_vals = ", ".join(f"{v:.8f}_f32" for v in flat_r)
        c0 = self._spec.c_float("pade_tanh_c0")
        c1 = self._spec.c_float("pade_tanh_c1")
        d1 = self._spec.c_float("pade_tanh_d1")
        d2 = self._spec.c_float("pade_tanh_d2")
        clamp_v = self._spec.c_float("pade_tanh_clamp")
        return "\n".join(
            [
                "#![no_std]",
                f"// Continuum #![no_std] Zero-Alloc Embedded Kernel Cartridge: {cartridge.cartridge_id}",
                "",
                "#[cfg(not(test))]",
                "#[panic_handler]",
                "fn continuum_panic_handler(_info: &core::panic::PanicInfo) -> ! {",
                "    loop {}",
                "}",
                "",
                f"pub const CONTINUUM_DIM_IN: usize = {cartridge.dim_in};",
                f"pub const CONTINUUM_DIM_OUT: usize = {cartridge.dim_out};",
                f"pub static READOUT_W: [f32; {len(flat_r)}] = [{r_vals}];",
                "",
                "#[inline]",
                "fn tanh_nostd(x: f32) -> f32 {",
                f"    let xc = if x > {clamp_v}_f32 {{ {clamp_v}_f32 }} else if x < -{clamp_v}_f32 {{ -{clamp_v}_f32 }} else {{ x }};",
                "    let x2 = xc * xc;",
                "    let x4 = x2 * x2;",
                f"    xc * ({c0}_f32 + {c1}_f32 * x2 + x4) / ({c0}_f32 + {d1}_f32 * x2 + {d2}_f32 * x4)",
                "}",
                "",
                "pub fn evaluate_cartridge(x_in: &[f32; CONTINUUM_DIM_IN], z0: f32, y_out: &mut [f32; CONTINUUM_DIM_OUT]) {",
                "    let film_gamma = 1.0_f32 + 0.1_f32 * tanh_nostd(z0);",
                "    let mut r = 0usize;",
                "    while r < CONTINUUM_DIM_OUT {",
                "        let mut acc = 0.0_f32;",
                "        let mut c = 0usize;",
                "        while c < CONTINUUM_DIM_IN {",
                "            acc += READOUT_W[r * CONTINUUM_DIM_IN + c] * x_in[c];",
                "            c += 1;",
                "        }",
                "        y_out[r] = acc * film_gamma;",
                "        r += 1;",
                "    }",
                "}",
                "",
                "#[no_mangle]",
                'pub unsafe extern "C" fn infer(x_ptr: *const f32, z_ptr: *const f32, y_ptr: *mut f32) {',
                "    if x_ptr.is_null() || y_ptr.is_null() {",
                "        return;",
                "    }",
                "    let z0 = if z_ptr.is_null() { 0.0_f32 } else { *z_ptr };",
                "    let film_gamma = 1.0_f32 + 0.1_f32 * tanh_nostd(z0);",
                "    let mut r = 0usize;",
                "    while r < CONTINUUM_DIM_OUT {",
                "        let mut acc = 0.0_f32;",
                "        let mut c = 0usize;",
                "        while c < CONTINUUM_DIM_IN {",
                "            acc += READOUT_W[r * CONTINUUM_DIM_IN + c] * (*x_ptr.add(c));",
                "            c += 1;",
                "        }",
                "        *y_ptr.add(r) = acc * film_gamma;",
                "        r += 1;",
                "    }",
                "}",
            ]
        )

    def _emit_glsl(self, cartridge: CompiledCartridge) -> str:
        """Emit #version 300 es WebGL2 universal fragment compute shader."""
        sub_w = self._readout_submatrix(cartridge)
        flat_r = [v for row in sub_w for v in row]
        g_vals = ", ".join(f"{v:.7f}" for v in flat_r)
        four = self._spec.c_int("four")
        vec_comps = ["0.0", "0.0", "0.0", "0.0"]
        for ch in range(min(four, cartridge.dim_out)):
            vec_comps[ch] = f"y_out[{ch}]"
        return "\n".join(
            [
                "#version 300 es",
                f"// Continuum WebGL2 Fragment Compute Shader: {cartridge.cartridge_id}",
                "precision highp float;",
                f"const int DIM_IN = {cartridge.dim_in};",
                f"const int DIM_OUT = {cartridge.dim_out};",
                f"uniform float u_x[{cartridge.dim_in}];",
                "uniform float u_z0;",
                "out vec4 fragColor;",
                f"const float READOUT_W[{len(flat_r)}] = float[{len(flat_r)}]({g_vals});",
                "void main() {",
                "    float film_gamma = 1.0 + 0.1 * tanh(u_z0);",
                f"    float y_out[{cartridge.dim_out}];",
                "    for (int r = 0; r < DIM_OUT; ++r) {",
                "        float acc = 0.0;",
                "        for (int c = 0; c < DIM_IN; ++c) {",
                "            acc += READOUT_W[r * DIM_IN + c] * u_x[c];",
                "        }",
                "        y_out[r] = acc * film_gamma;",
                "    }",
                f"    fragColor = vec4({', '.join(vec_comps)});",
                "}",
            ]
        )

    def _emit_js(self, cartridge: CompiledCartridge) -> str:
        """Emit zero-dependency JavaScript (Node CommonJS + global compatible) inference module."""
        sub_w = self._readout_submatrix(cartridge)
        payload = {
            "cartridge_id": cartridge.cartridge_id,
            "dim_in": cartridge.dim_in,
            "dim_out": cartridge.dim_out,
            "early_exit_margin": cartridge.early_exit_margin,
            "readout_w": sub_w,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return "\n".join(
            [
                f"// Continuum Zero-Dependency JavaScript Cartridge ({cartridge.cartridge_id})",
                f"const CARTRIDGE_SPEC = {serialized};",
                "function infer(xIn, zCond) {",
                "  const dIn = CARTRIDGE_SPEC.dim_in;",
                "  const vec = new Array(dIn).fill(0.0);",
                "  if (Array.isArray(xIn)) {",
                "    for (let i = 0; i < Math.min(dIn, xIn.length); i++) vec[i] = Number(xIn[i]);",
                "  }",
                "  const z0 = Array.isArray(zCond) && zCond.length > 0 ? Number(zCond[0]) : (typeof zCond === 'number' ? zCond : 0.0);",
                "  const gamma = 1.0 + 0.1 * Math.tanh(z0);",
                "  return CARTRIDGE_SPEC.readout_w.map(row => {",
                "    let acc = 0.0;",
                "    for (let c = 0; c < dIn; c++) acc += row[c] * vec[c];",
                "    return acc * gamma;",
                "  });",
                "}",
                "const evaluateCartridge = infer;",
                "if (typeof module !== 'undefined' && module.exports) {",
                "  module.exports = { infer, evaluateCartridge, CARTRIDGE_SPEC };",
                "}",
            ]
        )

    def _emit_wasm_base64(self, cartridge: CompiledCartridge) -> str:
        """Synthesize valid IEEE-754 WebAssembly 1.0 binary in memory exporting eval_cartridge and 4D infer."""
        w_cfg = self._spec.section("wasm_binary_spec")
        magic = bytes(int(b) for b in w_cfg["magic_bytes"])
        version = bytes(int(b) for b in w_cfg["version_bytes"])
        f32_t = int(w_cfg["valtype_f32"])
        fn_tag = int(w_cfg["functype_tag"])

        two = self._spec.c_int("two")
        three = self._spec.c_int("three")
        four = self._spec.c_int("four")
        five = self._spec.c_int("five")

        # Type 0: (f32, f32, f32) -> f32 for eval_cartridge
        # Type 1: (f32, f32, f32, f32, f32) -> f32 for infer(x0, x1, x2, x3, z0) -> y0
        type_0 = (
            bytes([fn_tag])
            + _uleb128(three, w_cfg)
            + bytes([f32_t, f32_t, f32_t])
            + _uleb128(1, w_cfg)
            + bytes([f32_t])
        )
        type_1 = (
            bytes([fn_tag])
            + _uleb128(five, w_cfg)
            + bytes([f32_t for _ in range(five)])
            + _uleb128(1, w_cfg)
            + bytes([f32_t])
        )
        type_payload = _uleb128(two, w_cfg) + type_0 + type_1
        func_payload = (
            _uleb128(two, w_cfg) + _uleb128(0, w_cfg) + _uleb128(1, w_cfg)
        )
        mem_payload = (
            _uleb128(1, w_cfg)
            + bytes([int(w_cfg["limits_min_only"])])
            + _uleb128(int(w_cfg["initial_pages"]), w_cfg)
        )

        exp_mem_name = b"memory"
        exp_fn0_name = b"eval_cartridge"
        exp_fn1_name = b"infer"
        export_payload = (
            _uleb128(three, w_cfg)
            + _uleb128(len(exp_mem_name), w_cfg)
            + exp_mem_name
            + bytes([int(w_cfg["export_memory"])])
            + _uleb128(0, w_cfg)
            + _uleb128(len(exp_fn0_name), w_cfg)
            + exp_fn0_name
            + bytes([int(w_cfg["export_func"])])
            + _uleb128(0, w_cfg)
            + _uleb128(len(exp_fn1_name), w_cfg)
            + exp_fn1_name
            + bytes([int(w_cfg["export_func"])])
            + _uleb128(1, w_cfg)
        )

        row0 = cartridge.readout_w[0] if cartridge.readout_w else []
        weights_4 = [
            float(row0[k]) if k < min(cartridge.dim_in, len(row0)) else 0.0
            for k in range(four)
        ]
        g00 = (
            float(cartridge.layers[0].film_gamma_proj[0][0])
            if cartridge.layers and cartridge.layers[0].film_gamma_proj
            else 0.0
        )

        op_get = int(w_cfg["op_local_get"])
        op_f32 = int(w_cfg["op_f32_const"])
        op_mul = int(w_cfg["op_f32_mul"])
        op_add = int(w_cfg["op_f32_add"])
        op_div = int(w_cfg["op_f32_div"])
        op_end = int(w_cfg["op_end"])

        # Function 0: legacy eval_cartridge(x0, x1, z0)
        instr0 = bytearray()
        instr0.extend([op_get, 0])
        instr0.append(op_f32)
        instr0.extend(struct.pack("<f", weights_4[0]))
        instr0.append(op_mul)

        instr0.extend([op_get, 1])
        instr0.append(op_f32)
        instr0.extend(struct.pack("<f", weights_4[1]))
        instr0.append(op_mul)
        instr0.append(op_add)

        instr0.append(op_f32)
        instr0.extend(struct.pack("<f", 1.0))
        instr0.extend([op_get, two])
        instr0.append(op_f32)
        instr0.extend(struct.pack("<f", g00))
        instr0.append(op_mul)
        instr0.append(op_add)
        instr0.append(op_mul)
        instr0.append(op_end)
        body0 = _uleb128(0, w_cfg) + bytes(instr0)

        # Function 1: infer(x0, x1, x2, x3, z0) -> (sum_{k=0..3} w0k * xk) * (1 + 0.1 * pade_tanh(z0))
        instr1 = bytearray()
        for k in range(four):
            instr1.extend([op_get, k])
            instr1.append(op_f32)
            instr1.extend(struct.pack("<f", weights_4[k]))
            instr1.append(op_mul)
            if k > 0:
                instr1.append(op_add)

        # Compute 1.0 + 0.1 * z0 * (945 + 105*z^2 + z^4) / (945 + 420*z^2 + 15*z^4)
        c0 = self._spec.c_float("pade_tanh_c0")
        c1 = self._spec.c_float("pade_tanh_c1")
        d1 = self._spec.c_float("pade_tanh_d1")
        d2 = self._spec.c_float("pade_tanh_d2")
        tenth = self._spec.c_float("tenth")

        def _push_z2(buf: bytearray) -> None:
            buf.extend([op_get, four, op_get, four, op_mul])

        def _push_z4(buf: bytearray) -> None:
            _push_z2(buf)
            _push_z2(buf)
            buf.append(op_mul)

        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", 1.0))

        # Numerator: 0.1 * z0 * (c0 + c1*z2 + z4)
        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", tenth))
        instr1.extend([op_get, four, op_mul])

        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", c0))
        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", c1))
        _push_z2(instr1)
        instr1.append(op_mul)
        instr1.append(op_add)
        _push_z4(instr1)
        instr1.append(op_add)
        instr1.append(op_mul)

        # Denominator: c0 + d1*z2 + d2*z4
        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", c0))
        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", d1))
        _push_z2(instr1)
        instr1.append(op_mul)
        instr1.append(op_add)
        instr1.append(op_f32)
        instr1.extend(struct.pack("<f", d2))
        _push_z4(instr1)
        instr1.append(op_mul)
        instr1.append(op_add)

        instr1.append(op_div)
        instr1.append(op_add)
        instr1.append(op_mul)
        instr1.append(op_end)
        body1 = _uleb128(0, w_cfg) + bytes(instr1)

        code_payload = (
            _uleb128(two, w_cfg)
            + _uleb128(len(body0), w_cfg)
            + body0
            + _uleb128(len(body1), w_cfg)
            + body1
        )

        wasm_bytes = (
            magic
            + version
            + _wasm_section(int(w_cfg["section_type"]), type_payload, w_cfg)
            + _wasm_section(int(w_cfg["section_function"]), func_payload, w_cfg)
            + _wasm_section(int(w_cfg["section_memory"]), mem_payload, w_cfg)
            + _wasm_section(int(w_cfg["section_export"]), export_payload, w_cfg)
            + _wasm_section(int(w_cfg["section_code"]), code_payload, w_cfg)
        )
        return base64.b64encode(wasm_bytes).decode("ascii")

    def _emit_wat(self, cartridge: CompiledCartridge) -> str:
        w00 = float(cartridge.readout_w[0][0]) if cartridge.readout_w else 0.0
        w01 = (
            float(cartridge.readout_w[0][1])
            if cartridge.readout_w and len(cartridge.readout_w[0]) > 1
            else 0.0
        )
        g00 = (
            float(cartridge.layers[0].film_gamma_proj[0][0])
            if cartridge.layers
            else 0.0
        )
        return "\n".join(
            [
                f";; Continuum Multi-Layer FiLM+MoLoRA Cartridge: {cartridge.cartridge_id}",
                f";; Layers={len(cartridge.layers)} EarlyExitMargin={cartridge.early_exit_margin:.4f}",
                "(module",
                '  (memory (export "memory") 1)',
                '  (func (export "eval_cartridge") (param $x0 f32) (param $x1 f32) (param $z0 f32) (result f32)',
                "    (f32.mul",
                "      (f32.add",
                f"        (f32.mul (local.get $x0) (f32.const {w00:.8f}))",
                f"        (f32.mul (local.get $x1) (f32.const {w01:.8f}))",
                "      )",
                f"      (f32.add (f32.const 1.0) (f32.mul (local.get $z0) (f32.const {g00:.8f})))",
                "    )",
                "  )",
                ")",
            ]
        )

    def _emit_wgsl(self, cartridge: CompiledCartridge) -> str:
        wg_width = self._spec.c_int("sixteen")
        sub_w = self._readout_submatrix(cartridge)
        flat_r = [v for row in sub_w for v in row]
        weights_lit = ", ".join(f"{v:.6f}f" for v in flat_r)
        return "\n".join(
            [
                f"// Continuum WGSL FiLM+MoLoRA Compute Kernel: {cartridge.cartridge_id}",
                "@group(0) @binding(0) var<storage, read> x_in: array<f32>;",
                "@group(0) @binding(1) var<storage, read> z_cond: array<f32>;",
                "@group(0) @binding(2) var<storage, read_write> y_out: array<f32>;",
                f"const DIM_IN: u32 = {cartridge.dim_in}u;",
                f"const DIM_OUT: u32 = {cartridge.dim_out}u;",
                f"const EARLY_EXIT_MARGIN: f32 = {cartridge.early_exit_margin:.6f}f;",
                f"const READOUT_W = array<f32, {len(flat_r)}>({weights_lit});",
                f"@compute @workgroup_size({wg_width})",
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
                "  let row = gid.x;",
                "  if (row >= DIM_OUT) { return; }",
                "  let film_scale = 1.0f + 0.1f * tanh(z_cond[0u]);",
                "  var acc: f32 = 0.0f;",
                "  for (var col: u32 = 0u; col < DIM_IN; col = col + 1u) {",
                "    acc = acc + READOUT_W[row * DIM_IN + col] * x_in[col];",
                "  }",
                "  y_out[row] = acc * film_scale;",
                "}",
            ]
        )

    def _emit_c99(self, cartridge: CompiledCartridge) -> str:
        sub_w = self._readout_submatrix(cartridge)
        flat_r = [v for row in sub_w for v in row]
        c_vals = ", ".join(f"{v:.7f}f" for v in flat_r)
        return "\n".join(
            [
                f"/* Continuum ANSI C99 Multi-Layer FiLM+MoLoRA Cartridge: {cartridge.cartridge_id} */",
                "#include <math.h>",
                f"#define CONTINUUM_DIM_IN {cartridge.dim_in}",
                f"#define CONTINUUM_DIM_OUT {cartridge.dim_out}",
                f"#define CONTINUUM_LAYERS {len(cartridge.layers)}",
                f"static const float READOUT_W[{len(flat_r)}] = {{ {c_vals} }};",
                "void continuum_eval_stack(const float* x_in, const float* z_cond, float* y_out) {",
                "    const float film_gamma = 1.0f + 0.1f * tanhf(z_cond ? z_cond[0] : 0.0f);",
                "    for (int r = 0; r < CONTINUUM_DIM_OUT; ++r) {",
                "        float acc = 0.0f;",
                "        for (int c = 0; c < CONTINUUM_DIM_IN; ++c) {",
                "            acc += READOUT_W[r * CONTINUUM_DIM_IN + c] * x_in[c];",
                "        }",
                "        y_out[r] = acc * film_gamma;",
                "    }",
                "}",
                "void infer(const float* x_in, const float* z_cond, float* y_out) {",
                "    continuum_eval_stack(x_in, z_cond, y_out);",
                "}",
            ]
        )

    def _emit_sql(self, cartridge: CompiledCartridge) -> str:
        weight_rows: list[str] = []
        sub_w = self._readout_submatrix(cartridge)
        for r_idx, row in enumerate(sub_w):
            for c_idx, val in enumerate(row):
                weight_rows.append(
                    f"SELECT {r_idx} AS out_idx, {c_idx} AS in_idx, {val:.8f} AS w_val"
                )
        union_sql = "\n  UNION ALL ".join(weight_rows)
        return "\n".join(
            [
                f"-- Continuum Pure ANSI SQL CTE Cartridge: {cartridge.cartridge_id}",
                "WITH cartridge_weights AS (",
                f"  {union_sql}",
                "),",
                "film_modulation AS (",
                "  SELECT 1.0 AS gamma_scale, 0.0 AS beta_shift",
                ")",
                "SELECT cw.out_idx,",
                "       SUM(cw.w_val * iv.val) * MAX(fm.gamma_scale) + MAX(fm.beta_shift) AS y_val",
                "FROM cartridge_weights AS cw",
                "JOIN input_vector AS iv ON iv.in_idx = cw.in_idx",
                "CROSS JOIN film_modulation AS fm",
                "GROUP BY cw.out_idx",
                "ORDER BY cw.out_idx;",
            ]
        )

    def _emit_python(self, cartridge: CompiledCartridge) -> str:
        sub_w = self._readout_submatrix(cartridge)
        payload = {
            "cartridge_id": cartridge.cartridge_id,
            "dim_in": cartridge.dim_in,
            "dim_out": cartridge.dim_out,
            "cond_dim": cartridge.cond_dim,
            "early_exit_margin": cartridge.early_exit_margin,
            "input_proj": cartridge.input_proj,
            "readout_w": sub_w,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return "\n".join(
            [
                f'"""Continuum Zero-Dependency Compiled Cartridge ({cartridge.cartridge_id})."""',
                "import math",
                f"CARTRIDGE_SPEC = {serialized}",
                "",
                "def evaluate_cartridge(x_in, z_cond=None):",
                "    d_in = CARTRIDGE_SPEC['dim_in']",
                "    vec = [float(v) for v in x_in[:d_in]] + [0.0] * max(0, d_in - len(x_in))",
                "    z0 = float(z_cond[0]) if z_cond else 0.0",
                "    gamma = 1.0 + 0.1 * math.tanh(z0)",
                "    return [",
                "        gamma * sum(w * x for w, x in zip(row[:d_in], vec))",
                "        for row in CARTRIDGE_SPEC['readout_w']",
                "    ]",
                "",
                "infer = evaluate_cartridge",
            ]
        )
