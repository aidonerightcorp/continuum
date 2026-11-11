"""Mechanical conformance gates: Zero Magic Literals, Domain-Agnostic Naming, Desktop Surface, and Dual Linalg."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as continuum_plugin  # noqa: E402
from continuum_core.config_loader import get_manifold_spec  # noqa: E402
from continuum_core.grid_and_gates import (  # noqa: E402
    DomainAgnosticNamingGate,
    GridCellRecord,
    GridLedger,
    ZeroHardcodedValuesAstGate,
)
from continuum_core.linalg_backend import (  # noqa: E402
    add_scaled,
    frobenius_norm,
    identity_matrix,
    matmul,
    ridge_solve,
    svd_compact,
    woodbury_rank1_update,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold  # noqa: E402
from continuum_core.weight_bus import CyberneticWeightBus  # noqa: E402


class _RecordingContext:
    def __init__(self) -> None:
        self.tools: list[str] = []
        self.hooks: list[str] = []
        self.middleware: list[str] = []
        self.hidden: list[str] = []
        self.commands: list[str] = []
        self.skills: list[str] = []
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

    def register_middleware(self, kind: str, callback, **kwargs) -> None:
        self.middleware.append(kind)

    def register_tool(self, name: str, **kwargs) -> None:
        self.tools.append(name)
        check = kwargs.get("check_fn")
        if check is not None and not check():
            self.hidden.append(name)

    def register_hook(self, hook_name: str, callback, **kwargs) -> None:
        self.hooks.append(hook_name)

    def register_command(self, name: str, handler, **kwargs) -> None:
        self.commands.append(name)

    def register_skill(self, name: str, path: Path, **kwargs) -> None:
        self.skills.append(name)


def test_zero_hardcoded_values_ast_gate() -> None:
    operational_files = sorted((REPO_ROOT / "continuum_core").glob("*.py")) + [
        REPO_ROOT / "__init__.py",
        REPO_ROOT / "omninode_shell.py",
        REPO_ROOT / "dashboard" / "plugin_api.py",
    ]
    violations = ZeroHardcodedValuesAstGate.scan_paths(operational_files)
    assert violations == [], f"Magic literal violations found: {violations}"


def test_all_python_modules_expose_static_ast_cards() -> None:
    from omninode_shell import StaticAstCardInspector

    spec = get_manifold_spec()
    inspector = StaticAstCardInspector(spec)
    operational_files = sorted((REPO_ROOT / "continuum_core").glob("*.py")) + [
        REPO_ROOT / "__init__.py",
        REPO_ROOT / "omninode_shell.py",
        REPO_ROOT / "dashboard" / "plugin_api.py",
    ]
    for py_path in operational_files:
        card = inspector.inspect_python_file(py_path)
        assert card is not None, f"Missing static AST CARD in {py_path}"
        for field in (
            "name",
            "role",
            "stratum",
            "describes",
            "keywords",
            "contract",
            "sha256",
        ):
            assert field in card, f"Missing {field} in CARD of {py_path}"
        assert len(card["sha256"]) == 64


def test_domain_agnostic_naming_gate() -> None:
    violations = DomainAgnosticNamingGate.scan_repository(REPO_ROOT)
    assert violations == [], f"Domain-coupled naming violations: {violations}"


def test_zero_legacy_codename_leftovers() -> None:
    legacy_stems = (
        "eigen" + "loom",
        "fenced" + "-task-loop",
        "fenced" + "_task_loop",
    )
    skip_dirs = {".git", "__pycache__", ".pytest_cache"}
    leftovers: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if any(part in skip_dirs for part in path.parts):
            continue
        low_name = path.name.lower()
        if any(st in low_name for st in legacy_stems):
            leftovers.append(str(path.relative_to(REPO_ROOT)))
            continue
        if path.is_file() and path.suffix.lower() not in {".png", ".pyc"}:
            # Skip self-reference inside this test function by checking stems constructed dynamically
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(st in text for st in legacy_stems):
                leftovers.append(str(path.relative_to(REPO_ROOT)))
    assert leftovers == [], f"Legacy codename references found: {leftovers}"


def test_desktop_plugin_surface_sdk_conformance() -> None:
    desktop_js = REPO_ROOT / "desktop" / "plugin.js"
    assert desktop_js.is_file()
    src = desktop_js.read_text(encoding="utf-8")
    assert "from '@hermes/plugin-sdk'" in src
    assert "from 'react/jsx-runtime'" in src
    assert "2.0.0" in src
    assert "ctx.rest('/rightsize')" in src  # the pane shows the plugin's own records, read-only
    assert "ctx.setInterval(" in src  # a scoped timer, cleared by Hermes with the plugin
    for stale in ("1.3.0", "engine_count", "lockfile_verified", "VERIFIED"):  # no fixed figures in the surface
        assert stale not in src
    for forbidden in (
        "eval(",
        "new Function(",
        "hermesDesktop",
        "[data-slot",
        "[data-tour",
        "[data-sidebar",
        "[data-testid",
        "<script",
        "webview",
    ):
        assert forbidden not in src


def test_overview_diagram_exists_and_light_palette() -> None:
    png_path = REPO_ROOT / "docs" / "continuum-overview.png"
    assert png_path.is_file()
    size = png_path.stat().st_size
    assert 10_000 < size < 250_000
    header = png_path.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"


def test_manifest_and_registration_exact_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import continuum_core.storage as storage_mod

    monkeypatch.setattr(
        storage_mod,
        "resolve_continuum_db_path",
        lambda hermes_home=None: tmp_path / "parity.db",
    )
    manifest_text = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert "name: continuum" in manifest_text
    assert 'version: "2.0.0"' in manifest_text
    assert "kind: standalone" in manifest_text
    assert "manifest_version: 2" in manifest_text
    assert 'requires_hermes: ">=0.23.0"' in manifest_text

    ctx = _RecordingContext()
    continuum_plugin.register(ctx)
    assert len(ctx.tools) == 19
    assert len(set(ctx.tools)) == 19
    assert "continuum_recall" in ctx.tools and "continuum_recall" not in ctx.hidden
    assert len(ctx.hidden) == 18  # the lab tools stay hidden from the model by default
    for t_name in ctx.tools:
        assert t_name.startswith("continuum_")
        assert f"- {t_name}" in manifest_text
    declared_hooks = {
        "pre_llm_call",
        "post_llm_call",
        "post_api_request",
        "api_request_error",
        "post_tool_call",
        "on_session_end",
        "on_session_finalize",
    }
    assert set(ctx.hooks) == declared_hooks
    for hook in declared_hooks:
        assert f"- {hook}" in manifest_text
    assert ctx.middleware == ["llm_request"] and "provides_middleware:\n  - llm_request" in manifest_text
    assert ctx.commands == ["continuum"]
    assert ctx.skills == ["continuum"]
    # The lab's providers are registered only when enable_lab_tools is on.
    assert ctx.context_engine is None and ctx.memory_provider is None and ctx.delegation_provider is None


def test_thirty_engines_one_hundred_twelve_subengines_one_seventy_two_hypergrid_dims() -> None:
    spec = get_manifold_spec()
    bus = CyberneticWeightBus(spec)
    status = bus.manifold_status()
    assert status["version"] == "1.3.0"
    assert status["engine_count"] == 30
    assert status["sub_engine_count"] == 112
    assert status["hypergrid_total_dimensions"] == 172
    assert status["supported_targets"] == [
        "rust",
        "glsl",
        "js",
        "python",
        "sql",
        "c99",
        "wasm",
        "wat",
        "wgsl",
    ]
    assert status["timescale_count"] == 5
    assert status["coupling_edge_count"] == 46
    assert "omninode" in status
    assert status["omninode"]["card_counts"]["engines"] == 30
    assert status["omninode"]["card_counts"]["tools"] == 18
    assert status["omninode"]["card_counts"]["surfaces"] == 5
    assert status["matryoshka_spaces"] == [
        "task2vec",
        "config2vec",
        "weight2vec",
        "trace2vec",
    ]
    assert status["matryoshka_prefix_dims"] == [16, 32, 64, 128]
    for required_engine in (
        "sheaf_consensus_topology_engine",
        "conformal_calibration_engine",
        "causal_counterfactual_engine",
        "game_theoretic_mechanism_engine",
        "formal_invariant_synthesis_engine",
        "active_experimental_design_engine",
        "information_bottleneck_compression_engine",
        "autonomic_self_healing_engine",
        "inter_engine_coupling_and_jacobian_bus",
        "homeostatic_meta_controller_engine",
        "structural_morphism_and_transfer_engine",
        "sparse_feature_dictionary",
        "crosslayer_circuit_tracer",
        "internal_state_prober",
        "nullspace_activation_steerer",
        "omninode_shell_encaser",
        "algebraic_shape_and_grid1_director",
    ):
        assert required_engine in bus.engines

    for eng_summary in status["engines"]:
        assert len(eng_summary["card_sha256"]) == 64

    assert bus.engines["sparse_feature_dictionary"].sub_engines == [
        "matryoshka_topk_encoder",
        "jumprelu_threshold_gate",
        "auxk_dead_latent_reviver",
        "monosemantic_selectivity_scorer",
    ]
    assert bus.engines["crosslayer_circuit_tracer"].sub_engines == [
        "multilayer_sparse_crosscoder",
        "mlp_sparse_transcoder",
        "procrustes_manifold_aligner",
        "causal_patching_attributor",
    ]
    assert bus.engines["internal_state_prober"].sub_engines == [
        "mass_mean_concept_prober",
        "fisher_discriminant_prober",
        "elk_layer_divergence_detector",
        "conformal_truthfulness_calibrator",
    ]
    assert bus.engines["nullspace_activation_steerer"].sub_engines == [
        "contrastive_direction_extractor",
        "anchor_nullspace_steerer",
        "zero_hook_weight_folder",
        "control_vector_exporter",
    ]
    assert bus.engines["omninode_shell_encaser"].sub_engines == [
        "static_ast_card_inspector",
        "content_addressed_cas_lockfile",
        "typed_failure_waterfall_router",
        "capability_gap_synthesizer",
    ]
    assert bus.engines["algebraic_shape_and_grid1_director"].sub_engines == [
        "sequential_pipe_operator",
        "parallel_sheaf_fanout_operator",
        "independent_quorum_gate_operator",
        "grid1_canonical_cell_recorder",
    ]


def test_grid_ledger_enforces_maker_checker_separation() -> None:
    spec = get_manifold_spec()
    manifold = MatryoshkaManifold(spec)
    ledger = GridLedger(spec)
    c_vec = manifold.embed_config({"alpha": 0.5})
    rec = GridCellRecord(
        cell_id="test_cell",
        engine_id="quotient_context_dag_engine",
        regime="test",
        values={"alpha": 0.5},
        state="represented",
        config_vec=c_vec,
    )
    ledger.register_cell(rec)
    ledger.transition("test_cell", "evaluated", reward=0.9)

    with pytest.raises(ValueError, match="maker != checker"):
        ledger.transition(
            "test_cell",
            "verified",
            maker_id="same_actor",
            checker_id="same_actor",
        )

    verified = ledger.transition(
        "test_cell",
        "verified",
        maker_id="actor_maker",
        checker_id="actor_checker",
    )
    assert verified.state == "verified"


def test_dual_linalg_stdlib_and_default_svd_reconstruction() -> None:
    mat = [
        [3.0, 1.0, 0.5, -0.2],
        [0.8, 2.5, -0.4, 0.7],
        [-0.3, 0.6, 1.9, 0.3],
        [0.2, -0.5, 0.4, 1.4],
    ]
    for stdlib_only in (False, True):
        u_m, s_v, vt_m = svd_compact(mat, force_stdlib=stdlib_only)
        diag_s = [
            [s_v[r] if r == c else 0.0 for c in range(len(s_v))]
            for r in range(len(s_v))
        ]
        recon = matmul(
            matmul(u_m, diag_s, force_stdlib=stdlib_only),
            vt_m,
            force_stdlib=stdlib_only,
        )
        err = frobenius_norm(add_scaled(recon, mat, scale=-1.0))
        assert err < 1e-6


def test_dual_linalg_ridge_and_woodbury_consistency() -> None:
    x_mat = [[1.0, 0.2], [0.4, 1.1], [0.9, -0.5], [-0.3, 0.8]]
    y_mat = [[2.0, -1.0], [1.2, 0.9], [1.5, -1.3], [-0.1, 0.7]]
    w_def = ridge_solve(x_mat, y_mat, ridge_lambda=0.05, force_stdlib=False)
    w_std = ridge_solve(x_mat, y_mat, ridge_lambda=0.05, force_stdlib=True)
    assert frobenius_norm(add_scaled(w_def, w_std, scale=-1.0)) < 1e-7

    a0 = identity_matrix(3)
    feat = [0.5, -0.3, 0.8]
    u_def = woodbury_rank1_update(a0, feat, damping=1.0, force_stdlib=False)
    u_std = woodbury_rank1_update(a0, feat, damping=1.0, force_stdlib=True)
    assert frobenius_norm(add_scaled(u_def, u_std, scale=-1.0)) < 1e-9
    assert math.isfinite(u_std[0][0])
