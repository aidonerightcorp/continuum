"""Continuum Dashboard FastAPI Router — Read-only SQLite WAL & OmniNode CARD / grid/1 telemetry endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from continuum_core.config_loader import get_manifold_spec  # noqa: E402
from continuum_core.storage import (  # noqa: E402
    ContinuumStore,
    resolve_continuum_db_path,
)
from continuum_core.weight_bus import CyberneticWeightBus  # noqa: E402

CARD = {
    "name": "dashboard_plugin_api",
    "role": "surface",
    "stratum": "omninode_shell_and_shape_composition",
    "contract": "hermes.desktop_dashboard/v1",
    "describes": "Read-only SQLite WAL FastAPI telemetry endpoints exposing 30-engine health, OmniNode CARD lockfiles, and grid/1 cells.",
    "keywords": ["dashboard", "fastapi", "status", "omninode", "grid1", "surface"],
}

try:
    from fastapi import APIRouter
except ImportError:

    class APIRouter:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.routes: list[tuple[str, Any]] = []

        def get(self, path: str) -> Any:
            def _decorator(fn: Any) -> Any:
                self.routes.append((path, fn))
                return fn

            return _decorator


router = APIRouter()
_BASELINE_BUS: CyberneticWeightBus | None = None


def _get_baseline_bus() -> CyberneticWeightBus:
    global _BASELINE_BUS
    if _BASELINE_BUS is None:
        _BASELINE_BUS = CyberneticWeightBus(get_manifold_spec())
    return _BASELINE_BUS


@router.get("/status")
def get_continuum_status() -> dict[str, Any]:
    """Return read-only SQLite WAL status merged with 30-engine OmniNode & grid/1 metadata."""
    db_path = resolve_continuum_db_path()
    ro = ContinuumStore.read_readonly_snapshot(db_path)
    base = (
        ro["snapshot"]
        if isinstance(ro.get("snapshot"), dict)
        else _get_baseline_bus().manifold_status()
    )
    merged = dict(base)
    if "omninode" not in merged:
        merged["omninode"] = _get_baseline_bus().omninode_director.status_summary()
    merged["wal_persisted"] = bool(ro["persisted"])
    merged["persisted_tasks"] = int(ro["tasks"])
    merged["persisted_verifications"] = int(ro["verifications"])
    merged["persisted_attractors"] = int(ro["attractors"])
    return merged


@router.get("/engines")
def get_continuum_engines() -> dict[str, Any]:
    """Return all 30 Functional Engines, 112 Sub-Engines, 9 Portable Runtimes, and 5 Adaptation Timescales."""
    spec = get_manifold_spec()
    status = get_continuum_status()
    return {
        "engine_count": status["engine_count"],
        "sub_engine_count": status["sub_engine_count"],
        "hypergrid_total_dimensions": status["hypergrid_total_dimensions"],
        "supported_targets": status.get("supported_targets", []),
        "strata": spec.section("strata"),
        "feedback_timescales": spec.section("feedback_timescales"),
        "omninode": status.get("omninode", {}),
        "engines": status["engines"],
    }


@router.get("/omninode")
def get_continuum_omninode() -> dict[str, Any]:
    """Return OmniNode CARD lockfile status, active grid/1 cell SHA-256, and supported >> // | @ operators."""
    bus = _get_baseline_bus()
    status = get_continuum_status()
    omni_summary = status.get("omninode", bus.omninode_director.status_summary())
    return {
        "omninode": omni_summary,
        "card_counts": omni_summary.get("card_counts", {}),
        "cards": {
            k: v
            for k, v in bus.omninode_encaser.catalog.items()
            if not k.startswith("module::")
        },
    }
