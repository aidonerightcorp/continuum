"""Public OmniNode Shell entrypoint re-exporting Continuum's parameterized CARD, CAS lockfile, >> // | @ director, and grid/1 recorder."""

from __future__ import annotations

from continuum_core.omninode_shell import (
    AlgebraicShapeAndGrid1Director,
    CapabilityGapSynthesizer,
    ContentAddressedCasLockfile,
    Grid1CanonicalCellRecorder,
    IndependentQuorumGateOperator,
    OmniNodeFailure,
    OmniNodeShellEncaser,
    ParallelSheafFanoutOperator,
    SequentialPipeOperator,
    ShapeSyntaxError,
    StaticAstCardInspector,
    TOOL_NAMES_CANONICAL,
    TypedFailureWaterfallRouter,
)

CARD = {
    "name": "omninode_shell_entrypoint",
    "role": "director",
    "stratum": "omninode_shell_and_shape_composition",
    "contract": "omninode.shell/v1",
    "describes": "Top-level OmniNode shell surface re-exporting static AST CARD inspection, CAS lockfile verification, >> // | @ shape composition, and grid/1 cell recording.",
    "keywords": [
        "omninode",
        "shell",
        "card",
        "cas",
        "lockfile",
        "shape",
        "grid1",
    ],
}

__all__ = [
    "AlgebraicShapeAndGrid1Director",
    "CARD",
    "CapabilityGapSynthesizer",
    "ContentAddressedCasLockfile",
    "Grid1CanonicalCellRecorder",
    "IndependentQuorumGateOperator",
    "OmniNodeFailure",
    "OmniNodeShellEncaser",
    "ParallelSheafFanoutOperator",
    "SequentialPipeOperator",
    "ShapeSyntaxError",
    "StaticAstCardInspector",
    "TOOL_NAMES_CANONICAL",
    "TypedFailureWaterfallRouter",
]
