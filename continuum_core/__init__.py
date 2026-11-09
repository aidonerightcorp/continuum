"""Continuum Core — OmniNode-Encased 30-Engine Manifold, 172-Axis TT-Cross HyperGrid, Mechanistic SAE/ELK Probing, Nullspace Activation Steering, >> // | @ Shape Composition, and 9-Target Compiler."""

from __future__ import annotations

from continuum_core.cartridge_compiler import CartridgeCompiler, CompiledCartridge
from continuum_core.config_loader import (
    ManifoldConfigSpec,
    OmniNodeGridSpec,
    get_manifold_spec,
    get_omninode_grid_spec,
)
from continuum_core.cybernetic_engines import (
    ActiveExperimentalDesignEngine,
    AutonomicSelfHealingEngine,
    CausalCounterfactualEngine,
    ConformalCalibrationEngine,
    FormalInvariantSynthesisEngine,
    GameTheoreticMechanismEngine,
    InformationBottleneckCompressionEngine,
    SheafConsensusTopologyEngine,
    TensorTrainCross,
    kronecker_product,
)
from continuum_core.grid_and_gates import (
    BilinearWoodburySurrogate,
    ComponentNodeGrid,
    DomainAgnosticNamingGate,
    FluidHyperGrid,
    GridCellRecord,
    GridLedger,
    NodeParameter,
    OperationCostLedger,
    ParameterBounds,
    ZeroHardcodedValuesAstGate,
)
from continuum_core.latent_engines import (
    CrosslayerCircuitTracerEngine,
    InternalStateProberEngine,
    NullspaceActivationSteererEngine,
    SparseFeatureDictionaryEngine,
)
from continuum_core.matryoshka_manifold import MatryoshkaManifold, MatryoshkaVector
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
    TypedFailureWaterfallRouter,
)
from continuum_core.storage import ContinuumStore, resolve_continuum_db_path
from continuum_core.weight_bus import CyberneticWeightBus, ParameterizedWeightSpaceEngine

CARD = {
    "name": "continuum_core",
    "role": "package",
    "stratum": "omninode_shell_and_shape_composition",
    "contract": "continuum.package/v1",
    "describes": "Core OmniNode-encased 30-engine cybernetic manifold, grid/1 recorder, and 9-runtime portable cartridge compiler.",
    "keywords": ["continuum", "omninode", "cybernetic", "hypergrid", "compiler"],
}

__all__ = [
    "ActiveExperimentalDesignEngine",
    "AlgebraicShapeAndGrid1Director",
    "AutonomicSelfHealingEngine",
    "BilinearWoodburySurrogate",
    "CARD",
    "CapabilityGapSynthesizer",
    "CartridgeCompiler",
    "CausalCounterfactualEngine",
    "CompiledCartridge",
    "ComponentNodeGrid",
    "ConformalCalibrationEngine",
    "ContentAddressedCasLockfile",
    "ContinuumStore",
    "CrosslayerCircuitTracerEngine",
    "CyberneticWeightBus",
    "DomainAgnosticNamingGate",
    "FluidHyperGrid",
    "FormalInvariantSynthesisEngine",
    "GameTheoreticMechanismEngine",
    "Grid1CanonicalCellRecorder",
    "GridCellRecord",
    "GridLedger",
    "IndependentQuorumGateOperator",
    "InformationBottleneckCompressionEngine",
    "InternalStateProberEngine",
    "ManifoldConfigSpec",
    "MatryoshkaManifold",
    "MatryoshkaVector",
    "NodeParameter",
    "NullspaceActivationSteererEngine",
    "OmniNodeFailure",
    "OmniNodeGridSpec",
    "OmniNodeShellEncaser",
    "OperationCostLedger",
    "ParallelSheafFanoutOperator",
    "ParameterBounds",
    "ParameterizedWeightSpaceEngine",
    "SequentialPipeOperator",
    "ShapeSyntaxError",
    "SheafConsensusTopologyEngine",
    "SparseFeatureDictionaryEngine",
    "StaticAstCardInspector",
    "TensorTrainCross",
    "TypedFailureWaterfallRouter",
    "ZeroHardcodedValuesAstGate",
    "get_manifold_spec",
    "get_omninode_grid_spec",
    "kronecker_product",
    "resolve_continuum_db_path",
]
