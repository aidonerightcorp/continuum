# Continuum — Unified Self-Improving Agent Engine (`v1.2.0`)

**Smart context compression, local episodic memory, Mechanistic SAE/ELK latent probing, Nullspace activation steering, Cellular Sheaf & Conformal verification, 160-axis TT-Cross configuration tuning, and 9-target skill compilation for Hermes Agent (`>=0.23.0`).**

![Continuum Architecture Overview](docs/continuum-overview.png)

**Continuum** gives Hermes Agent a unified, self-improving runtime that gets faster, sharper, and more interpretable as you work. Instead of bolting together disconnected utilities, Continuum connects **five native Hermes extension surfaces** over one shared local engine: **28 cooperating cybernetic engines** (`104` sub-engines across `7` functional layers), **shared 4-space Matryoshka embeddings** (`16d / 32d / 64d / 128d`), **Matryoshka Top-K Sparse Autoencoders (SAEs), Multi-Layer Crosscoders, ELK Layer-Divergence Probing, and `Null(A_anchor)` Activation Steering**, a **160-axis `TT-Cross` adaptive configuration grid** that tunes itself offline using **zero extra LLM tokens**, and a **portable skill & probe compiler** targeting nine zero-dependency runtimes.

Everything runs **100% locally in-process** backed by profile-scoped SQLite WAL storage and dual-mode linear algebra (pure Python standard library out of the box, with automatic NumPy acceleration when installed). Continuum makes **zero outbound network calls**, spawns **zero background TCP daemons**, requires **zero API keys**, and commits **zero binary blobs** to source control.

---

## Five Native Hermes Surfaces in One Plugin

| Surface | Registration API | What It Does for Hermes |
| :--- | :--- | :--- |
| **1. `ContextEngine`** | `ctx.register_context_engine` | Deduplicates repeated turns in a content-addressed quotient DAG, prunes low-saliency filler when approaching context limits, preserves the exact `request_messages` list reference when within budget for 100% prompt-cache hits, supports thread-safe `clone_for_agent()`, and provides `continuum_context_grep`. |
| **2. `MemoryProvider`** | `ctx.register_memory_provider` | Stores session memories in a profile-isolated SQLite WAL store (`HERMES_HOME/plugin-data/continuum/continuum.db`), retrieves relevant context via fast coarse-to-fine (`16d -> 32d -> 64d -> 128d`) search, suppresses writes during `cron` / `subagent` / `flush` runs, and consolidates pre-compaction turns into portable `.safetensors` micro-cartridges. |
| **3. `DelegationProvider`** | `ctx.register_delegation_provider` | Extends subtask routing with `engine_stratum`, `verification_role` (`maker` / `checker`), and `maker_id`; routes subtasks using online contextual LinUCB Woodbury scoring and enforces strict `maker != checker` separation (`DelegationRouteError` on collision). |
| **4. Tools, Hooks, Command & Skill** | `register_tool` / `register_hook` / `register_command` / `register_skill` | Registers **17 `continuum_*` agent tools**, **4 session lifecycle hooks** (`on_session_start`, `pre_tool_call`, `post_tool_call`, `on_session_end`), the `/continuum [status\|step]` slash command, and the bundled `continuum` workflow skill. |
| **5. Desktop & Web Dashboard** | `desktop/plugin.js` + `dashboard/` | Native Hermes Desktop status-bar chip, inspector pane, and settings page built exclusively on `@hermes/plugin-sdk` + `react/jsx-runtime` (`#f5f2ea` Paper & `#141413` Ink light palette), paired with read-only SQLite WAL FastAPI telemetry endpoints (`/status`, `/engines`). |

---

## What's New in `v1.2.0` (Mechanistic Interpretability & Nullspace Steering)

1. **28 Cybernetic Engines (`104` Sub-Engines across `7` Layers)**: adds a dedicated 7th layer (`latent_interpretability_and_steering`) featuring four mechanistic interpretability and activation-control engines (`SparseFeatureDictionaryEngine`, `CrosslayerCircuitTracerEngine`, `InternalStateProberEngine`, and `NullspaceActivationSteererEngine`).
2. **Matryoshka Top-K JumpReLU SAEs & Multi-Layer Sparse Crosscoders (`continuum_probe_latent_state`)**: decomposes residual vectors or session traces into nested coarse-to-fine monosemantic sparse features with AuxK dormant-latent revival, selectivity scoring, joint cross-layer circuit tracing, Orthogonal Procrustes layer alignment, and causal patching attribution.
3. **Internal Concept Probing, ELK Layer-Divergence Detection & Conformal Truthfulness (`continuum_probe_latent_state`)**: combines Mass-Mean and shrinkage-regularized Fisher Linear Discriminant (`LDA`) probes with Eliciting Latent Knowledge (`ELK`) intermediate-vs-surface layer divergence auditing and finite-sample Split-Conformal truthfulness sets (plus optional `9`-runtime portable probe compilation).
4. **Surgical `Null(A_anchor)` Activation Steering & Zero-Hook Weight Folding (`continuum_steer_nullspace_vector`)**: extracts Contrastive or monosemantic SAE steering vectors projected strictly onto $\operatorname{Null}(A_{\text{anchor}})$ — achieving **`0.0` anchor subspace drift** where naive residual steering disturbs protected behaviors — and folds them into zero-overhead rank-1 `LoRA-XS` + bias patches exportable across all **9 zero-dependency runtimes**.
5. **Retains Full `v1.1.0` Cybernetic Stack**: 160-axis `TT-Cross` epistatic `HyperGrid` (`0` extra LLM tokens), Cellular Sheaf consistency auditing (`continuum_sheaf_consistency_audit`), Split/Online ACI Conformal calibration (`continuum_conformal_calibrate`), Kronecker `HyperLoRA` (`continuum_hyperlora_modulate`), and 9 portable targets (`rust`, `glsl`, `js`, `python`, `sql`, `c99`, `wasm`, `wat`, `wgsl`, plus `safetensors`).

---

## Portable Skill, Probe & Steerer Compilation Across 9 Standalone Runtimes

`continuum_compile_cartridge`, `continuum_hyperlora_modulate`, `continuum_probe_latent_state`, and `continuum_steer_nullspace_vector` lower multi-layer (`1..6` layers) `FiLM` + `MoLoRA` + `HyperLoRA` operator stacks, linear/SAE latent probes, and folded `Null(A_anchor)` steerers into nine zero-dependency targets:

| Target | Output Artifact | Execution Environment |
| :--- | :--- | :--- |
| **`rust`** | `#![no_std]` zero-alloc module exporting `evaluate_cartridge` + C ABI `infer` | Embedded firmware, OS kernels, `cdylib` FFI |
| **`glsl`** | `#version 300 es` high-precision fragment compute shader | WebGL2 browsers & mobile GPUs |
| **`js`** | Zero-dependency ES/CommonJS module exporting `infer(xIn, zCond)` | Node.js, Cloudflare Workers, browsers |
| **`python`** | Standalone pure-stdlib Python module exporting `infer` | Any Python 3 runtime |
| **`sql`** | Pure ANSI SQL CTE matrix-vector query (`WITH cartridge_weights AS ...`) | SQLite, PostgreSQL, DuckDB |
| **`c99`** | Self-contained ANSI C99 source exporting `continuum_eval_stack` & `infer` | Native C/C++ binaries & microcontrollers |
| **`wasm`** | IEEE-754 WebAssembly 1.0 binary synthesized dynamically in memory (base64) | Wasmtime, V8, browser `WebAssembly.instantiate` |
| **`wat`** | Human-auditable WebAssembly Text S-expression module | WebAssembly toolchains & security review |
| **`wgsl`** | `@compute @workgroup_size(16)` storage-buffer compute kernel | WebGPU native & browser compute pipelines |

*(Auxiliary interchange format `safetensors` is also supported for IEEE-754 `float32` tensor serialization with embedded `Weight2Vec` headers.)*

---

## The 28 Cooperating Cybernetic Engines (`104` Sub-Engines)

| # | Engine ID | Functional Layer | Adaptation Cadence | Sub-Engines |
| :--- | :--- | :--- | :--- | :---: |
| 0 | `quotient_context_dag_engine` | Context & Memory | Fast turn reflex | 4 |
| 1 | `wanda_saliency_pruning_engine` | Context & Memory | Fast turn reflex | 4 |
| 2 | `hopfield_attractor_memory_engine` | Context & Memory | Session consolidation | 4 |
| 3 | `information_bottleneck_compression_engine` | Context & Memory | Fast turn reflex | 4 |
| 4 | `ridge_polytope_compiler_engine` | Skill Compilation | Session consolidation | 4 |
| 5 | `multilayer_film_molora_stack_engine` | Skill Compilation | Task episode | 4 |
| 6 | `svd_nullspace_surgical_editor_engine` | Skill Compilation | Task episode | 4 |
| 7 | `portable_target_lowering_engine` | Skill Compilation | Session consolidation | 4 |
| 8 | `contextual_linucb_router_engine` | Routing & Mechanism | Task episode | 4 |
| 9 | `cas_fission_lease_fencing_engine` | Routing & Mechanism | Fast turn reflex | 3 |
| 10 | `dawid_skene_quorum_adjudicator_engine` | Routing & Mechanism | Task episode | 3 |
| 11 | `game_theoretic_mechanism_engine` | Routing & Mechanism | Session consolidation | 4 |
| 12 | `sheaf_consensus_topology_engine` | Sheaf & Conformal | Task episode | 4 |
| 13 | `conformal_calibration_engine` | Sheaf & Conformal | Fast turn reflex | 4 |
| 14 | `formal_invariant_synthesis_engine` | Sheaf & Conformal | Session consolidation | 3 |
| 15 | `causal_counterfactual_engine` | Sheaf & Conformal | Task episode | 3 |
| 16 | `rmt_marchenko_pastur_spectral_engine` | TT-Cross & OED | Session consolidation | 4 |
| 17 | `fluid_hypergrid_interpolator_engine` | TT-Cross & OED | Fast turn reflex | 4 |
| 18 | `active_experimental_design_engine` | TT-Cross & OED | Fast turn reflex | 3 |
| 19 | `operation_cost_thermodynamic_engine` | TT-Cross & OED | Task episode | 3 |
| 20 | `inter_engine_coupling_and_jacobian_bus` | Autonomic Coupling | Cross-task transfer | 4 |
| 21 | `homeostatic_meta_controller_engine` | Autonomic Coupling | Grid range evolution | 3 |
| 22 | `structural_morphism_and_transfer_engine` | Autonomic Coupling | Cross-task transfer | 3 |
| 23 | `autonomic_self_healing_engine` | Autonomic Coupling | Task episode | 4 |
| 24 | `sparse_feature_dictionary` | Latent Mind & Steering | Fast turn reflex | 4 |
| 25 | `crosslayer_circuit_tracer` | Latent Mind & Steering | Task episode | 4 |
| 26 | `internal_state_prober` | Latent Mind & Steering | Fast turn reflex | 4 |
| 27 | `nullspace_activation_steerer` | Latent Mind & Steering | Task episode | 4 |

---

## Agent Tool Suite (17 Tools)

1. `continuum_fission_dag` — Split complex goals into a deduplicated subtask graph with clear interface contracts and minimal clarifying questions.
2. `continuum_claim_and_fence` — Claim a subtask with an atomic SQLite lease stamp or return an instant verified cache hit.
3. `continuum_verify_record` — Record independent `maker != checker` verification and update LinUCB, Dawid-Skene, and EigenTrust weights.
4. `continuum_compile_cartridge` — Compile rules or verified examples into a multi-layer skill cartridge across any of the 9 portable runtimes (`rust`, `glsl`, `js`, `python`, `sql`, `c99`, `wasm`, `wat`, `wgsl`, plus `safetensors`).
5. `continuum_analyze_spectrum` — Inspect Marchenko-Pastur noise floors, Roy-Vetterli effective rank, power-law $\alpha$, and Wanda input saliency.
6. `continuum_surgical_edit` — Apply targeted weight updates with CEGAR counterexample healing and `0.0` anchor drift.
7. `continuum_coupling_step` — Propagate Lyapunov-damped `28x28` Jacobian impulses and run zero-token `TT-Cross` + SHA offline config tuning.
8. `continuum_morphism_transfer` — Detect structurally matching task graphs via Laplacian spectra and transplant skills via Orthogonal Procrustes.
9. `continuum_calibrate_effort` — Compute Expected Value of Sample Information (`EVSI`) to pick reasoning effort, embedding slice width, and escalation thresholds.
10. `continuum_manifold_status` — Inspect live status across all 28 engines, 104 sub-engines, 160-axis parameter ledger counts, and compute savings.
11. `continuum_context_grep` — Search compressed and active session turns using hybrid keyword + nested vector matching.
12. `continuum_attractor_recall` — Retrieve relevant episodic memories from the local SQLite WAL Hopfield store.
13. `continuum_sheaf_consistency_audit` — Build a Cellular Sheaf Laplacian across multi-agent claims to localize contradictions and compute harmonic consensus.
14. `continuum_conformal_calibrate` — Construct Split Conformal + Online ACI `1 - alpha` prediction sets with selective abstention routing.
15. `continuum_hyperlora_modulate` — Generate input-conditioned Kronecker `HyperLoRA` weight modulations in the safety-anchor nullspace (`0.0` drift) across any of the 9 portable targets.
16. `continuum_probe_latent_state` — Run Matryoshka Top-K JumpReLU SAE sparse feature decomposition, Crosscoder circuit tracing, Fisher/Mass-Mean concept probing, ELK layer-divergence detection, and Split-Conformal truthfulness calibration (plus optional 9-runtime probe export).
17. `continuum_steer_nullspace_vector` — Extract Contrastive or SAE activation steering vectors projected strictly into `Null(A_anchor)` (`0.0` anchor drift), fold zero-hook rank-1 `LoRA-XS` + bias patches, and export across all 9 portable runtimes.

---

## Measured Local Benchmarks

Measured on Linux x86_64 across both **NumPy-accelerated** and **pure Python standard-library** (`python_dependencies: []`) runtimes:

| Operation | Latency (NumPy) | Key Mathematical Guarantee |
| :--- | :---: | :--- |
| **Initialize 28 Engines + 160-Axis `TT-Cross` Grid** | `~82 ms` | 84 verified baseline config cells indexed at startup |
| **Progressive `16d->32d->64d->128d` Recall (128 items)** | `~2.0 ms` | **`68.8%` MAC reduction** vs. exhaustive `128d` scan |
| **Matryoshka Top-K SAE + ELK State Probe** | `~1.1 ms` | Monosemantic selectivity + Split-Conformal `1 - alpha` truthfulness set |
| **`Null(A_anchor)` Activation Steering + `LoRA-XS` Fold** | `~1.3 ms` | **`0.0` anchor subspace drift** vs. non-zero naive steering drift |
| **`TT-Cross` + 3-Rung SHA Config Tournament** | `~92 ms` | **`0` extra LLM tokens** (replays verified local history) |
| **Cellular Sheaf Audit + Harmonic Consensus Repair** | `~0.9 ms` | Exact localization of contradictory edges + obstruction decay |
| **Cross-Runtime Compilation (`rust`, `c99`, `wasm`, `js`, `sql`, `python`)** | `< 4 ms` emit | **`< 1e-5` numerical parity** across native `rustc`, `gcc`, `node`, & SQLite |

---

## Installation & Configuration

Install via the Hermes plugin CLI:

```text
hermes plugins install continuum
```

Optional provider activation keys in your Hermes profile settings:
- Context Engine: `context.engine: continuum`
- Memory Provider: `memory.provider: continuum`
- Delegation Provider: `delegation.provider: continuum`

Plugin configuration options (`config_schema`):
- `compute_headroom_ratio` (`float`, default `0.6`): fraction of spare local CPU budget used to test `TT-Cross` and Sobol candidate configurations against saved history using zero extra LLM tokens.
- `default_embedding_dim` (`int`, default `64`): default nested embedding slice width (`16`, `32`, `64`, or `128`) for fast retrieval and routing.
- `enable_idle_exploration` (`bool`, default `true`): automatically evaluate and promote better verified configurations at session end.

---

## Security, Privacy & Isolation

- **Zero Network Egress**: Continuum performs zero HTTP, WebSocket, DNS, or raw socket calls.
- **Zero Background Daemons**: all logic runs synchronously in-process; dashboard endpoints read SQLite WAL state in read-only mode (`mode=ro`).
- **Zero External Secrets**: requires zero API keys (`requires_env: []`) and never reads environment variables directly for credentials.
- **Zero Binary Blobs in Git**: WebAssembly binaries (`wasm`) and `.safetensors` buffers are generated deterministically in memory when requested.
- **Whitelisted Public API Only**: imports exclusively documented Hermes v0.23.0+ extension interfaces (`agent.context_engine`, `agent.memory_provider`, `agent.delegation_provider`, `hermes_constants`, `plugins.plugin_storage`) with clean standalone fallbacks.

## License

MIT License — Copyright (c) 2026 `aidonerightcorp`.
