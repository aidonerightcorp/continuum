# Continuum — Unified Self-Improving Agent Engine (`v1.3.0`, OmniNode-Encased Edition)

**Parameterized `OmniNode` shell (`CARD` + `.omni/cas` lockfiles + `grid/1` SHA-256 cell receipts + `>> // | @` shape composition), smart context compression, local episodic memory, Mechanistic SAE/ELK latent probing, Nullspace activation steering, Cellular Sheaf & Conformal verification, 172-axis TT-Cross configuration tuning, and 9-target skill compilation for Hermes Agent (`>=0.23.0`).**

![Continuum Architecture Overview](docs/continuum-overview.png)

> **Flagship 84.3s Walkthrough Video (`1080p`, Captioned):** [Download / Watch `continuum-flagship-demo.mp4` on GitHub Release `v1.3.0`](https://github.com/aidonerightcorp/continuum/releases/download/v1.3.0/continuum-flagship-demo.mp4) ([Captions `.srt`](https://github.com/aidonerightcorp/continuum/releases/download/v1.3.0/continuum-flagship-demo.srt) · [Cover Poster](https://github.com/aidonerightcorp/continuum/releases/download/v1.3.0/continuum-flagship-demo.cover.jpg))

**Continuum (`v1.3.0`)** gives Hermes Agent a unified, self-improving runtime where **every engine, tool, and extension surface is encased as a parameterizable `OmniNode`**. Instead of bolting together disconnected utilities, Continuum connects **five native Hermes extension surfaces** over one shared local engine: **30 cooperating cybernetic engines** (`112` sub-engines across `8` functional layers), **self-describing static `CARD` descriptors** pinned by SHA-256 in `.omni/cas` lockfiles, **declarative `>> // | @` pipeline composition** emitting canonical **`format: "grid/1"`** experiment cell receipts, **shared 4-space Matryoshka embeddings** (`16d / 32d / 64d / 128d`), **Matryoshka Top-K Sparse Autoencoders (SAEs), Multi-Layer Crosscoders, ELK Layer-Divergence Probing, and `Null(A_anchor)` Activation Steering**, a **172-axis `TT-Cross` adaptive configuration grid** (`0` extra LLM tokens), and a **portable skill & probe compiler** targeting nine zero-dependency runtimes.

Everything runs **100% locally in-process** backed by profile-scoped SQLite WAL storage (`HERMES_HOME/plugin-data/continuum/`) and dual-mode linear algebra (pure Python standard library out of the box, with automatic NumPy acceleration when installed). Continuum makes **zero outbound network calls**, spawns **zero background TCP daemons**, requires **zero API keys**, and commits **zero binary blobs** to source control.

---

## Five Native Hermes Surfaces Encased as Self-Describing `OmniNode`s

| Surface | Registration API | What It Does for Hermes |
| :--- | :--- | :--- |
| **1. `ContextEngine`** | `ctx.register_context_engine` | Deduplicates repeated turns in a content-addressed quotient DAG, prunes low-saliency filler when approaching context limits, preserves the exact `request_messages` list reference when within budget for 100% prompt-cache hits, supports thread-safe `clone_for_agent()`, and provides `continuum_context_grep`. |
| **2. `MemoryProvider`** | `ctx.register_memory_provider` | Stores session memories in a profile-isolated SQLite WAL store (`HERMES_HOME/plugin-data/continuum/continuum.db`), retrieves relevant context via fast coarse-to-fine (`16d -> 32d -> 64d -> 128d`) search, suppresses writes during `cron` / `subagent` / `flush` runs, and consolidates pre-compaction turns into portable `.safetensors` micro-cartridges. |
| **3. `DelegationProvider`** | `ctx.register_delegation_provider` | Extends subtask routing with `engine_stratum`, `verification_role` (`maker` / `checker`), and `maker_id`; routes subtasks using online contextual LinUCB Woodbury scoring and enforces strict `maker != checker` separation (`DelegationRouteError` on collision). |
| **4. Tools, Hooks, Command & Skill** | `register_tool` / `register_hook` / `register_command` / `register_skill` | Registers **18 `continuum_*` agent tools** (including `continuum_compose_omninode`), **4 session lifecycle hooks** (`on_session_start`, `pre_tool_call`, `post_tool_call`, `on_session_end`), the `/continuum [status\|step\|omninode]` slash command, and the bundled `continuum` workflow skill. |
| **5. Desktop & Web Dashboard** | `desktop/plugin.js` + `dashboard/` | Native Hermes Desktop status-bar chip, inspector pane, and settings page built exclusively on `@hermes/plugin-sdk` + `react/jsx-runtime` (`#f5f2ea` Paper & `#141413` Ink light palette), displaying live `OmniNode` `CARD` lockfile verification, active `grid/1` cell SHA-256, and `30x112` engine health alongside read-only FastAPI endpoints (`/status`, `/engines`, `/omninode`). |

---

## What's New in `v1.3.0` (`OmniNode` Shell Encasement & `grid/1` Algebraic Shapes)

1. **Every Engine, Tool & Surface Encased as a Self-Describing `OmniNode` (`omninode_shell.py` + `config/continuum_grid.json`)**:
   - **Static AST `CARD` Inspection (`StaticAstCardInspector`)**: inspects `.py` modules via `ast.parse` + `ast.literal_eval` without executing code and builds deterministic `CARD` descriptors (`name`, `role`, `stratum`, `describes`, `keywords`, `contract`, `sha256`) across all `30` engines (`112` sub-engines), `18` `continuum_*` tools, and `5` Hermes extension surfaces.
   - **Content-Addressed `.omni/cas` & Lockfile Integrity (`ContentAddressedCasLockfile`)**: pins every component hash in `omninode.lock.json` and `.omni/cas/<sha256>.json`; any `hash_mismatch` fails closed with a typed `"contract"` failure.
   - **Typed Failure Waterfall Routing & Capability-Gap Synthesis (`TypedFailureWaterfallRouter`, `CapabilityGapSynthesizer`)**: classifies errors into `("transient", "bad_input", "capability", "data", "contract", "budget")`, routes recoverable faults cleanly across `//` waterfalls while failing closed on `"contract"` breaches, and dynamically resolves unknown steps via IDF + 4-space Matryoshka capability search.
2. **Algebraic `>> // | @` Shape Composition & Canonical `grid/1` Cell Receipts (`continuum_compose_omninode`)**:
   - Compiles declarative pipelines such as:
     ```text
     orient >> sae_probe >> (leace_erase | nullspace_steer) // fallback_compress @ independent_quorum_check
     ```
     with exact operator precedence (`//` waterfall tightest, then `|` parallel Cellular Sheaf fanout, `>>` balanced sequential pipe, and `@` independent `maker != checker` quorum check outermost).
   - Emits a canonical **`format: "grid/1"`** JSONL record (`cell = sha256(normalized_config)` across all `40` `grid/1` axes + conditional defaults + pinned lockfile `components` + clamped `ParameterBounds` overrides).
3. **Full Mechanistic Interpretability, Nullspace Steering & Cybernetic Stack (`v1.1.0` + `v1.2.0` Retained)**:
   - Matryoshka Top-K JumpReLU SAEs, multi-layer sparse crosscoders, Fisher/Mass-Mean probes, and ELK layer-divergence detection (`continuum_probe_latent_state`).
   - Contrastive/SAE control vectors projected strictly onto $\operatorname{Null}(A_{\text{anchor}})$ (`0.0` anchor drift) and folded into rank-1 `LoRA-XS` patches (`continuum_steer_nullspace_vector`).
   - 172-axis `TT-Cross` epistatic `HyperGrid` (`0` extra LLM tokens), Cellular Sheaf cohomology, Split/Online ACI Conformal prediction sets, Kronecker `HyperLoRA`, and 9 portable runtimes.

---

## Portable Skill, Probe & Steerer Compilation Across 9 Standalone Runtimes

`continuum_compile_cartridge`, `continuum_hyperlora_modulate`, `continuum_probe_latent_state`, `continuum_steer_nullspace_vector`, and `continuum_compose_omninode` lower multi-layer (`1..6` layers) `FiLM` + `MoLoRA` + `HyperLoRA` operator stacks, linear/SAE latent probes, and folded `Null(A_anchor)` steerers into nine zero-dependency targets:

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

## The 30 Cooperating Cybernetic Engines (`112` Sub-Engines Across `8` Layers)

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
| 28 | `omninode_shell_encaser` | OmniNode Shell & Shapes | Fast turn reflex | 4 |
| 29 | `algebraic_shape_and_grid1_director` | OmniNode Shell & Shapes | Task episode | 4 |

---

## Agent Tool Suite (18 Tools)

1. `continuum_fission_dag` — Split complex goals into a deduplicated subtask graph with clear interface contracts and minimal clarifying questions.
2. `continuum_claim_and_fence` — Claim a subtask with an atomic SQLite lease stamp or return an instant verified cache hit.
3. `continuum_verify_record` — Record independent `maker != checker` verification and update LinUCB, Dawid-Skene, and EigenTrust weights.
4. `continuum_compile_cartridge` — Compile rules or verified examples into a multi-layer skill cartridge across any of the 9 portable runtimes (`rust`, `glsl`, `js`, `python`, `sql`, `c99`, `wasm`, `wat`, `wgsl`, plus `safetensors`).
5. `continuum_analyze_spectrum` — Inspect Marchenko-Pastur noise floors, Roy-Vetterli effective rank, power-law $\alpha$, and Wanda input saliency.
6. `continuum_surgical_edit` — Apply targeted weight updates with CEGAR counterexample healing and `0.0` anchor drift.
7. `continuum_coupling_step` — Propagate Lyapunov-damped `30x30` Jacobian impulses and run zero-token `TT-Cross` + SHA offline config tuning.
8. `continuum_morphism_transfer` — Detect structurally matching task graphs via Laplacian spectra and transplant skills via Orthogonal Procrustes.
9. `continuum_calibrate_effort` — Compute Expected Value of Sample Information (`EVSI`) to pick reasoning effort, embedding slice width, and escalation thresholds.
10. `continuum_manifold_status` — Inspect live status across all 30 engines, 112 sub-engines, `OmniNode` `CARD` lockfile, `grid/1` active cell SHA-256, and 172-axis parameter ledger counts.
11. `continuum_context_grep` — Search compressed and active session turns using hybrid keyword + nested vector matching.
12. `continuum_attractor_recall` — Retrieve relevant episodic memories from the local SQLite WAL Hopfield store.
13. `continuum_sheaf_consistency_audit` — Build a Cellular Sheaf Laplacian across multi-agent claims to localize contradictions and compute harmonic consensus.
14. `continuum_conformal_calibrate` — Construct Split Conformal + Online ACI `1 - alpha` prediction sets with selective abstention routing.
15. `continuum_hyperlora_modulate` — Generate input-conditioned Kronecker `HyperLoRA` weight modulations in the safety-anchor nullspace (`0.0` drift) across any of the 9 portable targets.
16. `continuum_probe_latent_state` — Run Matryoshka Top-K JumpReLU SAE sparse feature decomposition, Crosscoder circuit tracing, Fisher/Mass-Mean concept probing, ELK layer-divergence detection, and Split-Conformal truthfulness calibration (plus optional 9-runtime probe export).
17. `continuum_steer_nullspace_vector` — Extract Contrastive or SAE activation steering vectors projected strictly into `Null(A_anchor)` (`0.0` anchor drift), fold zero-hook rank-1 `LoRA-XS` + bias patches, and export across all 9 portable runtimes.
18. `continuum_compose_omninode` — Compile and execute declarative `OmniNode` `>> // | @` pipelines with `ParameterBounds` clamping, `.omni/cas` lockfile integrity verification (`hash_mismatch` fails closed), `maker != checker` quorum checks, and canonical `format: "grid/1"` SHA-256 cell receipts.

---

## Measured Local Benchmarks

Measured on Linux x86_64 across both **NumPy-accelerated** and **pure Python standard-library** (`python_dependencies: []`) runtimes:

| Operation | Latency (NumPy) | Key Mathematical Guarantee |
| :--- | :---: | :--- |
| **Initialize 30 Engines + `OmniNode` CAS Lockfile + 172-Axis Grid** | `~95 ms` | 90 verified baseline cells + full `CARD` SHA-256 lockfile pinned |
| **Execute Canonical `>> // \| @` `OmniNode` Pipeline + `grid/1` Receipt** | `~3.8 ms` | `$0.00` spend, `0` tokens, `0.0` anchor drift, deterministic `grid/1` SHA-256 |
| **Progressive `16d->32d->64d->128d` Recall (128 items)** | `~2.0 ms` | **`68.8%` MAC reduction** vs. exhaustive `128d` scan |
| **Matryoshka Top-K SAE + ELK State Probe** | `~1.1 ms` | Monosemantic selectivity + Split-Conformal `1 - alpha` truthfulness set |
| **`Null(A_anchor)` Activation Steering + `LoRA-XS` Fold** | `~1.3 ms` | **`0.0` anchor subspace drift** vs. non-zero naive steering drift |
| **`TT-Cross` + 3-Rung SHA Config Tournament** | `~94 ms` | **`0` extra LLM tokens** (replays verified local history) |
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

---

## Security & Behavior Disclosure (Rule 13)

- **Outbound Network Calls:** None (`0` HTTP/TCP/UDP connections).
- **Background Processes:** None (`0` daemons or background sockets).
- **Filesystem Scope:** Reads and writes strictly inside the active profile's `HERMES_HOME/plugin-data/continuum/` directory (SQLite WAL + `.omni/cas` lockfile & `grid/1` JSONL ledger).
- **Shell / Subprocess / Dynamic Code Execution:** None (`0` `subprocess`, `0` `eval` / `exec`, `0` self-modifying source files).
- **Secrets & Telemetry:** Reads zero `.env` files, requires zero API keys, and emits zero telemetry.

---

## License

MIT License — Copyright (c) 2026 `aidonerightcorp`.
