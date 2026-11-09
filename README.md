# Continuum — Unified Self-Improving Agent Engine (`v1.0.0`)

**Smart context compression, local episodic memory, verified subtask routing, adaptive configuration tuning, and 7-target skill compilation for Hermes Agent (`>=0.23.0`).**

![Continuum Architecture Overview](docs/continuum-overview.png)

**Continuum** gives Hermes Agent a unified, self-improving runtime that gets faster, sharper, and more reliable as you work. Instead of bolting together disconnected tools, Continuum connects **five native Hermes extension surfaces** over one shared local engine: **20 cooperating functional engines** (`72` sub-engines across `5` layers), **shared 4-space nested embeddings** (`16d / 32d / 64d / 128d`), a **100-parameter adaptive configuration grid** that tunes itself offline using **zero extra LLM tokens**, and a **portable skill compiler** targeting seven standalone runtimes.

Everything runs **100% locally in-process** backed by profile-scoped SQLite WAL storage and dual-mode linear algebra (pure Python standard library out of the box, with automatic NumPy acceleration when installed). Continuum makes **zero outbound network calls**, spawns **zero background TCP daemons**, requires **zero API keys**, and commits **zero binary blobs** to source control.

---

## Five Native Hermes Surfaces in One Plugin

| Surface | Registration API | What It Does for Hermes |
| :--- | :--- | :--- |
| **1. `ContextEngine`** | `ctx.register_context_engine` | Deduplicates repeated turns in a content-addressed graph, prunes low-saliency filler when approaching context limits, preserves the exact `request_messages` list reference when within budget for 100% prompt-cache hits, supports thread-safe `clone_for_agent()`, and provides `continuum_context_grep`. |
| **2. `MemoryProvider`** | `ctx.register_memory_provider` | Stores session memories in a profile-isolated SQLite WAL store (`HERMES_HOME/plugin-data/continuum/continuum.db`), retrieves relevant context via fast coarse-to-fine (`16d -> 32d -> 64d -> 128d`) search, suppresses writes during `cron` / `subagent` / `flush` runs, and consolidates pre-compaction turns into portable `.safetensors` micro-cartridges. |
| **3. `DelegationProvider`** | `ctx.register_delegation_provider` | Extends subtask routing with `engine_stratum`, `verification_role` (`maker` / `checker`), and `maker_id`; routes subtasks using online contextual uncertainty scoring and enforces strict `maker != checker` separation (`DelegationRouteError` on collision). |
| **4. Tools, Hooks, Command & Skill** | `register_tool` / `register_hook` / `register_command` / `register_skill` | Registers **12 `continuum_*` agent tools**, **4 session lifecycle hooks** (`on_session_start`, `pre_tool_call`, `post_tool_call`, `on_session_end`), the `/continuum [status\|step]` slash command, and the bundled `continuum` workflow skill. |
| **5. Desktop & Web Dashboard** | `desktop/plugin.js` + `dashboard/` | Native Hermes Desktop status-bar chip, inspector pane, and settings page built exclusively on `@hermes/plugin-sdk` + `react/jsx-runtime` (`#f5f2ea` Paper & `#141413` Ink light palette), paired with read-only SQLite WAL FastAPI telemetry endpoints (`/status`, `/engines`). |

---

## How Continuum Self-Improves Without Extra Token Spend

### 1. Shared 4-Space Nested Embeddings (`16d / 32d / 64d / 128d`)
Tasks (`task2vec`), engine configurations (`config2vec`), compiled skill weights (`weight2vec`), and conversation memories (`trace2vec`) all project into one shared nested vector geometry where every prefix slice (`16d`, `32d`, `64d`, `128d`) is independently unit-normalized. Retrieval and routing scan `16d` prefixes first and refine only top candidates up to `128d`, cutting multiply-accumulate work by **`68.8%`**.

### 2. Adaptive 100-Parameter Grid & Zero-Token Offline Exploration
Rather than relying on fixed magic numbers or brittle static presets:
- **Continuous Per-Task Config Blending**: for any incoming task vector, Continuum smoothly blends verified parameter cells across its `100` tunable dimensions (`5` parameters × `20` engines) while clamping every value to safe schema bounds.
- **Zero-Token Candidate Testing**: when spare local CPU headroom is available (`compute_headroom_ratio`), Continuum generates candidate configurations using pairwise covering arrays plus low-discrepancy probes and filters them through a **3-stage tournament** (`16d` surrogate score -> `64d` replay against saved `maker != checker` records -> `128d` full check) at **`0` extra LLM tokens**.
- **Automatic Range Expansion**: when verified winning configurations consistently hit an expandable parameter boundary, Continuum automatically widens that parameter range into a new grid generation.

### 3. Independent `Maker != Checker` Task Leasing & Verification
Complex goals decompose into content-addressed subtask graphs (`continuum_fission_dag`) guarded by atomic compare-and-swap SQLite leases (`continuum_claim_and_fence`). Subtasks only transition to reusable verified cache entries (`continuum_verify_record`) when an independent checker (`maker_id != checker_id`) validates the output, updating actor reliability and routing weights in one step.

### 4. Portable Skill Compilation (7 Standalone Targets) & Zero-Drift Updates
`continuum_compile_cartridge` turns structured rules or verified examples into compact multi-layer (`1..6` layers) residual adapter cartridges with confidence-margin early exit, exporting directly to seven zero-dependency formats:
1. `safetensors` — IEEE-754 `float32` weight buffer + metadata header (base64-encoded)
2. `wasm_base64` — spec-compliant WebAssembly 1.0 binary generated dynamically in memory (zero `.wasm` files stored on disk)
3. `wat` — human-readable WebAssembly Text module
4. `wgsl` — WebGPU `@compute` shader kernel
5. `c99` — self-contained ANSI C99 evaluation source
6. `sql` — pure ANSI SQL CTE matrix-vector query
7. `python` — standalone zero-dependency Python module

When refining existing engine weights (`continuum_surgical_edit`), updates project onto the orthogonal complement of protected anchor behaviors with automatic counterexample refinement so existing verified behaviors experience **`0.0` anchor drift**.

---

## The 20 Cooperating Functional Engines (`72` Sub-Engines)

| # | Engine ID | Functional Layer | Adaptation Cadence | Sub-Engines |
| :--- | :--- | :--- | :--- | :---: |
| 0 | `quotient_context_dag_engine` | Context & Memory | Fast turn reflex | 4 |
| 1 | `wanda_saliency_pruning_engine` | Context & Memory | Fast turn reflex | 4 |
| 2 | `hopfield_attractor_memory_engine` | Context & Memory | Session consolidation | 4 |
| 3 | `matryoshka_manifold_encoder_engine` | Context & Memory | Fast turn reflex | 4 |
| 4 | `ridge_polytope_compiler_engine` | Skill Compilation | Session consolidation | 4 |
| 5 | `multilayer_film_molora_stack_engine` | Skill Compilation | Task episode | 4 |
| 6 | `svd_nullspace_surgical_editor_engine` | Skill Compilation | Task episode | 4 |
| 7 | `portable_target_lowering_engine` | Skill Compilation | Session consolidation | 4 |
| 8 | `contextual_linucb_router_engine` | Routing & Trust | Task episode | 4 |
| 9 | `cas_fission_lease_fencing_engine` | Routing & Trust | Fast turn reflex | 3 |
| 10 | `dawid_skene_quorum_adjudicator_engine` | Routing & Trust | Task episode | 3 |
| 11 | `eigentrust_reputation_flow_engine` | Routing & Trust | Session consolidation | 3 |
| 12 | `rmt_marchenko_pastur_spectral_engine` | Adaptive Tuning | Session consolidation | 4 |
| 13 | `fluid_hypergrid_interpolator_engine` | Adaptive Tuning | Fast turn reflex | 4 |
| 14 | `evsi_effort_cascade_calibrator_engine` | Adaptive Tuning | Fast turn reflex | 3 |
| 15 | `operation_cost_thermodynamic_engine` | Adaptive Tuning | Task episode | 3 |
| 16 | `inter_engine_coupling_and_jacobian_bus` | Engine Coupling | Cross-task transfer | 4 |
| 17 | `homeostatic_meta_controller_engine` | Engine Coupling | Grid range evolution | 3 |
| 18 | `structural_morphism_and_transfer_engine` | Engine Coupling | Cross-task transfer | 3 |
| 19 | `human_attention_and_escalation_engine` | Engine Coupling | Task episode | 3 |

---

## Agent Tool Suite (12 Tools)

1. `continuum_fission_dag` — Split complex goals into a deduplicated subtask graph with clear inputs/outputs and minimal clarifying questions.
2. `continuum_claim_and_fence` — Claim a subtask with an atomic SQLite lease stamp or return an instant verified cache hit.
3. `continuum_verify_record` — Record independent `maker != checker` verification and update routing & trust weights.
4. `continuum_compile_cartridge` — Compile rules or verified examples into a multi-layer skill cartridge across any of the 7 portable targets (`safetensors`, `wasm_base64`, `wat`, `wgsl`, `c99`, `sql`, `python`).
5. `continuum_analyze_spectrum` — Inspect engine capacity utilization, effective rank, and input feature saliency.
6. `continuum_surgical_edit` — Apply targeted weight updates that preserve protected baseline behaviors with `0.0` anchor drift.
7. `continuum_coupling_step` — Propagate stability-bounded cross-engine adjustments and run zero-token offline config tuning.
8. `continuum_morphism_transfer` — Detect structurally matching task graphs and transfer compiled skills across domains.
9. `continuum_calibrate_effort` — Recommend reasoning effort (`low` / `medium` / `high`), embedding slice width, and human review thresholds for task stakes.
10. `continuum_manifold_status` — Inspect live status across all 20 engines, 72 sub-engines, parameter ledger counts, and compute savings.
11. `continuum_context_grep` — Search compressed and active session turns using hybrid keyword + nested vector matching.
12. `continuum_attractor_recall` — Retrieve relevant episodic memories from the local SQLite WAL memory store.

---

## Measured Local Benchmarks

Measured on Linux x86_64 across both **NumPy-accelerated** (`Python 3.14.4`) and **pure Python standard-library** (`Python 3.11.15`, `python_dependencies: []`) runtimes:

| Operation | NumPy Accelerated | Pure Python Stdlib | Key Guarantee / Savings |
| :--- | :---: | :---: | :--- |
| **Initialize 20 Engines + 100-Param Grid** | `64.1 ms` | `149.8 ms` | 60 verified baseline config cells indexed at startup |
| **Progressive `16d->32d->64d->128d` Recall (128 items)** | `2.00 ms` | `21.35 ms` | **`68.8%` compute reduction** vs. full `128d` scan |
| **3-Stage Offline Config Tournament (`36 -> 12 -> 4`)** | `84.2 ms` | `263.5 ms` | **`0` extra LLM tokens** (replays verified local history) |
| **Protected Weight Update Across All 20 Engines** | `2.75 ms` | `10.92 ms` | **`0.0` anchor drift** on protected behaviors |
| **Compile 5-Layer Cartridge & Export All 7 Targets** | `3.01 ms` | `12.43 ms` | **`80.0%` layer savings** (`1/5` layers) on confident early exit |

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
- `compute_headroom_ratio` (`float`, default `0.6`): fraction of spare local CPU budget used to test candidate configurations against saved history using zero extra LLM tokens.
- `default_embedding_dim` (`int`, default `64`): default nested embedding slice width (`16`, `32`, `64`, or `128`) for fast retrieval and routing.
- `enable_idle_exploration` (`bool`, default `true`): automatically evaluate and promote better verified configurations at session end.

---

## Security, Privacy & Isolation

- **Zero Network Egress**: Continuum performs zero HTTP, WebSocket, DNS, or raw socket calls.
- **Zero Background Daemons**: all logic runs synchronously in-process; dashboard endpoints read SQLite WAL state in read-only mode (`mode=ro`).
- **Zero External Secrets**: requires zero API keys (`requires_env: []`) and never reads environment variables directly for credentials.
- **Zero Binary Blobs in Git**: WebAssembly binaries (`wasm_base64`) and `.safetensors` buffers are generated deterministically in memory when requested.
- **Whitelisted Public API Only**: imports exclusively documented Hermes v0.23.0+ extension interfaces (`agent.context_engine`, `agent.memory_provider`, `agent.delegation_provider`, `hermes_constants`, `plugins.plugin_storage`) with clean standalone fallbacks.

## License

MIT License — Copyright (c) 2026 `aidonerightcorp`.
