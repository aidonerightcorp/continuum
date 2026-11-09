---
name: continuum
description: Plan multi-step work, enforce independent review, recall past episodes, tune settings automatically, and compile repeatable logic into fast portable modules with Continuum.
---

# Continuum Workflow Guide

Use the `continuum_*` tools whenever you want structured multi-step execution, independent quality checks, fast memory lookup, or reusable compiled modules.

## Practical Step-by-Step Workflow

1. **Right-Size Effort First (`continuum_calibrate_effort`)**
   - Estimate task uncertainty and stakes to pick `low`, `medium`, or `high` reasoning effort, choose the right embedding slice width (`16`, `32`, `64`, or `128`), and check whether asking a clarifying question saves wasted work.

2. **Split Large Goals Into Clear Subtasks (`continuum_fission_dag`)**
   - Break a multi-step goal into ordered subtasks stored locally in profile-scoped SQLite WAL storage, each with an explicit interface contract.

3. **Claim Exclusive Task Leases (`continuum_claim_and_fence`)**
   - Acquire an atomic compare-and-swap lease (`fencing_stamp`) before executing a subtask so parallel workers never overwrite each other.
   - If a subtask was already verified by an independent reviewer, Continuum immediately returns the cached result.

4. **Verify Work Independently (`continuum_verify_record`)**
   - Always keep `maker_id != checker_id` so the reviewer is distinct from the author.
   - Recording a review automatically updates Continuum's routing models, reviewer reliability estimates, and actor trust scores.

5. **Compile Repeatable Rules Into Fast Modules (`continuum_compile_cartridge`)**
   - Turn recurring decision tables or verified input-output examples into fast multi-layer modules with early exit in any of seven zero-dependency formats: `safetensors`, `wasm_base64`, `wat`, `wgsl`, `c99`, `sql`, or `python`.

6. **Inspect & Continuously Improve (`continuum_analyze_spectrum`, `continuum_surgical_edit`, `continuum_coupling_step`, `continuum_morphism_transfer`, `continuum_manifold_status`)**
   - Check weight health and feature sensitivity, apply zero-drift updates that protect existing behavior, transfer structure across related workflows, and test new configurations offline against saved history using zero extra LLM tokens.
