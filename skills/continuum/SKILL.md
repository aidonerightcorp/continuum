---
name: continuum
description: Plan multi-step work, audit multi-agent consistency with Cellular Sheaves, calibrate uncertainty via Conformal prediction, tune 160 settings automatically, and compile repeatable logic across 9 portable runtimes with Continuum.
---

# Continuum v1.1.0 Workflow Guide

Use the 15 `continuum_*` tools whenever you want structured multi-step execution, cohomological consistency checks, distribution-free uncertainty calibration, fast memory lookup, or zero-dependency compiled modules.

## Practical Step-by-Step Workflow

1. **Right-Size Effort & Uncertainty First (`continuum_calibrate_effort`, `continuum_conformal_calibrate`)**
   - Estimate task uncertainty and stakes to pick `low`, `medium`, or `high` reasoning effort, select the right embedding slice width (`16`, `32`, `64`, or `128`), and check whether asking a clarifying question saves wasted work.
   - Use `continuum_conformal_calibrate` (Split Conformal + Online Adaptive Conformal Inference) to construct finite-sample `1 - alpha` prediction sets and decide automatically between fast cartridge execution, multi-layer operator stacks, or full deliberation.

2. **Split Large Goals & Claim Exclusive Leases (`continuum_fission_dag`, `continuum_claim_and_fence`)**
   - Break a multi-step goal into ordered subtasks stored locally in profile-scoped SQLite WAL storage, each with an explicit interface contract.
   - Acquire an atomic compare-and-swap lease (`fencing_stamp`) before executing a subtask so parallel workers never overwrite each other; if a subtask was already verified by an independent reviewer, Continuum immediately returns the cached result.

3. **Audit Multi-Agent Consistency & Verify Independently (`continuum_sheaf_consistency_audit`, `continuum_verify_record`)**
   - Run `continuum_sheaf_consistency_audit` across multi-agent claims and communication edges to measure Cellular Sheaf coboundary obstruction energy, pinpoint contradictory edges, and compute harmonic consensus stalks.
   - Record reviews via `continuum_verify_record` with strict `maker_id != checker_id` separation to update routing models, reviewer confusion matrices, VCG truthful incentives, and EigenTrust scores.

4. **Compile & Modulate Skills Across 9 Portable Runtimes (`continuum_compile_cartridge`, `continuum_hyperlora_modulate`)**
   - Turn recurring decision tables or verified input-output examples into fast multi-layer modules in any of nine zero-dependency runtimes: `rust` (`#![no_std]` embedded/kernel FFI), `glsl` (`#version 300 es` WebGL2 shader), `js`, `python`, `sql`, `c99`, `wasm`, `wat`, or `wgsl` (plus `safetensors`).
   - Use `continuum_hyperlora_modulate` to generate input-conditioned Kronecker weight modulations projected strictly onto the safety-anchor SVD nullspace (`0.0` anchor drift) and export directly to any portable target.

5. **Inspect & Continuously Improve (`continuum_analyze_spectrum`, `continuum_surgical_edit`, `continuum_coupling_step`, `continuum_morphism_transfer`, `continuum_manifold_status`, `continuum_context_grep`, `continuum_attractor_recall`)**
   - Check weight health and feature sensitivity, apply zero-drift surgical edits, transfer structure across related workflows, recall past session attractors, search deduplicated context history, and explore epistatic `TT-Cross` configurations offline using zero extra LLM tokens.
