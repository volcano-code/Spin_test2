# W6 upstream candidate project library — v0.1

Audit date: 2026-11-06  
Repository: volcano-code/Spin_test  
Branch: research/neuralmoments-w6-pcraft-20261019  
Parent W6 head: 1365be784e0e0ff04e7b19cbc7c86206d2499f8f

## Purpose

This is a pinned source-code candidate registry for the Spin-specific P-CRAFT U-FNO / neural moment operator. It is not a claim that all projects have been integrated. It distinguishes upstream code, method-only references, license holds, external benchmarks and existing W6 adapters.

The library contains **24 entries**:
- 12 repositories with a clear permissive/custom-permissive license suitable for a controlled direct-reuse review;
- 8 entries whose root license or permission must be resolved before copying;
- 1 GPL-3.0 quadrature package restricted to an isolated benchmark role;
- 3 paper/repository watchlist entries without verified reusable code.

## Current W6 baseline and external dependencies

- W6 current head: 1365be784e0e0ff04e7b19cbc7c86206d2499f8f (R4.2 box-event classification).
- R4.2 classifies exact-rational quadratic ranges and zero-set events on one axis-aligned box. It intentionally withholds J_D and training at every event status.
- R4.2 commit reports 9 standard-library tests and 36 seeded analytic comparisons. The full legacy suite reports 302 passed, 1 skipped and 1 failed; the M1 encoder density-gradient smoke failure was also present on the untouched parent 057087a7ae69e99f373b4ae0d5ff667fb39d15bb.
- W2 current manifest is M22 diagnostic-only: one native teacher run is recorded, but training_samples remains zero and J_D remains null/excluded. This is not a teacher-training release.

## Highest-value roles

1. **Exact geometry/integration reference:** Algoim; keep Cherchi as the authoritative Spin teacher.
2. **Learned quadrature:** QuadrANN (MIT code) and NQR (SIGGRAPH 2026 method/code reference, but root license not found).
3. **Geometry-conditioned operators:** NeuralOperator GINO/UQNO, Transolver++, PhysicsNeMo GeoTransolver, DoMINO/FIGConvNet.
4. **Large/sparse geometry:** fVDB and Kaolin; GPU validation required.
5. **Unstructured-mesh and equivariant baselines:** GAOT/GAOT-3D, GINOT, PhysGTO, EqGINO, e3nn and PyG. Several source repositories lack a root license and are therefore reference-only until rights are confirmed.
6. **Teacher-budget control:** PhysicsNeMo active-learning utilities and the 2026 active-operator-learning paper.
7. **Differentiable optimization:** TorchOpt as an optional unrolled/implicit-differentiation baseline, never as a replacement for the certified native Newton path.

## Files

- catalog.json — full machine-readable registry, pinned commits, file-level Git blob SHAs, license state and W6 status.
- ADOPTION_MATRIX.md — proposed adoption order and cloud/local execution.
- INTEGRATION_PLAN.md — architecture, original innovation hypotheses, interfaces and acceptance gates.
- LICENSE_AUDIT.md — code reuse and license restrictions.

## Update rule

When a candidate changes, add a new audit entry with the new upstream commit and exact source blob SHA. Do not silently update old pins. A candidate becomes INTEGRATED only after source import, license review, tests, matched ablation and an explicit W6 commit.

## M80 current adoption view

The file above remains the historical W6 registry snapshot and must not be rewritten as if it were a current integration report. For the current expanded review, see:

- M80_CANDIDATE_ADOPTION_MATRIX.md — 31 expanded candidates plus 14 extended research sources.
- M80_REFACTORING_PLAN.md — M80.4–M80.10 remediation roadmap.
- M80_ARCHITECTURE_CONTRACT.md — current and target model/physical/solver interfaces.

Important clarification: the current Spin_test2 tree contains an internal refined_quadrature.py numerical path. It is not the upstream QuadrANN adapter. The QuadrANN adapter status remains "re-audit required".
