# M80 Candidate Adoption Matrix — 31 Expanded Candidates + 14 Extended Sources

Date: 2026-11-13  
Authoritative registry: candidate_projects/catalog.json (24 historical entries, audit date 2026-11-06).  
This matrix expands composite registry entries into individual model/utility candidates. It does not rewrite the historical W6 catalog.

## Decision vocabulary

- **Integrated — limited:** code path exists, but quality promotion is not established.
- **Research next:** implement or reproduce a controlled adapter/ablation.
- **Conditional:** only if a measured bottleneck or failure mode justifies it.
- **Reference / benchmark:** use for comparison or independent verification, not as a production network dependency.
- **Hold:** no source/permission/identity clearance; do not copy code, weights or assets.

## A. The 31 expanded candidates

| # | Candidate | Current M80 status | Decision | Intended role / promotion condition | Source / license |
|---:|---|---|---|---|---|
| 1 | P-CRAFT PINO / U-FNO | M80.1 single-scale baseline; M80.4 shared 32/64/128 path implemented | **Integrated — core** | Permanent global geometry backbone. Preserve single-scale baseline and promote multiscale only after GPU/quality ablation. | Internal source reference volcano-code/test@24eda3e9b3f87ba16251a9025761986d6cfe3079; historical catalog found no root LICENSE. Verify internal redistribution rights. |
| 2 | GINO | Not integrated | **Research next** | Geometry-to-latent-grid baseline against M72 RLE and P-CRAFT multi-scale mapping. Same ten-moment contract and data budget. | neuraloperator/neuraloperator@a6648eeeb60fe0837f2d7d021517a945a159a0c8; MIT. |
| 3 | FNOGNO | Not integrated | **Research next — high** | Strong hybrid geometry/Fourier operator comparator to the hand-built P-CRAFT + GNO stack. | Same NeuralOperator commit; MIT. |
| 4 | UQNO | Not integrated | **Research next — high** | Uncertainty-aware moment/Jacobian predictor; calibrate risk-coverage and false-accept rate. UQ never overrides event validity. | Same NeuralOperator commit; MIT. |
| 5 | Transolver++ | Spin-specific mass-aware slicing implemented in M80.3 | **Integrated — limited** | Keep optional. Four-way Large resource preflight passed; quality benefit still requires matched training and downstream solver tests. | thuml/Transolver_plus@d5a23bc734a0ebac56384cf72049a26af9673452; MIT. Current code is a reimplementation/adaptation, not an upstream code or weight port. |
| 6 | Neural-Solver-Library | Not integrated | **Reference / benchmark** | Select representative UPT/GNOT/FNO/U-FNO/Transolver baselines; do not import the whole library or duplicate architectures. | thuml/Neural-Solver-Library@db942b72c6b3972bad217300e3cbdb8004a1a80b; MIT. |
| 7 | GeoTransolver | Not integrated | **Research next — high** | Geometry-aware latent embedding and physical attention comparator to current Transolver++ adaptation. Only one of GeoTransolver/DoMINO/FIGConvNet should be added at a time. | NVIDIA/physicsnemo@22bb35ec6e700c2f8d6088ddbf88af12b157e5f2; Apache-2.0. |
| 8 | DoMINO | Not integrated | **Conditional** | Multi-scale point/grid geometry alternative if current RLE-to-P-CRAFT mapping fails on irregular mesh families. | Same PhysicsNeMo commit; Apache-2.0. |
| 9 | FIGConvNet | Not integrated | **Conditional** | Alternative multi-scale geometric feature encoder; compare against DoMINO rather than stacking both. | Same PhysicsNeMo commit; Apache-2.0. |
| 10 | PhysicsNeMo Active Learning strategies | Not integrated | **Research next — high** | Reuse strategy patterns for teacher-budget-aware query selection after the formal teacher adapter is admitted. | Same PhysicsNeMo commit; Apache-2.0. |
| 11 | PhysicsNeMo GP utilities | Not integrated | **Research next** | GP baseline for uncertainty and acquisition; compare against ensembles/UQNO. | Same PhysicsNeMo commit; Apache-2.0. |
| 12 | AeroJEPA | Not integrated; exact source file/weights not pinned in the historical file list | **Conditional / source audit** | Self-supervised geometry pretraining only if geometry corpus size justifies it and downstream moment transfer is demonstrated. | PhysicsNeMo repository root Apache-2.0; exact module, data and checkpoint provenance must be pinned before reuse. |
| 13 | GAOT | Method inspected; no root LICENSE found | **Hold** | Paper-level comparator only until permission is obtained. | camlab-ethz/GAOT@549c5a5f7113e23ba5e91469f2f8bbb1567fae46; no root LICENSE in historical audit. |
| 14 | GAOT-3D | Method inspected; no root LICENSE found | **Hold** | Large-3D MAGNO/integral-transform comparator only after rights clearance. | Shizheng-Wen/GAOT-3D@67886040401687eed9e63fb2a2bda2c1d0784c96; no root LICENSE in historical audit. |
| 15 | GINOT | Method inspected; no root LICENSE found | **Hold** | Point-cloud grouping/cross-attention comparator only after rights and asset terms are cleared. | QibangLiu/GINOT@552674f978b3a317bf5bfb50ed81efe3b62ba5a1; no root LICENSE in historical audit. |
| 16 | PhysGTO | Method inspected; no root LICENSE found | **Hold** | Manifold-aware mesh operator for thin-wall/near-layer failure studies only after permission. | pengwei07/PhysGTO@fc2dbc54e6b8205252534afddc20ca1fd61b8b4b; no root LICENSE in historical audit. |
| 17 | EqGINO | Method inspected; no root LICENSE found; repository has modified NeuralOperator files | **Hold** | SO(3)/E(3) equivariance comparator only after file-level license and modification audit. | sung-won-kim/EqGINO@8465697cad54b7875dff4b5b65db5cdb71a332a6; no root LICENSE in historical audit. |
| 18 | UNOP | README MIT badge, no root LICENSE found | **Hold / method reference** | Study LIPE/GALA/curvature gating as ideas; no source copying until license text is confirmed. | chengxinrui/UNOP@7e3f4e9b332bc9d6207d74ccd3d52930386b24fc; license unresolved. |
| 19 | QuadrANN | Prior W6 adapter reported; production status requires re-audit | **Research next — high** | Nonnegative learned quadrature head. Compare fixed numerical quadrature, learned weights and exact-corrected learned weights on ten moments and Jacobians. | kesmarag/QuadrANN@111dd611aaeac1fb98c76385706b003ea8ef94be; MIT. |
| 20 | NQR | Paper/source inspected; no root LICENSE found | **Hold / method reference** | Adaptive sampling and neural quadrature concepts; do not vendor code/shaders/assets until permission. | SuikaSibyl/nqr at historical audited snapshot; no root LICENSE. |
| 21 | Algoim | R4 adapter exists; official build not verified | **Reference / exact numerical baseline** | Independent implicit-geometry quadrature comparator. Cherchi remains the authoritative Spin teacher. | Pinned Algoim source in historical catalog; custom BSD-style license with enhancement-grant clause. Preserve full notice. |
| 22 | fVDB | Not integrated | **Conditional** | Sparse SDF/voxel/ray backend only if 512+/1024+ geometry preprocessing, storage or query is measured as a bottleneck. | NVIDIA fVDB pinned source; Apache-2.0. Local CUDA build required. |
| 23 | NVIDIA Kaolin | Not integrated | **Reference / preprocessing tool** | Mesh/SPC conversion and SDF preprocessing; compare with fVDB, not a neural operator. | NVIDIA Kaolin pinned source; Apache-2.0. |
| 24 | PyTorch3D | Not integrated | **Reference / preprocessing tool** | Surface sampling, mesh distance and normals where needed. | PyTorch3D pinned source; BSD-style. |
| 25 | PyTorch Geometric | Dependency candidate, not a Spin operator | **Engineering dependency** | Use declared dependency for graph layers when needed; do not vendor as a research contribution. | PyG pinned source; MIT. |
| 26 | e3nn | Not integrated | **Research next — conditional** | Strict SO(3)/E(3) feature equivariance ablation after current arbitrary-rotation tests identify a real failure. | e3nn/e3nn pinned source; MIT. |
| 27 | TorchOpt | Not integrated | **Conditional / optimizer baseline** | Implicit/unrolled differentiation experiment only. Native Spin-It Faster Newton remains final solver. | TorchOpt pinned source; Apache-2.0. |
| 28 | torchquad | External baseline only | **Isolated benchmark** | Integration/runtime comparison in a separate process/environment. Do not vendor/link into the core without a separate compatibility review. | torchquad pinned source; GPL-3.0. |
| 29 | Active Operator Learning with Predictive UQ (JCP 2026) | Paper only; no verified code repository in historical audit | **Research next — high** | Implement independently from the paper to allocate expensive Cherchi queries by uncertainty, event margin and expected solver-risk reduction. | Paper reference; no verified code repository/license. |
| 30 | Transolver-3 | Announced but separate official code repository not verified | **Watchlist** | Revisit only when official code, commit and license are available and large-geometry bottleneck is measured. | No verified repository/commit/license. |
| 31 | Opt-RNO | Repository identity not verified | **Hold** | Do not cite as an available implementation until the exact official project is identified. | No verified repository/commit/license. |

## B. Fourteen extended research sources outside the 24-entry registry

These are tracked separately; they are not additional rows in the historical candidate catalog.

| # | Source | Decision | Role | Source/permission status |
|---:|---|---|---|---|
| 32 | Spin-It (2014) | **Permanent baseline** | Original voxel task and material-removal baseline | Pin the exact paper/code/data version used in experiments. |
| 33 | Spin-It Faster (2024) | **Permanent baseline** | Quadric, mass moments and native Newton reference | Pin exact source commit and solver contract before bridge coding. |
| 34 | Cherchi | **Authoritative teacher** | Native Spin moment/derivative evaluation and exact fallback | Preserve executable/source hash, precision and integration settings. |
| 35 | DensiCrafter | **Offline design teacher** | Material-layout proposals; every proposal must pass Spin physics and exact evaluation | Separate paper/code/weight/data rights audit. |
| 36 | TRELLIS / TRELLIS.2 | **Geometry data source** | Diverse 3D geometry generation/expansion | Validate mesh/solid provenance, conversion and model/data licenses. |
| 37 | Ray-reps | **Independent numerical baseline** | Ray/segment representation and integration comparison | Exact paper/source pin and license audit required. |
| 38 | NIFE | **Independent numerical baseline** | Implicit cut-cell integration comparison | Exact paper/source pin and license audit required. |
| 39 | PFEM | **Solver-initialization research** | Warm-start/initialization idea | Exact expansion, paper, implementation and license must be verified. |
| 40 | Neural-Initialized Newton | **Research next** | Learn q0, then retain native Newton refinement | Define task-specific objective and compare with native initial guess. |
| 41 | HyPINO | **Conditional validation reference** | Physical/manufacturing solution validation | Exact paper/repository/rights audit required. |
| 42 | PI-CViT | **Conditional training comparator** | Physics-loss training comparison | Exact paper/repository/rights audit required. |
| 43 | TensorPILS | **Conditional training method** | Multi-loss gradient/conditioning treatment | Trigger only after measured gradient conflict/conditioning problem. |
| 44 | Stress-constrained differentiable GNN | **Future branch** | Stress/manufacturability extension | Requires declared loads, material model and FEM teacher; not current rigid-body Spin core. |
| 45 | PI-GINOT | **Future branch** | Physics-informed point-cloud operator comparison | Exact source, task relevance and license audit required. |

## C. Promotion policy

A candidate is called integrated only after:
1. Exact upstream repository, commit and relevant file SHA are recorded.
2. License and bundled dependency/data/weight notices are reviewed.
3. A Spin adapter obeys the frozen geometry, coefficient and ten-moment contract.
4. The same data split, query/teacher budget, seeds and measured compute budget are used.
5. Moment, Jacobian, event and physical invariants pass.
6. Cold/warm latency, peak memory, fallback and end-to-end costs are reported.
7. A reproducible report and status update are committed.

A project can be useful as a paper/method reference without its code being legally or technically integrated. Do not confuse those states.
